# ------------------------------------------------------------------------------
# PoseBridge HPE: RTMPose backbone with hierarchical spatial refinement that
# yields a single pose-anchored semantic vector aligned to CLIP text.
#
# Architecture
#   CSPNeXt backbone                -> stage2, stage3, stage4 features
#   RTMCCHead (SimCC)               -> pred_x, pred_y (pose regression loss)
#   SpatialRefineBlock cascade      -> shallow features refine deeper maps
#       num_scales=2: stage3 refines stage4
#       num_scales=3: stage2 refines stage3, refined_stage3 refines stage4
#   PoseConditionedPoolingHead      -> body-attention pooling -> (B, C4)
#   Final MLP                       -> pose_sem  (B, D) for CLIP alignment
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cspnext import CSPNeXt

_WIDEN_FACTORS = {'t': 0.375, 's': 0.5, 'm': 0.75, 'l': 1.0}
_DEEPEN_FACTORS = {'t': 0.167, 's': 0.33, 'm': 0.67, 'l': 1.0}


def _stage_channels(model_size: str) -> Tuple[int, int, int]:
    """Return (stage2_ch, stage3_ch, stage4_ch) for P5 arch."""
    w = _WIDEN_FACTORS[model_size.lower()]
    return int(256 * w), int(512 * w), int(1024 * w)


# ---------------------------------------------------------------------------
# Body attention from SimCC predictions (from PCMG)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_body_attention(
    pred_x: Tensor,
    pred_y: Tensor,
    target_h: int,
    target_w: int,
) -> Tensor:
    """Derive a body spatial attention map from SimCC probability distributions."""
    B, K, Wx = pred_x.shape
    Wy = pred_y.shape[-1]

    soft_x = F.softmax(pred_x, dim=-1)
    soft_y = F.softmax(pred_y, dim=-1)

    soft_x = F.adaptive_avg_pool1d(
        soft_x.reshape(B * K, 1, Wx), target_w
    ).reshape(B, K, target_w)
    soft_y = F.adaptive_avg_pool1d(
        soft_y.reshape(B * K, 1, Wy), target_h
    ).reshape(B, K, target_h)

    soft_x = soft_x / (soft_x.sum(dim=-1, keepdim=True) + 1e-8)
    soft_y = soft_y / (soft_y.sum(dim=-1, keepdim=True) + 1e-8)

    heatmaps = soft_y.unsqueeze(-1) * soft_x.unsqueeze(-2)
    body_attn = heatmaps.mean(dim=1, keepdim=True)

    bmax = body_attn.flatten(1).max(dim=1)[0].view(B, 1, 1, 1)
    body_attn = body_attn / (bmax + 1e-8)

    return body_attn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SpatialRefineBlock(nn.Module):
    """U-Net style: shallow feature spatially refines a deeper feature map.

    1. Channel-align shallow to match deep channels (1x1 conv)
    2. Bilinear downsample to match deep spatial resolution
    3. 3x3 conv refinement (zero-init for stable start)
    4. Residual addition: deep + gate_weight * refined_shallow

    Gate modes:
      fixed_gate=None  -> learnable gate (sigmoid of nn.Parameter, init ~0.05)
      fixed_gate=0.5   -> constant 0.5, no learnable gate parameter

    In both modes, the refine conv is zero-initialized so the shallow
    contribution starts near zero and grows as the conv learns.
    """

    def __init__(self, shallow_ch: int, deep_ch: int, dropout: float = 0.0,
                 fixed_gate: float = None):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(shallow_ch, deep_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(deep_ch),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(deep_ch, deep_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(deep_ch),
            nn.GELU(),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.fixed_gate = fixed_gate
        if fixed_gate is not None:
            self.register_buffer('_gate_value', torch.tensor(float(fixed_gate)))
        else:
            self.gate_logit = nn.Parameter(torch.tensor(-3.0))

        nn.init.zeros_(self.refine[0].weight)

    def gate_value(self) -> float:
        if self.fixed_gate is not None:
            return self.fixed_gate
        return torch.sigmoid(self.gate_logit).item()

    def forward(self, shallow: Tensor, deep: Tensor) -> Tensor:
        """
        Args:
            shallow: (B, C_shallow, H_s, W_s)
            deep:    (B, C_deep, H_d, W_d)
        Returns:
            refined: (B, C_deep, H_d, W_d)
        """
        s = self.align(shallow)
        s = F.interpolate(s, size=deep.shape[2:], mode='bilinear', align_corners=False)
        s = self.refine(s)
        s = self.dropout(s)
        if self.fixed_gate is not None:
            return deep + self._gate_value * s
        return deep + torch.sigmoid(self.gate_logit) * s


class PoseConditionedPoolingHead(nn.Module):
    """Pool (B, C, H, W) to (B, C) using body-attention weighted spatial pooling.

    Uses a body attention map from SimCC predictions to focus on body-relevant
    regions. Includes an attention refine conv (residual, zero-init) so the
    model can slightly adjust the raw body attention.
    """

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.attn_refine = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.attn_refine.weight)
        nn.init.constant_(self.attn_refine.bias, 0.0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, feats: Tensor, body_attn: Tensor) -> Tensor:
        """
        Args:
            feats:     (B, C, H, W)
            body_attn: (B, 1, H, W)
        Returns:
            (B, C)
        """
        attn = torch.sigmoid(self.attn_refine(body_attn))
        pooled = (feats * attn).sum(dim=[2, 3]) / (attn.sum(dim=[2, 3]) + 1e-6)
        return self.dropout(pooled)


class TextProjectionHead(nn.Module):
    """Project frozen CLIP text embeddings to pose-aligned space."""

    def __init__(self, clip_dim: int, proj_dim: int = 512, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.proj = nn.Linear(clip_dim, proj_dim, bias=True)
        nn.init.normal_(self.proj.weight, std=clip_dim ** -0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, text_feat: Tensor) -> Tensor:
        x = self.proj(text_feat)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


# ---------------------------------------------------------------------------
# RTMCCHead (SimCC head)
# ---------------------------------------------------------------------------

class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Scale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1., trainable: bool = True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


def _rope(x: Tensor, dim: int) -> Tensor:
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]
    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.reshape(
        torch.arange(total_len, dtype=torch.int, device=x.device), spatial_shape)
    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)
    half_size = shape[-1] // 2
    freq_seq = -torch.arange(half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000 ** -freq_seq
    sinusoid = position[..., None] * inv_freq[None, None, :]
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class RTMCCBlock(nn.Module):
    def __init__(self, num_token, in_token_dims, out_token_dims,
                 expansion_factor=2, s=128, eps=1e-5, dropout_rate=0.,
                 drop_path=0., attn_type='self-attn', act_fn='SiLU',
                 bias=False, use_rel_bias=False, pos_enc=False):
        super().__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.drop_path = _DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.e = int(in_token_dims * expansion_factor)

        if use_rel_bias:
            if attn_type == 'self-attn':
                self.w = nn.Parameter(torch.rand([2 * num_token - 1], dtype=torch.float))
            else:
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))

        self.o = nn.Linear(self.e, out_token_dims, bias=bias)
        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)

        self.ln = ScaleNorm(in_token_dims, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)
        self.act_fn = nn.SiLU(True)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False
        self.sqrt_s = math.sqrt(s)

    def _rel_pos_bias(self, seq_len, k_len=None):
        if self.attn_type == 'self-attn':
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            return t[..., r:-r]
        else:
            a = _rope(self.a.repeat(seq_len, 1), dim=0)
            b = _rope(self.b.repeat(k_len, 1), dim=0)
            return torch.bmm(a, b.permute(0, 2, 1))

    def _forward(self, inputs):
        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs
        x = self.ln(x)
        uv = self.act_fn(self.uv(x))
        if self.attn_type == 'self-attn':
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
            if self.pos_enc:
                base = _rope(base, dim=1)
            q, k = torch.unbind(base, dim=2)
        else:
            u, q = torch.split(uv, [self.e, self.s], dim=2)
            k = self.k_fc(k)
            v = self.v_fc(v)
            if self.pos_enc:
                q = _rope(q, 1)
                k = _rope(k, 1)
        qk = torch.bmm(q, k.permute(0, 2, 1))
        if self.use_rel_bias:
            bias = self._rel_pos_bias(
                q.size(1), k.size(1) if self.attn_type != 'self-attn' else None)
            qk = qk + bias[:, :q.size(1), :k.size(1)]
        kernel = torch.square(F.relu(qk / self.sqrt_s))
        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)
        x = u * torch.bmm(kernel, v)
        return self.o(x)

    def forward(self, x):
        if self.shortcut:
            res = x[0] if self.attn_type == 'cross-attn' else x
            return self.res_scale(res) + self.drop_path(self._forward(x))
        return self.drop_path(self._forward(x))


class RTMCCHead(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, in_featuremap_size,
                 simcc_split_ratio=2.0, final_layer_kernel_size=7, gau_cfg=None):
        super().__init__()
        if gau_cfg is None:
            gau_cfg = dict(hidden_dims=256, s=128, expansion_factor=2,
                           dropout_rate=0., drop_path=0., act_fn='SiLU',
                           use_rel_bias=False, pos_enc=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.simcc_split_ratio = simcc_split_ratio

        flatten_dims = in_featuremap_size[0] * in_featuremap_size[1]
        self.final_layer = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=final_layer_kernel_size, stride=1,
            padding=final_layer_kernel_size // 2)
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))
        W = int(input_size[0] * simcc_split_ratio)
        H = int(input_size[1] * simcc_split_ratio)
        self.gau = RTMCCBlock(
            out_channels, gau_cfg['hidden_dims'], gau_cfg['hidden_dims'],
            s=gau_cfg['s'], expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'], drop_path=gau_cfg['drop_path'],
            attn_type='self-attn', act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'], pos_enc=gau_cfg['pos_enc'])
        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
        x = feats[-1]
        x = self.final_layer(x)
        x = torch.flatten(x, 2)
        x = self.mlp(x)
        x = self.gau(x)
        return self.cls_x(x), self.cls_y(x)


# ---------------------------------------------------------------------------
# Main model: PoseBridgeHPE
# ---------------------------------------------------------------------------

class PoseBridgeHPE(nn.Module):
    """PoseBridge HPE: pose estimator with pose-anchored semantic head.

    Fuses shallow features into the deep feature map in the spatial domain
    via a SpatialRefineBlock cascade, then applies body-attention pooling
    once on the final enriched feature map to produce a single semantic
    vector aligned to CLIP text.

    Scale configurations (MODEL.ROUTE.NUM_SCALES):
      2 -> stage3 refines stage4
      3 -> stage2 refines stage3, then refined_stage3 refines stage4
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_size = cfg.MODEL.SIZE.lower()
        w = _WIDEN_FACTORS[model_size]
        d = _DEEPEN_FACTORS[model_size]
        all_channels = _stage_channels(model_size)
        stage2_ch, stage3_ch, stage4_ch = all_channels

        self.num_scales = cfg.MODEL.ROUTE.NUM_SCALES
        assert self.num_scales in (2, 3), \
            f'NUM_SCALES must be 2 or 3, got {self.num_scales}'

        self.backbone = CSPNeXt(
            arch=cfg.MODEL.BACKBONE.ARCH,
            deepen_factor=d,
            widen_factor=w,
            out_indices=(2, 3, 4),
            expand_ratio=cfg.MODEL.BACKBONE.EXPAND_RATIO,
            channel_attention=cfg.MODEL.BACKBONE.CHANNEL_ATTENTION,
        )

        input_size = (cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
        in_featuremap_size = (input_size[0] // 32, input_size[1] // 32)

        gau_cfg = dict(
            hidden_dims=cfg.MODEL.HEAD.GAU_HIDDEN_DIMS,
            s=cfg.MODEL.HEAD.GAU_S,
            expansion_factor=cfg.MODEL.HEAD.GAU_EXPANSION_FACTOR,
            dropout_rate=cfg.MODEL.HEAD.GAU_DROPOUT_RATE,
            drop_path=cfg.MODEL.HEAD.GAU_DROP_PATH,
            act_fn=cfg.MODEL.HEAD.GAU_ACT_FN,
            use_rel_bias=cfg.MODEL.HEAD.GAU_USE_REL_BIAS,
            pos_enc=cfg.MODEL.HEAD.GAU_POS_ENC,
        )
        self.head = RTMCCHead(
            in_channels=stage4_ch,
            out_channels=cfg.MODEL.NUM_JOINTS,
            input_size=input_size,
            in_featuremap_size=in_featuremap_size,
            simcc_split_ratio=cfg.MODEL.SIMCC.SPLIT_RATIO,
            final_layer_kernel_size=cfg.MODEL.HEAD.FINAL_LAYER_KERNEL_SIZE,
            gau_cfg=gau_cfg,
        )

        proj_dim = cfg.MODEL.ROUTE.PROJ_DIM
        dropout = cfg.MODEL.ROUTE.DROPOUT
        normalize = cfg.MODEL.ROUTE.NORMALIZE

        fixed_gate_cfg = getattr(cfg.MODEL.ROUTE, 'FIXED_GATE', -1.0)
        fixed_gate = None if fixed_gate_cfg < 0 else float(fixed_gate_cfg)
        self._fixed_gate = fixed_gate
        self.detach_shallow = getattr(cfg.MODEL.ROUTE, 'DETACH_SHALLOW', False)

        # U-Net style spatial refinement cascade
        #   num_scales=2: [SpatialRefineBlock(C3 -> C4)]
        #   num_scales=3: [SpatialRefineBlock(C2 -> C3), SpatialRefineBlock(C3 -> C4)]
        if self.num_scales == 2:
            self.refine_blocks = nn.ModuleList([
                SpatialRefineBlock(stage3_ch, stage4_ch, dropout=0.0,
                                   fixed_gate=fixed_gate),
            ])
        else:
            self.refine_blocks = nn.ModuleList([
                SpatialRefineBlock(stage2_ch, stage3_ch, dropout=0.0,
                                   fixed_gate=fixed_gate),
                SpatialRefineBlock(stage3_ch, stage4_ch, dropout=0.0,
                                   fixed_gate=fixed_gate),
            ])

        # Body-attention pooling on the final refined stage4 map
        self.pool = PoseConditionedPoolingHead(channels=stage4_ch, dropout=dropout)

        # Final projection: pooled (B, stage4_ch) -> (B, proj_dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(stage4_ch, proj_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.normalize = normalize

    def get_gate_values(self):
        """Return list of (name, gate_value) for logging."""
        if self.num_scales == 2:
            names = ['s3->s4']
        else:
            names = ['s2->s3', 's3->s4']
        return [(n, blk.gate_value())
                for n, blk in zip(names, self.refine_blocks)]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            pred_x: (B, K_joints, Wx) SimCC x predictions
            pred_y: (B, K_joints, Wy) SimCC y predictions
            pose_sem: (B, proj_dim) L2-normalized semantic embedding
        """
        feats = self.backbone(x)  # (stage2, stage3, stage4)

        # SimCC predictions from stage4
        pred_x, pred_y = self.head((feats[2],))

        # Progressive spatial refinement (U-Net decoder style)
        # When detach_shallow=True, shallow features are detached so CLIP
        # gradients only flow through stage4 -> backbone, not through
        # stage2/stage3 -> backbone. The refine block params still get gradients.
        if self.num_scales == 2:
            s3 = feats[1].detach() if self.detach_shallow else feats[1]
            refined = self.refine_blocks[0](s3, feats[2])
        else:
            s2 = feats[0].detach() if self.detach_shallow else feats[0]
            s3 = feats[1].detach() if self.detach_shallow else feats[1]
            refined_s3 = self.refine_blocks[0](s2, s3)
            refined = self.refine_blocks[1](refined_s3, feats[2])

        # Body-attention pooling on the refined map
        H, W = refined.shape[2], refined.shape[3]
        body_attn = compute_body_attention(pred_x, pred_y, H, W)
        pooled = self.pool(refined, body_attn)  # (B, C4)

        # Final projection
        pose_sem = self.final_mlp(pooled)
        if self.normalize:
            pose_sem = F.normalize(pose_sem, dim=-1, eps=1e-6)

        return pred_x, pred_y, pose_sem

    def init_weights(self, pretrained: str = ''):
        if pretrained:
            print(f'=> Loading pretrained model from {pretrained}')
            checkpoint = torch.load(pretrained, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            backbone_sd = {
                k.replace('backbone.', ''): v
                for k, v in state_dict.items()
                if k.startswith('backbone.')
            }
            if backbone_sd:
                missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
                print(f'=> Loaded backbone: missing={len(missing)}, unexpected={len(unexpected)}')
            else:
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                print(f'=> Loaded model: missing={len(missing)}, unexpected={len(unexpected)}')


__all__ = [
    'PoseBridgeHPE',
    'TextProjectionHead',
    'SpatialRefineBlock',
    'PoseConditionedPoolingHead',
    'compute_body_attention',
    'get_posebridge_hpe',
]


def get_posebridge_hpe(cfg, is_train: bool = True) -> PoseBridgeHPE:
    model = PoseBridgeHPE(cfg)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model
