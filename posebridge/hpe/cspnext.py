# ------------------------------------------------------------------------------
# CSPNeXt Backbone for RTMPose
# Based on mmpose/mmdet official implementation
# Implemented in pure PyTorch without mmcv/mmengine dependencies
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class SiLU(nn.Module):
    """SiLU activation (also known as Swish)."""
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size.
        stride (int): Stride. Default: 1.
        padding (int): Padding. Default: 0.
        groups (int): Groups. Default: 1.
        bias (bool): Whether to use bias. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias if norm_cfg is None else False
        )
        
        # Normalization
        if norm_cfg is not None:
            momentum = norm_cfg.get('momentum', 0.03)
            eps = norm_cfg.get('eps', 0.001)
            self.bn = nn.BatchNorm2d(out_channels, momentum=momentum, eps=eps)
        else:
            self.bn = None
        
        # Activation
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'SiLU')
            if act_type == 'SiLU':
                self.act = SiLU()
            elif act_type == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            elif act_type == 'LeakyReLU':
                self.act = nn.LeakyReLU(0.1, inplace=True)
            else:
                self.act = SiLU()
        else:
            self.act = None
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.
    
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size.
        stride (int): Stride. Default: 1.
        padding (int): Padding. Default: 0.
        norm_cfg (dict): Config dict for normalization layer.
        act_cfg (dict): Config dict for activation layer.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: Optional[dict] = dict(type='SiLU'),
        **kwargs
    ):
        super().__init__()
        
        # Depthwise conv
        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        
        # Pointwise conv
        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ChannelAttention(nn.Module):
    """Channel attention module.
    
    Args:
        channels (int): The input channels.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)
    
    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast('cuda', enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPNeXtBlock(nn.Module):
    """CSPNeXt bottleneck block.
    
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        expansion (float): Expand ratio. Default: 0.5.
        add_identity (bool): Whether to add identity. Default: True.
        use_depthwise (bool): Whether to use depthwise conv. Default: False.
        kernel_size (int): Kernel size of the second conv. Default: 5.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        
        self.add_identity = add_identity and in_channels == out_channels
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(nn.Module):
    """Cross Stage Partial Layer.
    
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        expand_ratio (float): Ratio for hidden channels. Default: 0.5.
        num_blocks (int): Number of blocks. Default: 1.
        add_identity (bool): Whether to add identity in blocks. Default: True.
        use_depthwise (bool): Whether to use depthwise conv. Default: False.
        use_cspnext_block (bool): Whether to use CSPNeXt block. Default: True.
        channel_attention (bool): Whether to use channel attention. Default: True.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        use_cspnext_block: bool = True,
        channel_attention: bool = True,
        norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ) for _ in range(num_blocks)
        ])
        
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        
        x_final = torch.cat((x_main, x_short), dim=1)
        
        if self.channel_attention:
            x_final = self.attention(x_final)
        
        return self.final_conv(x_final)


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling bottleneck.
    
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_sizes (tuple): Kernel sizes for pooling. Default: (5, 9, 13).
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, ...] = (5, 9, 13),
        norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: Optional[dict] = dict(type='SiLU'),
    ):
        super().__init__()
        mid_channels = in_channels // 2
        
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class CSPNeXt(nn.Module):
    """CSPNeXt backbone used in RTMDet/RTMPose.
    
    Args:
        arch (str): Architecture of CSPNeXt. Default: 'P5'.
        deepen_factor (float): Depth multiplier. Default: 1.0.
        widen_factor (float): Width multiplier. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages. Default: (4,).
        frozen_stages (int): Stages to be frozen. Default: -1.
        use_depthwise (bool): Whether to use depthwise conv. Default: False.
        expand_ratio (float): Ratio for CSPLayer. Default: 0.5.
        spp_kernel_sizes (Sequence[int]): Kernel sizes for SPP. Default: (5, 9, 13).
        channel_attention (bool): Whether to use channel attention. Default: True.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
        norm_eval (bool): Whether to set norm to eval mode. Default: False.
    """
    
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }
    
    def __init__(
        self,
        arch: str = 'P5',
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (4,),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        spp_kernel_sizes: Sequence[int] = (5, 9, 13),
        channel_attention: bool = True,
        norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: Optional[dict] = dict(type='SiLU'),
        norm_eval: bool = False,
    ):
        super().__init__()
        
        arch_setting = self.arch_settings[arch]
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        
        # Build stem
        self.stem = nn.Sequential(
            ConvModule(
                3,
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )
        self.layers = ['stem']
        
        # Build stages
        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            
            stage = []
            
            # Downsample conv
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            stage.append(conv_layer)
            
            # SPP layer
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernel_sizes,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                stage.append(spp)
            
            # CSP layer
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            stage.append(csp_layer)
            
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _freeze_stages(self):
        """Freeze stages."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def train(self, mode=True):
        """Set train mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


def get_cspnext(cfg):
    """Build CSPNeXt backbone from config."""
    model_size = cfg.MODEL.SIZE.lower()
    
    # Size configurations (from mmpose)
    size_configs = {
        't': {'deepen_factor': 0.167, 'widen_factor': 0.375, 'in_channels': 384},  # tiny
        's': {'deepen_factor': 0.33, 'widen_factor': 0.5, 'in_channels': 512},    # small
        'm': {'deepen_factor': 0.67, 'widen_factor': 0.75, 'in_channels': 768},   # medium
        'l': {'deepen_factor': 1.0, 'widen_factor': 1.0, 'in_channels': 1024},    # large
    }
    
    config = size_configs.get(model_size, size_configs['m'])
    
    backbone = CSPNeXt(
        arch=cfg.MODEL.BACKBONE.ARCH,
        deepen_factor=config['deepen_factor'],
        widen_factor=config['widen_factor'],
        out_indices=tuple(cfg.MODEL.BACKBONE.OUT_INDICES),
        expand_ratio=cfg.MODEL.BACKBONE.EXPAND_RATIO,
        channel_attention=cfg.MODEL.BACKBONE.CHANNEL_ATTENTION,
    )
    
    return backbone, config['in_channels']
