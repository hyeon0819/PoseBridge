"""RTMDet-nano person detector in plain PyTorch (loads the official MMPose checkpoint)."""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from .cspnext import ConvModule, CSPLayer, CSPNeXt, DepthwiseSeparableConvModule

RTMDET_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
              "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth")


def _dw(cin, cout, stride=1):
    return DepthwiseSeparableConvModule(cin, cout, 3, stride=stride, padding=1)


class PAFPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256), out_channels=64):
        super().__init__()
        n = len(in_channels)

        def csp(cin, cout):
            return CSPLayer(cin, cout, num_blocks=1, add_identity=False, use_depthwise=True, channel_attention=False)

        self.reduce_layers = nn.ModuleList(ConvModule(in_channels[i], in_channels[i - 1], 1) for i in range(n - 1, 0, -1))
        self.top_down_blocks = nn.ModuleList(csp(in_channels[i - 1] * 2, in_channels[i - 1]) for i in range(n - 1, 0, -1))
        self.downsamples = nn.ModuleList(_dw(in_channels[i], in_channels[i], 2) for i in range(n - 1))
        self.bottom_up_blocks = nn.ModuleList(csp(in_channels[i] * 2, in_channels[i + 1]) for i in range(n - 1))
        self.out_convs = nn.ModuleList(_dw(c, out_channels) for c in in_channels)

    def forward(self, x):
        n = len(x)
        inner = [x[-1]]
        for j, i in enumerate(range(n - 1, 0, -1)):
            inner[0] = self.reduce_layers[j](inner[0])
            up = F.interpolate(inner[0], size=x[i - 1].shape[2:], mode="nearest")
            inner.insert(0, self.top_down_blocks[j](torch.cat([up, x[i - 1]], 1)))
        outs = [inner[0]]
        for i in range(n - 1):
            outs.append(self.bottom_up_blocks[i](torch.cat([self.downsamples[i](outs[-1]), inner[i + 1]], 1)))
        return [conv(o) for conv, o in zip(self.out_convs, outs)]


class SepBNHead(nn.Module):
    """Per-level conv towers; returns class logits and LTRB distances (in pixels)."""

    def __init__(self, in_channels=64, feat_channels=64, strides=(8, 16, 32)):
        super().__init__()
        self.strides = strides

        def tower():
            return nn.ModuleList(nn.ModuleList([_dw(in_channels, feat_channels), _dw(feat_channels, feat_channels)])
                                 for _ in strides)

        self.cls_convs, self.reg_convs = tower(), tower()
        self.rtm_cls = nn.ModuleList(nn.Conv2d(feat_channels, 1, 1) for _ in strides)
        self.rtm_reg = nn.ModuleList(nn.Conv2d(feat_channels, 4, 1) for _ in strides)

    def forward(self, feats):
        cls, reg = [], []
        for i, x in enumerate(feats):
            c, r = x, x
            for m in self.cls_convs[i]:
                c = m(c)
            for m in self.reg_convs[i]:
                r = m(r)
            cls.append(self.rtm_cls[i](c))
            reg.append(self.rtm_reg[i](r) * self.strides[i])
        return cls, reg


class RTMDetNano(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CSPNeXt(arch="P5", deepen_factor=0.33, widen_factor=0.25, out_indices=(2, 3, 4),
                                use_depthwise=True, channel_attention=True)
        self.neck = PAFPN()
        self.bbox_head = SepBNHead()
        for m in self.modules():  # mmdet SyncBN defaults
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-5, 0.1

    def forward(self, x):
        return self.bbox_head(self.neck(self.backbone(x)))


class RTMDetPerson:
    """Batched person detection -> list of [N, 5] (x1, y1, x2, y2, score) in image coordinates."""

    def __init__(self, checkpoint, device="cuda:0", score_thr=0.3, iou_thr=0.6, input_size=320,
                 nms_pre=1000, max_per_img=100):
        if not os.path.isfile(checkpoint):
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
            torch.hub.download_url_to_file(RTMDET_URL, checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = RTMDetNano()
        self.model.load_state_dict(state.get("state_dict", state))
        self.model.to(device).eval()
        self.device, self.size = device, input_size
        self.score_thr, self.iou_thr, self.nms_pre, self.max_per_img = score_thr, iou_thr, nms_pre, max_per_img
        self.mean = np.array([103.53, 116.28, 123.675], np.float32)  # BGR
        self.std = np.array([57.375, 57.12, 58.395], np.float32)

    def _prepare(self, img):
        h, w = img.shape[:2]
        s = min(self.size / h, self.size / w)
        nh, nw = int(round(h * s)), int(round(w * s))
        x = np.full((self.size, self.size, 3), 114, np.uint8)
        x[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return ((x.astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1), s

    @torch.no_grad()
    def __call__(self, frames):
        if not frames:
            return []
        batch, scales = zip(*(self._prepare(f) for f in frames))
        cls, reg = self.model(torch.from_numpy(np.stack(batch)).to(self.device))
        points = []
        for c, stride in zip(cls, self.model.bbox_head.strides):
            ys, xs = torch.meshgrid(torch.arange(c.shape[2], device=c.device), torch.arange(c.shape[3], device=c.device),
                                    indexing="ij")
            points.append(torch.stack([xs, ys], -1).reshape(-1, 2).float() * stride)
        points = torch.cat(points)
        scores = torch.cat([c.flatten(2) for c in cls], 2)[:, 0].sigmoid()                  # B, P
        dist = torch.cat([r.flatten(2) for r in reg], 2).permute(0, 2, 1)                  # B, P, 4
        boxes = torch.cat([points - dist[..., :2], points + dist[..., 2:]], -1)

        out = []
        for b, (img, s) in enumerate(zip(frames, scales)):
            sc, bx = scores[b], boxes[b]
            sc, idx = sc.topk(min(self.nms_pre, sc.numel()))
            keep = sc > self.score_thr
            sc, bx = sc[keep], bx[idx[keep]]
            keep = nms(bx, sc, self.iou_thr)[: self.max_per_img]
            bx = (bx[keep] / s).cpu().numpy()
            h, w = img.shape[:2]
            bx[:, 0::2] = bx[:, 0::2].clip(0, w)
            bx[:, 1::2] = bx[:, 1::2].clip(0, h)
            out.append(np.concatenate([bx, sc[keep].cpu().numpy()[:, None]], 1).astype(np.float32))
        return out
