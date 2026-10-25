# Frozen OpenCLIP text tower.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import torch
import torch.nn as nn


class OpenCLIPTextTower(nn.Module):
    """Returns raw (un-normalized) float32 text features."""

    def __init__(self, model_name, pretrained='openai', device=0):
        super(OpenCLIPTextTower, self).__init__()
        import open_clip

        with warnings.catch_warnings():  # keep original (non-QuickGELU) features
            warnings.filterwarnings('ignore', message='QuickGELU mismatch')
            mdl, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = mdl
        self.tokenizer = open_clip.get_tokenizer(model_name)
        dev = torch.device('cuda:%d' % device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(dev).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._device = dev

    def forward(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(self._device, non_blocking=True)
        elif isinstance(tokens, dict):
            tokens = {
                k: v.to(self._device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in tokens.items()
            }
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return feats.float()
