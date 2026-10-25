# ------------------------------------------------------------------------------
# COCO caption dataset with offline CLIP text features (encode_text output only).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import torch

from .coco_caption import COCOCaptionDataset

logger = logging.getLogger(__name__)

CLIP_TO_SUBDIR = {
    'ViT-B-32': 'vitb32',
    'ViT-L-14-336': 'vitl14-336',
    'ViT-L-14': 'vitl14',
}


class COCOCaptionPreextractDataset(COCOCaptionDataset):
    """Same as COCOCaptionDataset but adds meta['clip_text_feat'] (float32, D,)."""

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super(COCOCaptionPreextractDataset, self).__init__(cfg, root, image_set, is_train, transform)

        clip_name = cfg.TRAIN.TEXT_ENCODER.CLIP_MODEL
        if clip_name not in CLIP_TO_SUBDIR:
            raise ValueError(
                'CLIP_TEXT_PREEXTRACT: unsupported CLIP_MODEL %r (use %s)'
                % (clip_name, list(CLIP_TO_SUBDIR.keys()))
            )
        sub = CLIP_TO_SUBDIR[clip_name]
        feat_path = os.path.join(root, 'caption_features', sub, '{}.pt'.format(image_set))
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(
                'Pre-extracted CLIP text features not found:\n  %s\n'
                'Run: python tools/hpe/encode_captions.py --clip-model %s'
                % (feat_path, clip_name)
            )

        blob = torch.load(feat_path, map_location='cpu')
        self._feat_meta = blob['meta']
        self._features = blob['features']
        logger.info('=> Loaded CLIP text cache: %s (%d keys, meta=%s)'
                    % (feat_path, len(self._features), self._feat_meta))

        n_before = len(self.db)
        self.db = [
            r for r in self.db
            if r.get('image_id') in self.captions_dict
            and len(self.captions_dict[r['image_id']]) > 0
        ]
        logger.info('=> Preextract dataset: %d / %d samples kept (images with captions)'
                    % (len(self.db), n_before))

    def __getitem__(self, idx):
        input_tensor, target_x, target_y, target_weight, meta = super(
            COCOCaptionPreextractDataset, self
        ).__getitem__(idx)

        image_id = meta.get('image_id', None)
        cap = meta.get('caption', None)

        if image_id is None or cap is None:
            raise RuntimeError(
                'COCOCaptionPreextractDataset: missing image_id or caption at idx=%d' % idx
            )
        if isinstance(cap, list):
            raise NotImplementedError(
                'coco_caption_preextract does not support DATASET.CAPTION_INDEX==-1 '
                '(multi-caption list). Use CAPTION_INDEX in 0..4 or coco_caption dataset.'
            )

        caps_list = self.captions_dict[image_id]
        try:
            cap_idx = caps_list.index(cap)
        except ValueError:
            cap_idx = 0

        key = '{}_{}'.format(int(image_id), cap_idx)
        if key not in self._features:
            raise KeyError(
                'Missing pre-extracted key %r. Re-run tools/hpe/encode_captions.py '
                'for image_set=%s and CLIP %s.'
                % (key, self.image_set, self._feat_meta.get('clip_model'))
            )

        vec = self._features[key]
        if not torch.is_tensor(vec):
            vec = torch.tensor(vec, dtype=torch.float32)
        else:
            vec = vec.float().clone()
        meta['clip_text_feat'] = vec
        return input_tensor, target_x, target_y, target_weight, meta
