from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray):
    N, K, _ = simcc_x.shape
    sx = simcc_x.reshape(N * K, -1)
    sy = simcc_y.reshape(N * K, -1)
    x_locs = np.argmax(sx, axis=1)
    y_locs = np.argmax(sy, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vx = np.amax(sx, axis=1)
    vy = np.amax(sy, axis=1)
    mask = vx > vy
    vx[mask] = vy[mask]
    locs[vx <= 0.] = -1
    return locs.reshape(N, K, 2), vx.reshape(N, K)


def simcc_pck_accuracy(output, target, simcc_split_ratio=2.0, mask=None,
                       thr=0.05, normalize=None):
    pred_x, pred_y = output
    gt_x, gt_y = target
    N, K, Wx = pred_x.shape
    Wy = pred_y.shape[2]
    W = int(Wx / simcc_split_ratio)
    H = int(Wy / simcc_split_ratio)
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred_coords, _ = get_simcc_maximum(pred_x, pred_y)
    pred_coords /= simcc_split_ratio
    gt_coords, _ = get_simcc_maximum(gt_x, gt_y)
    gt_coords /= simcc_split_ratio

    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    nc = normalize.copy()
    nc[nc <= 0] = 1e6
    distances[_mask] = np.linalg.norm(
        ((pred_coords - gt_coords) / nc[:, None, :])[_mask], axis=-1)
    distances = distances.T

    def _acc(d, t):
        valid = d != -1
        n = valid.sum()
        return (d[valid] < t).sum() / n if n > 0 else -1.0

    acc = np.array([_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt


def decode_simcc(simcc_x, simcc_y, simcc_split_ratio):
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = keypoints / simcc_split_ratio
    return keypoints, scores


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=False):
    import cv2
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size
    rot_rad = np.pi * rot / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir_pt = np.array([0, src_w * -0.5])
    src_dir = np.array([src_dir_pt[0]*cs - src_dir_pt[1]*sn,
                        src_dir_pt[0]*sn + src_dir_pt[1]*cs])
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center + scale_tmp * shift
    src[1] = center + src_dir + scale_tmp * shift
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    direct = src[0] - src[1]
    src[2] = src[1] + np.array([-direct[1], direct[0]])
    direct = dst[0] - dst[1]
    dst[2] = dst[1] + np.array([-direct[1], direct[0]])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    return np.dot(t, new_pt)[:2]


def transform_preds(coords, center, scale, output_size):
    N, K, _ = coords.shape
    target_coords = np.zeros_like(coords)
    for n in range(N):
        trans = get_affine_transform(center[n], scale[n], 0, output_size, inv=True)
        for k in range(K):
            target_coords[n, k, :2] = affine_transform(coords[n, k, :2], trans)
    return target_coords


def _print_name_value(name_value, full_arch_name, print_log):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print_log('| Arch ' + ' '.join([f'| {n}' for n in names]) + ' |')
    print_log('|---' * (num_values + 1) + '|')
    arch = full_arch_name[:8] + '...' if len(full_arch_name) > 15 else full_arch_name
    print_log('| ' + arch + ' ' + ' '.join([f'| {v:.3f}' for v in values]) + ' |')


# ---------------------------------------------------------------------------
# CLIP contrastive loss + text feature helper
# ---------------------------------------------------------------------------

def clip_contrastive_loss(pose_emb: torch.Tensor, text_emb: torch.Tensor,
                          temperature=0.07) -> torch.Tensor:
    """Symmetric InfoNCE / CLIP contrastive loss between (B, D) pose and text."""
    if isinstance(temperature, torch.Tensor):
        temp = temperature
    else:
        temp = torch.tensor(temperature, dtype=pose_emb.dtype,
                            device=pose_emb.device)

    logits = torch.matmul(pose_emb, text_emb.t()) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) * 0.5


def get_raw_text_features(meta, clip_encoder, device):
    """Return raw CLIP text embeddings (B, clip_dim).

    Uses pre-extracted features when ``meta['clip_text_feat']`` is set
    (via ``COCOCaptionPreextractDataset``); otherwise encodes captions
    on the fly with ``clip_encoder``.
    """
    feat = meta.get('clip_text_feat', None)
    if feat is not None:
        if not isinstance(feat, torch.Tensor):
            feat = torch.tensor(feat, dtype=torch.float32)
        return feat.to(device, non_blocking=True).float()
    captions = meta['caption']
    with torch.no_grad():
        raw = clip_encoder(captions)
    return raw.to(device, non_blocking=True).float()
