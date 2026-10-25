"""SimCC decoding, crop-to-image mapping and gap utilities for extraction."""
from typing import Tuple

import cv2
import numpy as np


# =============================================================================
# SimCC decoding
# =============================================================================

def decode_simcc(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    simcc_split_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode SimCC predictions to keypoint coordinates.

    Returns (keypoints (B, K, 2), scores (B, K)).
    Score is the minimum of max(pred_x) and max(pred_y) at each keypoint.
    """
    B, K, _ = pred_x.shape

    x_locs = np.argmax(pred_x, axis=-1)
    y_locs = np.argmax(pred_y, axis=-1)
    max_x = np.amax(pred_x, axis=-1)
    max_y = np.amax(pred_y, axis=-1)
    scores = np.minimum(max_x, max_y)

    keypoints = np.zeros((B, K, 2), dtype=np.float32)
    keypoints[:, :, 0] = x_locs / simcc_split_ratio
    keypoints[:, :, 1] = y_locs / simcc_split_ratio

    invalid = scores <= 0.
    keypoints[invalid] = -1
    return keypoints, scores


# =============================================================================
# Affine transform back to image coords
# =============================================================================

def _get_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: np.ndarray = np.array([0, 0], dtype=np.float32),
    inv: bool = False,
) -> np.ndarray:
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180
    src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_dir = np.array(
        [src_dir[0] * cs - src_dir[1] * sn,
         src_dir[0] * sn + src_dir[1] * cs],
        dtype=np.float32,
    )

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2, :] = src[0, :] + np.array([src_dir[1], -src_dir[0]])
    dst[2, :] = dst[0, :] + np.array([dst_dir[1], -dst_dir[0]])

    if inv:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def transform_preds(
    coords: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Transform per-person predictions back to original image coordinates.

    Args:
        coords:      (K, 2 or more) keypoint predictions in input-image space.
        center:      (2,) crop centre.
        scale:       (2,) crop scale (200-pixel units).
        output_size: (W, H) of the network input.
    """
    target_coords = np.zeros_like(coords)
    trans = _get_affine_transform(center, scale, 0, output_size, inv=True)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = np.dot(
            trans, np.array([coords[p, 0], coords[p, 1], 1.0])
        )
    return target_coords


# =============================================================================
# Per-sequence gap fill / detection
# =============================================================================

def _interpolate_gaps(data: np.ndarray, max_gap: int = 3):
    """Linearly interpolate missing frames (all-zeros) in-place along axis 0.

    Works on any shape ``[T, ...]``. Gaps longer than ``max_gap`` are left
    untouched (still zeros).
    """
    T = data.shape[0]
    flat = data.reshape(T, -1)
    is_valid = np.any(flat != 0, axis=1)

    t = 0
    while t < T:
        if is_valid[t]:
            t += 1
            continue
        gap_start = t
        while t < T and not is_valid[t]:
            t += 1
        gap_end = t
        gap_len = gap_end - gap_start
        if gap_len > max_gap:
            continue
        left = gap_start - 1
        right = gap_end
        if left < 0 or right >= T:
            continue
        for g in range(gap_len):
            alpha = (g + 1) / (gap_len + 1)
            flat[gap_start + g] = flat[left] * (1 - alpha) + flat[right] * alpha


def _person_has_zero_frames(person_skel: np.ndarray) -> bool:
    """Return True if any frame of a person has all-zero (x, y) keypoints."""
    T = person_skel.shape[0]
    xy = person_skel[:, :, :2].reshape(T, -1)
    return bool(np.any(~np.any(xy != 0, axis=1)))


def sequence_has_gap(skeleton: np.ndarray) -> bool:
    """Decide whether a multi-person sequence should be flagged for gaps.

    Single-person (M=1): flag if the person has any all-zero frame.
    Multi-person (M>=2): flag only if *all* persons miss at least one frame.
    """
    M = skeleton.shape[0]
    return all(_person_has_zero_frames(skeleton[p]) for p in range(M))
