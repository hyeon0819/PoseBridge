"""Batched PoseBridge-HPE inference on tracked person boxes: 2D skeletons + pose-anchored cues."""
from collections import defaultdict

import cv2
import numpy as np
import torch

from .coco_caption import get_affine_transform
from .config import cfg as base_cfg
from .model import get_posebridge_hpe
from .postprocess import _interpolate_gaps, decode_simcc, transform_preds

PIXEL_STD = 200.0


class Extractor:
    def __init__(self, cfg_path, checkpoint, device="cuda:0", half=True):
        cfg = base_cfg.clone()
        cfg.defrost()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.INIT_WEIGHTS = False
        cfg.freeze()
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = get_posebridge_hpe(cfg, is_train=False)
        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", ""): v for k, v in sd.get("state_dict", sd).items()}
        self.model.load_state_dict(sd, strict=True)
        self.model = self.model.to(self.device).eval()
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.model = self.model.half()
        self.dim = int(cfg.MODEL.ROUTE.PROJ_DIM)
        self.input_size = tuple(cfg.MODEL.IMAGE_SIZE)  # (w, h)
        self.aspect = self.input_size[0] / float(self.input_size[1])
        self.split_ratio = cfg.MODEL.SIMCC.SPLIT_RATIO
        self.mean = np.array(cfg.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(cfg.DATASET.STD, dtype=np.float32)

    def crop(self, frame, box):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        w, h = x2 - x1, y2 - y1
        center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)
        if w > self.aspect * h:
            h = w / self.aspect
        elif w < self.aspect * h:
            w = h * self.aspect
        scale = np.array([w / PIXEL_STD, h / PIXEL_STD], dtype=np.float32) * 1.25
        trans = get_affine_transform(center, scale, 0.0, self.input_size)
        img = cv2.warpAffine(frame, trans, (int(self.input_size[0]), int(self.input_size[1])),
                             flags=cv2.INTER_LINEAR)
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), center, scale


def read_frames(path, max_frames=0):
    cap, frames = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(f)
    cap.release()
    return frames


def read_windows(path, windows):
    """One forward pass over a long video; windows = [(name, lo, hi)] (half-open, sorted)."""
    out = {n: [] for n, _, _ in windows}
    cap, pos = cv2.VideoCapture(path), 0
    for name, lo, hi in sorted(windows, key=lambda w: w[1]):
        while pos < lo and cap.grab():
            pos += 1
        while pos < hi:
            ok, f = cap.read()
            if not ok:
                break
            out[name].append(f)
            pos += 1
    cap.release()
    return out


def prepare(name, frames, bboxes, max_persons, dim, expected=None):
    """Keep the longest `M` tracks; allocate skeleton (M,T,17,3) and cue (M,T,D) buffers."""
    T = min(len(frames), len(bboxes))
    if T == 0:
        return None
    tracks = defaultdict(list)
    for t, dets in enumerate(bboxes[:T]):
        for row in dets:
            if len(row) >= 6:
                tracks[int(row[5])].append(t)
    if not tracks:
        return None
    M = min(max_persons, len(tracks), expected or max_persons)
    order = sorted(tracks, key=lambda k: (-len(tracks[k]), min(tracks[k])))[:M]
    return {"name": name, "T": T, "frames": frames[:T], "bboxes": bboxes[:T], "expected": expected,
            "tid": {k: i for i, k in enumerate(order)},
            "skeleton": np.zeros((M, T, 17, 3), np.float32), "cues": np.zeros((M, T, dim), np.float32)}


@torch.no_grad()
def run(states, ex):
    """Frame-synchronous batched forward over a chunk of videos."""
    states = [s for s in states if s is not None]
    for t in range(max((s["T"] for s in states), default=0)):
        entries = []
        for s in states:
            if t >= s["T"]:
                continue
            for row in s["bboxes"][t]:
                if len(row) >= 6 and int(row[5]) in s["tid"]:
                    crop, c, sc = ex.crop(s["frames"][t], row)
                    entries.append((s, s["tid"][int(row[5])], crop, c, sc))
        if not entries:
            continue
        batch = torch.stack([e[2] for e in entries]).to(ex.device)
        px, py, sem = ex.model(batch.half() if ex.half else batch)
        kpts, scores = decode_simcc(px.float().cpu().numpy(), py.float().cpu().numpy(), ex.split_ratio)
        sem = sem.float().cpu().numpy()
        for j, (s, p, _, c, sc) in enumerate(entries):
            xy = transform_preds(kpts[j], c, sc, ex.input_size)
            s["skeleton"][p, t] = np.concatenate([xy, scores[j][:, None]], axis=1)
            s["cues"][p, t] = sem[j]


def finalize(s, interp_max_gap=0):
    """Drop surplus persons (by mean keypoint score), optionally interpolate short gaps; float16 out."""
    skel, cues = s["skeleton"], s["cues"]
    if s["expected"] and skel.shape[0] > s["expected"]:
        keep = np.sort(np.argsort(skel.astype(np.float16)[..., 2].mean(axis=(1, 2)))[::-1][:s["expected"]])
        skel, cues = skel[keep], cues[keep]
    if interp_max_gap > 0:
        for p in range(skel.shape[0]):
            _interpolate_gaps(skel[p], max_gap=interp_max_gap)
            _interpolate_gaps(cues[p], max_gap=interp_max_gap)
    return skel.astype(np.float16), cues.astype(np.float16)
