"""Shift-GCN features + pose-anchored cues, aligned by video name."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils import resolve


class PoseCues:
    """Packed cues from tools/pack_pose_cues.py: <prefix>.npy + <prefix>_names.npy."""

    def __init__(self, prefix):
        prefix = resolve(prefix)
        self.cues = np.load(prefix + ".npy", mmap_mode="r")
        names = np.load(prefix + "_names.npy")
        self.index = {str(n): i for i, n in enumerate(names)}

    def gather(self, names):
        idx = np.array([self.index.get(str(n), -1) for n in names])
        out = np.zeros((len(names),) + self.cues.shape[1:], dtype=self.cues.dtype)
        ok = idx >= 0
        out[ok] = self.cues[np.sort(idx[ok])][np.argsort(np.argsort(idx[ok]))]
        return torch.from_numpy(out), int((~ok).sum())


class TrainSet(Dataset):
    """Seen-class training samples with Gaussian feature noise (numpy RNG, per worker)."""

    def __init__(self, split, seen, noise):
        self.ske = split["ske"].numpy()
        self.cues = split["cues"].numpy()
        self.label = split["label"].numpy()
        remap = {c: i for i, c in enumerate(sorted(seen))}
        self.label_seen = np.array([remap[int(c)] for c in self.label])
        self.noise = noise

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        ske = self.ske[i] + np.random.randn(*self.ske[i].shape).astype(np.float32) * self.noise
        return ske, self.cues[i].astype(np.float32), self.label_seen[i], self.label[i]


def _seed_worker(_):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def train_loader(split, seen, noise, batch_size, seed, num_workers=4):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(TrainSet(split, seen, noise), batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=True, worker_init_fn=_seed_worker, generator=g)


def load_split(features, cues, split):
    """split in {train, zsl, gzsl} -> dict(ske, cues, label) as CPU tensors."""
    data = np.load(resolve(features), allow_pickle=True)
    c, missing = cues.gather(data[f"{split}_seq"])
    return {
        "ske": torch.from_numpy(data[f"{split}_data"].astype(np.float32)),
        "cues": c,
        "label": torch.from_numpy(data[f"{split}_label"].astype(np.int64)),
        "missing": missing,
    }
