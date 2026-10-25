import numpy as np
from torch.utils.data import Dataset


class SkeletonSet(Dataset):
    """COCO-17 skeleton npz (x_{train,test}: (N, T, 2*17*3)) -> (3, T, 17, 2).

    split: train (seen) | test (seen, for model selection) | zsl (unseen test) | gzsl (all test)
    """

    def __init__(self, npz, unseen, split):
        part = "train" if split == "train" else "test"
        x, y = npz[f"x_{part}"], npz[f"y_{part}"]
        seq = np.array([str(s) for s in npz[f"{part}_seq"]])
        unseen = set(unseen)
        is_unseen = np.array([int(l) in unseen for l in y])
        keep = {"train": ~is_unseen, "test": ~is_unseen, "zsl": is_unseen, "gzsl": np.ones_like(is_unseen)}[split]
        self.x, self.y, self.seq = x[keep], y[keep].astype(np.int64), seq[keep]
        remap = {c: i for i, c in enumerate(sorted(set(self.y.tolist()) - unseen))}
        self.y_seen = np.array([remap.get(int(l), -1) for l in self.y]) if split in ("train", "test") else self.y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        t = self.x.shape[1]
        x = self.x[i].reshape(t, 2, 17, 3).transpose(3, 0, 2, 1)
        return np.ascontiguousarray(x, dtype=np.float32), self.y_seen[i]
