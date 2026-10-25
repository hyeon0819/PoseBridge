"""skeleton_results.pkl (tools/hpe/extract.py) -> Shift-GCN input npz under data/skeleton/.

Each npz holds x_{train,test}: (N, 300, 2*17*3), y_{train,test}, {train,test}_seq.
"""
import argparse
import gzip
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posebridge.utils import ROOT, resolve  # noqa: E402

META = os.path.join(ROOT, "posebridge", "data")
NTU_XSUB = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
NTU120_XSUB = NTU_XSUB + [45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85,
                          86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
MAX_T, J = 300, 17


def normalize(kp, score, height, top2_by_score=False):
    """(P, T, 17, 2) pixel coords -> (300, 2*17*3): first-frame pelvis centered, divided by height / 2."""
    kp, score = kp[:, :MAX_T], score[:, :MAX_T]
    if top2_by_score:  # Kinetics: keep the two most confident persons, zero-pad the rest
        order = np.argsort(-score.mean(axis=(1, 2)))[:2]
        kp, score = kp[order], score[order]
    P, T = kp.shape[:2]
    pelvis = kp[:min(P, 2), 0, 11:13].mean(axis=1).mean(axis=0)
    out = np.zeros((2, T, J, 3), np.float32)
    for p in range(2):
        src = p if p < P else (None if top2_by_score else 0)  # NTU/PKU duplicate a single actor
        if src is None:
            continue
        out[p, :, :, :2] = (kp[src] - pelvis) / (height / 2)
        out[p, :, :, 2] = score[src]
    flat = np.zeros((MAX_T, 2 * J * 3), np.float32)
    flat[:T] = np.hstack([out[0].reshape(T, -1), out[1].reshape(T, -1)])
    return flat


def build(names, labels, skel, kinetics=False):
    x, y, seq = [], [], []
    for n, l in tqdm(zip(names, labels), total=len(names)):
        s = skel.get(n)
        if s is None or s["skeleton"] is None or s["skeleton"].size == 0:
            continue
        s = np.asarray(s["skeleton"], np.float32)
        if kinetics:  # frame height estimated from keypoints (min 240)
            h = max(240, int(np.ceil(float(s.astype(np.float16)[..., 1].max()) + 1)))
            x.append(normalize(s[..., :2], s[..., 2], h, top2_by_score=True))
        else:
            x.append(normalize(s[..., :2], s[..., 2], 1080))
        y.append(l)
        seq.append(n)
    x = np.stack(x) if x else np.zeros((0, MAX_T, 2 * J * 3), np.float32)
    return x, np.array(y, np.int64), np.array(seq)


def save(path, train, test):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, x_train=train[0], y_train=train[1], train_seq=train[2],
             x_test=test[0], y_test=test[1], test_seq=test[2])
    print(f"{path}: train {len(train[1])}, test {len(test[1])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["ntu", "pku51", "kinetics"])
    p.add_argument("--skeleton-pkl", required=True)
    p.add_argument("--pku-root", default=None, help="PKU-MMD root containing split/cross-subject.txt")
    p.add_argument("--out-dir", default="data/skeleton")
    args = p.parse_args()

    out = resolve(args.out_dir)
    with open(args.skeleton_pkl, "rb") as f:
        skel = pickle.load(f)

    if args.dataset == "ntu":
        excluded = set(open(os.path.join(META, "ntu_excluded.txt")).read().split())
        names = sorted(n for n in skel if n not in excluded)
        lab = lambda n: int(n[17:20]) - 1  # noqa: E731
        rules = {"ntu60_xsub": (60, lambda n: int(n[9:12]) in NTU_XSUB),
                 "ntu60_xview": (60, lambda n: int(n[5:8]) in (2, 3)),
                 "ntu120_xsub": (120, lambda n: int(n[9:12]) in NTU120_XSUB),
                 "ntu120_xset": (120, lambda n: int(n[1:4]) % 2 == 0)}
        for name, (n_cls, is_train) in rules.items():
            sel = [n for n in names if lab(n) < n_cls]
            tr = [n for n in sel if is_train(n)]
            te = [n for n in sel if not is_train(n)]
            save(f"{out}/{name}.npz", build(tr, map(lab, tr), skel), build(te, map(lab, te), skel))

    elif args.dataset == "pku51":
        split = {}
        with open(os.path.join(args.pku_root, "split", "cross-subject.txt")) as f:
            cur = None
            for line in f:
                low = line.strip().lower().replace(" ", "")
                if low.startswith("training"):
                    cur = "train"
                elif low.startswith("valid"):
                    cur = "test"
                elif cur and line.strip():
                    split.update({c.strip(): cur for c in line.split(",") if c.strip()})
        lab = lambda n: int(n[-3:]) - 1  # noqa: E731
        parts = {k: sorted(n for n in skel if split.get(n.split("_")[0]) == k) for k in ("train", "test")}
        save(f"{out}/pku51_xsub.npz", *[build(parts[k], map(lab, parts[k]), skel) for k in ("train", "test")])

    else:
        lists = {}
        for k, s in (("train", "train"), ("test", "val")):
            with gzip.open(os.path.join(META, f"kinetics400_{s}.txt.gz"), "rt") as f:
                rows = [l.split() for l in f]
            lists[k] = ([r[0] for r in rows], [int(r[1]) for r in rows])
        k400 = {k: build(*lists[k], skel, kinetics=True) for k in lists}
        save(f"{out}/kinetics400.npz", k400["train"], k400["test"])
        k200 = {k: tuple(a[v[1] < 200] for a in v) for k, v in k400.items()}
        save(f"{out}/kinetics200.npz", k200["train"], k200["test"])


if __name__ == "__main__":
    main()
