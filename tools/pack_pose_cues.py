"""Pack per-video pose-anchored cues into one array for fast ZSSAR loading.

Input : <pose_emb_dir>/**/{name}.npz with key `pose_sem` of shape (M, T, D)
Output: <out>.npy (N, T_sub, D) float32 and <out>_names.npy (N,)
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm


def temporal_cue(path, key="pose_sem", t_sub=16):
    arr = np.load(path)[key].astype(np.float32)
    if arr.ndim == 3:  # (M, T, D) -> person mean
        arr = arr.mean(axis=0)
    t, d = arr.shape
    if t >= t_sub:
        out = arr[np.linspace(0, t - 1, t_sub, dtype=int)]
    else:
        out = np.zeros((t_sub, d), dtype=np.float32)
        out[:t] = arr
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-emb-dir", required=True)
    p.add_argument("--out", required=True, help="output prefix, e.g. data/pose_cues/ntu")
    p.add_argument("--t-sub", type=int, default=16)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    names = []
    for root, _, files in os.walk(args.pose_emb_dir):
        for f in files:
            if f.endswith(".npz"):
                rel = os.path.relpath(os.path.join(root, f), args.pose_emb_dir)
                names.append(rel[:-4])
    names.sort()
    paths = [os.path.join(args.pose_emb_dir, n + ".npz") for n in names]

    with ProcessPoolExecutor(args.workers) as ex:
        cues = list(tqdm(ex.map(temporal_cue, paths, chunksize=256), total=len(paths)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out + ".npy", np.stack(cues))
    np.save(args.out + "_names.npy", np.array(names))
    print(f"packed {len(names)} videos -> {args.out}.npy")


if __name__ == "__main__":
    main()
