"""Collect best ZSL / GZSL results from work_dirs/*/*/best.pth (random splits are averaged)."""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posebridge.utils import resolve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="work_dirs")
    args = p.parse_args()

    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(resolve(args.work_dir), "*", "*", "best.pth"))):
        ck = torch.load(f, map_location="cpu", weights_only=True)
        name = "/".join(f.split(os.sep)[-3:-1])
        zsl = ck["zsl"]["acc"]
        gzsl = ck["gzsl"]["acc"] if "gzsl" in ck else None
        groups[re.sub(r"_split\d$", " (random)", name)].append((zsl, gzsl))

    print(f"{'config':45s} {'ZSL':>6s} {'S':>6s} {'U':>6s} {'H':>6s}")
    for name, runs in groups.items():
        zsl = sum(r[0] for r in runs) / len(runs)
        row = f"{name:45s} {zsl:6.2f}"
        if runs[0][1] is not None:
            s, u, h = [sum(r[1][i] for r in runs) / len(runs) for i in range(3)]
            row += f" {s:6.2f} {u:6.2f} {h:6.2f}"
        print(row + (f"   ({len(runs)} splits)" if len(runs) > 1 else ""))


if __name__ == "__main__":
    main()
