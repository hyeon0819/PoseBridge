"""Evaluate a trained PoseBridge ZSSAR checkpoint (ZSL, and GZSL for NTU/PKU)."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posebridge.utils import load_config  # noqa: E402
from posebridge.zssar.trainer import build_model, load_eval_data, run_eval  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    data = load_eval_data(cfg)
    model = build_model(cfg, device)

    model.load_state_dict(ckpt["zsl"]["state_dict"])
    print(f"ZSL  acc : {run_eval(model, {'zsl': data['zsl']}, cfg, device)['zsl']:.2f}")
    if cfg["gzsl"]:
        model.load_state_dict(ckpt["gzsl"]["state_dict"])
        s, u, h = run_eval(model, data, cfg, device)["gzsl"]
        print(f"GZSL S/U/H : {s:.2f} / {u:.2f} / {h:.2f}")


if __name__ == "__main__":
    main()
