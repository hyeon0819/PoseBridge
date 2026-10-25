"""Train PoseBridge ZSSAR on seen classes; report best ZSL / GZSL epochs."""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posebridge.utils import config_name, get_logger, load_config, resolve  # noqa: E402
from posebridge.zssar.trainer import train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--work-dir", default=None, help="default: work_dirs/<config name>")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cfg = load_config(args.config)
    name = config_name(args.config)
    work_dir = args.work_dir or resolve(os.path.join("work_dirs", name))
    os.makedirs(work_dir, exist_ok=True)
    logger = get_logger(os.path.join(work_dir, "train.log"))
    logger.info(f"config: {args.config}")
    logger.info("\n" + yaml.safe_dump({k: v for k, v in cfg.items() if k != "shiftgcn"}, sort_keys=False, width=200))

    best = train(cfg, logger, torch.device(args.device))

    ckpt = {"config": cfg}
    z = best["zsl"]
    logger.info("=" * 60)
    logger.info(f"best ZSL  : {z['acc']:.2f}  (epoch {z['epoch']})")
    ckpt["zsl"] = {"state_dict": z["state"], "epoch": z["epoch"], "acc": z["acc"]}
    if best["gzsl"] is not None:
        g = best["gzsl"]
        logger.info("best GZSL : S {:.2f}  U {:.2f}  H {:.2f}".format(*g["acc"]) + f"  (epoch {g['epoch']})")
        ckpt["gzsl"] = {"state_dict": g["state"], "epoch": g["epoch"], "acc": list(g["acc"])}
    logger.info(f"inference: {cfg['inference']}")
    path = os.path.join(work_dir, "best.pth")
    torch.save(ckpt, path)
    rel = os.path.relpath(path)
    logger.info(f"saved {path if rel.startswith('..') else rel}")


if __name__ == "__main__":
    main()
