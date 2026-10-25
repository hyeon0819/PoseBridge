"""Pretrain Shift-GCN on seen classes, then dump 256-d features for train / zsl / gzsl."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from posebridge.skeleton.data import SkeletonSet  # noqa: E402
from posebridge.skeleton.shift_gcn import ShiftGCN  # noqa: E402
from posebridge.utils import class_split, config_name, get_logger, load_config, resolve, set_seed  # noqa: E402


def lr_at(epoch, c):
    if epoch < c["warmup_epochs"]:
        return c["lr"] * (epoch + 1) / c["warmup_epochs"]
    return c["lr"] * 0.1 ** np.sum(epoch >= np.array(c["steps"]))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return 100 * correct / total


def train(model, data, unseen, c, device, log, ckpt_path):
    loader = lambda s, bs, sh: DataLoader(SkeletonSet(data, unseen, s), batch_size=bs, shuffle=sh,  # noqa: E731
                                          num_workers=8, drop_last=sh, worker_init_fn=lambda _: set_seed(c["seed"]))
    train_ld, test_ld = loader("train", c["batch_size"], True), loader("test", c["test_batch_size"], False)
    groups = [{"params": p, "weight_decay": 1e-3 if "Linear_weight" in n else 0.0 if "Mask" in n else c["weight_decay"]}
              for n, p in model.named_parameters()]
    opt = torch.optim.SGD(groups, lr=c["lr"], momentum=0.9, nesterov=True)
    crit = torch.nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(c["epochs"]):
        for n, p in model.named_parameters():  # freeze shift / mask parameters in the first epoch
            if "PA" in n or "Linear_weight" in n or "Feature_Mask" in n:
                p.requires_grad = epoch >= 1
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch, c)
        model.train()
        losses = []
        for x, y in tqdm(train_ld, desc=f"epoch {epoch + 1}", leave=False):
            loss = crit(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        msg = f"epoch {epoch + 1}/{c['epochs']}  loss {np.mean(losses):.4f}"
        if epoch % 10 == 0 or epoch >= 50:
            acc = evaluate(model, test_ld, device)
            if acc > best:
                best = acc
                torch.save(model.state_dict(), ckpt_path)
            msg += f"  seen-test top1 {acc:.2f} (best {best:.2f})"
        log(msg)


def extract(model, data, unseen, out, device):
    model.eval()
    res = {}
    for split in ("train", "zsl", "gzsl"):
        ds = SkeletonSet(data, unseen, split)
        feats = [model.extract(x.to(device)).cpu().numpy()
                 for x, _ in tqdm(DataLoader(ds, batch_size=256, num_workers=8), desc=f"extract {split}")]
        res.update({f"{split}_data": np.concatenate(feats), f"{split}_label": ds.y, f"{split}_seq": ds.seq})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, **res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="default: checkpoints/shiftgcn/<config name>.pt")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    cfg = load_config(args.config)
    c = cfg["shiftgcn"]
    name = config_name(args.config).replace("/", "_")
    ckpt = resolve(args.checkpoint or f"checkpoints/shiftgcn/{name}.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    log = get_logger(resolve(f"work_dirs/shiftgcn/{name}.log"), name="shiftgcn").info
    set_seed(c["seed"])
    device = torch.device(args.device)

    n_cls, seen, unseen = class_split(cfg["dataset"], cfg["split"])
    data = dict(np.load(resolve(cfg["skeleton"]), allow_pickle=True))
    model = ShiftGCN(num_class=len(seen)).to(device)
    if not args.extract_only:
        train(model, data, unseen, c, device, log, ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    extract(model, data, unseen, resolve(cfg["features"]), device)
    log(f"features -> {cfg['features']}")


if __name__ == "__main__":
    main()
