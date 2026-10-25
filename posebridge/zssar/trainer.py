import math
import time

import torch

from ..utils import class_split, resolve, set_seed
from .data import PoseCues, load_split, train_loader
from .evaluate import embed, evaluate_gzsl, evaluate_zsl
from .model import PoseBridgeZSSAR
from .text import text_features

LOSS_KEYS = ["ske_cls", "ske_sem", "ske_con", "pose_cls", "pose_sem", "pose_con",
             "bridge_sem", "bridge_con", "s2p", "kd"]


def build_model(cfg, device):
    _, seen, unseen = class_split(cfg["dataset"], cfg["split"])
    model = PoseBridgeZSSAR(
        text_features(cfg["dataset"], cfg["clip_model"], device=device), seen, unseen,
        ske_dim=cfg["ske_dim"], pose_dim=cfg["pose_dim"], embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"], bridge_heads=cfg["bridge_heads"],
        text_neighbor_k=cfg["text_neighbor_k"], text_neighbor_beta=cfg["text_neighbor_beta"],
        multi_template_weight=cfg["multi_template_weight"],
        loss_weights={k: cfg[f"w_{k}"] for k in LOSS_KEYS},
        supcon_temp=cfg["supcon_temp"], distill_temp=cfg["distill_temp"])
    return model.to(device)


def eval_cfg(cfg, mode="zsl"):
    return {"rho": cfg["rho"], "topk": cfg["topk"], **cfg["inference"][mode]}


def load_eval_data(cfg):
    cues = PoseCues(cfg["pose_cues"])
    splits = ["zsl", "gzsl"] if cfg["gzsl"] else ["zsl"]
    return {s: load_split(cfg["features"], cues, s) for s in splits}


def run_eval(model, data, cfg, device):
    emb = {s: embed(model, d, device, cfg["test_batch_size"]) for s, d in data.items()}
    out = {"zsl": evaluate_zsl(model, emb["zsl"], eval_cfg(cfg, "zsl"))}
    if "gzsl" in emb:
        out["gzsl"] = evaluate_gzsl(model, emb["gzsl"], eval_cfg(cfg, "gzsl"))
    return out


class EMA:
    """Parameter EMA, updated once per epoch."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply(self, model):
        for n, p in model.named_parameters():
            self.backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            p.data.copy_(self.backup[n])
        self.backup = {}


def lr_at(epoch, cfg):
    """Linear warmup + cosine decay (epoch is 0-based)."""
    if epoch < cfg["warmup_epochs"]:
        return cfg["lr"] * (epoch + 1) / cfg["warmup_epochs"]
    t = (epoch - cfg["warmup_epochs"]) / max(1, cfg["epochs"] - cfg["warmup_epochs"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * t))


def train(cfg, logger, device, on_epoch=None):
    set_seed(cfg["seed"])
    feats = resolve(cfg["features"])
    cues = PoseCues(cfg["pose_cues"])
    train_set = load_split(feats, cues, "train")
    data = {s: load_split(feats, cues, s) for s in (["zsl", "gzsl"] if cfg["gzsl"] else ["zsl"])}
    for name, d in [("train", train_set)] + list(data.items()):
        if d["missing"]:
            logger.info(f"warning: {d['missing']} {name} videos have no pose cues (zero-filled)")
    logger.info("data: " + ", ".join(f"{k}={len(v['label'])}" for k, v in [("train", train_set)] + list(data.items())))

    model = build_model(cfg, device)
    if cfg["freeze_text"]:
        model.text_projector.requires_grad_(False)
    loader = train_loader(train_set, model.seen, cfg["feature_noise"], cfg["batch_size"], cfg["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ema, best = None, {"zsl": None, "gzsl": None}

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch, cfg)
        model.train()
        total = 0.0
        for ske, cues, y_seen, y in loader:
            loss = model.loss(ske.to(device, non_blocking=True), cues.to(device, non_blocking=True),
                              y_seen.to(device, non_blocking=True), y.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            total += loss.item()

        if epoch + 1 >= cfg["ema_start"]:
            if ema is None:
                ema = EMA(model, cfg["ema_decay"])
            else:
                ema.update(model)
        if ema:
            ema.apply(model)
        model.eval()
        model.set_pose_centroids(train_set["cues"], train_set["label"])
        res = run_eval(model, data, cfg, device)
        if on_epoch:
            on_epoch(epoch + 1, model, data)

        msg = f"epoch {epoch + 1:2d}/{cfg['epochs']}  loss {total / len(loader):.3f}  ZSL {res['zsl']:.2f}"
        if "gzsl" in res:
            msg += "  GZSL S/U/H {:.2f}/{:.2f}/{:.2f}".format(*res["gzsl"])
        logger.info(msg + f"  ({time.time() - t0:.0f}s)")

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best["zsl"] is None or res["zsl"] > best["zsl"]["acc"]:
            best["zsl"] = {"acc": res["zsl"], "epoch": epoch + 1, "state": state}
        if "gzsl" in res and (best["gzsl"] is None or res["gzsl"][2] > best["gzsl"]["acc"][2]):
            best["gzsl"] = {"acc": res["gzsl"], "epoch": epoch + 1, "state": state}
        if ema:
            ema.restore(model)
    return best
