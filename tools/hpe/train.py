"""Train PoseBridge-HPE (RTMPose-s + pose-anchored semantic head) on COCO keypoints + captions."""
import argparse
import math
import os
import sys

import torch
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posebridge.hpe.coco_caption_preextract import COCOCaptionPreextractDataset  # noqa: E402
from posebridge.hpe.config import cfg as base_cfg  # noqa: E402
from posebridge.hpe.engine import train_posebridge, validate_posebridge  # noqa: E402
from posebridge.hpe.loss import KLDiscretLoss  # noqa: E402
from posebridge.hpe.model import TextProjectionHead, get_posebridge_hpe  # noqa: E402
from posebridge.hpe.text_tower import OpenCLIPTextTower  # noqa: E402
from posebridge.utils import get_logger, resolve  # noqa: E402

CLIP_DIM = {"ViT-B-32": 512, "ViT-L-14": 768, "ViT-L-14-336": 768}


class WarmupCosineLR:
    """Per-iteration linear warmup, constant, then cosine over the second half."""

    def __init__(self, optimizer, total_epochs, steps_per_epoch, warmup_iters, warmup_factor, eta_min_factor):
        self.opt = optimizer
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.steps = steps_per_epoch
        self.warmup_iters, self.warmup_factor, self.eta_min_factor = warmup_iters, warmup_factor, eta_min_factor
        self.cos_start = (total_epochs // 2) * steps_per_epoch
        self.cos_iters = total_epochs * steps_per_epoch - self.cos_start

    def step(self, iteration, epoch):
        it = epoch * self.steps + iteration
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            if it < self.warmup_iters:
                a = it / self.warmup_iters
                g["lr"] = base * (self.warmup_factor + (1 - self.warmup_factor) * a)
            elif it >= self.cos_start:
                eta = base * self.eta_min_factor
                prog = min((it - self.cos_start) / self.cos_iters, 1.0)
                g["lr"] = eta + (base - eta) * (1 + math.cos(math.pi * prog)) / 2
            else:
                g["lr"] = base

    def get_lr(self):
        return [g["lr"] for g in self.opt.param_groups]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/hpe/posebridge_rtmpose-s.yaml")
    p.add_argument("--work-dir", default="work_dirs/hpe")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("opts", nargs=argparse.REMAINDER, help="extra yacs KEY VALUE pairs")
    args = p.parse_args()

    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(resolve(args.config))
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.DATASET.ROOT = resolve(cfg.DATASET.ROOT)
    cfg.MODEL.PRETRAINED = resolve(cfg.MODEL.PRETRAINED) if cfg.MODEL.PRETRAINED else ""
    cfg.GPUS = (args.device,)
    cfg.freeze()

    work_dir = resolve(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    logger = get_logger(os.path.join(work_dir, "train.log"), name="hpe")
    log = logger.info

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.cuda.set_device(args.device)

    model = get_posebridge_hpe(cfg, is_train=True).cuda()
    clip_name = cfg.TRAIN.TEXT_ENCODER.CLIP_MODEL
    clip_encoder = OpenCLIPTextTower(clip_name, pretrained=cfg.TRAIN.TEXT_ENCODER.OPENCLIP_PRETRAINED,
                                     device=args.device).eval()
    text_proj = TextProjectionHead(CLIP_DIM[clip_name], cfg.MODEL.ROUTE.PROJ_DIM, normalize=True).cuda()
    log(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
        f"scales={cfg.MODEL.ROUTE.NUM_SCALES}, gate={cfg.MODEL.ROUTE.FIXED_GATE}")

    criterion = KLDiscretLoss(beta=cfg.LOSS.BETA, label_softmax=cfg.LOSS.LABEL_SOFTMAX,
                              label_beta=cfg.LOSS.BETA, use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
    train_set = COCOCaptionPreextractDataset(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True)
    val_set = COCOCaptionPreextractDataset(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
                                               shuffle=True, num_workers=cfg.WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
                                             shuffle=False, num_workers=cfg.WORKERS, pin_memory=True)
    log(f"train {len(train_set)} / val {len(val_set)} person instances")

    base_lr = cfg.TRAIN.LR
    if cfg.TRAIN.AUTO_SCALE_LR:
        base_lr *= cfg.TRAIN.BATCH_SIZE_PER_GPU / cfg.TRAIN.BASE_BATCH_SIZE
    decay, no_decay = [], []
    for m in (model, text_proj):
        for n, prm in m.named_parameters():
            (no_decay if ("bias" in n or "bn" in n or "norm" in n) else decay).append(prm)
    log_temp = torch.nn.Parameter(torch.tensor(math.log(cfg.TRAIN.CLIP_TEMPERATURE), device="cuda"))
    no_decay.append(log_temp)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.TRAIN.WEIGHT_DECAY},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=base_lr)
    scheduler = WarmupCosineLR(optimizer, cfg.TRAIN.END_EPOCH, len(train_loader), cfg.TRAIN.WARMUP_ITERS,
                               cfg.TRAIN.WARMUP_FACTOR, cfg.TRAIN.ETA_MIN_FACTOR)

    start, best_ap = 0, 0.0
    last_path = os.path.join(work_dir, "last.pth")
    if os.path.isfile(last_path):
        ck = torch.load(last_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        text_proj.load_state_dict(ck["text_proj_state_dict"])
        log_temp.data = ck["log_temperature"]
        optimizer.load_state_dict(ck["optimizer"])
        start, best_ap = ck["epoch"], ck["best_ap"]
        log(f"resumed from epoch {start}")

    for epoch in range(start, cfg.TRAIN.END_EPOCH):
        if epoch >= cfg.TRAIN.STAGE2_EPOCH:
            train_set.set_stage2(enable=True)
        log(f"epoch {epoch}  lr {scheduler.get_lr()[0]:.6f}  temp {log_temp.exp().item():.4f}")
        train_posebridge(config=cfg, train_loader=train_loader, model=model, criterion=criterion,
                         optimizer=optimizer, epoch=epoch, output_dir=work_dir, print_log=log,
                         clip_encoder=clip_encoder, text_proj=text_proj, lr_scheduler=scheduler,
                         log_temperature=log_temp)

        if (epoch + 1) % cfg.TRAIN.VAL_INTERVAL == 0 or epoch >= cfg.TRAIN.END_EPOCH - 100:
            ap = validate_posebridge(config=cfg, val_loader=val_loader, val_dataset=val_set, model=model,
                                     criterion=criterion, output_dir=work_dir, print_log=log,
                                     clip_encoder=clip_encoder, text_proj=text_proj,
                                     temperature=log_temp.exp().item())
            if ap >= best_ap:
                best_ap = ap
                torch.save({"state_dict": model.state_dict(), "text_proj_state_dict": text_proj.state_dict(),
                            "log_temperature": log_temp.data.clone(), "perf": ap},
                           os.path.join(work_dir, "best.pth"))
            log(f"epoch {epoch}  AP {ap:.4f}  best {best_ap:.4f}")

        torch.save({"epoch": epoch + 1, "best_ap": best_ap, "state_dict": model.state_dict(),
                    "text_proj_state_dict": text_proj.state_dict(), "log_temperature": log_temp.data.clone(),
                    "optimizer": optimizer.state_dict()}, last_path)
    log(f"done. best AP {best_ap:.4f}")


if __name__ == "__main__":
    main()
