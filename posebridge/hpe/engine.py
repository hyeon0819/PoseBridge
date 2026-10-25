# ------------------------------------------------------------------------------
# Per-epoch train / validate routines for PoseBridge HPE.
#   - Pose loss     : KLDiscretLoss on SimCC outputs.
#   - Semantic loss : symmetric CLIP contrastive loss between pose_sem and
#                     CLIP text features (frozen text tower, learnable head).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .helpers import (
    AverageMeter,
    clip_contrastive_loss,
    decode_simcc,
    get_raw_text_features,
    simcc_pck_accuracy,
    transform_preds,
    _print_name_value,
)


def train_posebridge(
    config,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    output_dir,
    print_log,
    clip_encoder,
    text_proj,
    lr_scheduler=None,
    log_temperature=None,
):
    """Train one epoch of PoseBridge HPE."""

    losses_m = AverageMeter()
    pose_losses_m = AverageMeter()
    clip_m = AverageMeter()
    acc_m = AverageMeter()

    clip_w = config.TRAIN.CLIP_LOSS_WEIGHT
    use_learnable_temp = log_temperature is not None

    model.train()
    text_proj.train()
    clip_encoder.eval()

    pbar = tqdm(train_loader, desc=f'PoseBridge Train Epoch {epoch}',
                dynamic_ncols=True)

    for i, (input_tensor, target_x, target_y, target_weight, meta) in enumerate(pbar):
        if lr_scheduler is not None:
            lr_scheduler.step(iteration=i, epoch=epoch)

        input_tensor = input_tensor.cuda(non_blocking=True)
        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        pred_x, pred_y, pose_sem = model(input_tensor)

        pose_loss = criterion((pred_x, pred_y), (target_x, target_y), target_weight)

        raw_text_feat = get_raw_text_features(meta, clip_encoder, input_tensor.device)
        text_feat_proj = text_proj(raw_text_feat)

        temp = torch.exp(log_temperature) if use_learnable_temp else config.TRAIN.CLIP_TEMPERATURE
        clip_loss = clip_contrastive_loss(pose_sem, text_feat_proj, temp)

        loss = pose_loss + clip_w * clip_loss

        optimizer.zero_grad()
        loss.backward()

        if config.TRAIN.CLIP_GRAD:
            all_params = list(model.parameters()) + list(text_proj.parameters())
            if use_learnable_temp and log_temperature is not None:
                all_params.append(log_temperature)
            torch.nn.utils.clip_grad_norm_(
                all_params,
                config.TRAIN.CLIP_MAX_NORM,
                norm_type=config.TRAIN.CLIP_NORM_TYPE,
            )

        optimizer.step()

        bs = input_tensor.size(0)
        losses_m.update(loss.item(), bs)
        pose_losses_m.update(pose_loss.item(), bs)
        clip_m.update(clip_loss.item(), bs)

        _, avg_acc, cnt = simcc_pck_accuracy(
            output=(pred_x.detach().cpu().numpy(), pred_y.detach().cpu().numpy()),
            target=(target_x.detach().cpu().numpy(), target_y.detach().cpu().numpy()),
            simcc_split_ratio=config.MODEL.SIMCC.SPLIT_RATIO,
            mask=target_weight.detach().cpu().numpy() > 0,
        )
        acc_m.update(avg_acc, cnt)

        pbar.set_postfix(
            loss=f'{losses_m.avg:.5f}',
            pose=f'{pose_losses_m.avg:.5f}',
            clip=f'{clip_m.avg:.5f}',
            acc=f'{acc_m.avg:.3f}',
        )

    gate_str = ' '.join(f'{n}={v:.4f}' for n, v in model.get_gate_values())
    print_log(
        f'PoseBridge Train Epoch {epoch} | '
        f'Loss: {losses_m.avg:.6f} | Pose: {pose_losses_m.avg:.6f} | '
        f'CLIP: {clip_m.avg:.6f} | Acc: {acc_m.avg:.3f} | '
        f'Gate({gate_str})'
    )


def validate_posebridge(
    config,
    val_loader,
    val_dataset,
    model,
    criterion,
    output_dir,
    print_log,
    clip_encoder,
    text_proj,
    temperature=None,
):
    """Validate PoseBridge HPE on the COCO val set. Returns COCO AP."""

    losses_m = AverageMeter()
    pose_losses_m = AverageMeter()
    clip_m = AverageMeter()
    acc_m = AverageMeter()

    temp = temperature if temperature is not None else config.TRAIN.CLIP_TEMPERATURE
    clip_w = config.TRAIN.CLIP_LOSS_WEIGHT

    model.eval()
    text_proj.eval()
    clip_encoder.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6), dtype=np.float32)
    image_path = []
    idx = 0

    _FLIP_INDICES_COCO17 = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    with torch.no_grad():
        for i, (input_tensor, target_x, target_y, target_weight, meta) in enumerate(
                tqdm(val_loader, desc='PoseBridge Valid', dynamic_ncols=True)):

            input_tensor = input_tensor.cuda(non_blocking=True)
            target_x = target_x.cuda(non_blocking=True)
            target_y = target_y.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            pred_x, pred_y, pose_sem = model(input_tensor)

            if config.TEST.FLIP_TEST:
                inp_flip = input_tensor.flip(3)
                pred_x_f, pred_y_f, pose_sem_f = model(inp_flip)
                fi = _FLIP_INDICES_COCO17
                pred_x_f = pred_x_f[:, fi].flip(-1)
                pred_y_f = pred_y_f[:, fi]
                pred_x = (pred_x + pred_x_f) * 0.5
                pred_y = (pred_y + pred_y_f) * 0.5
                pose_sem = F.normalize((pose_sem + pose_sem_f) * 0.5, dim=-1, eps=1e-6)

            pose_loss = criterion((pred_x, pred_y), (target_x, target_y), target_weight)

            raw_text_feat = get_raw_text_features(meta, clip_encoder, input_tensor.device)
            text_feat_proj = text_proj(raw_text_feat)
            clip_loss = clip_contrastive_loss(pose_sem, text_feat_proj, temp)
            loss = pose_loss + clip_w * clip_loss

            bs = input_tensor.size(0)
            losses_m.update(loss.item(), bs)
            pose_losses_m.update(pose_loss.item(), bs)
            clip_m.update(clip_loss.item(), bs)

            _, avg_acc, cnt = simcc_pck_accuracy(
                output=(pred_x.cpu().numpy(), pred_y.cpu().numpy()),
                target=(target_x.cpu().numpy(), target_y.cpu().numpy()),
                simcc_split_ratio=config.MODEL.SIMCC.SPLIT_RATIO,
                mask=target_weight.cpu().numpy() > 0,
            )
            acc_m.update(avg_acc, cnt)

            preds, scores = decode_simcc(
                pred_x.cpu().numpy(), pred_y.cpu().numpy(),
                config.MODEL.SIMCC.SPLIT_RATIO,
            )
            center = meta['center'].numpy()
            scale_ = meta['scale'].numpy()
            score = meta.get('score', np.ones(bs))
            if isinstance(score, torch.Tensor):
                score = score.numpy()

            preds = transform_preds(
                preds, center, scale_,
                [config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]])

            all_preds[idx:idx + bs, :, :2] = preds[:, :, :2]
            all_preds[idx:idx + bs, :, 2] = scores
            all_boxes[idx:idx + bs, 0:2] = center
            all_boxes[idx:idx + bs, 2:4] = scale_
            all_boxes[idx:idx + bs, 4] = np.prod(scale_ * 200, axis=1)
            all_boxes[idx:idx + bs, 5] = score if not isinstance(score, (int, float)) \
                else np.full(bs, score)

            if 'image' in meta:
                image_path.extend(meta['image'])
            else:
                image_path.extend([''] * bs)
            idx += bs

    gate_str = ' '.join(f'{n}={v:.4f}' for n, v in model.get_gate_values())
    print_log(
        f'PoseBridge Valid | Loss: {losses_m.avg:.6f} | '
        f'Pose: {pose_losses_m.avg:.6f} | CLIP: {clip_m.avg:.6f} | '
        f'Acc: {acc_m.avg:.3f} | Gate({gate_str})'
    )

    name_values, perf_indicator = val_dataset.evaluate(
        config, all_preds, output_dir, all_boxes, image_path)

    model_name = f'{config.MODEL.NAME}-{config.MODEL.SIZE}'
    if isinstance(name_values, list):
        for nv in name_values:
            _print_name_value(nv, model_name, print_log)
    else:
        _print_name_value(name_values, model_name, print_log)

    return perf_indicator
