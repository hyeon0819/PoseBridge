"""ZSL / GZSL evaluation with a fixed inference config."""
import torch


@torch.no_grad()
def embed(model, split, device, batch_size=256):
    """Returns (z_s, z_b, z_p, labels) on device."""
    model.eval()
    zs, zb, zp = [], [], []
    for i in range(0, len(split["ske"]), batch_size):
        ske = split["ske"][i:i + batch_size].to(device)
        cues = split["cues"][i:i + batch_size].to(device).float()
        zs.append(model.encode_skeleton(ske))
        zb.append(model.encode_bridge(ske, cues))
        zp.append(model.encode_pose(cues))
    return torch.cat(zs), torch.cat(zb), torch.cat(zp), split["label"].to(device)


def scores(model, emb, cfg):
    """Query = beta * z_s + gamma * z_b + alpha * z_p, matched to adapted prototypes."""
    z_s, z_b, z_p, _ = emb
    query = cfg["beta"] * z_s + cfg["gamma"] * z_b + cfg["alpha"] * z_p
    protos = model.adapted_prototypes(cfg["rho"], cfg["topk"], neighbor_beta=cfg["neighbor_beta"],
                                      mt_weight=cfg["mt_weight"])
    sim = query @ protos.T
    k = cfg["csls_k"]
    if k > 0:  # CSLS hubness correction
        sim = 2 * sim - sim.topk(k, dim=0).values.mean(0, keepdim=True)
    return sim


def per_class_acc(pred, labels, classes):
    accs = [(pred[labels == c] == c).float().mean() for c in classes if (labels == c).any()]
    return 100 * torch.stack(accs).mean().item()


@torch.no_grad()
def evaluate_zsl(model, emb, cfg):
    s = scores(model, emb, cfg)
    unseen = model._unseen_idx
    pred = unseen[s[:, unseen].argmax(1)]
    return per_class_acc(pred, emb[3], unseen.tolist())


@torch.no_grad()
def evaluate_gzsl(model, emb, cfg):
    """Calibrated stacking: subtract kappa from seen-class scores."""
    s = scores(model, emb, cfg)
    s[:, model._seen_idx] -= cfg["kappa"]
    pred = s.argmax(1)
    labels = emb[3]
    acc_s = per_class_acc(pred, labels, model._seen_idx.tolist())
    acc_u = per_class_acc(pred, labels, model._unseen_idx.tolist())
    h = 2 * acc_s * acc_u / (acc_s + acc_u) if acc_s + acc_u > 0 else 0.0
    return acc_s, acc_u, h
