import torch
import torch.nn.functional as F


def ce_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def supcon_loss(z, labels, temperature=0.07):
    """Supervised contrastive loss on L2-normalized embeddings."""
    n = z.size(0)
    pos = (labels[None] == labels[:, None]).float()
    pos.fill_diagonal_(0)
    logits = z @ z.T / temperature - 1e9 * torch.eye(n, device=z.device)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return (-(pos * log_prob).sum(1) / pos.sum(1).clamp(min=1)).mean()
