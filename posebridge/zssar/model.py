"""PoseBridge ZSSAR: skeleton-conditioned semantic bridge + prototype adaptation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import ce_loss, supcon_loss


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, norm="bn", dropout=0.1, n_layers=2):
        super().__init__()
        layers, prev = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(prev, hidden_dim),
                       nn.BatchNorm1d(hidden_dim) if norm == "bn" else nn.LayerNorm(hidden_dim),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalPooling(nn.Module):
    """Gated mix of attention pooling and mean pooling over T cues."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 4), nn.Tanh(),
                                  nn.Dropout(dropout), nn.Linear(dim // 4, 1))
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        w = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        g = torch.sigmoid(self.gate)
        return g * (x * w.unsqueeze(-1)).sum(1) + (1 - g) * x.mean(1)


class SemanticBridge(nn.Module):
    """z_s queries temporal pose cues (Eq. 5)."""

    def __init__(self, embed_dim, pose_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.kv_proj = nn.Linear(pose_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(embed_dim * 2, embed_dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.bridge_gate = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, z_s, cues):
        kv = self.kv_proj(cues)
        attn, _ = self.cross_attn(z_s.unsqueeze(1), kv, kv)
        h = self.norm1(z_s + torch.sigmoid(self.bridge_gate) * attn.squeeze(1))
        return self.norm2(h + self.ffn(h))


class PoseBridgeZSSAR(nn.Module):
    def __init__(self, text_feats, seen, unseen, ske_dim=256, pose_dim=512, embed_dim=512,
                 dropout=0.1, bridge_heads=4, text_neighbor_k=5, text_neighbor_beta=0.3,
                 multi_template_weight=0.1, loss_weights=None, supcon_temp=0.07, distill_temp=4.0):
        super().__init__()
        self.seen, self.unseen = list(seen), list(unseen)
        self.num_classes = len(self.seen) + len(self.unseen)
        self.text_neighbor_k = text_neighbor_k
        self.text_neighbor_beta = text_neighbor_beta
        self.mt_weight = multi_template_weight
        self.w = loss_weights or {}
        self.supcon_temp = supcon_temp
        self.distill_temp = distill_temp
        clip_dim = text_feats["llm"].shape[1]

        self.ske_projector = MLP(ske_dim, embed_dim, embed_dim, "bn", dropout, 2)
        self.temporal_pose_attn = TemporalPooling(pose_dim, dropout)
        self.pose_adapter = MLP(pose_dim, embed_dim, embed_dim, "bn", dropout, 1)
        self.text_projector = MLP(clip_dim, embed_dim, embed_dim, "ln", dropout, 2)
        self.bridge_module = SemanticBridge(embed_dim, pose_dim, bridge_heads, dropout)
        self.ske_classifier = nn.Linear(embed_dim, len(self.seen))
        self.pose_classifier = nn.Linear(embed_dim, len(self.seen))
        init = torch.log(torch.tensor(1 / 0.07))
        self.logit_scale_ske = nn.Parameter(init.clone())
        self.logit_scale_pose = nn.Parameter(init.clone())
        self.logit_scale_bridge = nn.Parameter(init.clone())
        self.class_gate_logit = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("_seen_idx", torch.tensor(self.seen))
        self.register_buffer("_unseen_idx", torch.tensor(self.unseen))
        self.register_buffer("csv_text_feats", text_feats["label"].float())
        self.register_buffer("llm_text_feats", text_feats["llm"].float())
        self.register_buffer("mt_text_feats", text_feats["multi_template"].float())
        self.register_buffer("pose_centroids", torch.zeros(self.num_classes, embed_dim))

    # ----- text prototypes -----
    def raw_text(self):
        g = torch.sigmoid(self.class_gate_logit)
        t = g * self.csv_text_feats + (1 - g) * self.llm_text_feats
        if self.mt_weight > 0:
            t = (1 - self.mt_weight) * t + self.mt_weight * self.mt_text_feats
        return F.normalize(t, dim=-1)

    def text_prototypes(self, neighbor_beta=0.0, mt_weight=0.0):
        t = self.raw_text()
        if neighbor_beta > 0 and self.text_neighbor_k > 0:
            sim = t @ t.T
            sim.fill_diagonal_(-1e9)
            vals, idx = sim.topk(self.text_neighbor_k, dim=-1)
            t = t + neighbor_beta * (F.softmax(vals, -1).unsqueeze(-1) * t[idx]).sum(1)
        protos = F.normalize(self.text_projector(t), dim=-1)
        if mt_weight > 0:
            mt = F.normalize(self.text_projector(self.mt_text_feats), dim=-1)
            protos = F.normalize((1 - mt_weight) * protos + mt_weight * mt, dim=-1)
        return protos

    def adapted_prototypes(self, rho=0.2, topk=5, tau=0.1, neighbor_beta=0.0, mt_weight=0.0):
        """Semantic prototype adaptation (Eq. 6-7)."""
        t = self.text_prototypes(neighbor_beta, mt_weight)
        if rho <= 0:
            return t
        seen = self._seen_idx
        mu = self.pose_centroids[seen]
        delta = mu - t[seen]
        out = t.clone()
        out[seen] = F.normalize((1 - rho) * t[seen] + rho * mu, dim=-1)
        u = self._unseen_idx
        vals, idx = (t[u] @ t[seen].T).topk(min(topk, len(seen)), dim=-1)
        w = F.softmax(vals / tau, dim=-1).unsqueeze(-1)
        out[u] = F.normalize(t[u] + rho * (w * delta[idx]).sum(1), dim=-1)
        return out

    # ----- encoders -----
    def encode_skeleton(self, ske):
        return F.normalize(self.ske_projector(ske), dim=-1)

    def encode_pose(self, cues):
        return F.normalize(self.pose_adapter(self.temporal_pose_attn(cues)), dim=-1)

    def encode_bridge(self, ske, cues):
        return F.normalize(self.bridge_module(self.ske_projector(ske), cues), dim=-1)

    @torch.no_grad()
    def set_pose_centroids(self, cues, labels, batch_size=4096):
        z = torch.cat([self.encode_pose(cues[i:i + batch_size].to(self.pose_centroids.device).float())
                       for i in range(0, len(cues), batch_size)])
        labels = labels.to(z.device)
        for c in labels.unique().tolist():
            self.pose_centroids[c] = F.normalize(z[labels == c].mean(0), dim=0)

    # ----- training -----
    def loss(self, ske, cues, y_seen, y):
        w, T = self.w, self.distill_temp
        z_s = self.encode_skeleton(ske)
        z_p = self.encode_pose(cues)
        seen_p = self.text_prototypes(self.text_neighbor_beta)[self._seen_idx]
        seen_p_det = seen_p.detach()
        s_s = self.logit_scale_ske.exp().clamp(max=200)
        s_p = self.logit_scale_pose.exp().clamp(max=200)
        s_b = self.logit_scale_bridge.exp().clamp(max=200)
        z_b = self.encode_bridge(ske, cues)

        teacher = F.softmax(z_p.detach() @ seen_p_det.T / T, dim=-1)
        kd = F.kl_div(F.log_softmax(z_s @ seen_p.T / T, dim=-1), teacher, reduction="batchmean") * T * T
        loss = (
            w["ske_cls"] * ce_loss(self.ske_classifier(z_s), y_seen)
            + w["ske_sem"] * ce_loss(s_s * z_s @ seen_p.T, y_seen)
            + w["ske_con"] * supcon_loss(z_s, y, self.supcon_temp)
            + w["pose_cls"] * ce_loss(self.pose_classifier(z_p), y_seen)
            + w["pose_sem"] * ce_loss(s_p * z_p @ seen_p_det.T, y_seen)
            + w["pose_con"] * supcon_loss(z_p, y, self.supcon_temp)
            + w["bridge_sem"] * ce_loss(s_b * z_b @ seen_p_det.T, y_seen)
            + w["bridge_con"] * supcon_loss(z_b, y, self.supcon_temp)
            + w["s2p"] * (1 - F.cosine_similarity(z_s, z_p)).mean()
            + w["kd"] * kd
        )
        return loss
