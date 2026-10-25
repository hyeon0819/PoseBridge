"""Frozen CLIP text features for class descriptions (cached on disk)."""
import csv
import os

import torch
import torch.nn.functional as F

from ..utils import ROOT

SEMANTIC_DIR = os.path.join(ROOT, "semantics")
TEMPLATES = [
    "A person is {}.",
    "Someone performs the action of {}.",
    "The human action: {}.",
    "A video of a person {}.",
    "An individual doing the motion of {}.",
    "The activity of {}.",
    "A person demonstrating {}.",
    "This shows a person {}.",
]


def _read_label_desc(dataset):
    with open(os.path.join(SEMANTIC_DIR, f"{dataset}.csv"), encoding="utf-8") as f:
        rows = sorted((int(r["idx"]), r["label"].strip()) for r in csv.DictReader(f))
    return [r[1] for r in rows]


def _read_llm_desc(dataset):
    with open(os.path.join(SEMANTIC_DIR, f"{dataset}_llm.txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _action_name(desc):
    if '"' in desc:
        return desc.split('"')[1]
    return desc.split(":")[0].strip() if ":" in desc else desc.split(".")[0].strip()


@torch.no_grad()
def _encode(model, tokenizer, texts, device):
    return model.encode_text(tokenizer(texts).to(device)).float()


def text_features(dataset, clip_model="ViT-L-14", pretrained="openai", device="cpu"):
    """Returns dict(label, llm, multi_template), each (C, D) L2-normalized."""
    cache = os.path.join(SEMANTIC_DIR, "clip_features", f"{dataset}_{clip_model}_{pretrained}.pt")
    if os.path.isfile(cache):
        return torch.load(cache, map_location="cpu", weights_only=True)

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(clip_model)

    llm = _read_llm_desc(dataset)
    mt = []
    for name in map(_action_name, llm):
        f = _encode(model, tok, [t.format(name.lower()) for t in TEMPLATES], device)
        mt.append(F.normalize(f.mean(0), dim=-1))
    feats = {
        "label": F.normalize(_encode(model, tok, _read_label_desc(dataset), device), dim=-1).cpu(),
        "llm": F.normalize(_encode(model, tok, llm, device), dim=-1).cpu(),
        "multi_template": torch.stack(mt).cpu(),
    }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(feats, cache)
    return feats
