"""Pre-encode COCO captions with the frozen CLIP text tower (person-keypoint images only)."""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posebridge.hpe.coco_caption_preextract import CLIP_TO_SUBDIR  # noqa: E402
from posebridge.hpe.text_tower import OpenCLIPTextTower  # noqa: E402
from posebridge.utils import resolve  # noqa: E402


def encode_split(root, split, encoder, clip_model, batch_size, out_dir):
    with open(os.path.join(root, "annotations", f"captions_{split}.json")) as f:
        caps = defaultdict(list)
        for a in json.load(f)["annotations"]:
            caps[a["image_id"]].append(a["caption"])
    with open(os.path.join(root, "annotations", f"person_keypoints_{split}.json")) as f:
        person_ids = {a["image_id"] for a in json.load(f)["annotations"] if a.get("category_id") == 1}
    rows = [(i, j, t) for i in sorted(caps) if i in person_ids for j, t in enumerate(caps[i])]

    feats = {}
    with torch.no_grad():
        for s in tqdm(range(0, len(rows), batch_size), desc=split):
            chunk = rows[s:s + batch_size]
            out = encoder([t for _, _, t in chunk]).float().cpu()
            for k, (i, j, _) in enumerate(chunk):
                feats[f"{i}_{j}"] = out[k].clone()
    path = os.path.join(out_dir, f"{split}.pt")
    torch.save({"meta": {"clip_model": clip_model, "image_set": split}, "features": feats}, path)
    print(f"{split}: {len(feats)} captions -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", default="data/coco")
    p.add_argument("--clip-model", default="ViT-B-32")
    p.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    root = resolve(args.coco_root)
    out_dir = os.path.join(root, "caption_features", CLIP_TO_SUBDIR[args.clip_model])
    os.makedirs(out_dir, exist_ok=True)
    encoder = OpenCLIPTextTower(args.clip_model, pretrained="openai", device=args.device).eval()
    for split in args.splits:
        encode_split(root, split, encoder, args.clip_model, args.batch_size, out_dir)


if __name__ == "__main__":
    main()
