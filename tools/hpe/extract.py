"""Run PoseBridge-HPE on tracked boxes -> skeleton_results.pkl + per-video pose cues (pose_emb/)."""
import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posebridge.hpe.detection import TWO_PERSON  # noqa: E402
from posebridge.hpe.extraction import Extractor, finalize, prepare, read_frames, read_windows, run  # noqa: E402
from posebridge.utils import resolve  # noqa: E402

# dataset -> (max persons, expected persons from label, gap interpolation)
SETTINGS = {"ntu60": (2, True, 0), "ntu120": (2, True, 0), "pku51": (2, False, 0), "kinetics": (5, False, 3)}


def load_boxes(box_dir):
    boxes = {}
    for f in sorted(glob.glob(os.path.join(box_dir, "bbox_results*.pkl"))):
        with open(f, "rb") as fh:
            boxes.update(pickle.load(fh))
    return boxes


def make_chunks(dataset, boxes, video_root, size):
    """Yields lists of (name, frames_loader, bboxes)."""
    if dataset == "pku51":
        by_capture = defaultdict(list)
        for n, v in boxes.items():
            by_capture[v["capture_id"]].append(n)
        caps = sorted(by_capture)
        for i in range(0, len(caps), max(1, size // 8)):
            items = []
            for cap in caps[i:i + max(1, size // 8)]:
                names = sorted(by_capture[cap])
                windows = [(n, boxes[n]["window_lo"], boxes[n]["window_hi"]) for n in names]
                load = (lambda c=cap, w=windows: read_windows(os.path.join(video_root, "rgb", f"{c}.avi"), w))
                items.append((names, load))
            yield items
        return
    names = sorted(boxes)
    ext = "_rgb.avi" if dataset.startswith("ntu") else ".mp4"
    for i in range(0, len(names), size):
        items = []
        for n in names[i:i + size]:
            path = os.path.join(video_root, n + ext)
            items.append(([n], (lambda p=path, k=n: {k: read_frames(p, len(boxes[k]["bboxes"]))})))
        yield items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(SETTINGS))
    p.add_argument("--video-root", required=True, help="same as tools/hpe/detect.py")
    p.add_argument("--box-dir", required=True, help="output of tools/hpe/detect.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", default="configs/hpe/posebridge_rtmpose-s.yaml")
    p.add_argument("--checkpoint", default="checkpoints/hpe/posebridge_rtmpose-s.pth")
    p.add_argument("--batch-videos", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="debug: first N videos/segments")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    max_persons, by_label, gap = SETTINGS[args.dataset]
    ex = Extractor(resolve(args.config), resolve(args.checkpoint), args.device)
    boxes = load_boxes(args.box_dir)
    if args.limit:
        boxes = {k: boxes[k] for k in sorted(boxes)[:args.limit]}
    emb_dir = os.path.join(args.out_dir, "pose_emb")
    os.makedirs(emb_dir, exist_ok=True)

    def expected(name):
        if not by_label:
            return max_persons if args.dataset == "pku51" else None
        return 2 if int(name[17:20]) - 1 in TWO_PERSON[args.dataset] else 1

    results, pool = {}, ThreadPoolExecutor(8)
    chunks = list(make_chunks(args.dataset, boxes, args.video_root, args.batch_videos))
    futures = None
    for ci in tqdm(range(len(chunks)), desc="extract"):
        loaded = futures or [pool.submit(load) for _, load in chunks[ci]]
        futures = [pool.submit(load) for _, load in chunks[ci + 1]] if ci + 1 < len(chunks) else None
        frames = {}
        for f in loaded:
            frames.update(f.result())
        states = [prepare(n, frames.get(n, []), boxes[n]["bboxes"], max_persons, ex.dim, expected(n))
                  for names, _ in chunks[ci] for n in names]
        run(states, ex)
        for s in states:
            if s is None:
                continue
            skel, cues = finalize(s, gap)
            results[s["name"]] = {"skeleton": skel, "num_persons": skel.shape[0], "frame_count": s["T"]}
            path = os.path.join(emb_dir, s["name"] + ".npz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, pose_sem=cues)

    with open(os.path.join(args.out_dir, "skeleton_results.pkl"), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(results)}/{len(boxes)} videos -> {args.out_dir}")


if __name__ == "__main__":
    main()
