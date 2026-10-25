"""Person detection + tracking (YOLO11x + ByteTrack) -> <out-dir>/bbox_results.pkl.

NTU / PKU keep the 1-2 main actors (label-aware); Kinetics keeps the top-5 tracks.
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posebridge.hpe.detection import (PersonDetector, _interpolate_bbox_gaps, filter_by_confidence,  # noqa: E402
                                      filter_small_bboxes, pick_top_k_tracks, track_fill_by_main_tids_two_person)

# dataset -> (score thr, nms iou thr)
THRESHOLDS = {"ntu60": (0.3, 0.7), "ntu120": (0.3, 0.7), "pku51": (0.3, 0.7), "kinetics": (0.25, 0.5)}


def ntu_videos(root):
    return [(f[:-8], os.path.join(root, f)) for f in sorted(os.listdir(root)) if f.endswith("_rgb.avi")]


def kinetics_videos(root):
    out = []
    for split in ("train", "val", "test"):
        d = os.path.join(root, split)
        if os.path.isdir(d):
            out += [(f"{split}/{f[:-4]}", os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".mp4")]
    return out


def pku_segments(root):
    """Yields (capture_id, [(name, action, start, end)]) from label/<capture>.txt."""
    for txt in sorted(os.listdir(os.path.join(root, "label"))):
        cap, segs = txt[:-4], []
        with open(os.path.join(root, "label", txt)) as f:
            for line in f:
                tok = [t.strip() for t in line.strip().split(",") if t.strip()]
                if len(tok) < 3:
                    continue
                a, s, e = int(tok[0]), int(tok[1]), int(tok[2])
                if 1 <= a <= 51 and 0 < e and s <= e:
                    segs.append((f"{cap}_S{len(segs):03d}A{a:03d}", a, s, e))
        yield cap, segs


def entry(bboxes, **extra):
    tids = sorted({int(r[5]) for fd in bboxes for r in fd})
    return {"bboxes": [b.astype(np.float32) for b in bboxes], "n_frames": len(bboxes), "track_ids": tids, **extra}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(THRESHOLDS))
    p.add_argument("--video-root", required=True, help="NTU: *_rgb.avi dir | PKU: root with rgb/, label/ | "
                                                        "Kinetics: root with train/, val/")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="yolo11x.pt")
    p.add_argument("--part", default="1/1", help="shard k/n for multi-GPU runs")
    p.add_argument("--limit", type=int, default=0, help="debug: first N videos/captures")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    thr, iou = THRESHOLDS[args.dataset]
    det = PersonDetector(args.model, args.device, thr, iou)
    k, n = map(int, args.part.split("/"))
    results, ds = {}, args.dataset

    if ds == "pku51":
        caps = list(pku_segments(args.video_root))
        caps = caps[(k - 1) * len(caps) // n: k * len(caps) // n][: args.limit or None]
        for cap, segs in tqdm(caps, desc="detect"):
            video = cv2.VideoCapture(os.path.join(args.video_root, "rgb", f"{cap}.avi"))
            total, pos = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9, 0
            for name, action, s, e in sorted(segs, key=lambda x: x[2]):
                lo = max(0, s - 1)
                hi = max(lo + 1, min(total, e))
                if hi - lo > 300:  # center 300 frames
                    lo += (hi - lo - 300) // 2
                    hi = lo + 300
                if pos > lo:
                    video.release()
                    video, pos = cv2.VideoCapture(os.path.join(args.video_root, "rgb", f"{cap}.avi")), 0
                while pos < lo and video.grab():
                    pos += 1
                frames = []
                while pos < hi:
                    ok, f = video.read()
                    if not ok:
                        break
                    frames.append(f)
                    pos += 1
                if not frames:
                    continue
                b = det.track(frames, persist=False, batch=32)
                b = track_fill_by_main_tids_two_person(b, action - 1, ds, force_num_person=2)
                results[name] = entry(_interpolate_bbox_gaps(b, max_gap=3), start=s, end=e, action=action,
                                      window_lo=lo, window_hi=hi, capture_id=cap)
            video.release()
    else:
        videos = ntu_videos(args.video_root) if ds.startswith("ntu") else kinetics_videos(args.video_root)
        videos = videos[(k - 1) * len(videos) // n: k * len(videos) // n][: args.limit or None]
        for name, path in tqdm(videos, desc="detect"):
            if ds.startswith("ntu"):
                b = det.track(path, max_frames=300)
                if not b:
                    continue
                b = track_fill_by_main_tids_two_person(b, int(name[17:20]) - 1, ds)
            else:
                cap = cv2.VideoCapture(path)
                frames = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    frames.append(f)
                cap.release()
                if not frames:
                    continue
                h, w = frames[0].shape[:2]
                b = det.track(frames, batch=8)
                b = filter_small_bboxes(filter_by_confidence(b, thr), 0.005, (w, h))
                b = pick_top_k_tracks(b, K=5, img_wh=(w, h))
            results[name] = entry(_interpolate_bbox_gaps(b, max_gap=3))

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if n == 1 else f"_part{k}"
    with open(os.path.join(args.out_dir, f"bbox_results{suffix}.pkl"), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{len(results)} samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
