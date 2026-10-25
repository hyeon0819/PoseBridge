"""Person detection + tracking (RTMDet-nano or YOLO11, ByteTrack) and per-dataset track post-processing."""
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch

from ..utils import resolve

RTMDET_CKPT = resolve("checkpoints/hpe/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth")

TWO_PERSON = {
    "ntu60": set(range(49, 60)),
    "ntu120": set(range(49, 60)) | set(range(105, 120)),
    "pku51": set(),
}


def is_two_person_action(label, dataset):
    return label in TWO_PERSON.get(dataset, set())


def read_video(path, max_frames=0):
    cap, frames = cv2.VideoCapture(path), []
    while not max_frames or len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def bytetrack(dets, img_hw):
    """Per-frame [N, 5] detections -> per-frame [N, 6] (x1, y1, x2, y2, score, track_id)."""
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))
    out = []
    for d in dets:
        boxes = Boxes(np.concatenate([d, np.zeros((len(d), 1), np.float32)], 1), img_hw)  # cls 0 = person
        t = tracker.update(boxes)
        out.append(t[:, [0, 1, 2, 3, 5, 4]].astype(np.float32) if len(t) else np.zeros((0, 6), np.float32))
    return out


class PersonDetector:
    """Person detection (RTMDet-nano or YOLO11) + ByteTrack; per-frame [N, 6] (x1, y1, x2, y2, score, track_id)."""

    def __init__(self, detector="rtmdet", checkpoint=None, device="cuda:0", score_thr=0.3, iou_thr=0.7,
                 img_size=640):
        self.detector = detector
        self.device, self.score_thr, self.iou_thr, self.img_size = device, score_thr, iou_thr, img_size
        if detector == "rtmdet":
            from .rtmdet import RTMDetPerson
            self.model = RTMDetPerson(checkpoint or RTMDET_CKPT, device, score_thr, iou_thr)
        else:
            from ultralytics import YOLO
            self.model = YOLO(checkpoint or "yolo11x.pt")
            self.half = "cuda" in device and torch.cuda.is_available()
            self._warm = False

    def _warmup(self, tracker):
        if not self._warm:  # first-frame detections drop without a track() warmup
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in self.model.track(source=[dummy, dummy], imgsz=self.img_size, conf=self.score_thr,
                                      iou=self.iou_thr, classes=[0], device=self.device, half=self.half,
                                      verbose=False, persist=False, tracker=f"{tracker}.yaml", stream=True):
                pass
            self._warm = True

    @torch.no_grad()
    def track(self, source, tracker="bytetrack", persist=True, batch=1, max_frames=0):
        """source: video path or list of BGR frames."""
        if self.detector == "rtmdet":
            frames = read_video(source, max_frames) if isinstance(source, str) else source
            if not frames:
                return []
            dets = [d for i in range(0, len(frames), 64) for d in self.model(frames[i:i + 64])]
            return bytetrack(dets, frames[0].shape[:2])

        self._warmup(tracker)
        out = []
        for i, r in enumerate(self.model.track(source=source, imgsz=self.img_size, conf=self.score_thr,
                                               iou=self.iou_thr, classes=[0], device=self.device,
                                               half=self.half, verbose=False, persist=persist,
                                               tracker=f"{tracker}.yaml", stream=isinstance(source, str),
                                               batch=batch)):
            if max_frames and i >= max_frames:
                break
            if r.boxes is None or len(r.boxes) == 0:
                out.append(np.zeros((0, 6), dtype=np.float32))
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = r.boxes.conf.cpu().numpy().astype(np.float32)
            tid = (r.boxes.id.cpu().numpy() if r.boxes.id is not None else np.arange(len(xyxy))).astype(np.float32)
            out.append(np.concatenate([xyxy, conf[:, None], tid[:, None]], axis=1))
        if not isinstance(source, str):
            out += [np.zeros((0, 6), dtype=np.float32)] * (len(source) - len(out))
        return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0])); iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2])); iy2 = min(float(a[3]), float(b[3]))
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    aa = max(float(a[2]-a[0]), 0.0) * max(float(a[3]-a[1]), 0.0)
    bb = max(float(b[2]-b[0]), 0.0) * max(float(b[3]-b[1]), 0.0)
    return float(inter / (aa + bb - inter + 1e-6))

def _center_dist_norm(a: np.ndarray, b: np.ndarray) -> float:
    acx = (float(a[0]) + float(a[2])) * 0.5
    acy = (float(a[1]) + float(a[3])) * 0.5
    bcx = (float(b[0]) + float(b[2])) * 0.5
    bcy = (float(b[1]) + float(b[3])) * 0.5
    dist = float(np.hypot(acx - bcx, acy - bcy))
    w = max(float(a[2] - a[0]), 1.0)
    h = max(float(a[3] - a[1]), 1.0)
    scale = float(np.sqrt(w * h))
    return dist / (scale + 1e-6)

def _area(a: np.ndarray) -> float:
    w = max(float(a[2]-a[0]), 0.0)
    h = max(float(a[3]-a[1]), 0.0)
    return float(w * h)

def _log_area_ratio(a: np.ndarray, b: np.ndarray) -> float:
    aa = _area(a) + 1e-6
    bb = _area(b) + 1e-6
    return float(abs(np.log(aa / bb)))

def _representative_per_tid(frame_dets: np.ndarray) -> Dict[int, np.ndarray]:
    """
    frame_dets: [N,6] -> dict tid -> row (keep highest conf per tid)
    row format: [x1,y1,x2,y2,conf,tid]
    """
    rep: Dict[int, np.ndarray] = {}
    if frame_dets is None or len(frame_dets) == 0:
        return rep
    arr = np.asarray(frame_dets, dtype=np.float32)
    for row in arr:
        tid = int(row[5])
        if (tid not in rep) or (float(row[4]) > float(rep[tid][4])):
            rep[tid] = row.copy()
    return rep

def _pick_main_tids(
    reps_per_frame: List[Dict[int, np.ndarray]],
    K: int,
    img_wh: Optional[Tuple[float, float]] = (1920, 1080),
    w_count: float = 0.15,
    w_center: float = 0.15,
) -> List[int]:
    """Score = sum_conf + w_count * count + w_center * centrality * count; returns top-K tids."""

    total_conf: Dict[int, float] = {}
    count: Dict[int, int] = {}
    sum_center: Dict[int, np.ndarray] = {}
    all_x1 = []; all_y1 = []; all_x2 = []; all_y2 = []

    for rep in reps_per_frame:
        for tid, row in rep.items():
            tid = int(tid)
            row = np.asarray(row, dtype=np.float32)

            c = float(row[4])
            total_conf[tid] = total_conf.get(tid, 0.0) + c
            count[tid] = count.get(tid, 0) + 1

            cx = (float(row[0]) + float(row[2])) * 0.5
            cy = (float(row[1]) + float(row[3])) * 0.5
            if tid not in sum_center:
                sum_center[tid] = np.array([cx, cy], dtype=np.float32)
            else:
                sum_center[tid] += np.array([cx, cy], dtype=np.float32)

            all_x1.append(float(row[0])); all_y1.append(float(row[1]))
            all_x2.append(float(row[2])); all_y2.append(float(row[3]))

    if not total_conf:
        return []

    if img_wh is not None:
        W, H = float(img_wh[0]), float(img_wh[1])
        icx, icy = W * 0.5, H * 0.5
        diag = float(np.hypot(W, H)) + 1e-6
    else:
        minx = float(min(all_x1)) if all_x1 else 0.0
        miny = float(min(all_y1)) if all_y1 else 0.0
        maxx = float(max(all_x2)) if all_x2 else 1.0
        maxy = float(max(all_y2)) if all_y2 else 1.0
        W = max(maxx - minx, 1.0)
        H = max(maxy - miny, 1.0)
        icx, icy = minx + W * 0.5, miny + H * 0.5
        diag = float(np.hypot(W, H)) + 1e-6

    img_center = np.array([icx, icy], dtype=np.float32)

    items = []
    for tid in total_conf.keys():
        m = sum_center[tid] / float(count[tid])  # mean center of tid
        dist = float(np.linalg.norm(m - img_center)) / diag
        bonus = max(0.0, 1.0 - dist)
        sc = float(total_conf[tid]) + (w_count * float(count[tid])) + (w_center * bonus * float(count[tid]))
        items.append((sc, tid))

    items.sort(key=lambda x: x[0], reverse=True)
    picked = [int(tid) for _, tid in items[:K]]

    if len(picked) < K:
        for _, tid in items:
            if int(tid) not in picked:
                picked.append(int(tid))
            if len(picked) == K:
                break

    return picked[:K]

def _nearest_anchor(slots_out: List[List[Optional[np.ndarray]]], s: int, f: int, max_gap: int) -> Optional[np.ndarray]:
    """
    Find nearest existing bbox row for slot s around frame f within +-max_gap.
    Prefer closer in time; tie-break by higher conf.
    """
    T = len(slots_out[s])
    best = None
    best_dt = 1e9
    best_conf = -1e9

    for dt in range(1, max_gap + 1):
        for ff in (f - dt, f + dt):
            if ff < 0 or ff >= T:
                continue
            r = slots_out[s][ff]
            if r is None:
                continue
            conf = float(r[4])
            if dt < best_dt or (dt == best_dt and conf > best_conf):
                best = r
                best_dt = dt
                best_conf = conf
        if best_dt == 1:
            break
    return best

def _bruteforce_assign_K2(cost: np.ndarray) -> Tuple[int, int]:
    """
    cost: [2, M]
    returns (j0, j1) minimizing cost[0,j0]+cost[1,j1] with j0!=j1
    """
    _, M = cost.shape
    best_sum = 1e18
    best = (0, 1 if M > 1 else 0)
    for j0 in range(M):
        for j1 in range(M):
            if j1 == j0:
                continue
            s = float(cost[0, j0] + cost[1, j1])
            if s < best_sum:
                best_sum = s
                best = (j0, j1)
    return best


def track_fill_by_main_tids_two_person(
    bboxes_per_frame: List[np.ndarray],
    label: int,
    dataset: str,
    force_num_person: Optional[int] = None,
    min_conf: float = 0.0,
    max_gap: int = 30,               # anchor search radius (+/-)
    w_iou: float = 0.45,
    w_dist: float = 0.45,
    w_size: float = 0.10,
    w_conf: float = 0.10,            # subtract term
    tid_match_bonus: float = 0.25,   # strong preference for main tid when it appears
    seen_tid_bonus: float = 0.06,    # mild preference for tids already attached to the slot
    other_main_penalty: float = 0.25,# avoid stealing the other slot's main tid
    dup_iou_thr: float = 0.7,        # suppress duplicates if they collapse
) -> List[np.ndarray]:
    """
    Offline post-processing for detector + ByteTrack outputs (per-frame [N,6]):
      - Pick main tids (K=1 or 2)
      - For each frame, jointly assign detections to K slots using nearest anchors
      - If frame has >=K detections: FORCE assign K detections (no "unassigned" among chosen K)
      - Output tids are rewritten to main tids (slot canonical tids)
      - If frame has <K detections: assign what exists; remaining slots are None
    """

    if bboxes_per_frame is None or len(bboxes_per_frame) == 0:
        return []

    T = len(bboxes_per_frame)

    reps_per_frame: List[Dict[int, np.ndarray]] = []
    for f in range(T):
        rep = _representative_per_tid(bboxes_per_frame[f])
        if min_conf > 0:
            rep = {tid: row for tid, row in rep.items() if float(row[4]) >= min_conf}
        reps_per_frame.append(rep)

    try:
        is_two = is_two_person_action(label, dataset)
    except Exception:
        is_two = False

    if force_num_person is not None:
        K = 2 if int(force_num_person) >= 2 else 1
    else:
        if is_two:
            K = 2
        else:
            K = 1
    K = 2 if K >= 2 else 1

    if K == 2:
        ever2 = any(len(rep) >= 2 for rep in reps_per_frame)
        if not ever2:
            K = 1

    main_tids = _pick_main_tids(reps_per_frame, K)
    if len(main_tids) < K:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(T)]

    slots_out: List[List[Optional[np.ndarray]]] = [[None for _ in range(T)] for _ in range(K)]
    slot_attached_raw_tids: List[Set[int]] = [set([main_tids[s]]) for s in range(K)]

    for f in range(T):
        rep = reps_per_frame[f]
        for s in range(K):
            mt = int(main_tids[s])
            if mt in rep:
                r = rep[mt].copy().astype(np.float32)
                r[5] = float(mt)  # rewrite tid to main tid (already same)
                slots_out[s][f] = r

    for f in range(T):
        det_list = list(reps_per_frame[f].values())
        M = len(det_list)

        if M == 0:
            continue

        det_boxes = np.stack([d[:4] for d in det_list], axis=0).astype(np.float32)
        det_conf = np.array([float(d[4]) for d in det_list], dtype=np.float32)
        det_tid  = np.array([int(d[5]) for d in det_list], dtype=np.int32)

        if K == 1:
            if slots_out[0][f] is not None:
                slot_attached_raw_tids[0].add(int(det_tid[np.argmax(det_conf)]))
                continue

            anchor = _nearest_anchor(slots_out, 0, f, max_gap)
            if anchor is None:
                j = int(np.argmax(det_conf))
            else:
                ab = anchor[:4].astype(np.float32)
                best_j = 0
                best_c = 1e18
                for j in range(M):
                    iou = _iou(ab, det_boxes[j])
                    dist = _center_dist_norm(ab, det_boxes[j])
                    size = _log_area_ratio(ab, det_boxes[j])
                    c = (w_iou * (1.0 - iou)) + (w_dist * dist) + (w_size * size) - (w_conf * float(det_conf[j]))
                    if int(det_tid[j]) == int(main_tids[0]):
                        c -= tid_match_bonus
                    if int(det_tid[j]) in slot_attached_raw_tids[0]:
                        c -= seen_tid_bonus
                    if c < best_c:
                        best_c = c
                        best_j = j
                j = best_j

            r = det_list[j].copy().astype(np.float32)
            slot_attached_raw_tids[0].add(int(r[5]))
            r[5] = float(main_tids[0])
            slots_out[0][f] = r
            continue

        anchors = [None, None]
        for s in range(2):
            if slots_out[s][f] is not None:
                anchors[s] = slots_out[s][f]
            else:
                anchors[s] = _nearest_anchor(slots_out, s, f, max_gap)

        cost = np.zeros((2, M), dtype=np.float32)
        for s in range(2):
            mt = int(main_tids[s])
            other_mt = int(main_tids[1 - s])
            if anchors[s] is None:
                for j in range(M):
                    c = - (w_conf * float(det_conf[j]))
                    if int(det_tid[j]) == mt:
                        c -= tid_match_bonus
                    if int(det_tid[j]) in slot_attached_raw_tids[s]:
                        c -= seen_tid_bonus
                    if int(det_tid[j]) == other_mt:
                        c += other_main_penalty
                    cost[s, j] = float(c)
            else:
                ab = anchors[s][:4].astype(np.float32)
                for j in range(M):
                    iou = _iou(ab, det_boxes[j])
                    dist = _center_dist_norm(ab, det_boxes[j])
                    size = _log_area_ratio(ab, det_boxes[j])
                    c = (w_iou * (1.0 - iou)) + (w_dist * dist) + (w_size * size) - (w_conf * float(det_conf[j]))
                    if int(det_tid[j]) == mt:
                        c -= tid_match_bonus
                    if int(det_tid[j]) in slot_attached_raw_tids[s]:
                        c -= seen_tid_bonus
                    if int(det_tid[j]) == other_mt:
                        c += other_main_penalty
                    cost[s, j] = float(c)

        if M == 1:
            j = 0
            s_pick = 0 if float(cost[0, j]) <= float(cost[1, j]) else 1
            r = det_list[j].copy().astype(np.float32)
            slot_attached_raw_tids[s_pick].add(int(r[5]))
            r[5] = float(main_tids[s_pick])
            slots_out[s_pick][f] = r
        else:
            j0, j1 = _bruteforce_assign_K2(cost)
            r0 = det_list[j0].copy().astype(np.float32)
            r1 = det_list[j1].copy().astype(np.float32)

            slot_attached_raw_tids[0].add(int(r0[5]))
            slot_attached_raw_tids[1].add(int(r1[5]))

            r0[5] = float(main_tids[0])
            r1[5] = float(main_tids[1])

            slots_out[0][f] = r0
            slots_out[1][f] = r1

            if _iou(slots_out[0][f][:4], slots_out[1][f][:4]) >= dup_iou_thr:
                if float(slots_out[0][f][4]) >= float(slots_out[1][f][4]):
                    slots_out[1][f] = None
                else:
                    slots_out[0][f] = None

    out: List[np.ndarray] = []
    for f in range(T):
        rows = []
        for s in range(K):
            r = slots_out[s][f]
            if r is not None:
                rows.append(r.astype(np.float32))
        if not rows:
            out.append(np.zeros((0, 6), dtype=np.float32))
        else:
            arr = np.stack(rows, axis=0).astype(np.float32)
            if arr.shape[0] > 1:
                if _iou(arr[0, :4], arr[1, :4]) >= dup_iou_thr:
                    arr = arr[[0]] if float(arr[0, 4]) >= float(arr[1, 4]) else arr[[1]]
            out.append(arr)
    return out


def _interpolate_bbox_gaps(bboxes_per_frame: list, max_gap: int = 3) -> list:
    """
    Linear interpolation for short bbox gaps (<=max_gap frames).
    Per track: if frame t has no bbox but t-1 and t+1 do, fill with interpolated bbox.
    Same logic as inference._interpolate_gaps but for bbox structure.
    """
    if max_gap <= 0 or not bboxes_per_frame:
        return bboxes_per_frame

    T = len(bboxes_per_frame)
    track_ids = set()
    for fd in bboxes_per_frame:
        for row in fd:
            if len(row) >= 6:
                track_ids.add(int(row[5]))
    track_ids = sorted(track_ids)

    per_track = {}
    for tid in track_ids:
        arr = np.zeros((T, 6), dtype=np.float32)
        for t, fd in enumerate(bboxes_per_frame):
            for row in fd:
                if len(row) >= 6 and int(row[5]) == tid:
                    arr[t] = row[:6]
                    break
        per_track[tid] = arr

    for tid in track_ids:
        data = per_track[tid]
        flat = data.reshape(T, -1)
        is_valid = np.any(flat != 0, axis=1)

        t = 0
        while t < T:
            if is_valid[t]:
                t += 1
                continue
            gap_start = t
            while t < T and not is_valid[t]:
                t += 1
            gap_end = t
            gap_len = gap_end - gap_start

            if gap_len > max_gap:
                continue

            left = gap_start - 1
            right = gap_end
            if left < 0 or right >= T:
                continue

            for g in range(gap_len):
                alpha = (g + 1) / (gap_len + 1)
                flat[gap_start + g] = flat[left] * (1 - alpha) + flat[right] * alpha

        per_track[tid] = data

    out = []
    for t in range(T):
        rows = []
        for tid in track_ids:
            row = per_track[tid][t]
            if np.any(row != 0):
                rows.append(row)
        out.append(np.array(rows, dtype=np.float32) if rows else np.zeros((0, 6), dtype=np.float32))
    return out



# ----- Kinetics: confidence / size filtering and top-K track selection -----
def filter_by_confidence(bboxes_per_frame: List[np.ndarray], min_conf: float) -> List[np.ndarray]:
    """Keep only detections with conf >= min_conf."""
    out = []
    for arr in bboxes_per_frame:
        if arr is None or len(arr) == 0:
            out.append(np.zeros((0, 6), dtype=np.float32))
            continue
        arr = np.asarray(arr, dtype=np.float32)
        mask = arr[:, 4] >= min_conf
        out.append(arr[mask] if mask.any() else np.zeros((0, 6), dtype=np.float32))
    return out


def filter_small_bboxes(
    bboxes_per_frame: List[np.ndarray],
    min_area_ratio: float = 0.005,
    img_wh: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Filter out very small bboxes (background/crowd).
    min_area_ratio: min bbox_area / (img_area). Default 0.0005 = 0.05%.
    img_wh: (W,H). If None, infer from max bbox extent.
    """
    if min_area_ratio <= 0:
        return bboxes_per_frame

    out = []
    for arr in bboxes_per_frame:
        if arr is None or len(arr) == 0:
            out.append(np.zeros((0, 6), dtype=np.float32))
            continue
        arr = np.asarray(arr, dtype=np.float32)
        x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        img_area = float(img_wh[0] * img_wh[1]) if img_wh else 1920.0 * 1080.0
        ratio = area / (img_area + 1e-9)
        keep = ratio >= min_area_ratio
        out.append(arr[keep] if keep.any() else np.zeros((0, 6), dtype=np.float32))
    return out


def _reps_per_frame(bboxes_per_frame: List[np.ndarray]) -> List[Dict[int, np.ndarray]]:
    """Per-frame: dict tid -> best row (highest conf). row format [x1,y1,x2,y2,conf,tid]."""
    reps = []
    for arr in bboxes_per_frame:
        d = {}
        if arr is not None and len(arr) > 0:
            for row in np.asarray(arr, dtype=np.float32):
                tid = int(row[5])
                if tid not in d or float(row[4]) > float(d[tid][4]):
                    d[tid] = row.copy()
        reps.append(d)
    return reps


def pick_top_k_tracks(
    bboxes_per_frame: List[np.ndarray],
    K: int = 5,
    w_mean_conf: float = 0.30,
    w_max_conf: float = 0.25,
    w_mean_area: float = 0.10,
    w_max_area: float = 0.10,
    w_count: float = 0.10,
    w_center: float = 0.10,
    img_wh: Optional[Tuple[float, float]] = None,
) -> List[np.ndarray]:
    if K <= 0:
        return bboxes_per_frame

    reps = _reps_per_frame(bboxes_per_frame)

    total_conf: Dict[int, float] = {}
    total_area: Dict[int, float] = {}
    max_conf: Dict[int, float] = {}
    max_area: Dict[int, float] = {}
    sum_center: Dict[int, np.ndarray] = {}
    count: Dict[int, int] = {}

    for rep in reps:
        for tid, row in rep.items():
            tid = int(tid)
            conf = float(row[4])
            area = _area(row)
            cx = (float(row[0]) + float(row[2])) * 0.5
            cy = (float(row[1]) + float(row[3])) * 0.5

            total_conf[tid] = total_conf.get(tid, 0.0) + conf
            total_area[tid] = total_area.get(tid, 0.0) + area
            max_conf[tid] = max(max_conf.get(tid, 0.0), conf)
            max_area[tid] = max(max_area.get(tid, 0.0), area)

            if tid not in sum_center:
                sum_center[tid] = np.array([cx, cy], dtype=np.float32)
            else:
                sum_center[tid] += np.array([cx, cy], dtype=np.float32)

            count[tid] = count.get(tid, 0) + 1

    if not total_conf:
        return bboxes_per_frame

    max_mean_area = max((total_area[tid] / count[tid]) for tid in count) + 1e-9
    max_peak_area = max(max_area.values()) + 1e-9
    max_log_count = max(np.log1p(v) for v in count.values()) + 1e-9

    if img_wh is not None:
        W, H = float(img_wh[0]), float(img_wh[1])
        img_center = np.array([W * 0.5, H * 0.5], dtype=np.float32)
        diag = np.hypot(W, H) + 1e-6
    else:
        x1_list = [r[0] for rep in reps for r in rep.values()]
        x2_list = [r[2] for rep in reps for r in rep.values()]
        y1_list = [r[1] for rep in reps for r in rep.values()]
        y2_list = [r[3] for rep in reps for r in rep.values()]
        if x1_list and y1_list:
            W = max(x2_list) - min(x1_list) + 1.0
            H = max(y2_list) - min(y1_list) + 1.0
            img_center = np.array(
                [(min(x1_list) + max(x2_list)) * 0.5,
                 (min(y1_list) + max(y2_list)) * 0.5],
                dtype=np.float32,
            )
        else:
            img_center = np.array([960.0, 540.0], dtype=np.float32)
        W, H = 1920.0, 1080.0
        diag = np.hypot(W, H) + 1e-6

    items = []
    for tid in total_conf.keys():
        c = count[tid]
        mean_conf = total_conf[tid] / c
        mean_area = total_area[tid] / c
        peak_conf = max_conf[tid]
        peak_area = max_area[tid]
        log_c = np.log1p(c)
        center = sum_center[tid] / c
        dist = np.hypot(center[0] - img_center[0], center[1] - img_center[1]) / diag
        score = (
            w_mean_conf * (mean_conf / 1.0) +
            w_max_conf * (peak_conf / 1.0) +
            w_mean_area * (mean_area / max_mean_area) +
            w_max_area * (peak_area / max_peak_area) +
            w_count * (log_c / max_log_count) +
            w_center * (1.0 - min(dist, 1.0))
        )
        items.append((tid, score))

    items.sort(key=lambda x: -x[1])
    keep_tids = set(tid for tid, _ in items[:K])

    out = []
    for arr in bboxes_per_frame:
        if arr is None or len(arr) == 0:
            out.append(np.zeros((0, 6), dtype=np.float32))
            continue
        arr = np.asarray(arr, dtype=np.float32)
        mask = np.array([int(arr[i, 5]) in keep_tids for i in range(len(arr))])
        out.append(arr[mask] if mask.any() else np.zeros((0, 6), dtype=np.float32))
    return out


