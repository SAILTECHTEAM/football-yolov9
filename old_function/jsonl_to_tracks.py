#!/usr/bin/env python3
import numpy as np
import os, json, argparse, glob
from pathlib import Path
from types import SimpleNamespace
from multiprocessing import Pool, cpu_count
from collections import defaultdict, OrderedDict
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracker.byte_tracker import STrack
from collections import defaultdict as _dd

class TrackJsonlStreamer:
    """
    Streams per-track data to disk as JSONL.
    Each track is written as a single line once it becomes stale.
    """
    def __init__(self, out_path: str, flush_interval: int = 500, lost_thresh: int = 100):
        self.out_path = Path(out_path)
        self.flush_interval = flush_interval
        self.lost_thresh = lost_thresh
        self.records = _dd(lambda: {
            "track_id": None,
            "frame_id": [],
            "conf": [],
            "bbox": [],
            "projected": [],
        })
        self.last_seen = {}
        self.fh = self.out_path.open("w", encoding="utf-8")

    def update(self, tid, frame_idx, bbox, conf, proj_pt):
        rec = self.records[tid]
        if rec["track_id"] is None:
            rec["track_id"] = tid
        rec["frame_id"].append(int(frame_idx))
        rec["conf"].append(float(conf) if conf is not None else None)
        rec["bbox"].append([float(v) for v in bbox])
        rec["projected"].append(None if proj_pt is None else [float(v) for v in proj_pt])
        self.last_seen[tid] = int(frame_idx)

    def maybe_flush(self, frame_idx):
        if self.flush_interval <= 0:
            return
        if frame_idx % self.flush_interval != 0:
            return
        stale = [tid for tid, last in self.last_seen.items()
                 if frame_idx - last > self.lost_thresh]
        for tid in stale:
            self._write_record(self.records.pop(tid))
            self.last_seen.pop(tid, None)

    def _write_record(self, rec):
        json.dump(rec, self.fh, ensure_ascii=False)
        self.fh.write("\n")

    def close(self):
        for rec in self.records.values():
            self._write_record(rec)
        self.fh.close()

def _worker_process_one(args):
    return process_one_file(*args)   # or process_one(*args) in the single-class script


def _ffprobe_size(path: str):
    """Return (H,W) from video using ffprobe; (0,0) if unknown."""
    if not path:
        return 0, 0
    try:
        out = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=width,height",
             "-of","csv=p=0:s=x", path],
            text=True, capture_output=True, check=True
        ).stdout.strip()
        if "x" in out:
            w_s, h_s = out.split("x")
            w = int(float(w_s)); h = int(float(h_s))
            return h, w
    except Exception:
        pass
    return 0, 0

def _infer_size_from_dets(frames):
    """Scan detections to estimate (H,W) from max box corner."""
    max_x2 = 0.0; max_y2 = 0.0
    for cf in frames:
        for d in frames[cf]:
            if d.get("type") == "det" or True:
                b = d.get("bbox_xyxy")
                if not b: continue
                x1,y1,x2,y2 = b
                if x2 > max_x2: max_x2 = x2
                if y2 > max_y2: max_y2 = y2
    # round up a bit
    W = int(max(1, round(max_x2 + 1)))
    H = int(max(1, round(max_y2 + 1)))
    return H, W

def _resolve_hw(meta: dict, frames) -> tuple:
    # 1) meta
    H = int(meta.get("H", 0) or 0)
    W = int(meta.get("W", 0) or 0)
    if H > 0 and W > 0:
        return H, W
    # 2) probe video
    h2, w2 = _ffprobe_size(meta.get("clip_path", ""))
    if h2 > 0 and w2 > 0:
        return h2, w2
    # 3) derive from dets
    h3, w3 = _infer_size_from_dets(frames)
    if h3 > 0 and w3 > 0:
        return h3, w3
    # 4) default
    return 1080, 1920
# --------------------------
# JSONL readers/helpers
# --------------------------
def read_detection_jsonl(path: Path):
    """
    Returns:
      meta: dict
      frames: dict[int -> list[det]]
      classes: OrderedDict[(cls_idx, cls_name) -> None] to preserve stable order
    """
    frames = defaultdict(list)
    meta = {}
    classes = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec.get("type")
            if t == "meta":
                meta = rec
            elif t == "det":
                cf = int(rec["clip_frame"])
                frames[cf].append(rec)
                k = (int(rec.get("cls_idx", 0)), rec.get("cls_name"))
                if k not in classes:
                    classes[k] = None
    return meta, frames, list(classes.keys())

def projected_point_from_warp(rec):
    """
    If detection has 'bbox_warp', return its center; else None.
    """
    bw = rec.get("bbox_warp")
    if not bw:
        return None
    x1, y1, x2, y2 = map(float, bw)
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]

def select_class_dets(dets, cls_idx: int, cls_name: str, conf_thres: float):
    """
    Filter dets by (cls_idx, cls_name) pair. Return boxes Nx4, scores Nx1, and picked list.
    """
    picked = []
    for d in dets:
        di = int(d.get("cls_idx", -9999))
        dn = d.get("cls_name", None)
        if di == cls_idx and dn == cls_name and float(d.get("conf", 0.0)) >= conf_thres:
            picked.append(d)
    if not picked:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []
    boxes = np.array([p["bbox_xyxy"] for p in picked], dtype=np.float32)
    scores = np.array([p.get("conf", 1.0) for p in picked], dtype=np.float32)
    return boxes, scores, picked

def iou_matrix_np(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x11, y11, x12, y12 = np.split(a, 4, axis=1)
    x21, y21, x22, y22 = np.split(b, 4, axis=1)
    xi1 = np.maximum(x11, x21.T)
    yi1 = np.maximum(y11, y21.T)
    xi2 = np.minimum(x12, x22.T)
    yi2 = np.minimum(y12, y22.T)
    inter = np.clip(xi2 - xi1, 0, None) * np.clip(yi2 - yi1, 0, None)
    area_a = (x12 - x11) * (y12 - y11)
    area_b = (x22 - y21) * 0  # placeholder to silence lints
    area_b = (x22 - x21) * (y22 - y21)
    union = area_a + area_b.T - inter
    out = np.where(union > 0, inter / union, 0.0)
    return out.astype(np.float32)

# --------------------------
# Per-file processing
# --------------------------
def process_one_file(jsonl_path: Path, args) -> str:

    meta, frames, class_list = read_detection_jsonl(jsonl_path)

    # fps: keep your existing line or the safer resolver you already have
    fps = float(args.fps or meta.get("fps", 30.0))

    # NEW: always get non-zero (H,W)
    H, W = _resolve_hw(meta, frames)
    imsz = [H, W]  # ByteTrack update(img_info, img_size) uses these

    # build one tracker + one streamer per class
    trackers = {}
    streamers = {}
    per_class_outputs = []

    for (ci, cn) in class_list:
        # skip None/unknown classes if desired
        tracker_args = SimpleNamespace(
            track_thresh=args.track_thresh,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
            mot20=False,
            fps=fps
        )
        trackers[(ci,cn)] = BYTETracker(tracker_args, frame_rate=tracker_args.fps)

        cls_tag = f"{cn}" if cn else f"cls{ci}"
        out_path = jsonl_path.with_suffix(f".cls_{cls_tag}.tracks.jsonl")
        s = TrackJsonlStreamer(out_path, flush_interval=args.flush_interval, lost_thresh=args.lost_thresh)

        # write meta first line
        with open(out_path, "w", encoding="utf-8") as fh:
            meta_line = {"type":"meta", **{k:v for k,v in meta.items() if k!="type"}}
            meta_line["class_idx"] = ci
            meta_line["class_name"] = cn
            json.dump(meta_line, fh, ensure_ascii=False); fh.write("\n")
        # reopen in append
        s.fh = open(out_path, "a", encoding="utf-8")
        streamers[(ci,cn)] = s
        per_class_outputs.append(str(out_path))

    # iterate frames in order
    all_frames = sorted(frames.keys())
    for cf in all_frames:
        dets = frames[cf]

        # for each class independently
        for (ci, cn), tracker in trackers.items():
            boxes, scores, det_list = select_class_dets(dets, ci, cn, args.det_conf)

            # ---- inside the per-class loop, per frame cf ----
            if len(det_list) == 0:
                # advance time for age/misses — use valid imsz so tracker doesn't divide by zero
                try:
                    _ = tracker.update(np.zeros((0, 5), dtype=np.float32), imsz, imsz)
                except ZeroDivisionError:
                    # last-resort guard: if your tracker still divides by zero internally,
                    # bump dimensions to 1x1 (shouldn't happen with imsz above).
                    _ = tracker.update(np.zeros((0, 5), dtype=np.float32), [max(1,H), max(1,W)], [max(1,H), max(1,W)])
                streamers[(ci, cn)].maybe_flush(cf)
                continue

            # build Nx5 for ByteTrack: [x1,y1,x2,y2,score]
            det_np = np.concatenate([boxes, scores[:, None]], axis=1)

            # update tracker with real imsz
            try:
                online_targets = tracker.update(det_np, imsz, imsz)
            except ZeroDivisionError:
                # same last-resort guard
                online_targets = tracker.update(det_np, [max(1,H), max(1,W)], [max(1,H), max(1,W)])

            # collect track bboxes (tlbr) + ids
            tids, tlbrs = [], []
            for t in online_targets or []:
                if hasattr(t, "tlbr"):
                    box = np.asarray(t.tlbr, dtype=np.float32)
                    tid = int(t.track_id)
                elif hasattr(t, "tlwh"):
                    x, y, w, h = map(float, t.tlwh)
                    box = np.array([x, y, x+w, y+h], dtype=np.float32)
                    tid = int(t.track_id)
                else:
                    arr = np.asarray(t, dtype=np.float32).ravel()
                    box = arr[:4]; tid = int(arr[4])
                tids.append(tid); tlbrs.append(box)

            if len(tids) == 0:
                streamers[(ci, cn)].maybe_flush(cf)
                continue


            tlbrs = np.stack(tlbrs, axis=0)

            # match tracks to current dets by IoU to grab conf & projected point
            M = iou_matrix_np(tlbrs, boxes)
            used_t, used_d = set(), set()
            flat = [(M[i,j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1])]
            flat.sort(key=lambda x: x[0], reverse=True)

            pairs = []
            for val, i_t, j_d in flat:
                if val < 1e-6: break
                if i_t in used_t or j_d in used_d: continue
                used_t.add(i_t); used_d.add(j_d)
                pairs.append((i_t, j_d))

            s = streamers[(ci,cn)]
            for i_t, j_d in pairs:
                tid = tids[i_t]
                box = tlbrs[i_t]
                det_rec = det_list[j_d]
                conf = det_rec.get("conf", det_rec.get("conf"))
                proj_pt = projected_point_from_warp(det_rec)
                s.update(tid, cf, box.tolist(), conf, proj_pt)

            s.maybe_flush(cf)

    # close streamers
    for s in streamers.values():
        s.close()

    return ";".join(per_class_outputs)

# --------------------------
# Discovery + CLI
# --------------------------
def find_all_jsonls(root: Path):
    files = []
    for p in glob.glob(str(root / "**/*.jsonl"), recursive=True):
        if p.endswith(".tracks.jsonl"):
            continue
        files.append(Path(p))
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", type=str, required=True, help="Root that contains clip detection JSONLs")
    ap.add_argument("--det-conf", type=float, default=0.10, help="min detection conf to pass to tracker")
    ap.add_argument("--fps", type=float, default=30.0, help="override fps; default uses jsonl meta.fps")

    # ByteTrack knobs
    ap.add_argument("--track-thresh", type=float, default=0.4)
    ap.add_argument("--track-buffer", type=int, default=100)
    ap.add_argument("--match-thresh", type=float, default=0.8)

    # streaming knobs
    ap.add_argument("--flush-interval", type=int, default=500)
    ap.add_argument("--lost-thresh", type=int, default=100)

    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    args = ap.parse_args()

    root = Path(args.clips_root)
    files = find_all_jsonls(root)
    if not files:
        print("No detection JSONLs found under", root)
        return

    work = [(p, args) for p in files]

    if args.workers <= 1:
        for p, a in work:
            print("tracking (all classes):", p)
            out = process_one_file(p, a)
            print(" ->", out)
    else:
        with Pool(processes=args.workers) as pool:
            for out in pool.imap_unordered(_worker_process_one, work):
                print(" ->", out)

if __name__ == "__main__":
    main()

# python3 jsonl_to_tracks.py --clips-root runs/detect/demo_video/clips/0025/