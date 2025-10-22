#!/usr/bin/env python3
import json, math, argparse, os
from pathlib import Path
from collections import defaultdict, OrderedDict
import numpy as np
import cv2
from typing import Optional


# your classifier
from tools.goalkeeper_motion_classification import classify_goalkeeper_behavior

# ---------------- I/O helpers ----------------
def read_tracks_jsonl(path: Path):
    meta = None
    tracks = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("type") == "meta":
                meta = rec
                continue
            if "track_id" in rec and isinstance(rec.get("frame_id"), list):
                tracks[int(rec["track_id"])] = rec
    if meta is None:
        raise ValueError(f"No meta in {path}")
    return meta, tracks

def valid_number(x):
    return (x is not None) and (not (isinstance(x, float) and math.isnan(x)))

def select_goalkeeper_track(person_tracks: dict):
    best_tid, best_score = None, -1.0
    for tid, tr in person_tracks.items():
        scores = tr.get("team_score") or []
        s = sum(float(v) for v in scores if valid_number(v))
        if s > best_score:
            best_score = s
            best_tid = tid
    return best_tid

def choose_single_ball_track(ball_tracks: dict):
    """Prefer fused -1, else sole track, else longest by frame count."""
    if not ball_tracks:
        return None
    if -1 in ball_tracks:
        return -1
    if len(ball_tracks) == 1:
        return next(iter(ball_tracks.keys()))
    return max(ball_tracks.keys(), key=lambda tid: len(ball_tracks[tid].get("frame_id", [])))

# ---------------- homography utils ----------------
def _ensure_H(H_like) -> Optional[np.ndarray]:
    if H_like is None:
        return None
    H = np.asarray(H_like, dtype=np.float64)
    if H.shape != (3,3):
        raise ValueError(f"Homography must be 3x3, got {H.shape}")
    return H

def warp_points_xy(points, H):
    if H is None:
        return [p if p is None else [float(p[0]), float(p[1])] for p in points]
    pts = np.array([[p] for p in points if p is not None], dtype=np.float32)
    if pts.size == 0:
        return [None if p is None else p for p in points]
    warped = cv2.perspectiveTransform(pts, H).reshape(-1,2)
    out = []
    j = 0
    for p in points:
        if p is None:
            out.append(None)
        else:
            out.append([float(warped[j,0]), float(warped[j,1])]); j += 1
    return out

def warp_bbox_xyxy(box, H):
    x1,y1,x2,y2 = map(float, box)
    if H is None:
        return [x1,y1,x2,y2]
    corners = np.array([[[x1,y1]], [[x2,y1]], [[x2,y2]], [[x1,y2]]], dtype=np.float32)
    wc = cv2.perspectiveTransform(corners, H).reshape(4,2)
    xmin, ymin = float(np.min(wc[:,0])), float(np.min(wc[:,1]))
    xmax, ymax = float(np.max(wc[:,0])), float(np.max(wc[:,1]))
    return [xmin, ymin, xmax, ymax]

# ------------- build classifier input (warped) -------------
def build_all_frame_detections_warped(gk_track, ball_track, H, frame_window=None):
    """
    Build frames list where each entry has:
      - GK:   {'cls':0,  'bbox_warp': warped bbox, 'keypoints': warped skel, 'score': team_score}
      - Ball: {'cls':32, 'bbox_warp': warped bbox}
    If frame_window (set of frame indices) is provided, only include those frames.
    """
    frames = set()
    f2i_gk, f2i_ball = {}, {}

    if gk_track:
        for i, f in enumerate(gk_track.get("frame_id", [])):
            ff = int(f)
            if (frame_window is None) or (ff in frame_window):
                frames.add(ff); f2i_gk[ff] = i
    if ball_track:
        for i, f in enumerate(ball_track.get("frame_id", [])):
            ff = int(f)
            if (frame_window is None) or (ff in frame_window):
                frames.add(ff); f2i_ball[ff] = i

    if not frames:
        return []

    ordered   = sorted(frames)
    index_of  = {f: i for i, f in enumerate(ordered)}
    all_frames = [[] for _ in ordered]

    # GK
    if gk_track:
        skel_arr = gk_track.get("skel") or []
        tscore   = gk_track.get("team_score") or []
        boxes    = gk_track.get("bbox") or []

        for f, ei in f2i_gk.items():
            if ei >= len(boxes): 
                continue
            dst_idx = index_of[f]
            box_w   = warp_bbox_xyxy(boxes[ei], H)

            kp   = skel_arr[ei] if ei < len(skel_arr) else None
            kp_w = warp_points_xy(kp, H) if kp is not None else None

            raw = tscore[ei] if ei < len(tscore) else None
            score = float(raw) if isinstance(raw, (int, float)) and not (isinstance(raw, float) and math.isnan(raw)) else 0.0

            all_frames[dst_idx].append({
                "cls": 0,
                "bbox_warp": box_w,
                "keypoints": kp_w,
                "score": score,
            })

    # Ball
    if ball_track:
        boxes = ball_track.get("bbox") or []
        for f, ei in f2i_ball.items():
            if ei >= len(boxes):
                continue
            dst_idx = index_of[f]
            box_w   = warp_bbox_xyxy(boxes[ei], H)
            all_frames[dst_idx].append({"cls": 32, "bbox_warp": box_w})

    return all_frames

# ---------------- discovery: pair people/ball per clip ----------------
def _read_meta_head(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("type") == "meta":
                return rec
            # if first line isn't meta, stop early
            break
    return {}

def _clip_id_from_meta_or_name(path: Path, meta: dict):
    # Prefer meta.clip_path basename (without extension)
    cp = meta.get("clip_path")
    if cp:
        try:
            return Path(cp).stem
        except Exception:
            pass
    # Fallback: file stem up to ".cls_"
    stem = path.stem
    cut = stem.split(".cls_")[0]
    return cut

def _center_frame_window(gk_track, ball_track, fps: float, seconds: float = 2.0):
    """Return a set of frame indices covering the `seconds` before the midpoint
    up to the midpoint of the union of GK+ball frames. 
    If no frames, return None (means 'no filter')."""
    frames = []
    if gk_track:
        frames.extend(int(f) for f in gk_track.get("frame_id", []))
    if ball_track:
        frames.extend(int(f) for f in ball_track.get("frame_id", []))
    if not frames:
        return None

    fmin, fmax = min(frames), max(frames)
    fmid = (fmin + fmax) // 2   # midpoint frame

    # start = seconds before fmid
    start = max(fmin, fmid - int(seconds * fps))

    return set(range(start, fmid + 1))   # inclusive up to fmid

def is_person_meta(meta: dict):
    ci = int(meta.get("class_idx", -9999))
    cn = str(meta.get("class_name", "") or "").lower()
    return (ci == 0) or (cn == "person")

def is_ball_meta(meta: dict):
    ci = int(meta.get("class_idx", -9999))
    cn = str(meta.get("class_name", "") or "").lower()
    return (ci == 32) or (cn == "ball")

def discover_pairs(root: Path):
    """
    Returns a list of (clip_id, people_jsonl, ball_jsonl, fps_guess)
    """
    people_for = {}
    ball_for = {}
    fps_guess_for = {}

    for p in root.rglob("*.tracks.jsonl"):
        meta = _read_meta_head(p)
        if not meta: 
            continue
        clip_id = _clip_id_from_meta_or_name(p, meta)

        # remember fps guess if present
        fps_guess_for.setdefault(clip_id, float(meta.get("fps", 0.0) or 0.0))

        if is_person_meta(meta):
            # prefer class_name person
            people_for[clip_id] = p
        elif is_ball_meta(meta):
            ball_for[clip_id] = p

    # Build jobs: only clips that have people (GK needed). Ball may be missing.
    pairs = []
    for clip_id, ppl_path in sorted(people_for.items()):
        ball_path = ball_for.get(clip_id, None)
        fps_guess = fps_guess_for.get(clip_id, 0.0)
        pairs.append((clip_id, ppl_path, ball_path, fps_guess))
    return pairs

# ---------------- speeds parsing ----------------
def parse_speeds_arg(speeds_arg: str, expected_n: int):
    """
    Accept:
      - "12,34,18" (comma separated)
      - file path: each line a number
    Enforces length match with expected_n.
    """
    speeds = []
    if speeds_arg is None:
        raise ValueError("--speeds is required")
    cand = Path(speeds_arg)
    if cand.exists() and cand.is_file():
        for ln in cand.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln: continue
            speeds.append(float(ln))
    else:
        speeds = [float(x) for x in speeds_arg.split(",") if x.strip()]

    if len(speeds) != expected_n:
        raise ValueError(f"Speeds count ({len(speeds)}) does not match number of clips ({expected_n}).")
    return speeds

# ---------------- per-clip analyze ----------------
def analyze_one_clip(people_jsonl: Path,
                     maybe_ball_jsonl: Path,
                     ball_speed_value: float,
                     H: Optional[np.ndarray],
                     center_seconds: float = 4.0):
    
    ppl_meta, ppl_tracks = read_tracks_jsonl(people_jsonl)
    # fps
    fps = float(ppl_meta.get("fps", 0.0) or 0.0)
    if fps <= 0:
        fps = 30.0  # fallback

    # select GK (max team_score)
    gk_tid = select_goalkeeper_track(ppl_tracks)
    if gk_tid is None:
        return {
            "warped": bool(H is not None),
            "gk_tid": None, "ball_tid": None,
            "fps": fps, "ball_speed": 0.0, "speed_units": "km/h",
            "tags": [0]
        }

    gk_track = ppl_tracks[gk_tid]

    ball_tid = None
    ball_track = None
    if maybe_ball_jsonl is not None and Path(maybe_ball_jsonl).exists():
        _, ball_tracks = read_tracks_jsonl(maybe_ball_jsonl)
        ball_tid = choose_single_ball_track(ball_tracks)
        if ball_tid is not None:
            ball_track = ball_tracks[ball_tid]

    if ball_track is None:
        return {
            "warped": bool(H is not None),
            "gk_tid": int(gk_tid),
            "ball_tid": None,
            "fps": fps, "ball_speed": 0.0, "speed_units": "km/h",
            "tags": [0]
        }

    # compute window (set of frame indices) centered on the union of GK+ball frames
    frame_window = None
    if center_seconds and center_seconds > 0:
        frame_window = _center_frame_window(gk_track, ball_track, fps=fps, seconds=center_seconds)

    all_frame_detections = build_all_frame_detections_warped(
        gk_track, ball_track, H, frame_window=frame_window
    )

    tags = classify_goalkeeper_behavior(
        all_frame_detections=all_frame_detections,
        ball_speed=float(ball_speed_value),
        distance_threshold=120.0,
        movement_threshold=60.0,
        speed_threshold=15.0,
        jump_threshold=20.0,
        elbow_angle_threshold=110.0,
    )

    return {
        "warped": bool(H is not None),
        "gk_tid": int(gk_tid),
        "ball_tid": int(ball_tid),
        "fps": fps,
        "ball_speed": float(ball_speed_value),
        "speed_units": "km/h",
        "tags": tags,
    }

# ---------------- CLI + main ----------------
def main():
    ap = argparse.ArgumentParser("Batch GK behavior analysis over a root folder")
    ap.add_argument("--root", required=True, type=str,
                    help="Root directory containing *.tracks.jsonl files")
    ap.add_argument("--speeds", required=True, type=str,
                    help="Comma-separated list OR a text file with one speed per clip (order printed)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Directory to write per-clip JSON results (optional)")
    ap.add_argument("--homography", type=str, default=None,
                    help="Single 3x3 homography .npy applied to all (optional)")
    ap.add_argument("--homography-dir", type=str, default=None,
                    help="Directory containing per-clip homographies named <clip_id>.npy (optional)")
    ap.add_argument("--center-seconds", type=float, default=2.0,
                help="Analyze only the middle N seconds of the clip (default 2.0). Use 0 to analyze full span.")
    # thresholds are inside analyze_one_clip; expose here if you want to override
    args = ap.parse_args()

    root = Path(args.root)
    pairs = discover_pairs(root)
    if not pairs:
        print("No people tracks discovered under", root)
        return

    # parse speeds
    speeds = parse_speeds_arg(args.speeds, expected_n=len(pairs))

    # output dir
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # base homography (fallback)
    H_global = _ensure_H(np.load(args.homography)) if args.homography else None
    H_dir = Path(args.homography_dir) if args.homography_dir else None

    # show the processing order so the user can line up speeds properly
    print("Processing order (clip_id) -> files:")
    for i, (clip_id, ppl_p, ball_p, fps_guess) in enumerate(pairs):
        print(f"{i:02d}: {clip_id} | people={ppl_p.name} | ball={ball_p.name if ball_p else 'NONE'}")

    # run
    for i, (clip_id, ppl_p, ball_p, fps_guess) in enumerate(pairs):
        # per-clip H: homography-dir/<clip_id>.npy > global > None
        H = None
        if H_dir:
            cand = H_dir / f"{clip_id}.npy"
            if cand.exists():
                try:
                    H = _ensure_H(np.load(cand))
                except Exception:
                    H = None
        if H is None:
            H = H_global

        result = analyze_one_clip(
            people_jsonl=ppl_p,
            maybe_ball_jsonl=ball_p,
            ball_speed_value=speeds[i],
            H=H,
            center_seconds=args.center_seconds,
        )

        # include identifiers
        result["clip_id"] = clip_id
        result["people_tracks_jsonl"] = str(ppl_p)
        result["ball_tracks_jsonl"] = str(ball_p) if ball_p else None

        if out_dir:
            out_path = out_dir / f"{clip_id}.analysis.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[OK] {clip_id} -> {out_path}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

# python3 analyze_goalkeeper_behavior.py --root ./runs/detect/demo_video/clips/0025/ --speeds 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45 --out-dir ./runs/detect/demo_video/ananly --homography ./runs/detect/demo_video/homography_matrix.npy 