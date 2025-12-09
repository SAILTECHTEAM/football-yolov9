import json
import cv2
import math
from pathlib import Path
from collections import defaultdict
import subprocess
import numpy as np
from typing import List, Optional

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def _norm_stem(p: Path) -> str:
    # strip common extra suffixes if you have patterns like GX010025_clip_002
    return p.stem


def _pick_best_match(cands: List[Path]) -> Optional[Path]:
    if not cands:
        return None
    # pick shortest filename (closest match); tie-breaker by newest mtime
    cands = sorted(cands, key=lambda p: (len(p.name), -p.stat().st_mtime))
    return cands[0]


def _find_jsonl_for_video(video_path: Path, token: str) -> Optional[Path]:
    """
    Find a jsonl in the same directory whose name:
      - shares the video stem as prefix (loosely), and
      - contains the token substring.
    """
    stem = _norm_stem(video_path)
    folder = video_path.parent
    cands = []
    for p in folder.glob("*.jsonl"):
        name = p.name
        if token in name and name.startswith(stem):
            cands.append(p)
    # fallback: if strict startswith fails, allow contains(stem)
    if not cands:
        for p in folder.glob("*.jsonl"):
            name = p.name
            if token in name and stem in name:
                cands.append(p)
    return _pick_best_match(cands)


def _rel_to(base: Path, path: Path) -> Path:
    try:
        return path.relative_to(base)
    except Exception:
        return Path(path.name)


def process_root(
    root: str,
    people_token: str = ".cls_0.tracks.jsonl",
    ball_token: str = ".cls_32.tracks.jsonl",
    dets_token: Optional[str] = ".dets.jsonl",
    output_dir: Optional[str] = None,
    people_color=(0, 220, 0),
    ball_color=(40, 40, 230),
    det_color=(200, 200, 200),
    thickness: int = 2,
    show_scores: bool = True,
    score_field: str = "team_score",
):
    root_path = Path(root).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve() if output_dir else None

    videos = [p for p in root_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"[WARN] No videos found under {root_path}")
        return

    print(f"[INFO] Found {len(videos)} videos. Starting batch rendering...")

    for vid in videos:
        try:
            people_jsonl = _find_jsonl_for_video(vid, people_token)
            ball_jsonl = _find_jsonl_for_video(vid, ball_token)
            dets_jsonl = _find_jsonl_for_video(vid, dets_token) if dets_token else None

            if people_jsonl is None and ball_jsonl is None:
                print(f"[SKIP] {vid} — no matching people/ball JSONL found.")
                continue

            # decide output path
            if out_root:
                rel = _rel_to(root_path, vid.parent)
                out_folder = out_root / rel
                out_folder.mkdir(parents=True, exist_ok=True)
                out_path = out_folder / f"{vid.stem}_render.mp4"
            else:
                out_path = vid.with_name(f"{vid.stem}_render.mp4")

            print(f"[RUN ] {vid.name}")
            print(f"       people: {people_jsonl.name if people_jsonl else 'None'}")
            print(f"       ball  : {ball_jsonl.name if ball_jsonl else 'None'}")
            if dets_jsonl:
                print(f"       dets  : {dets_jsonl.name}")

            render_tracks_on_video(
                video_path=str(vid),
                people_tracks_jsonl=str(people_jsonl) if people_jsonl else None,
                ball_tracks_jsonl=str(ball_jsonl) if ball_jsonl else None,
                output_path=str(out_path),
                dets_jsonl=str(dets_jsonl) if dets_jsonl else None,
                people_color=people_color,
                ball_color=ball_color,
                det_color=det_color,
                thickness=thickness,
                show_scores=show_scores,
                score_field=score_field,
            )
        except Exception as e:
            print(f"[ERR ] Failed on {vid}: {e}")


# ---------- helpers: probe video fps/size ----------
def _ffprobe_fps(path: str) -> float:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                path,
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if "/" in out:
            a, b = out.split("/")
            a = float(a)
            b = float(b) if float(b) else 0.0
            return a / b if b > 0 else 0.0
        return float(out or 0.0)
    except Exception:
        return 0.0


def _ffprobe_size(path: str):
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                path,
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if "x" in out:
            w_s, h_s = out.split("x")
            return int(h_s), int(w_s)
    except Exception:
        pass
    return 0, 0


# ---------- JSONL readers ----------
def load_tracks_jsonl(jsonl_path: str):
    """
    Supports two schemas:
      A) per-track rows (your TrackJsonlStreamer): one JSON per track with arrays:
         { "track_id": 7, "frame_id":[...], "bbox":[[x1,y1,x2,y2],...], ... }
      B) per-detection rows (type='det'): one JSON per det with 'track_id' and 'clip_frame'
         { "type":"det", "clip_frame":123, "bbox_xyxy":[...], "track_id":7, ... }
    Returns: dict[frame_idx] -> list of items: {"tid", "bbox", "det_conf", "assoc_score"}
    (det_conf/assoc_score may be None if not present)
    """
    per_frame = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                continue
            # Schema B: per-det with 'clip_frame'
            if rec.get("type") == "det" and "clip_frame" in rec:
                cf = int(rec["clip_frame"])
                bbox = rec.get("bbox_xyxy") or rec.get("bbox")  # some variants emit 'bbox'
                if bbox is None:
                    continue
                item = {
                    "tid": int(rec.get("track_id", -1)),
                    "bbox": [float(v) for v in bbox],
                    "det_conf": float(rec.get("conf", rec.get("det_conf", np.nan))),
                    "assoc_score": float(rec.get("assoc_score", np.nan)),
                }
                per_frame[cf].append(item)
                continue

            # Schema A: per-track arrays
            track_id = rec.get("track_id")
            fids = rec.get("frame_id")
            boxes = rec.get("bbox")
            det_confs = rec.get("det_conf") or []
            assoc_scores = rec.get("assoc_score") or []
            if track_id is not None and isinstance(fids, list) and isinstance(boxes, list):
                for i, fid in enumerate(fids):
                    bbox = boxes[i]
                    dc = det_confs[i] if i < len(det_confs) else np.nan
                    ascr = assoc_scores[i] if i < len(assoc_scores) else np.nan
                    per_frame[int(fid)].append(
                        {
                            "tid": int(track_id),
                            "bbox": [float(v) for v in bbox],
                            "det_conf": float(dc) if dc is not None else np.nan,
                            "assoc_score": float(ascr) if ascr is not None else np.nan,
                        }
                    )
    return per_frame


def load_dets_jsonl(jsonl_path: str):
    """
    Optional: load original detections (to show conf if tracks missing).
    Returns dict[frame_idx] -> list of {"bbox","conf","cls_name","cls_idx"}
    """
    per_frame = defaultdict(list)
    if not jsonl_path:
        return per_frame
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") != "det":
                continue
            cf = int(rec["clip_frame"])
            per_frame[cf].append(
                {
                    "bbox": [float(v) for v in rec.get("bbox_xyxy", rec.get("bbox", []))],
                    "conf": float(rec.get("conf", np.nan)),
                    "cls_name": rec.get("cls_name"),
                    "cls_idx": rec.get("cls_idx"),
                }
            )
    return per_frame


def load_people_tracks_with_scores(jsonl_path: str, score_field: str = "team_score"):
    """
    Supports:
      A) per-track arrays: {track_id, frame_id[], bbox[], det_conf[], assoc_score[], team_score[], skel[]}
      B) per-detection rows (type='det') — no team_score/skel expected; renders as before.

    Returns:
      per_frame: dict[frame_idx] -> list of {
        'tid','bbox','det_conf','assoc_score','team_score','skel'
      }
      best_tid: track_id with the highest sum of (non-None) team_score for schema A; else None.
    """
    per_frame = defaultdict(list)
    sum_by_tid = defaultdict(float)
    has_per_track_schema = False

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                continue

            # B) per-det rows
            if rec.get("type") == "det" and "clip_frame" in rec:
                cf = int(rec["clip_frame"])
                bbox = rec.get("bbox_xyxy") or rec.get("bbox")
                if bbox is None:
                    continue
                item = {
                    "tid": int(rec.get("track_id", -1)),
                    "bbox": [float(v) for v in bbox],
                    "det_conf": float(rec.get("conf", rec.get("det_conf", np.nan))),
                    "assoc_score": float(rec.get("assoc_score", np.nan)),
                    "team_score": None,
                    "skel": None,
                }
                per_frame[cf].append(item)
                continue

            # A) per-track arrays
            track_id = rec.get("track_id")
            fids = rec.get("frame_id")
            boxes = rec.get("bbox")
            if track_id is not None and isinstance(fids, list) and isinstance(boxes, list):
                has_per_track_schema = True
                det_confs = rec.get("det_conf") or []
                assoc_scores = rec.get("assoc_score") or []
                team_scores = rec.get(score_field) or []  # <-- team_score OR team_score_b
                skels = rec.get("skel") or []

                for i, fid in enumerate(fids):
                    bbox = boxes[i]
                    dc = det_confs[i] if i < len(det_confs) else np.nan
                    ascr = assoc_scores[i] if i < len(assoc_scores) else np.nan
                    ts = team_scores[i] if i < len(team_scores) else None
                    sk = skels[i] if i < len(skels) else None
                    if isinstance(sk, np.ndarray):
                        sk = sk.tolist()
                    per_frame[int(fid)].append(
                        {
                            "tid": int(track_id),
                            "bbox": [float(v) for v in bbox],
                            "det_conf": float(dc) if dc is not None else np.nan,
                            "assoc_score": float(ascr) if ascr is not None else np.nan,
                            "team_score": ts,
                            "skel": sk,
                        }
                    )
                    if ts is not None:
                        try:
                            sum_by_tid[int(track_id)] += float(ts)
                        except Exception:
                            pass

    best_tid = None
    if has_per_track_schema and sum_by_tid:
        # track with the largest total team_score
        best_tid = max(sum_by_tid.items(), key=lambda kv: kv[1])[0]

    return per_frame, best_tid


# ---------- drawing ----------
def _draw_box_with_label(img, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, img.shape[1] - 1))
    y1 = max(0, min(y1, img.shape[0] - 1))
    x2 = max(0, min(x2, img.shape[1] - 1))
    y2 = max(0, min(y2, img.shape[0] - 1))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    # Filled label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
    th_pad = th + 6
    y_bg = max(0, y1 - th_pad)
    cv2.rectangle(img, (x1, y_bg), (x1 + tw + 6, y_bg + th_pad), color, -1)
    cv2.putText(img, label, (x1 + 3, y_bg + th + 1), font, fs, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (x1 + 3, y_bg + th + 1), font, fs, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_skeleton(img, kpts, color=(0, 220, 220), radius=3, thickness=2):
    """
    kpts: list/array of shape (K,2) or (K,3) [x,y,(conf)] in image coords.
    Uses a COCO17-style edge set if K >= 17; otherwise just dots.
    """
    if kpts is None:
        return
    kpts = np.asarray(kpts, dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return

    H, W = img.shape[:2]
    K = kpts.shape[0]

    # draw joints
    for i in range(K):
        x, y = int(round(kpts[i, 0])), int(round(kpts[i, 1]))
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(img, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

    # try to draw limbs if we have at least COCO 17
    if K >= 17:
        # minimal COCO-style connectivity
        edges = [
            (5, 7),
            (7, 9),  # left arm
            (6, 8),
            (8, 10),  # right arm
            (5, 11),
            (6, 12),  # shoulders to hips
            (11, 13),
            (13, 15),  # left leg
            (12, 14),
            (14, 16),  # right leg
            (5, 6),
            (11, 12),  # shoulders bar, hips bar
        ]
        for a, b in edges:
            if a < K and b < K:
                xa, ya = int(round(kpts[a, 0])), int(round(kpts[a, 1]))
                xb, yb = int(round(kpts[b, 0])), int(round(kpts[b, 1]))
                if 0 <= xa < W and 0 <= ya < H and 0 <= xb < W and 0 <= yb < H:
                    cv2.line(img, (xa, ya), (xb, yb), color, thickness, lineType=cv2.LINE_AA)


# ---------- main function ----------
def render_tracks_on_video(
    video_path: str,
    people_tracks_jsonl: str,
    ball_tracks_jsonl: str,
    output_path: str,
    dets_jsonl: str = None,  # optional, to draw raw dets when no tracks
    people_color=(0, 220, 0),  # green
    ball_color=(40, 40, 230),  # red-ish (BGR)
    det_color=(200, 200, 200),  # light gray for unmatched dets
    skeleton_color=(0, 220, 220),  # cyan-ish for best track skeleton
    thickness=2,
    show_scores=True,  # show det_conf/assoc_score on label
    score_field="team_score",  # which field to pick for "best track"
):
    video_path = str(video_path)
    out_path = str(output_path)

    # Load tracks (per frame)
    # People: NEW loader that returns best_tid and, when available, provides team_score + skel
    ppl_tracks = defaultdict(list)
    best_tid = None
    if people_tracks_jsonl:
        ppl_tracks, best_tid = load_people_tracks_with_scores(
            people_tracks_jsonl, score_field=score_field
        )

    # Ball: same as before
    ball_tracks = load_tracks_jsonl(ball_tracks_jsonl) if ball_tracks_jsonl else defaultdict(list)
    dets = load_dets_jsonl(dets_jsonl) if dets_jsonl else defaultdict(list)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # robust fallback if metadata missing
    if fps <= 0 or math.isnan(fps):
        probed = _ffprobe_fps(video_path)
        fps = probed if probed > 0 else 30.0
    if W <= 0 or H <= 0:
        h2, w2 = _ffprobe_size(video_path)
        W = w2 if w2 > 0 else 1920
        H = h2 if h2 > 0 else 1080

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Ball tracks first
        for item in ball_tracks.get(frame_idx, []):
            tid = item["tid"]
            bbox = item["bbox"]
            if show_scores:
                dc = item.get("det_conf")
                ascr = item.get("assoc_score")
                label = f"ball #{tid}"
                if dc == dc:  # not NaN
                    label += f" | conf {dc:.2f}"
                if ascr == ascr:
                    label += f" | iou* {ascr:.2f}"
            else:
                label = f"ball #{tid}"
            _draw_box_with_label(frame, bbox, label, ball_color, thickness)

        # 2) People: draw bbox for everyone; if tid==best_tid and skel present -> draw skeleton too
        for item in ppl_tracks.get(frame_idx, []):
            tid = item["tid"]
            bbox = item["bbox"]
            ts = item.get("team_score", None)  # might be None if old schema

            if show_scores:
                dc = item.get("det_conf")
                ascr = item.get("assoc_score")
                label = f"person #{tid}"
                if dc == dc:
                    label += f" | conf {dc:.2f}"
                if ascr == ascr:
                    label += f" | iou* {ascr:.2f}"
                if ts is not None:
                    try:
                        label += f" | team {float(ts):.2f}"
                    except Exception:
                        pass
            else:
                label = f"person #{tid}"

            _draw_box_with_label(frame, bbox, label, people_color, thickness)

            # Skeleton only for the best track (and only if we actually have kpts for this frame)
            if best_tid is not None and tid == best_tid:
                skel = item.get("skel", None)
                if skel is not None:
                    _draw_skeleton(frame, skel, color=skeleton_color, radius=3, thickness=2)

        # 3) Optional raw dets as light hints if not covered by tracks
        if dets:
            existing = (
                np.array(
                    [
                        it["bbox"]
                        for it in ball_tracks.get(frame_idx, []) + ppl_tracks.get(frame_idx, [])
                    ],
                    dtype=np.float32,
                )
                if (ball_tracks.get(frame_idx) or ppl_tracks.get(frame_idx))
                else np.zeros((0, 4), np.float32)
            )

            for d in dets.get(frame_idx, []):
                bbox = np.array(d["bbox"], dtype=np.float32)
                draw = True
                if existing.size:
                    xi1 = np.maximum(existing[:, 0], bbox[0])
                    yi1 = np.maximum(existing[:, 1], bbox[1])
                    xi2 = np.minimum(existing[:, 2], bbox[2])
                    yi2 = np.minimum(existing[:, 3], bbox[3])
                    inter = np.clip(xi2 - xi1, 0, None) * np.clip(yi2 - yi1, 0, None)
                    area_a = (existing[:, 2] - existing[:, 0]) * (existing[:, 3] - existing[:, 1])
                    area_b = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    iou = inter / np.maximum(area_a + area_b - inter, 1e-6)
                    if iou.max() > 0.5:
                        draw = False
                if draw:
                    cname = d.get("cls_name", "det")
                    c = d.get("conf", None)
                    label = f"{cname}"
                    if c is not None and c == c:
                        label += f" {c:.2f}"
                    _draw_box_with_label(frame, bbox.tolist(), label, det_color, 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    if best_tid is None:
        print(f"[OK] wrote {out_path} (no team_score found; rendered bboxes only)")
    else:
        print(
            f"[OK] wrote {out_path} (best person track with skeleton: tid={best_tid}, score_field={score_field})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    # single-file mode (original)
    mode.add_argument("--video", type=str, help="input video path (single-file mode)")
    # batch mode
    mode.add_argument("--root", type=str, help="root directory to batch render all videos")

    parser.add_argument(
        "--people-tracks",
        type=str,
        required=False,
        default=None,
        help="[single-file] people tracks JSONL path",
    )
    parser.add_argument(
        "--ball-tracks",
        type=str,
        required=False,
        default=None,
        help="[single-file] ball tracks JSONL path",
    )
    parser.add_argument(
        "--dets",
        type=str,
        required=False,
        default=None,
        help="[single-file] original detections JSONL path",
    )
    parser.add_argument(
        "--output", type=str, required=False, help="[single-file] output video path"
    )

    # batch options
    parser.add_argument(
        "--people-token",
        type=str,
        default=".cls_0.tracks.jsonl",
        help="[batch] substring to identify people tracks JSONL",
    )
    parser.add_argument(
        "--ball-token",
        type=str,
        default=".cls_32.tracks.jsonl",
        help="[batch] substring to identify ball tracks JSONL",
    )
    parser.add_argument(
        "--dets-token",
        type=str,
        default=".dets.jsonl",
        help="[batch] substring to identify dets JSONL; set empty to disable",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="[batch] write outputs under this directory; mirrors subfolders from root",
    )

    parser.add_argument(
        "--people-color",
        type=str,
        default="0,220,0",
        help="BGR color for people boxes (default green)",
    )
    parser.add_argument(
        "--ball-color",
        type=str,
        default="40,40,230",
        help="BGR color for ball boxes (default red-ish)",
    )
    parser.add_argument(
        "--det-color",
        type=str,
        default="200,200,200",
        help="BGR color for unmatched dets (default light gray)",
    )
    parser.add_argument("--thickness", type=int, default=2, help="box thickness")
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="do not show det_conf/assoc_score on labels",
    )
    parser.add_argument(
        "--score-field",
        type=str,
        default="team_score",
        help="per-track array score field to pick best person (e.g., team_score_b)",
    )

    args = parser.parse_args()

    ppl_color = tuple(int(v) for v in args.people_color.split(","))
    ball_color = tuple(int(v) for v in args.ball_color.split(","))
    det_color = tuple(int(v) for v in args.det_color.split(","))

    if args.root:
        dets_token = args.dets_token if args.dets_token else None
        process_root(
            root=args.root,
            people_token=args.people_token,
            ball_token=args.ball_token,
            dets_token=dets_token,
            output_dir=args.output_dir,
            people_color=ppl_color,
            ball_color=ball_color,
            det_color=det_color,
            thickness=args.thickness,
            show_scores=not args.no_scores,
            score_field=args.score_field,
        )
    else:
        if not args.output:
            raise SystemExit("--output is required in single-file mode")
        render_tracks_on_video(
            video_path=args.video,
            people_tracks_jsonl=args.people_tracks,
            ball_tracks_jsonl=args.ball_tracks,
            output_path=args.output,
            dets_jsonl=args.dets,
            people_color=ppl_color,
            ball_color=ball_color,
            det_color=det_color,
            thickness=args.thickness,
            show_scores=not args.no_scores,
            score_field=args.score_field,
        )

# python3 render_track_on_video.py --root ../runs/detect/demo_video/clips/0025/
