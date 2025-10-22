import json
import cv2
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse
from tools.detect_onepose_v5_re import load_pose_model, process_single_detection

# ---- You already have these ----
# def load_pose_model(): ...
# def process_single_detection(...): ...

def _read_person_tracks_jsonl(jsonl_path: str):
    """
    Read TrackJsonlStreamer schema (per-track rows).
    Returns:
      meta (dict or None),
      tracks (list[dict]),
      frame_to_entries: dict[frame_idx] -> list[(track_index, element_index)]
    """
    meta = None
    tracks = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "meta":
                meta = rec
                continue
            if "track_id" in rec and isinstance(rec.get("frame_id"), list) and isinstance(rec.get("bbox"), list):
                tracks.append(rec)

    if not tracks:
        raise ValueError(f"No per-track rows found in {jsonl_path} (expect TrackJsonlStreamer schema).")

    frame_to_entries = defaultdict(list)
    for ti, tr in enumerate(tracks):
        fids = tr.get("frame_id", [])
        for ei, cf in enumerate(fids):
            frame_to_entries[int(cf)].append((ti, ei))

    return meta, tracks, frame_to_entries


def _ensure_team_score_arrays(tracks: list):
    """
    Ensure every track has team_score (and optionally team_id) arrays aligned to frame_id length.
    """
    for tr in tracks:
        n = len(tr.get("frame_id", []))
        # team_score
        if "team_score" not in tr or not isinstance(tr["team_score"], list):
            tr["team_score"] = [None] * n
        elif len(tr["team_score"]) != n:
            tr["team_score"] = (tr["team_score"][:n] + [None] * max(0, n - len(tr["team_score"])))
        # team_id (optional; may remain absent unless we assign below)
        if "team_id" in tr:
            if not isinstance(tr["team_id"], list) or len(tr["team_id"]) != n:
                tr["team_id"] = (list(tr["team_id"])[:n] + [None] * max(0, n - len(tr["team_id"])))


def _iter_needed_frames(cap: cv2.VideoCapture, needed_frames_sorted):
    """
    Efficiently step through a video and yield only frames we need.
    Uses cap.grab() to skip frames without decoding.
      Yields (frame_idx, frame_bgr)
    """
    current = 0
    for target in needed_frames_sorted:
        if target < current:
            # If caller feeds non-monotonic indices, we can’t seek backwards with OpenCV reliably.
            # In practice frame indices from JSONL are sorted; assert to catch logic bugs.
            raise RuntimeError(f"Non-monotonic frame index requested: {target} < {current}")
        # fast-skip (grab) to target-1
        while current < target:
            ok = cap.grab()
            if not ok:
                return
            current += 1
        # decode the target
        ok, frame = cap.read()
        if not ok:
            return
        yield target, frame
        current += 1

def add_team_scores_to_person_tracks_jsonl(
    video_path: str,
    person_tracks_jsonl: str,
    clothes_colors_histogram,                # your reference histogram for team (or "home")
    *,
    # Optional second team support: if provided, we store both scores and assign team_id
    clothes_colors_histogram_b=None,         # e.g., "away" team
    team_id_threshold: float = 0.0,          # if B is provided: team_id = argmax(scoreA, scoreB) unless both < threshold -> None
    names=('person',),                       # class list; person at idx 0
    pose_model=None,                         # pass one to reuse GPU; else we'll load lazily
    inplace: bool = True,                    # overwrite the same JSONL
    backup: bool = True,                     # keep a .bak of the original JSONL if inplace
    out_jsonl_path: str = None,              # if not inplace, write here
    skip_small_area: int = 5000,            # same as your guard; skip tiny boxes
    verbose: bool = True,
):
    """
    For each bbox of each track (person class), compute clothing color match score via your pose pipeline
    and write:
        - 'team_score'     (and optionally 'team_score_b', 'team_id')
        - 'skel'           per-frame skeleton (as returned by process_single_detection['keypoints'])
        - 'skel_conf'      per-frame keypoint confidences (process_single_detection['keypoints_conf'])
    All arrays are aligned to 'frame_id' / 'bbox' length and order.
    """
    jsonl_p = Path(person_tracks_jsonl)
    if not jsonl_p.exists():
        raise FileNotFoundError(f"tracks jsonl not found: {person_tracks_jsonl}")

    # 1) Load tracks + index frames
    meta, tracks, frame_to_entries = _read_person_tracks_jsonl(str(jsonl_p))

    # Ensure arrays exist and match length
    def _ensure_len(arr, n, fill=None):
        if not isinstance(arr, list):
            return [fill] * n
        if len(arr) != n:
            return (arr[:n] + [fill] * max(0, n - len(arr)))
        return arr

    for tr in tracks:
        n = len(tr.get("frame_id", []))
        tr["team_score"] = _ensure_len(tr.get("team_score"), n, None)
        # NEW: skeleton arrays
        tr["skel"]      = _ensure_len(tr.get("skel"), n, None)
        tr["skel_conf"] = _ensure_len(tr.get("skel_conf"), n, None)
        # If you later return original pixel keypoints from process_single_detection as 'keypoints_src',
        # you can also keep a copy here:
        # tr["skel_src"]  = _ensure_len(tr.get("skel_src"), n, None)

    if clothes_colors_histogram_b is not None:
        for tr in tracks:
            n = len(tr.get("frame_id", []))
            tr["team_score_b"] = _ensure_len(tr.get("team_score_b"), n, None)
            tr["team_id"]      = _ensure_len(tr.get("team_id"), n, None)

    needed_frames = sorted(frame_to_entries.keys())

    # 2) Open video once
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # 3) Pose model (lazy load to CUDA if not provided)
    pm = pose_model if pose_model is not None else load_pose_model()

    # 4) Main loop: only frames we need
    processed = 0
    for cf, frame in _iter_needed_frames(cap, needed_frames):
        entries = frame_to_entries.get(cf, [])
        if not entries:
            continue

        for (t_idx, e_idx) in entries:
            tr = tracks[t_idx]
            if e_idx >= len(tr.get("bbox", [])):
                continue

            x1, y1, x2, y2 = map(int, tr["bbox"][e_idx])

            # Small bbox skip (keeps None placeholders)
            if (x2 - x1) * (y2 - y1) < skip_small_area:
                continue

            # ----- A: compute pose + team score for team A -----
            det = process_single_detection(
                frame,                       # im0s (BGR)
                (x1, y1, x2, y2),            # xyxy
                1.0,                          # conf (unused)
                0,                            # cls=0 (person)
                pm,                           # pose model
                save_crop=False,
                hide_labels=True,
                hide_conf=True,
                names=names,
                clothes_colors_histogram=clothes_colors_histogram,
                skip_small_area=skip_small_area,
            )

            score_a = None
            skel = None
            skel_conf = None
            # skel_src = None  # uncomment if your process_single_detection returns 'keypoints_src'

            if det is not None:
                score_a    = float(det.get("score", 0.0))
                # process_single_detection in your code sets:
                #   detection_result['keypoints'] = points
                #   detection_result['keypoints_conf'] = keypoints_dict['confidence']
                # save what we have:
                if det.get("keypoints") is not None:
                    # ensure it's JSON-serializable list of [x,y] or similar
                    skel = det["keypoints"]
                    if isinstance(skel, np.ndarray):
                        skel = skel.tolist()
                if det.get("keypoints_conf") is not None:
                    sc = det["keypoints_conf"]
                    skel_conf = sc.tolist() if isinstance(sc, np.ndarray) else sc
                # If you modify process_single_detection to add original pixels:
                # if det.get("keypoints_src") is not None:
                #     ks = det["keypoints_src"]
                #     skel_src = ks.tolist() if isinstance(ks, np.ndarray) else ks

            tr["team_score"][e_idx] = score_a
            tr["skel"][e_idx]       = skel
            tr["skel_conf"][e_idx]  = skel_conf
            # tr["skel_src"][e_idx] = skel_src  # if you choose to store source-space keypoints

            # ----- B: optionally compute second team score and assign team_id -----
            if clothes_colors_histogram_b is not None:
                det_b = process_single_detection(
                    frame,
                    (x1, y1, x2, y2),
                    1.0, 0, pm, False, True, True, names,
                    clothes_colors_histogram=clothes_colors_histogram_b
                )
                score_b = None
                if det_b is not None:
                    score_b = float(det_b.get("score", 0.0))
                tr["team_score_b"][e_idx] = score_b

                # assign team_id using threshold gate
                if (score_a is None or score_a < team_id_threshold) and (score_b is None or score_b < team_id_threshold):
                    tr["team_id"][e_idx] = None
                else:
                    if (score_b is not None and (score_a is None or score_b >= score_a)):
                        tr["team_id"][e_idx] = "B"
                    else:
                        tr["team_id"][e_idx] = "A"

            processed += 1

        if verbose and (processed % 200 == 0):
            print(f"[team-score+skel] frames done: up to {cf}, entries processed: {processed}")

    cap.release()

    # 5) Write back
    def _write_jsonl(path: Path):
        with open(path, "w", encoding="utf-8") as fh:
            if meta:
                json.dump(meta, fh, ensure_ascii=False); fh.write("\n")
            for tr in tracks:
                json.dump(tr, fh, ensure_ascii=False); fh.write("\n")

    if inplace:
        out_path = jsonl_p
        if backup and not Path(str(jsonl_p) + ".bak").exists():
            os.replace(str(jsonl_p), str(jsonl_p) + ".bak")
            _write_jsonl(out_path)
        else:
            _write_jsonl(out_path)
        print(f"[OK] wrote team_score(s) + skeletons to {out_path} (backup={backup})")
    else:
        out_p = Path(out_jsonl_path) if out_jsonl_path else jsonl_p.with_suffix(".with_team_skel.jsonl")
        _write_jsonl(out_p)
        print(f"[OK] wrote team_score(s) + skeletons to {out_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--tracks', type=str, required=True, help='Path to person tracks JSONL file')
    parser.add_argument('--hist_a', type=str, required=True, help='Path to clothes colors histogram for team A (pickle)')
    parser.add_argument('--hist_b', type=str, default=None, help='(optional) Path to clothes colors histogram for team B (pickle)')
    parser.add_argument('--team_id_threshold', type=float, default=0.0, help='Threshold for assigning team_id when both scores are low')
    parser.add_argument('--inplace', action='store_true', help='Overwrite the original JSONL file')
    parser.add_argument('--no_backup', action='store_true', help='Do not create a .bak backup when using --inplace')
    parser.add_argument('--out_jsonl', type=str, default=None, help='Output JSONL path if not using --inplace')
    parser.add_argument('--skip_small_area', type=int, default=5000, help='Skip boxes with area smaller than this')
    args = parser.parse_args()

    # Load histograms
    hist_a = np.load(args.hist_a)
    hist_b = np.load(args.hist_b) if args.hist_b else None

    add_team_scores_to_person_tracks_jsonl(
        video_path=args.video,
        person_tracks_jsonl=args.tracks,
        clothes_colors_histogram=hist_a,
        clothes_colors_histogram_b=hist_b,
        team_id_threshold=args.team_id_threshold,
        inplace=args.inplace,
        backup=not args.no_backup,
        out_jsonl_path=args.out_jsonl,
        skip_small_area=args.skip_small_area,
        verbose=True
    )