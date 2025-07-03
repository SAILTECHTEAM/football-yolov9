import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict, deque
from scipy.signal import savgol_filter
import cv2
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import ijson.backends.python as ijson_python
from typing import List, Dict, Any, Tuple, Iterator, Union

def assign_team_by_majority_vote(team_conf_list):
    team_count = defaultdict(float)
    for conf in team_conf_list:
        for k, v in conf.items():
            team_count[k] += v
    return max(team_count, key=team_count.get) if team_count else "ball"

def index_to_letter_suffix(idx):
    """Return 'a', 'b', ..., 'z', 'aa', 'ab', ... as suffix."""
    letters = []
    while True:
        letters.append(chr(97 + (idx % 26)))
        idx = idx // 26
        if idx == 0:
            break
        idx -= 1  # offset for 0-based index
    return ''.join(reversed(letters))

def split_track_by_sliding_window(
    obj: Dict[str, Any],
    window_size: int = 20,
    threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Splits a track when a new team dominates a sliding window.

    Args:
        obj: Original track.
        window_size: Size of the sliding window.
        threshold: Ratio of frames in the window needed for a team to trigger a split.

    Returns:
        List of split track segments.
    """
    team_conf_list = obj["team_conf"]
    frame_ids = obj["frame_id"]
    projected = obj["projected"]
    bboxes = obj.get("bbox", [])

    # Get dominant team label for each frame
    dominant_team_list = [
        max(conf, key=conf.get) if conf else "ball"
        for conf in team_conf_list
    ]

    segments = []
    buffer = []
    i = 0
    current_team = assign_team_by_majority_vote(team_conf_list)

    while i < len(dominant_team_list):
        if dominant_team_list[i] == current_team:
            buffer.append(i)
            i += 1
            continue

        # Only check if enough room for a full window
        if i + window_size <= len(dominant_team_list):
            window = dominant_team_list[i:i + window_size]
            counter = defaultdict(int)
            for t in window:
                counter[t] += 1
            dominant_in_window = max(counter, key=counter.get)
            ratio = counter[dominant_in_window] / window_size

            if dominant_in_window != current_team and ratio >= threshold:
                segment_conf_list = [team_conf_list[j] for j in buffer]
                if segment_conf_list:
                    team_score = sum(conf.get(current_team, 0.0) for conf in segment_conf_list) / len(segment_conf_list)
                else:
                    team_score = 0.0
                # Commit segment
                segment = {
                    "track_id": f"{obj['track_id']}{chr(97 + len(segments))}",
                    "frame_id": [frame_ids[j] for j in buffer],
                    "projected": [projected[j] for j in buffer],
                    "bbox": [bboxes[j] for j in buffer] if bboxes else [],
                    "team_conf": team_score,
                    "team": current_team
                }
                segments.append(segment)
                buffer = []
                current_team = dominant_in_window
                # move window forward
                continue

        buffer.append(i)
        i += 1

    # Final segment
    if buffer:
        segment_conf_list = [team_conf_list[j] for j in buffer]
        team_score = sum(conf.get(current_team, 0.0) for conf in segment_conf_list) / len(segment_conf_list)
        segment = {
            "track_id": f"{obj['track_id']}{chr(97 + len(segments))}",
            "frame_id": [frame_ids[j] for j in buffer],
            "projected": [projected[j] for j in buffer],
            "bbox": [bboxes[j] for j in buffer] if bboxes else [],
            "team_conf": team_score,
            "team": current_team
        }
        segments.append(segment)

    return segments

def interpolate_full_track(frames: List[int], points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate full track to fill in all missing frames using linear interpolation.

    Args:
        frames (List[int]): List of frame indices.
        points (np.ndarray): Corresponding points (N, 2) for each frame.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated frames and points (in same order).
    """
    if len(frames) < 2:
        return np.array(frames), points

    all_frames = np.arange(frames[0], frames[-1] + 1)
    xs_interp = np.interp(all_frames, frames, points[:, 0])
    ys_interp = np.interp(all_frames, frames, points[:, 1])
    full_points = np.stack([xs_interp, ys_interp], axis=1)

    return all_frames, full_points

def hybrid_merge_stream_fixed(
    jsonl_path: str,
    output_path: str,
    max_merge_gap: int = 5,
    max_merge_overlap_frames: int = 3,
    max_merge_distance: float = 10,
    smoothing_window: int = 11,
    polyorder: int = 3,
    max_step: int = 20,
):

    final_output = open(output_path, 'w')
    frame_to_tracks = defaultdict(list)
    active_tracks = {}
    done_tracks = set()

    # Load all segments from JSONL (inlined stream_jsonl_segments)
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            seg = json.loads(line)
            start_frame = seg['frames'][0]
            frame_to_tracks[start_frame].append(seg)

    max_buffer_frame = max(frame_to_tracks.keys()) if frame_to_tracks else 0
    current_frame = 0

    while current_frame <= max_buffer_frame:
        # Load candidate segments for current frame window
        candidates = []
        for offset in range(-max_merge_overlap_frames, max_merge_gap + 1):
            f = current_frame + offset
            candidates.extend(frame_to_tracks.get(f, []))

        merged_this_round = set()
        for seg in candidates:
            tid = seg['track_id']
            if tid in done_tracks or tid in merged_this_round:
                continue

            best_match = None
            best_dist = float('inf')
            for mtid, m in active_tracks.items():
                if seg['team'] != m['team']:
                    continue
                last_frame = m['frames'][-1]
                gap = seg['frames'][0] - last_frame
                if not ((0 <= gap <= max_merge_gap) or (0 < -gap <= max_merge_overlap_frames)):
                    continue
                dist = np.linalg.norm(
                    np.array(m['points'][-1]) - np.array(seg['points'][0])
                )
                if dist <= max_merge_distance and dist < best_dist:
                    best_match = mtid
                    best_dist = dist

            if best_match:
                m = active_tracks[best_match]
                m['frames'].extend(seg['frames'])
                m['points'].extend(seg['points'])
                m['team_conf_total'] += seg.get("team_conf", 0.0) * len(seg['frames'])
                m['team_conf_len'] += len(seg['frames'])
                merged_this_round.add(tid)
                done_tracks.add(tid)
            else:
                active_tracks[tid] = {
                    "track_id": tid,
                    "team": seg['team'],
                    "frames": seg['frames'],
                    "points": seg['points'],
                    "team_conf_total": seg.get("team_conf", 0.0) * len(seg['frames']),
                    "team_conf_len": len(seg['frames']),
                }
                merged_this_round.add(tid)

        # Finalize stale tracks
        to_remove = []
        for tid, m in active_tracks.items():
            if m['frames'][-1] < current_frame - max_merge_gap:
                frames, points = interpolate_full_track(m['frames'], np.array(m['points']))
                if len(points) >= smoothing_window:
                    xs = savgol_filter(points[:, 0], smoothing_window, polyorder)
                    ys = savgol_filter(points[:, 1], smoothing_window, polyorder)
                    points = np.stack([xs, ys], axis=1)
                team_conf = m['team_conf_total'] / m['team_conf_len'] if m['team_conf_len'] else 0.0
                output = {
                    "track_id": m['track_id'],
                    "team": m['team'],
                    "frame_range": [int(frames[0]), int(frames[-1])],
                    "frames": frames.tolist(),
                    "projected": points.tolist(),
                    "team_conf": team_conf,
                }
                final_output.write(json.dumps(output) + '\n')
                to_remove.append(tid)
                done_tracks.add(tid)

        for tid in to_remove:
            del active_tracks[tid]

        current_frame += 1

    # Final flush
    for tid, m in active_tracks.items():
        frames = np.array(m['frames'])
        points = np.array(m['points'])
        if len(points) >= smoothing_window:
            xs = savgol_filter(points[:, 0], smoothing_window, polyorder)
            ys = savgol_filter(points[:, 1], smoothing_window, polyorder)
            points = np.stack([xs, ys], axis=1)
        team_conf = m['team_conf_total'] / m['team_conf_len'] if m['team_conf_len'] else 0.0
        output = {
            "track_id": m['track_id'],
            "team": m['team'],
            "frame_range": [int(frames[0]), int(frames[-1])],
            "frames": frames.tolist(),
            "projected": points.tolist(),
            "team_conf": team_conf,
        }
        final_output.write(json.dumps(output) + '\n')

    final_output.close()
    print(f"✅ Merged and saved to: {output_path}")

def load_and_merge_tracks(
    json_path,
    output_path,
    field_size,
    min_track_length,
    smoothing_window,
    polyorder,
    max_step,
    max_merge_gap,
    max_merge_overlap_frames,
    max_merge_distance,
    window_size,
    threshold,
):
    """
    Draw smoothed 2D trajectories from tracking JSON with optional merging of fragmented tracks.

    Args:
        json_path (str): Path to tracking JSON.
        image_path (str): Path to field background image.
        field_size (tuple): Field dimension (width, height).
        min_track_length (int): Minimum track length to visualize.
        smoothing_window (int): Window size for Savitzky-Golay smoothing.
        polyorder (int): Polynomial order for smoothing.
        max_merge_gap (int): Max frame gap between track ends to consider merging.
        max_merge_distance (float): Max distance in projected space to consider merging.


    Returns:
        merged_tracks (list): List of merged track dicts with keys 'track_id', 'team', 'frames', and 'points'.
        bg_img (np.ndarray): The resized field background image.
    """

    track_dict = {}

    with open(json_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            # print(type(obj['projected'][0][0]))  # e.g., <class 'float'>
            projected_points = obj.get("projected", [])
            if len(projected_points) < min_track_length:
                continue

            pts = np.array([pt for pt in projected_points if pt is not None])
            if len(pts) < min_track_length:
                continue

            xs, ys = pts[:, 0], pts[:, 1]
            in_bounds = (xs >= 0) & (xs <= field_size[0]) & (ys >= 0) & (ys <= field_size[1])
            if in_bounds.sum() < min_track_length:
                continue

            obj["frame_id"] = np.array(obj["frame_id"])[in_bounds].tolist()
            obj["projected"] = pts[in_bounds].tolist()
            if "bbox" in obj:
                obj["bbox"] = np.array(obj["bbox"])[in_bounds].tolist()
            if "team_conf" in obj:
                obj["team_conf"] = np.array(obj["team_conf"])[in_bounds].tolist()

            # Now we split the clean long track
            split_objects = split_track_by_sliding_window(obj, window_size, threshold)

            for split_obj in split_objects:
                tid = split_obj["track_id"]
                frames = split_obj["frame_id"]
                projected_points = split_obj["projected"]

                pts = np.array([pt for pt in projected_points if pt is not None])
                if len(pts) == 0:
                    continue  # skip this segment
                xs, ys = pts[:, 0], pts[:, 1]

                frs = np.array(frames)

                track_dict = {
                    "track_id": tid,
                    "team": split_obj.get("team", "ball"),
                    "team_conf": split_obj.get("team_conf", []),
                    "frames": frs.tolist(),
                    "points": np.stack([xs, ys], axis=1).tolist(),
                }

                # save the track_dicrt to jsonl
                # output_json_path = os.path.splitext(json_path)[0] + "_spilt.jsonl"
                with open(output_path, 'a') as out_f:
                    out_f.write(json.dumps(track_dict) + '\n')   

def remove_referee_near_boundary_stream(jsonl_path, output_jsonl_path, field_size, margin_meter=3.0):
    """
    Stream version that writes filtered tracks directly to a new .jsonl file.

    Args:
        jsonl_path (str): Input path to merged .jsonl file.
        output_jsonl_path (str): Path to write filtered output.
        field_size (tuple): Field dimensions (length, width) in 0.1 meters.
        margin_meter (float): Distance from boundary (in meters) considered "near".
    """
    with open(jsonl_path, 'r') as f, open(output_jsonl_path, 'w') as out_f:
        for line in f:
            track = json.loads(line)

            if track["team"] != "referee":
                out_f.write(json.dumps(track) + '\n')
                continue

            points = np.array(track["projected"])
            xs, ys = points[:, 0], points[:, 1]

            near_left = (xs < margin_meter).sum()
            near_right = (xs > field_size[0] - margin_meter).sum()
            near_top = (ys < margin_meter).sum()
            near_bottom = (ys > field_size[1] - margin_meter).sum()

            near_edge_ratio = (near_left + near_right + near_top + near_bottom) / len(points)

            if near_edge_ratio < 0.7:
                out_f.write(json.dumps(track) + '\n')

def prepare_background_and_tracks(
    json_path,
    image_path,
    field_size,
    min_track_length,
    smoothing_window,
    polyorder,
    max_step,
    max_merge_gap,
    max_merge_overlap_frames,
    max_merge_distance,
    window_size,
    threshold
):
    # Load and resize background
    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    bg_img = cv2.resize(bg_img, field_size)

    # Merge and filter tracks
    load_and_merge_tracks(
        json_path=json_path,
        output_path=json_path.replace('.jsonl', '_spilt.jsonl'),
        field_size=field_size,
        min_track_length=min_track_length,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        max_merge_gap=max_merge_gap,
        max_merge_distance=max_merge_distance,
        max_merge_overlap_frames=max_merge_overlap_frames,
        window_size=window_size,
        threshold=threshold,
        max_step=max_step,
    )

    hybrid_merge_stream_fixed(
        jsonl_path=json_path.replace('.jsonl', '_spilt.jsonl'),
        output_path=json_path.replace('.jsonl', '_merged.jsonl'),
        max_merge_gap=max_merge_gap,
        max_merge_overlap_frames=max_merge_overlap_frames,
        max_merge_distance=max_merge_distance,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
        max_step=max_step,
    )

    remove_referee_near_boundary_stream(
        jsonl_path=json_path.replace('.jsonl', '_merged.jsonl'),
        output_jsonl_path=json_path.replace('.jsonl', '_merged_filtered.jsonl'),
        field_size=field_size,
        margin_meter=30
    )

    detect_team_size_violations_streaming(
        jsonl_path=json_path.replace('.jsonl', '_merged_filtered.jsonl'),
        save_path=json_path.replace('.jsonl', '_team_size_violations.jsonl'),
        max_team_size=10,
        allowed_goalkeepers=1,
        allowed_referees=1
    )

    merged_window = merge_problem_windows_by_reason(
        jsonl_path=json_path.replace('.jsonl', '_team_size_violations.jsonl'),
        min_gap=3
    )

    print("Merged problem windows by reason:")
    for reason, windows in merged_window.items():
        print(f"{reason}: {windows}")

    relabel_count = relabel_tracks_by_overlap_and_confidence(
        track_jsonl_path=json_path.replace('.jsonl', '_merged_filtered.jsonl'),
        reason_to_windows=merged_window,
        output_jsonl_path=json_path.replace('.jsonl', '_relabeled.jsonl'),
        overlap_threshold=0.4,
        conf_threshold=0.007,
        not_sure_label="unsure"
    )

    print(f"Total tracks relabeled: {relabel_count}")

    return bg_img

def render_to_image_from_jsonl(jsonl_path, bg_img, field_size, min_track_length, output_path="trajectory_plot.png"):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(bg_img[..., ::-1], extent=[0, field_size[0], 0, field_size[1]])
    team_colors = {
        'eastern': 'blue',
        'easterngoalkeeper': 'green',
        'kitchee': 'pink',
        'kitcheegoalkeeper': 'orange',
        'referee': 'yellow',
        'ball': 'black',
        'unsure': 'red',  # For relabeled tracks
    }

    with open(jsonl_path, 'r') as f:
        for line in f:
            track = json.loads(line)
            points = np.array(track.get("projected", track.get("points", [])))

            # Remove None points
            points = np.array([pt for pt in points if pt is not None])
            if len(points) < min_track_length:
                continue

            xs, ys = points[:, 0], points[:, 1]
            color = team_colors.get(track["team"], 'gray')
            ax.plot(xs, ys, color=color, alpha=0.8)

            # Draw last valid point
            if len(xs) > 0 and len(ys) > 0:
                ax.scatter(xs[-1], ys[-1], color=color)
                ax.text(xs[-1], ys[-1], str(track["track_id"]), fontsize=8, color='black')

    ax.set_xlim(0, field_size[0])
    ax.set_ylim(0, field_size[1])
    ax.set_title("Smoothed & Merged Trajectories")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved image to: {output_path}")

def render_to_video_from_jsonl(jsonl_path, bg_img, field_size, output_path, fps=30):
    height, width, _ = bg_img.shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    team_colors = {
        'eastern': (255, 0, 0),
        'easterngoalkeeper': (0, 255, 0),
        'kitchee': (255, 192, 203),
        'kitcheegoalkeeper': (0, 165, 255),
        'referee': (0, 255, 255),
        'ball': (0, 0, 0),
        'unsure': (0, 0, 255),  # For relabeled tracks
    }

    tracks = [json.loads(line) for line in open(jsonl_path)]
    min_frame = min(min(t["frames"]) for t in tracks)
    max_frame = max(max(t["frames"]) for t in tracks)

    frame_to_objects = defaultdict(list)
    for t in tracks:
        for i, f in enumerate(t["frames"]):
            if i >= len(t["projected"]):
                continue
            pt = t["projected"][i]
            if pt is None:
                continue
            frame_to_objects[f].append((pt, t["track_id"], t["team"]))

    for f in range(min_frame, max_frame + 1):
        frame_img = bg_img.copy()
        for pt, tid, team in frame_to_objects.get(f, []):
            x, y = int(pt[0]), field_size[1] - int(pt[1])
            color = team_colors.get(team, (128, 128, 128))
            cv2.circle(frame_img, (x, y), 5, color, -1)
            cv2.putText(frame_img, str(tid), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        writer.write(frame_img)

    writer.release()
    print(f"✅ Saved video to: {output_path}")

def process_merged_tracks(
    json_path,
    image_path,
    field_size,
    min_track_length,
    smoothing_window,
    polyorder,
    max_step,
    max_merge_gap,
    max_merge_overlap_frames,
    max_merge_distance,
    window_size,
    threshold,
    output_type='image',
    output_name='trajectory_plot',
    fps=30
):
    # Auto-generate full path with extension
    if output_type == 'image':
        output_path_image = f"{output_name}.png"
    elif output_type == 'video':
        output_path_video = f"{output_name}.mp4"
    elif output_type == 'all':
        output_path_image = f"{output_name}.png"
        output_path_video = f"{output_name}.mp4"
    else:
        raise ValueError("Unsupported output type. Use 'image', 'video' or 'all'.")

    # Shared logic
    bg_img = prepare_background_and_tracks(
        json_path, image_path, field_size,
        min_track_length, smoothing_window, polyorder, max_step,
        max_merge_gap, max_merge_overlap_frames, max_merge_distance,
        window_size, threshold
    )

    if output_type in ['image', 'all']:
        render_to_image_from_jsonl(
            jsonl_path=json_path.replace('.jsonl', '_relabeled.jsonl'),
            bg_img=bg_img,
            field_size=field_size,
            min_track_length=min_track_length,
            output_path=output_path_image
        )
    if output_type in ['video', 'all']:
        render_to_video_from_jsonl(
            jsonl_path=json_path.replace('.jsonl', '_merged_filtered.jsonl'),
            bg_img=bg_img,
            field_size=field_size,
            output_path=output_path_video,
            fps=fps
        )


def detect_team_size_violations_streaming(jsonl_path, save_path,
                                          max_team_size=10,
                                          allowed_goalkeepers=1,
                                          allowed_referees=1):
    """
    Streaming version: detects violations and saves directly to JSONL.

    Args:
        jsonl_path (str): Input .jsonl file path.
        save_path (str): Output .jsonl for violations.
    """
    frame_map = defaultdict(list)  # frame_id -> list of (team, track_id)

    print("🔄 Indexing frames...")
    # First pass: build minimal frame index
    with open(jsonl_path, "r") as f:
        for line in f:
            track = json.loads(line)
            team = track.get("team", "")
            tid = track.get("track_id")
            frame_range = track.get("frame_range", [])

            if not frame_range or len(frame_range) != 2:
                continue

            start, end = frame_range
            for frame_id in range(start, end + 1):
                frame_map[frame_id].append((team, tid))

    print("✅ Frame indexing complete. Writing violations...")

    # Second pass: detect violations and write line-by-line
    with open(save_path, "w") as out_f:
        for frame_id in sorted(frame_map.keys()):
            team_counter = defaultdict(list)
            for team, tid in frame_map[frame_id]:
                team_counter[team].append(tid)

            violations = {}

            for team, tids in team_counter.items():
                if team == "referee" and len(tids) > allowed_referees:
                    violations[team] = tids
                elif team.endswith("goalkeeper") and len(tids) > allowed_goalkeepers:
                    violations[team] = tids
                elif not team.endswith("goalkeeper") and team != "referee" and len(tids) > max_team_size:
                    violations[team] = tids

            if violations:
                out_f.write(json.dumps({
                    "frame_id": frame_id,
                    "violations": violations
                }) + "\n")

    print(f"📄 Saved violations to {save_path}")

def merge_problem_windows_by_reason(jsonl_path: str, min_gap: int = 1) -> Dict[str, List[Tuple[int, int]]]:
    """
    Merges frame_ids with violations from a JSONL file into continuous frame ranges per violation reason.

    Args:
        jsonl_path (str): Path to the JSONL file, each line is a dict with 'frame_id' and 'violations'.
        min_gap (int): Minimum gap between frames to merge into the same window.

    Returns:
        Dict[str, List[Tuple[int, int]]]: Dictionary where each key is a violation reason and value is list of (start, end) frame ranges.
    """
    reason_to_frames = defaultdict(list)

    with open(jsonl_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            if "violations" in obj and obj["violations"]:
                frame_id = obj["frame_id"]
                for reason, ids in obj["violations"].items():
                    if ids:
                        reason_to_frames[reason].append(frame_id)

    reason_to_merged = {}
    for reason, frames in reason_to_frames.items():
        if not frames:
            continue
        frames = sorted(set(frames))
        merged = []
        start = prev = frames[0]
        for frame in frames[1:]:
            if frame <= prev + min_gap:
                prev = frame
            else:
                merged.append((start, prev))
                start = prev = frame
        merged.append((start, prev))
        reason_to_merged[reason] = merged

    return reason_to_merged

def relabel_tracks_by_overlap_and_confidence(
    track_jsonl_path: str,
    reason_to_windows: Dict[str, List[Tuple[int, int]]],
    output_jsonl_path: str,
    overlap_threshold: float = 0.8,
    conf_threshold: float = 0.007,
    not_sure_label: str = "unsure"
) -> int:
    """
    Relabel low-confidence tracks that significantly overlap with problem windows by reason.

    Args:
        track_jsonl_path (str): Path to the input track JSONL file.
        reason_to_windows (dict): Dictionary of violation reason -> list of (start, end) frame ranges.
        output_jsonl_path (str): Path to save the modified tracks as a JSONL file.
        overlap_threshold (float): Minimum overlap ratio to consider for relabeling.
        conf_threshold (float): Maximum confidence to be considered low-confidence.
        not_sure_label (str): Label to assign for uncertain tracks.

    Returns:
        int: Number of tracks relabeled.
    """
    # Load all tracks
    tracks = []
    with open(track_jsonl_path, "r") as f:
        for line in f:
            track = json.loads(line.strip())
            tracks.append(track)

    relabel_count = 0

    for reason, windows in reason_to_windows.items():
        pending_windows = windows[:]

        while pending_windows:
            current_window = pending_windows.pop(0)
            window_start, window_end = current_window

            # Find candidate tracks from the same team
            candidate_tracks = []
            for track in tracks:
                if track.get("team") != reason:
                    continue
                frame_range = track.get("frame_range", [])
                if not frame_range or len(frame_range) != 2:
                    continue
                track_start, track_end = frame_range
                # Compute intersection
                inter_start = max(window_start, track_start)
                inter_end = min(window_end, track_end)
                if inter_end >= inter_start:
                    overlap = inter_end - inter_start + 1
                    duration = track_end - track_start + 1
                    overlap_ratio = overlap / duration
                    if overlap_ratio >= overlap_threshold:
                        candidate_tracks.append((track, overlap, inter_start, inter_end))

            # Sort by team_conf
            candidate_tracks.sort(key=lambda x: x[0].get("team_conf", 1.0))

            for track, overlap, inter_start, inter_end in candidate_tracks:
                if track.get("team_conf", 1.0) <= conf_threshold:
                    team = track.get("team", "")
                    if not team.startswith("referee") and not team.endswith("goalkeeper"):
                        track["team"] = not_sure_label
                        relabel_count += 1

                        # Add remaining segments of the current window back
                        if window_start < inter_start:
                            pending_windows.append((window_start, inter_start - 1))
                        if inter_end < window_end:
                            pending_windows.append((inter_end + 1, window_end))
                        break  # Only one relabel per window

    # Save modified tracks
    with open(output_jsonl_path, "w") as f:
        for track in tracks:
            json.dump(track, f)
            f.write("\n")

    print(f"✅ Relabeled {relabel_count} tracks and saved to {output_jsonl_path}")
    return relabel_count

if  __name__ == "__main__":
    start = time.time()

    process_merged_tracks(
        json_path="./runs/detect/test_4k/team_tracking.jsonl",
        image_path="./data/images/mongkok_football_field.png",
        field_size=(1060, 660),
        min_track_length=10,
        smoothing_window=90,
        polyorder=7,
        max_step=20,
        max_merge_gap=20,
        max_merge_overlap_frames=15,
        max_merge_distance=50,
        window_size=20,
        threshold=0.9,
        output_type='image', # 'image', 'video', 'jsonl', or 'all'
        output_name='trajectory_plot' 
    )

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")