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
from heapq import nsmallest

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

def frame_to_time(frame: int, fps: float = 29.97, format_output: bool = True) -> str:
    """
    Convert frame index to time based on FPS.
    
    Args:
        frame (int): Frame index.
        fps (float): Frames per second. Default is 29.97.
        format_output (bool): If True, return formatted time (HH:MM:SS.ms), else return seconds.
    
    Returns:
        str or float: Formatted timestamp or raw seconds.
    """
    seconds = frame / fps
    if not format_output:
        return seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"  # includes milliseconds

def render_to_image_from_jsonl(
    jsonl_path,
    bg_img,
    field_size,
    output_path="trajectory_plot.png",
    start_frame: int = None,
    end_frame: int = None
):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(bg_img[..., ::-1], extent=[0, field_size[0], 0, field_size[1]])

    team_colors = {
        'eastern': 'blue',
        'easterngoalkeeper': 'green',
        'kitchee': 'pink',
        'kitcheegoalkeeper': 'orange',
        'referee': 'yellow',
        'ball': 'black',
        'unsure': 'red',
    }

    with open(jsonl_path, 'r') as f:
        for line in f:
            track = json.loads(line)
            frames = track.get("frames", [])
            points = np.array(track.get("projected", track.get("points", [])))

            if len(frames) != len(points):
                continue  # skip bad track

            # Filter by selected time window
            if start_frame is not None and end_frame is not None:
                filtered = [
                    (f, pt) for f, pt in zip(frames, points)
                    if pt is not None and start_frame <= f <= end_frame
                ]
                if not filtered:
                    continue
                frames, points = zip(*filtered)
                points = np.array(points)
            else:
                # fallback: remove None
                points = np.array([pt for pt in points if pt is not None])

            xs, ys = points[:, 0], points[:, 1]
            color = team_colors.get(track["team"], 'gray')
            ax.plot(xs, ys, color=color, alpha=0.8)

            # Draw last point and label
            ax.scatter(xs[-1], ys[-1], color=color)
            ax.text(xs[-1], ys[-1], str(track["track_id"]), fontsize=8, color='black')

    ax.set_xlim(0, field_size[0])
    ax.set_ylim(0, field_size[1])
    ax.set_title(f"Trajectories from time {frame_to_time(start_frame)} to {frame_to_time(end_frame)}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved image to: {output_path}")

def render_to_video_from_jsonl(jsonl_path, bg_img, field_size, output_path, fps=29.97):
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
    min_frame = 0
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

def remove_tracks_near_boundary_stream(
    jsonl_path, 
    output_jsonl_path, 
    field_size, 
    margin_meter=30, 
    near_ratio_threshold=0.9
):
    """
    Removes tracks that stay near the field boundary for most of the time.

    Args:
        jsonl_path (str): Input path to .jsonl file.
        output_jsonl_path (str): Output path to write filtered tracks.
        field_size (tuple): Field dimensions (length, width) in 0.1 meters.
        margin_meter (float): Distance from edge considered "near".
        near_ratio_threshold (float): Ratio of points near edge to consider it a boundary-only track.
    """
    with open(jsonl_path, 'r') as f_in, open(output_jsonl_path, 'w') as f_out:
        for line in f_in:
            track = json.loads(line)
            team = track.get("team", "")
            points = np.array(track.get("projected", []))

            if len(points) == 0:
                continue  # skip empty tracks

            if team == "ball":
                f_out.write(json.dumps(track) + "\n")
                continue  # Always keep ball

            xs, ys = points[:, 0], points[:, 1]
            near_left = xs < margin_meter
            near_right = xs > (field_size[0] - margin_meter)
            near_top = ys < margin_meter
            near_bottom = ys > (field_size[1] - margin_meter)

            near_edge_mask = near_left | near_right | near_top | near_bottom
            near_edge_ratio = near_edge_mask.sum() / len(points)

            if near_edge_ratio < near_ratio_threshold:
                f_out.write(json.dumps(track) + "\n")

def remove_static_ball_tracks(
    jsonl_path,
    output_jsonl_path,
    movement_threshold=10  # in meters (10 = 1m if 0.1m units)
):
    """
    Remove ball tracks that don't move significantly.

    Args:
        jsonl_path (str): Input path to .jsonl file.
        output_jsonl_path (str): Output path to write filtered tracks.
        movement_threshold (float): Minimum total movement (Euclidean) to keep.
    """
    with open(jsonl_path, 'r') as f_in, open(output_jsonl_path, 'w') as f_out:
        for line in f_in:
            track = json.loads(line)
            if track.get("team") != "ball":
                f_out.write(json.dumps(track) + "\n")
                continue

            points = np.array(track.get("projected", []))
            if len(points) < 2:
                continue  # skip too short

            # Compute total movement
            deltas = np.diff(points, axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            total_distance = distances.sum()

            if total_distance >= movement_threshold:
                f_out.write(json.dumps(track) + "\n")

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

def merge_violation_windows_with_track_counts(jsonl_path: str, min_gap: int = 1) -> Dict[str, List[Dict]]:
    """
    Merge consecutive violation frames into windows grouped by team and number of violating tracks.
    
    Args:
        jsonl_path (str): Path to the input JSONL file with per-frame violations.
        min_gap (int): Allowed gap between frames to merge into the same window.
    
    Returns:
        Dict[str, List[Dict]]: Dictionary with team as key and list of merged windows as value.
    """
    team_count_to_frames = defaultdict(lambda: defaultdict(list))  # team -> count -> list of frame info

    # Read and categorize frames
    with open(jsonl_path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            frame_id = obj["frame_id"]
            violations = obj.get("violations", {})
            for team, ids in violations.items():
                track_count = len(ids)
                team_count_to_frames[team][track_count].append((frame_id, set(ids)))

    merged_result = defaultdict(list)

    for team, count_to_frames in team_count_to_frames.items():
        for count, frames in count_to_frames.items():
            if not frames:
                continue
            frames = sorted(frames, key=lambda x: x[0])
            merged = []
            start, prev_frame, current_ids = frames[0][0], frames[0][0], frames[0][1].copy()
            
            for i in range(1, len(frames)):
                frame_id, ids = frames[i]
                if frame_id <= prev_frame + min_gap:
                    current_ids.update(ids)
                    prev_frame = frame_id
                else:
                    merged.append({
                        "range": [start, prev_frame],
                        "count": count,
                        "track_ids": sorted(current_ids)
                    })
                    start = prev_frame = frame_id
                    current_ids = ids.copy()
            
            # Append the last segment
            merged.append({
                "range": [start, prev_frame],
                "count": count,
                "track_ids": sorted(current_ids)
            })

            merged_result[team].extend(merged)

    return merged_result

def relabel_tracks_by_confidence_and_decrement_windows_streaming(
    track_jsonl_path: str,
    team_windows: Dict[str, List[Dict]],
    output_jsonl_path: str,
    conf_threshold: float = 0.007,
    not_sure_label: str = "unsure"
) -> int:
    """
    Efficiently relabel low-confidence tracks that violate team size constraints,
    and decrement violation windows. Uses a one-pass preload strategy.

    Args:
        track_jsonl_path (str): Path to the input track JSONL file.
        team_windows (dict): Team -> List of dicts with keys: 'range', 'count', 'track_ids'.
        output_jsonl_path (str): Path to save the modified track JSONL.
        conf_threshold (float): Confidence threshold for relabeling.
        not_sure_label (str): Label to assign to uncertain tracks.

    Returns:
        int: Number of tracks relabeled.
    """

    print("📦 Preloading track data...")
    track_map = {}
    with open(track_jsonl_path, "r") as f:
        for line in f:
            track = json.loads(line.strip())
            tid = track.get("track_id")
            if tid:
                track_map[tid] = track
    print(f"✅ Loaded {len(track_map)} tracks.")

    relabel_count = 0
    relabel_map = {}  # tid -> new label

    team_windows = {k: [w.copy() for w in v] for k, v in team_windows.items()}

    for team, windows in team_windows.items():
        while windows:
            print(f"🔄 Processing team: {team}, remaining windows: {len(windows)}")
            window = windows.pop(0)
            win_start, win_end = window["range"]
            count = window["count"]
            track_ids = set(window["track_ids"])

            allowed_count = 1 if team.endswith("goalkeeper") or team == "referee" else 10
            excess = count - allowed_count
            if excess <= 0:
                continue

            # Filter and collect candidate tracks from preload
            candidate_tracks = []
            for tid in track_ids:
                if tid in relabel_map:
                    continue  # already relabeled
                track = track_map.get(tid)
                if not track or track.get("team") != team:
                    continue
                conf = track.get("team_conf", 1.0)
                frame_range = track.get("frame_range", [0, 0])
                track_start, track_end = frame_range
                # Overlap check
                if win_end < track_start or win_start > track_end:
                    continue
                overlap_start = max(win_start, track_start)
                overlap_end = min(win_end, track_end)
                overlap = overlap_end - overlap_start + 1
                duration = track_end - track_start + 1
                overlap_ratio = overlap / duration if duration > 0 else 0
                if overlap_ratio > 0.1:
                    candidate_tracks.append((conf, tid, track_start, track_end))

            # Only take lowest confidence ones
            candidate_tracks = nsmallest(excess, candidate_tracks)

            relabeled_in_window = 0
            for conf, tid, t_start, t_end in candidate_tracks:
                if conf > conf_threshold or tid in relabel_map:
                    continue
                relabel_map[tid] = not_sure_label
                relabel_count += 1
                relabeled_in_window += 1

                # Decrement other windows that overlap
                new_windows = []
                for other in windows:
                    ow_start, ow_end = other["range"]
                    if ow_end < t_start or ow_start > t_end:
                        new_windows.append(other)
                        continue
                    if t_start > ow_start:
                        new_windows.append({
                            "range": [ow_start, t_start - 1],
                            "count": other["count"],
                            "track_ids": other["track_ids"]
                        })
                    if t_end < ow_end:
                        new_windows.append({
                            "range": [t_end + 1, ow_end],
                            "count": other["count"],
                            "track_ids": other["track_ids"]
                        })
                windows = new_windows

            # If still unresolved, re-add current window
            window["count"] -= relabeled_in_window
            if window["count"] > allowed_count and relabeled_in_window > 0:
                windows.append(window)

    # Final pass to write relabeled file
    with open(track_jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
        for line in f_in:
            track = json.loads(line.strip())
            tid = track.get("track_id")
            if tid in relabel_map:
                if track.get("team") not in [not_sure_label, "referee"] and not track.get("team", "").endswith("goalkeeper"):
                    track["team"] = relabel_map[tid]
            json.dump(track, f_out)
            f_out.write("\n")

    print(f"✅ Relabeled {relabel_count} tracks.")
    return relabel_count

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

    remove_tracks_near_boundary_stream(
        jsonl_path=json_path.replace('.jsonl', '_merged.jsonl'),
        output_jsonl_path=json_path.replace('.jsonl', '_merged_filtered_near_boundary.jsonl'),
        field_size=field_size,
        margin_meter=30
    )

    remove_static_ball_tracks(
        json_path.replace('.jsonl', '_merged_filtered_near_boundary.jsonl'),
        json_path.replace('.jsonl', '_merged_filtered.jsonl'),
        movement_threshold=10  # in meters (10 = 1m if 0.1m units)
    )

    detect_team_size_violations_streaming(
        jsonl_path=json_path.replace('.jsonl', '_merged_filtered.jsonl'),
        save_path=json_path.replace('.jsonl', '_team_size_violations.jsonl'),
        max_team_size=10,
        allowed_goalkeepers=1,
        allowed_referees=1
    )

    start_merged = time.time()
    merged_window = merge_violation_windows_with_track_counts(
        jsonl_path=json_path.replace('.jsonl', '_team_size_violations.jsonl'),
        min_gap=3
    )
    end_merged = time.time()

    for team, windows in merged_window.items():
        print(f"🟢 Team {team} → {len(windows)} merged windows")
    print(f"✅ Merged windows in {end_merged - start_merged:.2f} seconds")

    relabel_count = relabel_tracks_by_confidence_and_decrement_windows_streaming(
        track_jsonl_path=json_path.replace('.jsonl', '_merged_filtered.jsonl'),
        team_windows=merged_window,
        output_jsonl_path=json_path.replace('.jsonl', '_relabeled.jsonl'),
        conf_threshold=0.007,
        not_sure_label="unsure"
    )
    end_relabel = time.time()
    print(f"✅ Relabeled {relabel_count} tracks in {end_relabel - end_merged:.2f} seconds")
    return bg_img

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
    output_name,
    fps=29.97
):
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(json_path))[0]

    output_path_video = f"{output_name}.mp4"

    # Shared logic
    bg_img = prepare_background_and_tracks(
        json_path, image_path, field_size,
        min_track_length, smoothing_window, polyorder, max_step,
        max_merge_gap, max_merge_overlap_frames, max_merge_distance,
        window_size, threshold
    )

    render_to_video_from_jsonl(
        jsonl_path=json_path.replace('.jsonl', '_relabeled.jsonl'),
        bg_img=bg_img,
        field_size=field_size,
        output_path=output_path_video,
        fps=fps
    )


if  __name__ == "__main__":
    start = time.time()

    process_merged_tracks(
        json_path="./runs/detect/test_4k-2h/team_tracking.jsonl",
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
        output_name='./runs/detect/test_4k-2h/team_tracking_output'  # without extension
    )

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")