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
from tools.remove_track_sharp import process_jsonl_detect_replace
import argparse

from scipy import interpolate
from scipy.signal import find_peaks

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor

from scipy.linalg import expm
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

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

def load_and_spilt_tracks(
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
    end_frame: int = None,
    highlight_ids: List[str] = None  # 👈 new param to highlight suspicious tracks
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
        'unsure': 'gray',
    }

    highlight_ids = set(highlight_ids or [])  # ensure it's a set

    with open(jsonl_path, 'r') as f:
        for line in f:
            track = json.loads(line)
            track_id = track["track_id"]
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
                if len(points) == 0:
                    continue

            xs, ys = points[:, 0], points[:, 1]

            color = team_colors.get(track.get("team", "unsure"), 'gray')

            ax.plot(xs, ys, color=color, alpha=0.8)
            ax.scatter(xs[-1], ys[-1], color=color)
            label_color = 'red' if track_id in highlight_ids else 'black'
            ax.text(xs[-1], ys[-1], str(track_id), fontsize=8, color=label_color)

    ax.set_xlim(0, field_size[0])
    ax.set_ylim(0, field_size[1])
    if start_frame is not None and end_frame is not None:
        ax.set_title(f"Trajectories from time {frame_to_time(start_frame)} to {frame_to_time(end_frame)}")
    else:
        ax.set_title("Trajectories (full match)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved image to: {output_path}")

def render_to_video_from_jsonl(
    jsonl_path,
    bg_img,
    field_size,
    output_path,
    fps=29.97,
    start_frame: int = None,
    end_frame: int = None,
    suspicious_track_ids: set = None
):
    height, width, _ = bg_img.shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    team_colors = {
        'eastern': (255, 0, 0),
        'easterngoalkeeper': (0, 255, 0),
        'kitchee': (255, 192, 203),
        'kitcheegoalkeeper': (0, 165, 255),
        'referee': (0, 255, 255),
        'ball': (0, 0, 0),
        'unsure': (128, 128, 128),
    }

    suspicious_track_ids = suspicious_track_ids or set()

    tracks = [json.loads(line) for line in open(jsonl_path)]

    # Determine valid frame range
    all_frames = [f for t in tracks for f in t["frames"]]
    min_frame = min(all_frames) if start_frame is None else start_frame
    max_frame = max(all_frames) if end_frame is None else end_frame

    # Index all objects by frame
    frame_to_objects = defaultdict(list)
    for t in tracks:
        tid = t["track_id"]
        team = t["team"]
        for i, f in enumerate(t["frames"]):
            if i >= len(t["projected"]):
                continue
            pt = t["projected"][i]
            if pt is None:
                continue
            frame_to_objects[f].append((pt, tid, team))

    # Render video
    for f in range(min_frame, max_frame + 1):
        frame_img = bg_img.copy()
        for pt, tid, team in frame_to_objects.get(f, []):
            x, y = int(pt[0]), field_size[1] - int(pt[1])
            color = team_colors.get(team, (128, 128, 128))
            text_color = (0, 0, 255) if tid in suspicious_track_ids else (0, 0, 0)
            cv2.circle(frame_img, (x, y), 5, color, -1)
            cv2.putText(frame_img, str(tid), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
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
    movement_threshold=20  # in meters (10 = 1m if 0.1m units)
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
            # print(f"🔄 Processing team: {team}, remaining windows: {len(windows)}")
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
    threshold,
    detector_kwargs=None
):
    # Load and resize background
    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    bg_img = cv2.resize(bg_img, field_size)

    filter_static_and_multiple_balls(
        json_path, 
        json_path.replace('.jsonl', '_filtered.jsonl'),
    )

    remove_ball_false_detection(
        json_path.replace('.jsonl', '_filtered.jsonl'),
        json_path.replace('.jsonl', '_removed.jsonl')
    )

    smoothen_ball_tracking(
        json_path.replace('.jsonl', '_removed.jsonl'), 
        json_path.replace('.jsonl', '_smoothed.jsonl'),
        field_size
    )

    convert_ball_tracking_format(
        json_path.replace('.jsonl', '_smoothed.jsonl'),
        json_path.replace('.jsonl', '_smoothed_temp.jsonl'),
    )

    process_jsonl_detect_replace(
        json_path.replace('.jsonl', '_smoothed_temp.jsonl'),
        json_path.replace('.jsonl', '_processed.jsonl'),
        detector_kwargs=detector_kwargs,
        overwrite_projected=True
    )

    refine_ball_tracking_with_ransac(
        json_path.replace('.jsonl', '_processed.jsonl'),
        json_path.replace('.jsonl', '_refined.jsonl'),
    )

    convert_ball_tracking_format(
        json_path.replace('.jsonl', '_refined.jsonl'),
        json_path.replace('.jsonl', '_final.jsonl'),
    )

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
    fps=29.97,
    detector_kwargs=None
):
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(json_path))[0]

    # output_path_video = f"{output_name}.mp4"

    start = time.time()
    # Shared logic
    bg_img = prepare_background_and_tracks(
        json_path, image_path, field_size,
        min_track_length, smoothing_window, polyorder, max_step,
        max_merge_gap, max_merge_overlap_frames, max_merge_distance,
        window_size, threshold, detector_kwargs
    )
    end = time.time()
    print(f"✅ Processed tracks in {end - start:.2f} seconds")

def convert_ball_tracking_format(json_path, output_path, max_frame_gap=10):
    """
    Convert input ball tracking JSONL with per-frame data to a consolidated track format.
    
    Args:
        json_path (str): Path to input JSONL file with per-frame ball detections
        output_path (str): Path to output JSONL file with consolidated track format
        
    The input format is:
    {
        "frame_id": 1,
        "projected": [[x, y], ...]
    }
    
    The output format is:
    {
        "track_id": 0,  # 0 for ball
        "team": "ball",
        "frame_range": [start_frame, end_frame],
        "frames": [frame_ids],
        "projected": [[x, y], ...]
    }
    """
    # Load and sort all frames
    frame_data = []
    
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    frame_data.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")
    
    # Extract frames and projected points
    frames = []
    projected = []
    
    for frame in frame_data:
        frame_id = frame.get('frame_id')
        proj_points = frame.get('projected', [])
        
        # Skip frames without data
        if not frame_id or not proj_points:
            continue
            
        # In case there are multiple ball detections in a frame, take the first one
        # You could implement more complex logic here if needed
        if proj_points and len(proj_points) > 0:
            frames.append(frame_id)
            projected.append(proj_points)

    # Generate output track
    if frames:
        output_track = {
            "track_id": 0,  # Fixed for ball
            "team": "ball",  # Fixed for ball
            "frames": frames,
            "projected": projected
        }
        
        # Write to output file
        with open(output_path, 'w') as out_f:
            out_f.write(json.dumps(output_track) + '\n')
        
        # print(f"✅ Converted {len(frames)} frames to ball track in {output_path}")
    else:
        print("⚠️ No valid frames found to convert")

def convert_ball_tracking_json_to_numpy(json_file):
    """
    Convert ball tracking data from JSONL format to a sorted NumPy array.
    
    Args:
        json_file (str): Path to the input JSONL file containing ball tracking data
        output_file (str): Path to save the output .npy file
    
    Returns:
        np.ndarray: The sorted array with format [[frame_id, x, y], ...]
    """
    # List to store the ball positions
    ball_positions = []
    
    # Read the JSONL file
    with open(json_file, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    if 'frame_id' in data: # Format in {frame_id, x, y}
                        frame_id = data.get('frame_id')
                        
                        # Extract the projected coordinates (x, y)
                        if 'projected' in data:
                            x, y = data['projected']
                            
                            # Add to our list as [frame_id, x, y]
                            ball_positions.append([frame_id, x, y])
                    elif 'frames' in data: # Format in {track_id, team, frame_range, frames, projected}
                        frames = data.get('frames', [])
                        projected = data.get('projected', [])
                        for f, p in zip(frames, projected):
                            if p is not None and len(p) == 2:
                                ball_positions.append([f, p[0], p[1]])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
                except Exception as e:
                    print(f"Error processing line: {e}")
    
    # Convert to NumPy array
    ball_positions_array = np.array(ball_positions)
    
    # Sort by frame_id
    sorted_indices = np.argsort(ball_positions_array[:, 0])
    sorted_ball_positions = ball_positions_array[sorted_indices]
    
    return sorted_ball_positions

def convert_ball_tracking_numpy_to_json(numpy_arr, output_file):
    """
    Convert ball tracking data from NumPy format back to JSONL format.

    Args:
        numpy_arr (np.ndarray): Array containing ball tracking data [[frame_id, x, y], ...]
        output_file (str): Path to save the output JSONL file

    Returns:
        None
    """

    # Write to JSONL file
    with open(output_file, 'w') as file:
        for frame_id, x, y in numpy_arr:
            json_line = json.dumps({
                "frame_id": int(frame_id),
                "projected": [float(x), float(y)]
            })
            file.write(json_line + "\n")

    # print(f"Converted ball positions from NumPy array to JSONL format")
    # print(f"Saved to {output_file}")

def filter_multiple_detections(ball_xy, max_speed=150, static_threshold=5, window_size=5, field_size=(1060, 660)):
    """
    Filter out false ball detections when multiple candidates exist in the same frame.
    
    Parameters:
    -----------
    ball_xy : np.ndarray
        Array with shape (n, 3) where each row is [frame_idx, x, y]
    max_speed : float
        Maximum plausible speed of the ball (pixels/frame)
    static_threshold : float
        Minimum movement required to not be considered static
    window_size : int
        Size of window to analyze for trajectory consistency
        
    Returns:
    --------
    np.ndarray
        Filtered ball_xy with only one detection per frame
    """
    # Group detections by frame
    frame_to_detections = defaultdict(list)
    for i in range(len(ball_xy)):
        frame_id = int(ball_xy[i, 0])
        frame_to_detections[frame_id].append((i, ball_xy[i, 1:]))
    
    # Initialize Kalman filter for trajectory prediction
    kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
    dt = 1.0  # Time step (1 frame)
    
    # State transition matrix (constant velocity model)
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Measurement matrix (we only observe position, not velocity)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # Initial state uncertainty
    kf.P *= 100
    
    # Measurement noise
    kf.R = np.diag([50, 50])
    
    # Process noise
    kf.Q = np.eye(4) * 10
    
    # Track frame-to-frame movement for each detection path
    movements = defaultdict(list)
    
    # Create a mask for points to keep
    keep_indices = []
    last_position = None
    
    # Process frames in order
    sorted_frames = sorted(frame_to_detections.keys())
    
    # Initialize with the first frame
    if sorted_frames:
        first_frame = sorted_frames[0]
        # If only one detection in first frame, use it
        if len(frame_to_detections[first_frame]) == 1:
            idx, pos = frame_to_detections[first_frame][0]
            keep_indices.append(idx)
            last_position = pos
            # Initialize Kalman filter
            kf.x = np.array([[pos[0]], [pos[1]], [0], [0]])
        # If multiple detections, use the one closest to the center
        else:
            center = np.array([field_size[0] / 2, field_size[1] / 2])  # Approximate field center based on size of field [1060, 660]
            dists = [np.linalg.norm(pos - center) for _, pos in frame_to_detections[first_frame]]
            best_idx = np.argmin(dists)
            idx, pos = frame_to_detections[first_frame][best_idx]
            keep_indices.append(idx)
            last_position = pos
            # Initialize Kalman filter
            kf.x = np.array([[pos[0]], [pos[1]], [0], [0]])
    
    # Process remaining frames
    for i in range(1, len(sorted_frames)):
        current_frame = sorted_frames[i]
        detections = frame_to_detections[current_frame]
        
        # If only one detection in this frame, easy case
        if len(detections) == 1:
            idx, pos = detections[0]
            
            # Check if this is a plausible continuation
            if last_position is not None:
                distance = np.linalg.norm(pos - last_position)
                if distance > max_speed:
                    # Skip implausibly fast movements
                    continue
            
            keep_indices.append(idx)
            last_position = pos
            
            # Update Kalman filter
            kf.predict()
            kf.update(pos)
            
        # If multiple detections, choose the most likely one
        else:
            # Predict next position using Kalman filter
            kf.predict()
            predicted_pos = kf.x[:2].flatten()
            
            # Score each detection based on multiple criteria
            scores = []
            for idx, pos in detections:
                # 1. Distance from predicted position
                pred_distance = np.linalg.norm(pos - predicted_pos)
                
                # 2. Check if this is a static detection point
                static_score = 0
                for prev_frame in range(max(0, current_frame-window_size), current_frame):
                    if prev_frame in frame_to_detections:
                        for _, prev_pos in frame_to_detections[prev_frame]:
                            if np.linalg.norm(pos - prev_pos) < static_threshold:
                                static_score += 1
                
                # 3. Distance from previous position
                if last_position is not None:
                    prev_distance = np.linalg.norm(pos - last_position)
                    # Heavily penalize positions too far from previous
                    if prev_distance > max_speed:
                        prev_distance_score = 1000  # Large penalty
                    else:
                        prev_distance_score = prev_distance
                else:
                    prev_distance_score = 0
                
                # Combined score (lower is better)
                score = pred_distance + (static_score * 50) + prev_distance_score
                scores.append((idx, score))
            
            # Choose detection with lowest score
            scores.sort(key=lambda x: x[1])
            best_idx, _ = scores[0]
            best_pos = ball_xy[best_idx, 1:]
            
            keep_indices.append(best_idx)
            last_position = best_pos
            
            # Update Kalman filter with chosen position
            kf.update(best_pos)
    
    # Return the filtered array
    return ball_xy[keep_indices]

def remove_detections_near_high_density_region(ball_xy, field_size=(1060, 660), 
                           bin_size=(20, 20), primary_threshold=8, 
                           secondary_threshold=4, neighbor_radius=2):
    """
    Density-based filter with multi-level thresholding and spatial awareness.
    Divide the football field into grids, allocate all detections to their respective grids,
    identify high-density grids, and remove points in and around these grids.

    Parameters:
    -----------
    ball_xy : np.ndarray
        Array with shape (n, 3) where each row is [frame_idx, x, y]
    field_size : list
        Size of the football field [width, height]
    bin_size : tuple
        Size of histogram bins (x, y)
    primary_threshold : int
        Primary threshold for high-density regions
    secondary_threshold : int
        Secondary threshold for neighboring regions
    neighbor_radius : int
        Radius for considering neighboring bins
        
    Returns:
    --------
    filtered_ball_xy : np.ndarray
        Filtered ball tracking data with high-density points removed
    """
    # Create a 2D histogram as before
    x_bins = np.linspace(0, field_size[0], int(field_size[0]/bin_size[0])+1)
    y_bins = np.linspace(0, field_size[1], int(field_size[1]/bin_size[1])+1)
    
    hist, x_edges, y_edges = np.histogram2d(
        ball_xy[:, 1], ball_xy[:, 2], 
        bins=[x_bins, y_bins]
    )
    
    # Find the bin for each data point
    x_indices = np.digitize(ball_xy[:, 1], x_edges) - 1
    y_indices = np.digitize(ball_xy[:, 2], y_edges) - 1
    
    # Bound indices to valid range
    x_indices = np.clip(x_indices, 0, len(x_edges)-2)
    y_indices = np.clip(y_indices, 0, len(y_edges)-2)
    
    # Create masks for primary and extended regions
    primary_mask = np.zeros(len(ball_xy), dtype=bool)
    extended_mask = np.zeros(len(ball_xy), dtype=bool)
    
    # Check density for each point and its neighborhood
    for i in range(len(ball_xy)):
        xi, yi = x_indices[i], y_indices[i]
        
        # Check if this point is in a high-density bin
        if hist[xi, yi] >= primary_threshold:
            primary_mask[i] = True
            continue
            
        # Check neighborhood for secondary threshold
        x_min, x_max = max(0, xi-neighbor_radius), min(hist.shape[0]-1, xi+neighbor_radius)
        y_min, y_max = max(0, yi-neighbor_radius), min(hist.shape[1]-1, yi+neighbor_radius)
        
        neighborhood = hist[x_min:x_max+1, y_min:y_max+1]
        if np.any(neighborhood >= primary_threshold) and hist[xi, yi] >= secondary_threshold:
            extended_mask[i] = True
    
    # Combine masks for complete filtering
    remove_mask = primary_mask | extended_mask
    
    # # Visualize before and after
    # plt.figure(figsize=(12, 6))
    
    # plt.subplot(1, 2, 1)
    # plt.scatter(ball_xy[:, 1], ball_xy[:, 2], c=ball_xy[:, 0], s=2, cmap='viridis')
    # plt.scatter(ball_xy[remove_mask, 1], ball_xy[remove_mask, 2], 
    #            c='red', s=10, alpha=0.5, marker='x')
    # plt.title(f'Original with Removed Points ({np.sum(remove_mask)} points)')
    # plt.axis('equal')
    
    # plt.subplot(1, 2, 2)
    # keep_mask = ~remove_mask
    # plt.scatter(ball_xy[keep_mask, 1], ball_xy[keep_mask, 2], 
    #            c=ball_xy[keep_mask, 0], s=2, cmap='viridis')
    # plt.title(f'Filtered ({np.sum(keep_mask)} points)')
    # plt.axis('equal')
    
    # plt.tight_layout()
    # plt.show()
    
    return ball_xy[~remove_mask]

def filter_static_and_multiple_balls(
    json_path,
    save_path,
    field_size=(1060, 660), # Standard football field size in 0.1m units
    bin_size=(5, 5),        # smaller bins for finer resolution
    primary_threshold=50, 
    secondary_threshold=10,
    neighbor_radius=10,
    max_speed=120,          # Adjust based on max ball speed in your videos
    static_threshold=5,     # Minimum movement to not be considered static
    window_size=7):         # Look back this many frames to check for static patterns

    ball_xy = convert_ball_tracking_json_to_numpy(json_path)

    # sort ball_xy
    ball_xy = ball_xy[np.argsort(ball_xy[:, 0])]

    # Filter out elements with x < 0 or x > 1060
    ball_xy = ball_xy[(ball_xy[:, 1] >= 0) & (ball_xy[:, 1] <= field_size[0])]
    # Filter out elements with y < 0 or y > 660
    ball_xy = ball_xy[(ball_xy[:, 2] >= 0) & (ball_xy[:, 2] <= field_size[1])]

    filtered_ball_xy = remove_detections_near_high_density_region(
        ball_xy,
        field_size=field_size,
        bin_size=bin_size,
        primary_threshold=primary_threshold,
        secondary_threshold=secondary_threshold,
        neighbor_radius=neighbor_radius,
    )

    filtered_ball_xy = filter_multiple_detections(
        filtered_ball_xy,
        max_speed=max_speed,
        static_threshold=static_threshold,
        window_size=window_size
    )

    convert_ball_tracking_numpy_to_json(filtered_ball_xy, save_path)


def remove_ball_false_detection(json_path, save_path, field_size=[1060, 660]):
    ball_xy = convert_ball_tracking_json_to_numpy(json_path)

    fs = 30 # frame rate
    Xraw = ball_xy.copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(Xraw)
    X[:,0] *= 6 # weight of temporal dimension, tune this if needed
    tree = KDTree(X)
    dist, ind = tree.query(X, k=max(fs//3, 10))
    thresh = 2*np.quantile(dist[:,-1], 0.5)

    knn_graph = kneighbors_graph(X[:,0].reshape(-1,1), 5, include_self=False)
    clustering = AgglomerativeClustering(n_clusters=None, linkage='single', 
                                        connectivity=knn_graph, distance_threshold=thresh)
    clustering.fit(X)

    labs, n_in_labs = np.unique(clustering.labels_, return_counts=True)
    outs = np.array([])
    for lab, n_in in zip(labs, n_in_labs):
        if n_in < 3:
            outs = np.append(outs, np.where(clustering.labels_== lab)[0])
    outs = np.sort(outs).astype(np.int16)

    # plt.scatter(ball_xy[:,1], ball_xy[:,2], s=0.5)
    # plt.plot(ball_xy[outs,1], ball_xy[outs,2],'r.')
    # plt.legend(['detections','outliers'])
    # plt.axis('equal')
    # plt.axis('off')
    # plt.show()

    ball_xy_val = np.delete(ball_xy, outs, axis=0)

    convert_ball_tracking_numpy_to_json(ball_xy_val, save_path)
    # print(f"Saved filtered ball tracking data to {save_path}")


def remove_static_clusters(ball_xy, time_window=30, spatial_threshold=10, min_points=5, max_displacement=20):
    """
    Remove static clusters from ball tracking data.
    
    Parameters:
    -----------
    filtered_ball_xy : np.ndarray
        Array with shape (n, 3) where each row is [frame_idx, x, y]
    time_window : int
        Time window (in frames) to consider for static cluster detection
    spatial_threshold : float
        Maximum radius for points to be considered in the same spatial cluster
    min_points : int
        Minimum number of points required to form a static cluster
    max_displacement : float
        Maximum allowed displacement within a cluster to be considered "static"
    
    Returns:
    --------
    np.ndarray
        Filtered filtered_ball_xy array with static clusters removed
    """
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    # Create a copy to avoid modifying the original
    filtered_ball_xy = ball_xy.copy()
    
    # Create a mask for points to keep
    keep_mask = np.ones(len(filtered_ball_xy), dtype=bool)
    
    # Get unique frame numbers
    frame_nums = np.unique(filtered_ball_xy[:, 0])
    
    # Process each time window
    for start_frame in range(0, int(frame_nums[-1]), time_window//2):  # Overlapping windows
        end_frame = start_frame + time_window
        
        # Get points in this time window
        window_mask = (filtered_ball_xy[:, 0] >= start_frame) & (filtered_ball_xy[:, 0] < end_frame)
        window_points = filtered_ball_xy[window_mask]
        
        if len(window_points) < min_points:
            continue
            
        # Cluster based on spatial coordinates only
        spatial_clustering = DBSCAN(eps=spatial_threshold, min_samples=min_points).fit(window_points[:, 1:3])
        
        # For each spatial cluster, check if it's static
        for cluster_id in np.unique(spatial_clustering.labels_):
            if cluster_id == -1:  # Skip noise points
                continue
                
            # Get points in this spatial cluster
            cluster_mask = spatial_clustering.labels_ == cluster_id
            cluster_points = window_points[cluster_mask]
            
            # Check if this cluster is static (small displacement over time)
            if len(cluster_points) >= min_points:
                # Calculate the maximum displacement within the cluster
                x_range = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
                y_range = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
                displacement = np.sqrt(x_range**2 + y_range**2)
                
                # Calculate frame span
                frame_span = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
                
                # If displacement is small and cluster spans significant time, it's likely static
                if displacement < max_displacement and frame_span > min_points:
                    # Find these points in the original array
                    for point in cluster_points:
                        # Find matching points in the original array
                        matches = np.where(
                            (filtered_ball_xy[:, 0] == point[0]) & 
                            (filtered_ball_xy[:, 1] == point[1]) & 
                            (filtered_ball_xy[:, 2] == point[2])
                        )[0]
                        keep_mask[matches] = False
    
    # Return the filtered data
    return filtered_ball_xy[keep_mask]


def smoothen_ball_tracking(json_path, save_path, field_size=[1060, 660]):
    ball_xy = convert_ball_tracking_json_to_numpy(json_path)

    first_frame =  ball_xy[ball_xy[:, 0] == np.min(ball_xy[:, 0])]
    filtered_ball_xy = remove_static_clusters(
        ball_xy, 
        time_window=30,  # Adjust based on your frame rate
        spatial_threshold=15,  # Maximum distance for points to be in same cluster
        min_points=5,  # Minimum points to consider a cluster
        max_displacement=10  # Maximum allowed movement within a static cluster
    )
    # Append the first frame to the filtered data if the first frame is not already included
    if not np.any(np.all(filtered_ball_xy == first_frame, axis=1)):
        filtered_ball_xy = np.vstack([first_frame, filtered_ball_xy])
    # Get the actual frame range from the filtered data
    min_frame = int(filtered_ball_xy[:, 0].min())
    max_frame = int(filtered_ball_xy[:, 0].max())

    f = interpolate.interp1d(filtered_ball_xy[:,0], filtered_ball_xy[:,1:], axis=0, fill_value='extrapolate')
    frames = np.arange(min_frame, max_frame + 1)
    ball_int = f(frames)

    # Kalman Filter setup
    dim_x = 4
    A = np.zeros((dim_x,dim_x))
    for i in range(dim_x-1): A[i, i+1] = 1
    dt = 1

    f = KalmanFilter(dim_x=dim_x, dim_z=1)
    f.F = expm(A * dt)

    f.H = np.zeros((1,dim_x))
    f.H[0,0] = 1
    f.P *= 10. 
    f.R = 100**2 

    varQ = 10
    f.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=varQ**2)

    zs = ball_int[:,1].copy()
    f.x = np.zeros((dim_x,1))
    f.x[0,0] = zs[0]
    mu, cov, _, _ = f.batch_filter(zs)
    xs, Ps, Ks,_ = f.rts_smoother(mu, cov)

    traj = xs.squeeze()[:,0]
    deltas = traj - zs

    ball_int_save = ball_int.copy()
    prev_thresholds = np.array([np.inf, np.inf])  # Initialize with large values
    threshold_change_tolerance = 0.05  # 5% change tolerance
    max_iterations = 10  # Safety cap on iterations

    for i_loop in range(max_iterations):
        for i in range(2):
            zs = ball_int[:,i].copy()
            f.x = np.zeros((dim_x,1))
            f.x[0,0] = zs[0]
            mu, cov, _, _ = f.batch_filter(zs)
            xs, Ps, Ks,_ = f.rts_smoother(mu, cov)
            if i==0:
                traj = xs.squeeze()[:,0].copy()
                d_ball = xs.squeeze()[:,1].copy()
                d2_ball = xs.squeeze()[:,2].copy()
            else:
                traj = np.vstack((traj, xs.squeeze()[:,0])).T
                d_ball = np.vstack((d_ball, xs.squeeze()[:,1])).T
                d2_ball = np.vstack((d2_ball, xs.squeeze()[:,2])).T
                
        deltas = np.abs(traj - ball_int)
        thresholds = 5 * np.median(deltas, axis=0)
        
        # Calculate relative change in thresholds
        if i_loop > 0:
            rel_change = np.abs(thresholds - prev_thresholds) / prev_thresholds
            max_change = np.max(rel_change)
            print(f"Iteration {i_loop}: thresholds = {thresholds}, relative change = {max_change:.4f}")
            
            # Check termination condition: change is less than tolerance
            if max_change < threshold_change_tolerance:
                print(f"Converged after {i_loop+1} iterations (threshold change < {threshold_change_tolerance})")
                break
        else:
            print(f"Iteration {i_loop}: initial thresholds = {thresholds}")
        
        # Store current thresholds for next iteration comparison
        prev_thresholds = thresholds.copy()
        
        # Update ball_int with filtered values where outliers are detected
        for i in range(2):
            ball_int[:,i] = np.where(deltas[:,i] < thresholds[i], ball_int[:,i], traj[:,i])

    # If we exited due to max iterations
    if i_loop == max_iterations - 1:
        print(f"Reached maximum iterations ({max_iterations}) without convergence")
    
    ball_xy_smooth = np.column_stack((frames, traj))
    convert_ball_tracking_numpy_to_json(ball_xy_smooth, save_path)


def refine_ball_tracking_with_ransac(json_path, save_path):
    ball_xy = convert_ball_tracking_json_to_numpy(json_path)
    traj = ball_xy[:,1:]
    frames = ball_xy[:,0].astype(np.int16)

    all_ins=[]

    traj_reg = traj.copy()
    reg_frames = frames.copy()
    ins = []
    long_seg = True
    i = 0
    while long_seg and len(traj_reg) > 2:  # Add check for minimum points

        # Check if enough points remain
        if len(traj_reg) < 3:
            break
            
        reg1 = RANSACRegressor(random_state=i, residual_threshold = 10 )
        reg1.fit(traj_reg[:,0].reshape(-1,1), traj_reg[:,1])
        frames_in_reg_1 = reg_frames[reg1.inlier_mask_]
        
        # Check if any inliers were found
        if len(frames_in_reg_1) == 0:
            break

        clustering1 = AgglomerativeClustering(n_clusters=None, linkage='single', 
                                            distance_threshold=5)
        clustering1.fit(frames_in_reg_1.reshape(-1,1))
        labs1, n_in_labs1 = np.unique(clustering1.labels_, return_counts=True)
        
        reg2 = RANSACRegressor(random_state=i, residual_threshold = 10 )
        reg2.fit(traj_reg[:,1].reshape(-1,1), traj_reg[:,0])
        frames_in_reg_2 = reg_frames[reg2.inlier_mask_]
        
        # Check if any inliers were found
        if len(frames_in_reg_2) == 0:
            break

        clustering2 = AgglomerativeClustering(n_clusters=None, linkage='single', 
                                            distance_threshold=5)
        clustering2.fit(frames_in_reg_2.reshape(-1,1))
        labs2, n_in_labs2 = np.unique(clustering2.labels_, return_counts=True)
        
        n_max = max(n_in_labs1.max() if len(n_in_labs1) > 0 else 0, 
                    n_in_labs2.max() if len(n_in_labs2) > 0 else 0)
        long_seg = n_max > 25
        
        if long_seg:
            
            if (len(n_in_labs1) > 0 and len(n_in_labs2) > 0 and 
                n_in_labs1.max() > n_in_labs2.max()) or len(n_in_labs2) == 0:
                new_ins = frames_in_reg_1[clustering1.labels_== np.argmax(n_in_labs1)].astype(np.int16)
            else:
                new_ins = frames_in_reg_2[clustering2.labels_== np.argmax(n_in_labs2)].astype(np.int16)
        
            ins.append(new_ins)
            indexes = np.searchsorted(reg_frames, new_ins)
            reg_frames = np.delete(reg_frames, indexes)
            traj_reg = np.delete(traj_reg, indexes, axis=0)
            i += 1
        else:
            break

    # Make sure we have at least one segment before trying to stack
    if len(ins) > 0:
        all_ins = np.hstack(ins)
    else:
        all_ins = np.array([])

    # fs = 30
    # acc = np.linalg.norm(d2_ball, axis=1)
    # peaks, properties = find_peaks(acc, distance= int(0.8*fs), prominence=10)

    debs = np.sort(np.array([seg[0] for seg in ins]))
    ends = np.sort(np.array([seg[-1] for seg in ins]))

    # peaks_in = []
    # for end, deb in zip(ends[:-1], debs[1:]):
    #     in_between = np.logical_and(peaks >= end + 8, peaks <= deb - 8)
    #     peaks_in.append(peaks[in_between])
    # new_pts = np.hstack(peaks_in)
    new_pts = []

    vertices = np.hstack((debs, ends, new_pts, [1, frames[-1]]))
    vertices = np.sort(np.unique(vertices)).astype(np.int16)

    f = interpolate.interp1d(vertices, traj[vertices-1], axis=0, fill_value='extrapolate')
    ball_radar = f(frames)
    ball_xy_final = np.column_stack((frames, traj))
    convert_ball_tracking_numpy_to_json(ball_xy_final, save_path)




def parse_args():
    parser = argparse.ArgumentParser(description="Process merged tracks from tracking JSONL")
    parser.add_argument('--json-path', type=str, required=True, help='Path to the merged tracking JSONL file')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the field image')
    parser.add_argument('--field-size', type=int, nargs=2, default=[1060, 660], help='Field size (length, width) as 0.1m')
    parser.add_argument('--min-track-length', type=int, default=10, help='Minimum track length to keep')
    parser.add_argument('--smoothing-window', type=int, default=90, help='Savitzky-Golay filter window size')
    parser.add_argument('--polyorder', type=int, default=7, help='Polynomial order for smoothing')
    parser.add_argument('--max-step', type=int, default=20, help='Max distance (in pixels) allowed per frame')
    parser.add_argument('--max-merge-gap', type=int, default=20, help='Max allowed gap (frames) between mergeable tracks')
    parser.add_argument('--max-merge-overlap-frames', type=int, default=15, help='Max allowed overlap for merging')
    parser.add_argument('--max-merge-distance', type=int, default=50, help='Max spatial distance for merging')
    parser.add_argument('--window-size', type=int, default=20, help='Window size for velocity consistency check')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for velocity consistency')
    parser.add_argument('--output-name', type=str, required=True, help='Base name of the output file (without extension)')
    return parser.parse_args()

def main():
    args = parse_args()
    start = time.time()

    BALL_DETECTOR = dict(
        window_size=501,
        step=250,
        prominence=7,
        min_wave_len=10,
        max_wave_len=60,
        speed_std_factor=0.7,
        smooth_window=7,
        savgol_poly=2,
        min_steepness=0.2,
        min_quad_curv=0.7,
        min_monotonic_ratio=0.7,
        max_gap_size=5
    )

    process_merged_tracks(
        json_path=args.json_path,
        image_path=args.image_path,
        field_size=tuple(args.field_size),
        min_track_length=args.min_track_length,
        smoothing_window=args.smoothing_window,
        polyorder=args.polyorder,
        max_step=args.max_step,
        max_merge_gap=args.max_merge_gap,
        max_merge_overlap_frames=args.max_merge_overlap_frames,
        max_merge_distance=args.max_merge_distance,
        window_size=args.window_size,
        threshold=args.threshold,
        output_name=args.output_name,
        detector_kwargs=BALL_DETECTOR
    )

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()

# example usage:
# python3 post-processing-ball.py --json-path "./runs/detect/test_4k-2h-crop/team_tracking.jsonl" --image-path "./data/images/mongkok_football_field.png" --output-name './runs/detect/test_4k-2h-crop/team_tracking_output'