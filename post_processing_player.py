import argparse
from dataclasses import dataclass, field
from itertools import compress
import json
import numpy as np
from collections import defaultdict
from scipy.signal import savgol_filter

import time
from typing import List, Dict, Any, Tuple, Iterator, Optional

from tools.remove_track_sharp import process_jsonl_detect_replace

@dataclass
class RawTrack:
    """
    Raw track data directly from object detection/tracking inference.
    Frame-level predictions before any aggregation.
    """
    track_id: str
    frames: List[int]
    projected: List[List[float]]  # [[x, y], ...]
    bbox: List[List[float]]  # [[x1, y1, x2, y2, conf], ...]
    team_conf: List[Dict[str, float]]  # [{"home": 0.8, "away": 0.2, ...}, ...] per frame
    jersey_num: List[int]  # [1, 10, 10, -1, ...] per frame (-1 = no detection)
    jersey_conf: List[List[float]]  # [[0.9], [0.95, 0.99], ...] per frame
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawTrack":
        """Load from JSONL dict."""
        return cls(
            track_id=data["track_id"],
            frames=data["frame_id"], # idk why it's named frame_id in jsonl
            projected=data["projected"],
            bbox=data.get("bbox", []),
            team_conf=data.get("team_conf", []),
            jersey_num=data.get("jersey_num", []),
            jersey_conf=data.get("jersey_conf", []),
        )


@dataclass
class TrackSegment:
    """
    Track segment after splitting by team changes.
    Team is now AGGREGATED (single value), but still frame-level jersey data.
    """
    track_id: str  # e.g., "123a", "123b"
    team: str  # "home" or "away" (voted)
    team_conf: float  # Aggregated confidence for this segment
    frames: List[int]
    points: List[List[float]]  # [[x, y], ...]
    bbox_area: List[float]  # Derived from bbox
    jersey_num: List[int]  # Still frame-level
    jersey_conf: List[List[float]]  # Still frame-level
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "team": self.team,
            "team_conf": self.team_conf,
            "frames": self.frames,
            "points": self.points,
            "bbox_area": self.bbox_area,
            "jersey_num": self.jersey_num,
            "jersey_conf": self.jersey_conf,
        }

def calculate_bbox_area(bboxes: List[List[float]]) -> List[float]:
    """
    Calculate area of multiple bounding boxes (vectorized version).

    Args:
        bboxes: List of bounding boxes, each as [x1, y1, x2, y2]

    Returns:
        List of areas corresponding to each bounding box
    """
    if not bboxes:
        return []

    bboxes_array = np.array(bboxes)  # Shape: (N, 4)
    widths = np.maximum(0.0, bboxes_array[:, 2] - bboxes_array[:, 0])
    heights = np.maximum(0.0, bboxes_array[:, 3] - bboxes_array[:, 1])
    areas = (widths * heights).tolist()
    return areas


def assign_team_by_majority_vote(team_conf_list: List[Dict[str, float]]) -> str:
    """
    Assign team label based on majority vote from team confidence list.

    Args:
        team_conf_list: List of dictionaries with team confidence scores.
    
    Returns:
        Assigned team label.
    """

    team_count = defaultdict(float)
    for conf in team_conf_list:
        for k, v in conf.items():
            team_count[k] += v
    return max(team_count, key=team_count.get) if team_count else "ball"


def index_to_letter_suffix(idx: int) -> str:
    """Return 'a', 'b', ..., 'z', 'aa', 'ab', ... as suffix."""
    letters = []
    while True:
        letters.append(chr(97 + (idx % 26)))
        idx = idx // 26
        if idx == 0:
            break
        idx -= 1  # offset for 0-based index
    return "".join(reversed(letters))


def split_track_by_sliding_window(
    obj: Dict[str, Any], window_size: int = 20, threshold: float = 0.8
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
    jersey_nums = obj.get("jersey_num", [])
    jersey_confs = obj.get("jersey_conf", [])

    # Get dominant team label for each frame
    dominant_team_list = [max(conf, key=conf.get) if conf else "ball" for conf in team_conf_list]

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
            window = dominant_team_list[i : i + window_size]
            counter = defaultdict(int)
            for t in window:
                counter[t] += 1
            dominant_in_window = max(counter, key=counter.get)
            ratio = counter[dominant_in_window] / window_size

            if dominant_in_window != current_team and ratio >= threshold:
                segment_conf_list = [team_conf_list[j] for j in buffer]
                if segment_conf_list:
                    team_score = sum(
                        conf.get(current_team, 0.0) for conf in segment_conf_list
                    ) / len(segment_conf_list)
                else:
                    team_score = 0.0
                # Commit segment
                segment = {
                    "track_id": f"{obj['track_id']}{chr(97 + len(segments))}",
                    "frame_id": [frame_ids[j] for j in buffer],
                    "projected": [projected[j] for j in buffer],
                    "bbox": [bboxes[j] for j in buffer] if bboxes else [],
                    "team_conf": team_score,
                    "team": current_team,
                    "jersey_num": ([jersey_nums[j] for j in buffer] if jersey_nums else []),
                    "jersey_conf": ([jersey_confs[j] for j in buffer] if jersey_confs else []),
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
        team_score = sum(conf.get(current_team, 0.0) for conf in segment_conf_list) / len(
            segment_conf_list
        )
        segment = {
            "track_id": f"{obj['track_id']}{chr(97 + len(segments))}",
            "frame_id": [frame_ids[j] for j in buffer],
            "projected": [projected[j] for j in buffer],
            "bbox": [bboxes[j] for j in buffer] if bboxes else [],
            "team_conf": team_score,
            "team": current_team,
            "jersey_num": [jersey_nums[j] for j in buffer] if jersey_nums else [],
            "jersey_conf": [jersey_confs[j] for j in buffer] if jersey_confs else [],
        }
        segments.append(segment)

    return segments


def interpolate_full_track(
    frames: List[int], points: np.ndarray, areas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate full track to fill in all missing frames using linear interpolation.

    Args:
        frames (List[int]): List of frame indices.
        points (np.ndarray): Corresponding points (N, 2) for each frame.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated frames and points (in same order).
    """
    if len(frames) < 2:
        return np.array(frames), points, areas  # Not enough points to interpolate

    all_frames = np.arange(frames[0], frames[-1] + 1)
    xs_interp = np.interp(all_frames, frames, points[:, 0])
    ys_interp = np.interp(all_frames, frames, points[:, 1])
    full_points = np.stack([xs_interp, ys_interp], axis=1)
    if areas is None or len(areas) == 0:
        areas_interp = np.zeros_like(all_frames, dtype=float)
    else:
        areas_interp = np.interp(all_frames, frames, areas)

    return all_frames, full_points, areas_interp


def hybrid_merge_stream_fixed(
    jsonl_path: str,
    output_path: str,
    max_merge_gap: int = 5,
    max_merge_overlap_frames: int = 3,
    max_merge_distance: float = 10,
    smoothing_window: int = 11,
    polyorder: int = 3,
):

    final_output = open(output_path, "w")
    frame_to_tracks = defaultdict(list)
    active_tracks = {}
    done_tracks = set()

    # Load all segments from JSONL (inlined stream_jsonl_segments)
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            seg = json.loads(line)
            start_frame = seg["frames"][0]
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
            tid = seg["track_id"]
            if tid in done_tracks or tid in merged_this_round or tid in active_tracks:
                continue

            best_match = None
            best_dist = float("inf")
            for mtid, m in active_tracks.items():
                if seg["team"] != m["team"]:
                    continue
                last_frame = m["frames"][-1]
                gap = seg["frames"][0] - last_frame
                if not ((0 <= gap <= max_merge_gap) or (0 < -gap <= max_merge_overlap_frames)):
                    continue
                dist = np.linalg.norm(np.array(m["points"][-1]) - np.array(seg["points"][0]))
                if dist <= max_merge_distance and dist < best_dist:
                    best_match = mtid
                    best_dist = dist

            if best_match:
                m = active_tracks[best_match]

                combined_frames = m["frames"] + seg["frames"]
                combined_points = m["points"] + seg["points"]
                combined_bbox_areas = m.get("bbox_area", []) + seg.get("bbox_area", [])

                # Create frame->point mapping and remove duplicates
                frame_point_map = {}
                frame_area_map = {}
                for frame, point, area in zip(
                    combined_frames, combined_points, combined_bbox_areas
                ):
                    if frame not in frame_point_map:
                        frame_point_map[frame] = point
                        frame_area_map[frame] = area
                    # If duplicate frame, keep the one with higher confidence or average them
                    # For simplicity, we keep the first occurrence

                # Sort by frame and update
                sorted_frames = sorted(frame_point_map.keys())
                m["frames"] = sorted_frames
                m["points"] = [frame_point_map[f] for f in sorted_frames]
                m["bbox_area"] = [frame_area_map[f] for f in sorted_frames]

                team = m.get("team", "")
                m_jersey = m.get("jersey_num", "unsure")
                seg_jersey = seg.get("jersey_num", "unsure")
                m_conf = m.get("jersey_conf", 0.0)
                seg_conf = seg.get("jersey_conf", 0.0)
                m_count = m.get("count", 0)
                seg_count = seg.get("count", 0)

                # print(f"Merging jersey_num for track {tid}: {m_jersey} + {seg_jersey}")

                # CASE 1: Referee team - always keep as "NA"
                if team == "referee":
                    m["jersey_num"] = "NA"
                    m["jersey_conf"] = 0.0
                    m["count"] = 0

                # CASE 2: Either side is "unsure" - use the other side
                elif m_jersey == "unsure" and seg_jersey != "unsure":
                    m["jersey_num"] = seg_jersey
                    m["jersey_conf"] = seg_conf
                    m["count"] = seg_count
                    # print(f"  → Using seg jersey: {seg_jersey}")

                elif seg_jersey == "unsure" and m_jersey != "unsure":
                    # Keep m's values (no change needed)
                    # print(f"  → Keeping m jersey: {m_jersey}")
                    pass

                # CASE 3: Both have data - merge intelligently
                elif m_jersey != "unsure" and seg_jersey != "unsure":
                    # Normalize both to lists
                    m_nums = m_jersey if isinstance(m_jersey, list) else [m_jersey]
                    seg_nums = seg_jersey if isinstance(seg_jersey, list) else [seg_jersey]
                    
                    m_confs = m_conf if isinstance(m_conf, list) else [m_conf]
                    seg_confs = seg_conf if isinstance(seg_conf, list) else [seg_conf]

                    m_counts = m_count if isinstance(m_count, list) else [m_count]
                    seg_counts = seg_count if isinstance(seg_count, list) else [seg_count]
                    # Create mapping: jersey_num -> {"confs": [...], "counts": [...]}
                    jersey_data_map = defaultdict(lambda: {"confs": [], "counts": []})
                    
                    for num, conf, count in zip(m_nums, m_confs, m_counts):
                        jersey_data_map[num]["confs"].append(conf)
                        jersey_data_map[num]["counts"].append(count)
                    
                    for num, conf, count in zip(seg_nums, seg_confs, seg_counts):
                        jersey_data_map[num]["confs"].append(conf)
                        jersey_data_map[num]["counts"].append(count)

                    # Average confidences and sum counts for each jersey number
                    merged_jerseys = []
                    merged_confs = []
                    merged_counts = []
                    
                    for jersey_num in sorted(jersey_data_map.keys()):
                        conf_list = jersey_data_map[jersey_num]["confs"]
                        count_list = jersey_data_map[jersey_num]["counts"]
                        
                        avg_conf = sum(conf_list) / len(conf_list)
                        total_count = sum(count_list)
                        
                        merged_jerseys.append(jersey_num)
                        merged_confs.append(avg_conf)
                        merged_counts.append(total_count)

                    # Sort by count (descending), then by confidence (descending)
                    sorted_triplets = sorted(
                        zip(merged_jerseys, merged_confs, merged_counts), 
                        key=lambda x: (x[2], x[1]),  # Sort by count first, then confidence
                        reverse=True
                    )
                    
                    m["jersey_num"] = [num for num, _, _ in sorted_triplets]
                    m["jersey_conf"] = [conf for _, conf, _ in sorted_triplets]
                    m["count"] = [count for _, _, count in sorted_triplets]
                    
                    # print(f"  → Merged result: {m['jersey_num']} with confs {m['jersey_conf']}")

                # CASE 4: Both are "unsure"
                else:
                    m["jersey_num"] = "unsure"
                    m["jersey_conf"] = 0.0
                    m["count"] = 0
                    # print(f"  → Both unsure, keeping unsure")

                # Update team confidence
                m["team_conf_total"] += seg.get("team_conf", 0.0) * len(seg["frames"])
                m["team_conf_len"] += len(seg["frames"])
                merged_this_round.add(tid)
                done_tracks.add(tid)
            else:
                active_tracks[tid] = {
                    "track_id": tid,
                    "team": seg["team"],
                    "frames": seg["frames"],
                    "points": seg["points"],
                    "jersey_num": seg.get("jersey_num", []),
                    "jersey_conf": seg.get("jersey_conf", []),
                    "count": seg.get("count", 0),
                    "bbox_area": seg.get("bbox_area", []),
                    "team_conf_total": seg.get("team_conf", 0.0) * len(seg["frames"]),
                    "team_conf_len": len(seg["frames"]),
                }
                merged_this_round.add(tid)

        # Finalize stale tracks
        to_remove = []
        for tid, m in active_tracks.items():
        #     print(f"Checking finalization for track {tid} at frame {current_frame}: last frame = {m['frames'][-1]}")
        #     print(f"  → Frames: {len(m['frames'])}")
        #     print(f"  → Points: {len(m['points'])}")
        #     print(f"  → BBox Areas: {len(m.get('bbox_area', []))}")
            if m["frames"][-1] < current_frame - max_merge_gap:
                frames = m["frames"]
                points = m["points"]
                areas = m.get("bbox_area", [])

                frames, points, areas = interpolate_full_track(
                    m["frames"], np.array(m["points"]), np.array(m.get("bbox_area", []))
                )

                if len(points) >= smoothing_window:
                    xs = savgol_filter(points[:, 0], smoothing_window, polyorder)
                    ys = savgol_filter(points[:, 1], smoothing_window, polyorder)
                    points = np.stack([xs, ys], axis=1)
                team_conf = m["team_conf_total"] / m["team_conf_len"] if m["team_conf_len"] else 0.0
                output = {
                    "track_id": m["track_id"],
                    "team": m["team"],
                    "jersey_num": m["jersey_num"],
                    "jersey_conf": m["jersey_conf"],
                    "count": m.get("count", 0),
                    "frame_range": [int(frames[0]), int(frames[-1])],
                    "frames": frames.tolist(),
                    "projected": points.tolist(),
                    "bbox_area": areas.tolist(),
                    "team_conf": team_conf,
                }
                final_output.write(json.dumps(output) + "\n")
                to_remove.append(tid)
                done_tracks.add(tid)

        for tid in to_remove:
            del active_tracks[tid]

        current_frame += 1

    # Final flush
    for tid, m in active_tracks.items():
        frames = m["frames"]
        points = m["points"]
        areas = m.get("bbox_area", [])

        frames, points, areas = interpolate_full_track(
            m["frames"], np.array(m["points"]), np.array(m.get("bbox_area", []))
        )
        if len(points) >= smoothing_window:
            xs = savgol_filter(points[:, 0], smoothing_window, polyorder)
            ys = savgol_filter(points[:, 1], smoothing_window, polyorder)
            points = np.stack([xs, ys], axis=1)
        team_conf = m["team_conf_total"] / m["team_conf_len"] if m["team_conf_len"] else 0.0
        output = {
            "track_id": m["track_id"],
            "team": m["team"],
            "jersey_num": m["jersey_num"],
            "jersey_conf": m["jersey_conf"],
            "count": m.get("count", 0),
            "frame_range": [int(frames[0]), int(frames[-1])],
            "frames": frames.tolist(),
            "projected": points.tolist(),
            "bbox_area": areas.tolist(),
            "team_conf": team_conf,
        }
        final_output.write(json.dumps(output) + "\n")

    final_output.close()
    print(f"✅ Merged and saved to: {output_path}")


def detect_endpoint_anomalies(positions: np.ndarray, frames: np.ndarray, 
                              window_size: int = 15,
                              speed_threshold_factor: float = 3.0,
                              acceleration_threshold_factor: float = 3.0,
                              direction_change_threshold: float = 60,
                              check_start: bool = True,
                              check_end: bool = True):
    """
    Detect anomalous movements at the start or end of a track.

    Args:
        positions (np.ndarray): Array of shape (N, 2) with x, y projected positions.
        frames (np.ndarray): Array of shape (N,) with frame indices.
        window_size (int): Number of frames to consider at start/end.
        speed_threshold_factor (float): Multiplier for speed threshold.
        acceleration_threshold_factor (float): Multiplier for acceleration threshold.
        direction_change_threshold (float): Angle in degrees for direction change.
        check_start (bool): Whether to check the start of the track.
        check_end (bool): Whether to check the end of the track.

    Returns:
        Tuple[int, int]: Indices to trim the start and end of the track.
    """
    n = len(positions)
    
    if n < window_size * 2:
        return 0, n - 1
    
    # Compute velocities and speeds
    dt = np.diff(frames)
    dt[dt == 0] = 1
    
    velocities = np.diff(positions, axis=0) / dt[:, np.newaxis]
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Check for valid speeds    
    if len(speeds) == 0:
        return 0, n - 1
    
    # Compute accelerations
    accelerations = np.diff(velocities, axis=0) / dt[:-1, np.newaxis]
    acc_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    # Compute global statistics (excluding endpoints)
    mid_start = window_size
    mid_end = n - window_size
    
    if mid_end <= mid_start:
        mid_speeds = speeds
        mid_accs = acc_magnitudes
    else:
        mid_speeds = speeds[mid_start:mid_end]
        mid_accs = acc_magnitudes[mid_start:mid_end-1]

    # Check for empty mid_speeds and mid_accs before computing statistics
    if len(mid_speeds) == 0:
        # Use full speeds if mid section is empty
        mid_speeds = speeds
    if len(mid_accs) == 0:
        # Use full accelerations if mid section is empty
        mid_accs = acc_magnitudes
    
    # Final check - if still empty, return no trimming
    if len(mid_speeds) == 0:
        return 0, n - 1
    
    mean_speed = np.mean(mid_speeds)
    std_speed = np.std(mid_speeds)

    # Handle case where std_speed is 0 or NaN
    if np.isnan(std_speed) or std_speed == 0:
        std_speed = 1.0  # Default value to avoid division issues
    
    if len(mid_accs) > 0:
        mean_acc = np.mean(mid_accs)
        std_acc = np.std(mid_accs)
        if np.isnan(std_acc) or std_acc == 0:
            std_acc = 1.0
    else:
        mean_acc = 0.0
        std_acc = 1.0
    
    speed_threshold = mean_speed + speed_threshold_factor * std_speed
    acc_threshold = mean_acc + acceleration_threshold_factor * std_acc
    
    # Check start
    start_trim_idx = 0
    if check_start:
        for i in range(min(window_size, len(speeds))):
            if speeds[i] > speed_threshold:
                start_trim_idx = i + 1
                continue
            
            if i < len(acc_magnitudes) and acc_magnitudes[i] > acc_threshold:
                start_trim_idx = i + 1
                continue
            
            if i + 5 < len(velocities):
                current_dir = velocities[i]
                future_dir = velocities[i + 5]
                
                norm_current = np.linalg.norm(current_dir)
                norm_future = np.linalg.norm(future_dir)
                
                # Check for zero vectors
                if norm_current > 1e-6 and norm_future > 1e-6:
                    cos_angle = np.dot(current_dir, future_dir) / (norm_current * norm_future)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    
                    if angle_deg > direction_change_threshold:
                        start_trim_idx = i + 1
                        continue
            
            break
    
    # Check end
    end_trim_idx = n - 1
    if check_end:
        for i in range(min(window_size, len(speeds))):
            idx = len(speeds) - 1 - i
            
            if speeds[idx] > speed_threshold:
                end_trim_idx = idx
                continue
            
            if idx - 1 >= 0 and idx - 1 < len(acc_magnitudes):
                if acc_magnitudes[idx - 1] > acc_threshold:
                    end_trim_idx = idx
                    continue
            
            if idx - 5 >= 0:
                current_dir = velocities[idx]
                past_dir = velocities[idx - 5]
                
                norm_current = np.linalg.norm(current_dir)
                norm_past = np.linalg.norm(past_dir)
                
                # Check for zero vectors
                if norm_current > 1e-6 and norm_past > 1e-6:
                    cos_angle = np.dot(current_dir, past_dir) / (norm_current * norm_past)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    
                    if angle_deg > direction_change_threshold:
                        end_trim_idx = idx
                        continue
            
            break
    
    return start_trim_idx, end_trim_idx


def trim_track_endpoints_streaming(
    jsonl_path: str,
    output_jsonl_path: str,
    window_size: int = 61,
    speed_threshold_factor: float = 2.5,
    acceleration_threshold_factor: float = 2.5,
    direction_change_threshold: float = 60,
    check_start: bool = True,
    check_end: bool = True,
    min_track_length: int = 30,
):
    """
    Stream-process tracks and trim endpoint anomalies.
    
    Args:
        jsonl_path: Input track JSONL path
        output_jsonl_path: Output path for trimmed tracks
        window_size: Number of frames to check at each endpoint (default: 61 frames ≈ 2 seconds)
        speed_threshold_factor: Std multiplier for speed anomaly (lower = stricter)
        acceleration_threshold_factor: Std multiplier for acceleration anomaly
        direction_change_threshold: Degrees threshold for direction change
        check_start: Check start of track
        check_end: Check end of track
        min_track_length: Minimum track length after trimming to keep
    """
    trimmed_count = 0
    removed_count = 0
    total_count = 0
    skipped_count = 0  # ⭐ NEW: Track skipped tracks
    
    with open(jsonl_path, "r") as fin, open(output_jsonl_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                track = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Skip ball and referee tracks (they have different motion patterns)
            team = track.get("team", "")
            if team in ["ball", "referee"]:
                fout.write(json.dumps(track) + "\n")
                continue
            
            total_count += 1
            
            frames = np.array(track.get("frames", []))
            positions = np.array(track.get("projected", []))

            # ⭐ FIX: Validate data before converting to numpy
            if frames.size == 0 or positions.size == 0:
                skipped_count += 1
                fout.write(json.dumps(track) + "\n")
                continue
            
            if len(frames) != len(positions):
                skipped_count += 1
                fout.write(json.dumps(track) + "\n")
                continue
            
            # ⭐ FIX: Convert and validate
            try:
                frames_array = np.array(frames)
                positions_array = np.array(positions)
            except (ValueError, TypeError):
                skipped_count += 1
                fout.write(json.dumps(track) + "\n")
                continue
            
            # ⭐ FIX: Check for valid positions (not all zeros or invalid)
            if positions_array.size == 0 or frames_array.size == 0:
                skipped_count += 1
                fout.write(json.dumps(track) + "\n")
                continue
            
            # ⭐ FIX: Check minimum track length requirement
            if len(frames_array) < window_size * 2 or len(frames_array) < min_track_length:
                skipped_count += 1
                fout.write(json.dumps(track) + "\n")
                continue            
            # Validate track has sufficient data
            if len(frames) < window_size * 2:
                fout.write(json.dumps(track) + "\n")
                continue
            
            # Detect endpoint anomalies
            try:
                start_trim, end_trim = detect_endpoint_anomalies(
                    positions_array, frames_array,
                    window_size=window_size,
                    speed_threshold_factor=speed_threshold_factor,
                    acceleration_threshold_factor=acceleration_threshold_factor,
                    direction_change_threshold=direction_change_threshold,
                    check_start=check_start,
                    check_end=check_end
                )
            except Exception as e:
                # ⭐ FIX: Catch any unexpected errors and skip this track
                print(f"⚠️  Error processing track {track.get('track_id', 'unknown')}: {e}")
                skipped_count += 1
                fout.write(json.dumps(track) + "\n")
                continue
            
            # Check if trimming would remove entire track
            if start_trim >= end_trim or (end_trim - start_trim + 1) < min_track_length:
                removed_count += 1
                continue  # Skip this track
            
            # Check if track was actually trimmed
            if start_trim > 0 or end_trim < len(frames) - 1:
                trimmed_count += 1
                
                # Trim all corresponding fields
                trimmed_frames = frames[start_trim:end_trim + 1]
                trimmed_positions = positions[start_trim:end_trim + 1]
                
                track["frames"] = trimmed_frames.tolist()
                track["projected"] = trimmed_positions.tolist()
                track["frame_range"] = [int(trimmed_frames[0]), int(trimmed_frames[-1])]
                
                # Trim bbox_area if present
                if "bbox_area" in track and len(track["bbox_area"]) == len(frames):
                    track["bbox_area"] = np.array(track["bbox_area"])[start_trim:end_trim + 1].tolist()
            
            fout.write(json.dumps(track) + "\n")
    
    print(f"✅ Trimmed endpoint anomalies:")
    print(f"   Processed: {total_count} tracks")
    print(f"   Modified: {trimmed_count} tracks ({trimmed_count/total_count*100:.1f}%)")
    print(f"   Removed (too short): {removed_count} tracks ({removed_count/total_count*100:.1f}%)")


def determine_track_jersey_number(
    jsonl_path: str,
    output_path: str,
    confidence_threshold: float = 0.99,
    min_accepted_entries: int = 3,
):
    """
    Determines the jersey number for each track based on jersey number recognition confidence.

    Args:
        jsonl_path: Path to the track data JSONL file that already contains jersey_num and jersey_conf
        output_path: Path to save the output JSONL with finalized jersey numbers
        confidence_threshold: Minimum confidence threshold for accepting a jersey number prediction
        min_accepted_entries: Minimum number of accepted predictions to make a final decision
    """
    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    # Read track data from JSONL
    with open(jsonl_path, "r") as f, open(output_path, "w") as out_file:
        for line in f:
            if not line.strip():
                continue

            track = json.loads(line)
            track_id = track.get("track_id", "")

            # Skip if this isn't a player track or no jersey number data
            if track.get("team", "") == "ball" or "jersey_num" not in track:
                out_file.write(json.dumps(track, default=_json_default, allow_nan=False) + "\n")
                continue

            # Skip if the team is referee
            if track.get("team", "") == "referee":
                track["jersey_num"] = "NA"
                track["jersey_conf"] = 0.0
                out_file.write(json.dumps(track, default=_json_default, allow_nan=False) + "\n")
                continue

            # Get all frame-by-frame jersey number predictions for this track
            jersey_entries = track.get("jersey_num", [])
            jersey_conf_entries = track.get("jersey_conf", [])

            if len(jersey_entries) != len(jersey_conf_entries):
                print(f"⚠️  Mismatched jersey_num and jersey_conf lengths for track {track_id}. Skipping jersey determination.")
                track["jersey_num"] = "unsure"
                track["jersey_conf"] = 0.0
                out_file.write(json.dumps(track, default=_json_default, allow_nan=False) + "\n")
                continue

            # Skip if no jersey number data
            if not isinstance(jersey_entries, list) or not isinstance(jersey_conf_entries, list):
                track["jersey_num"] = "unsure"
                track["jersey_conf"] = 0.0
                out_file.write(json.dumps(track, default=_json_default, allow_nan=False) + "\n")
                continue

            # Validate jersey predictions
            candidates = []
            for i, (jersey_num, conf_list) in enumerate(zip(jersey_entries, jersey_conf_entries)):
                # No valid prediction
                if not isinstance(conf_list, list):
                    continue
                # No valid prediction (-1 indicates no jersey detected as defined in inference)
                if jersey_num == -1:
                    continue
                # Skip if any confidence of a jersey number prediction is below threshold
                if any(conf < confidence_threshold for conf in conf_list):
                    continue

                # All confidences meet threshold, add to candidates
                avg_conf = sum(conf_list) / len(conf_list) if conf_list else 0
                candidates.append({"jersey_num": jersey_num, "jersey_conf": avg_conf})

            # Group by jersey number and calculate statistics
            jersey_stats = defaultdict(list)
            for candidate in candidates:
                jersey_stats[candidate["jersey_num"]].append(candidate["jersey_conf"])

            # Filter jersey numbers that meet min_accepted_entries
            accepted_jerseys = []
            for jersey_num, conf_list in jersey_stats.items():
                if len(conf_list) >= min_accepted_entries:
                    avg_confidence = sum(conf_list) / len(conf_list)
                    count = len(conf_list)
                    accepted_jerseys.append({
                        "jersey_num": jersey_num,
                        "jersey_conf": avg_confidence,
                        "count": count
                    })

            # Sort by count (descending) then by confidence (descending)
            accepted_jerseys.sort(key=lambda x: (x["count"], x["jersey_conf"]), reverse=True)

            # Save all accepted jerseys as lists
            if accepted_jerseys:
                track["jersey_num"] = [j["jersey_num"] for j in accepted_jerseys]
                track["jersey_conf"] = [j["jersey_conf"] for j in accepted_jerseys]
                track["count"] = [j["count"] for j in accepted_jerseys]
            else:
                track["jersey_num"] = "unsure"
                track["jersey_conf"] = 0.0
                track["count"] = 0

            track_dict = {
                "track_id": track.get("track_id", ""),
                "team": track.get("team", "ball"),
                "team_conf": track.get("team_conf", 0.0),
                "jersey_num": track.get("jersey_num", "unsure"),
                "jersey_conf": track.get("jersey_conf", 0.0),
                "count": track.get("count", 0),
                "bbox_area": track.get("bbox_area", []),
                "frames": track.get("frames", []),
                "points": track.get("points", []),
            }
            try:
                out_file.write(json.dumps(track_dict, default=_json_default, allow_nan=False) + "\n")
            except Exception as e:
                # If there's a bad value, fail with context (should NOT segfault; should error cleanly)
                raise RuntimeError(f"Failed to JSON-encode track_id={track_id}: {e}") from e    

    print(f"✅ Jersey numbers determined and saved to: {output_path}")


def filter_valid_points(
    track: RawTrack,
    field_size: Tuple[int, int] = (1060, 660),
) -> RawTrack:
    """
    Pure function: Filters points outside field boundaries.
    
    Returns:
        track: Filtered track with only in-bounds points.
    """

    # 1. Validation
    if len(track.projected) == 0:
        return None

    # 2. Geometry Calculation
    w, h = field_size
    
    # This generates a list of booleans: [True, False, True, ...]
    in_bounds = [
        (0 <= p[0] <= w) and (0 <= p[1] <= h)
        for p in track.projected
    ]

    # 3. Check if we filtered everything out
    # 'any' is faster than sum() for lists because it stops at the first True
    if not any(in_bounds): 
        return None

    # 4. Helper for List Slicing (Solves the "Unbound Variable" & "Repetition" issues)
    def filter_list(data: list) -> list:
        # Only try to compress if data exists, otherwise return empty list
        return list(compress(data, in_bounds)) if data else []

    # 5. Constructor with Inlining
    return RawTrack(
        track_id=track.track_id,
        frames=filter_list(track.frames),
        projected=filter_list(track.projected),
        bbox=filter_list(track.bbox),
        team_conf=filter_list(track.team_conf),
        jersey_num=filter_list(track.jersey_num),
        jersey_conf=track.jersey_conf,
    )


def load_and_split_tracks(
    json_path: str,
    output_path: str,
    field_size: List[int],
    min_track_length: int,
    window_size: int,
    threshold: float,
):
    """
    Draw smoothed 2D trajectories from tracking JSON with optional merging of fragmented tracks.

    Args:
        json_path: Path to tracking JSON.
        output_path: Path to save processed tracks JSONL.
        field_size: Field dimension (width, height).
        min_track_length: Minimum track length to visualize.
        window_size: Size of the sliding window.
        threshold: Ratio of frames in the window needed for a team to trigger a split.

    """

    track_dict = {}

    # Make sure the output file is empty/overwritten at start
    with open(output_path, "w") as out_f:
        pass  # This creates an empty file or truncates existing file

    with open(json_path, "r") as f, open(output_path, "a") as out_f:
        for line in f:
            obj = json.loads(line)
            # print(type(obj['projected'][0][0]))  # e.g., <class 'float'>
            projected_points = obj.get("projected", [])
            if len(projected_points) < min_track_length:
                continue

            raw_track = RawTrack.from_dict(obj)
            
            raw_track.projected = [pt for pt in raw_track.projected if pt is not None]
            
            filtered_track = filter_valid_points(
                raw_track,
                field_size=field_size,
            )

            if filtered_track is None:
                continue
            
            if len(filtered_track.projected) < min_track_length:
                continue
            
            # Step 4: Reconstruct track object for splitter
            cleaned_track = {
                "track_id": filtered_track.track_id,
                "frame_id": filtered_track.frames,
                "projected": filtered_track.projected,
                "bbox": filtered_track.bbox,
                "team_conf": filtered_track.team_conf,
                "jersey_num": filtered_track.jersey_num,
                "jersey_conf": filtered_track.jersey_conf,
            }

            # Now we split the clean long track
            split_objects = split_track_by_sliding_window(cleaned_track, window_size, threshold)

            for split_obj in split_objects:
                tid = split_obj["track_id"]
                frames = split_obj["frame_id"]
                projected_points = split_obj["projected"]
                bboxes = split_obj.get("bbox", [])

                pts = np.array([pt for pt in projected_points if pt is not None])
                if len(pts) == 0:
                    continue  # skip this segment
                xs, ys = pts[:, 0], pts[:, 1]

                frs = np.array(frames)

                # Calculate bbox areas for later track matching
                bboxes_area = calculate_bbox_area(bboxes) if bboxes else []

                track_dict = {
                    "track_id": tid,
                    "team": split_obj.get("team", "ball"),
                    "team_conf": split_obj.get("team_conf", []),
                    "jersey_num": split_obj.get("jersey_num", []),
                    "jersey_conf": split_obj.get("jersey_conf", []),
                    "bbox_area": bboxes_area,
                    "frames": frs.tolist(),
                    "points": np.stack([xs, ys], axis=1).tolist(),
                }

                # save the track_dict to jsonl
                out_f.write(json.dumps(track_dict) + "\n")


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


def remove_tracks_near_boundary_stream(
    jsonl_path, output_jsonl_path, field_size, margin_meter=30, near_ratio_threshold=0.9
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
    with open(jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
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


def remove_static_tracks(
    jsonl_path,
    output_jsonl_path,
    movement_threshold=20,  # in meters (10 = 1m if 0.1m units)
):
    """
    Remove tracks that don't move significantly.

    Args:
        jsonl_path (str): Input path to .jsonl file.
        output_jsonl_path (str): Output path to write filtered tracks.
        movement_threshold (float): Minimum total movement (Euclidean) to keep.
    """
    with open(jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
        for line in f_in:
            track = json.loads(line)
            # if track.get("team") != "ball":
            #     f_out.write(json.dumps(track) + "\n")
            #     continue

            points = np.array(track.get("projected", []))
            if len(points) < 2:
                continue  # skip too short

            # Compute total movement
            deltas = np.diff(points, axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            total_distance = distances.sum()

            if total_distance >= movement_threshold:
                f_out.write(json.dumps(track) + "\n")


def allocate_jersey_numbers_by_confidence(
    jsonl_path: str,
    output_path: str,
    home_jersey_numbers: List[int],
    away_jersey_numbers: List[int],
):
    """
    Allocate jersey numbers to tracks based on confidence scores and handle conflicts.

    Args:
        jsonl_path: Path to the input JSONL file with track data
        output_path: Path to save the output JSONL with allocated jersey numbers
        home_jersey_numbers: List of valid jersey numbers for the home team
        away_jersey_numbers: List of valid jersey numbers for the away team
    """

    # Create sets for O(1) lookup
    valid_jerseys = {
        "home": set(home_jersey_numbers),
        "away": set(away_jersey_numbers),
    }

    # Read all tracks
    tracks = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))

    print(f"📊 Processing {len(tracks)} tracks for jersey number allocation...")

    # Step 1: Allocate jersey numbers based on confidence and validity
    for track in tracks:
        track_id = track.get("track_id", "")
        team = track.get("team", "")

        # Skip non-player tracks
        if team in ["ball", "referee", "unsure"]:
            track["allocated_jersey"] = "NA" if team == "referee" else "unsure"
            track["allocated_conf"] = 0.0
            continue

        # Determine team key
        team_key = "home" if "home" in team.lower() else "away" if "away" in team.lower() else None
        if team_key is None:
            track["allocated_jersey"] = "unsure"
            track["allocated_conf"] = 0.0
            continue

        jersey_nums = track.get("jersey_num", "unsure")
        jersey_confs = track.get("jersey_conf", 0.0)

        # Handle unsure cases
        if jersey_nums == "unsure" or not jersey_nums:
            track["allocated_jersey"] = "unsure"
            track["allocated_conf"] = 0.0
            continue

        # Normalize to lists
        if not isinstance(jersey_nums, list):
            jersey_nums = [jersey_nums]
        if not isinstance(jersey_confs, list):
            jersey_confs = [jersey_confs]

        # Sort by confidence (descending)
        jersey_conf_pairs = sorted(zip(jersey_nums, jersey_confs), key=lambda x: x[1], reverse=True)

        # Find first valid jersey number
        allocated = False
        for jersey_num, conf in jersey_conf_pairs:
            if jersey_num in valid_jerseys[team_key]:
                track["allocated_jersey"] = jersey_num
                track["allocated_conf"] = conf
                allocated = True
                break

        if not allocated:
            track["allocated_jersey"] = "unsure"
            track["allocated_conf"] = 0.0

    print("✅ Initial allocation complete. Resolving conflicts...")

    # Step 2: Resolve conflicts where same jersey number is assigned to overlapping tracks
    # Build frame-to-track mapping for efficient conflict detection
    frame_team_jersey_tracks = defaultdict(lambda: defaultdict(list))

    for track in tracks:
        allocated_jersey = track.get("allocated_jersey")
        team = track.get("team", "")

        if allocated_jersey == "unsure" or allocated_jersey == "NA":
            continue

        frames = track.get("frames", [])
        track_id = track.get("track_id", "")
        allocated_conf = track.get("allocated_conf", 0.0)

        for frame in frames:
            frame_team_jersey_tracks[frame][(team, allocated_jersey)].append(
                {"track_id": track_id, "conf": allocated_conf}
            )

    # Find conflicting tracks (same team + jersey in overlapping frames)
    conflicts = defaultdict(set)  # track_id -> set of conflicting track_ids

    for frame, team_jersey_dict in frame_team_jersey_tracks.items():
        for (team, jersey), track_list in team_jersey_dict.items():
            if len(track_list) > 1:
                # Multiple tracks with same jersey in this frame
                for track_info in track_list:
                    other_tracks = [
                        t["track_id"] for t in track_list if t["track_id"] != track_info["track_id"]
                    ]
                    conflicts[track_info["track_id"]].update(other_tracks)

    print(f"⚠️  Found {len(conflicts)} tracks with conflicts")

    # Step 3: Resolve conflicts - keep highest confidence, reassign others
    track_map = {track["track_id"]: track for track in tracks}
    resolved_count = 0

    for track_id in conflicts:
        track = track_map[track_id]
        conflicting_ids = conflicts[track_id]

        # Get all conflicting tracks including self
        all_conflicting = [track] + [track_map[cid] for cid in conflicting_ids if cid in track_map]

        # Filter to only those with same allocated jersey and team
        same_jersey_tracks = [
            t
            for t in all_conflicting
            if t.get("allocated_jersey") == track.get("allocated_jersey")
            and t.get("team") == track.get("team")
        ]

        if len(same_jersey_tracks) <= 1:
            continue

        # Sort by confidence
        same_jersey_tracks.sort(key=lambda t: t.get("allocated_conf", 0.0), reverse=True)

        # Keep highest confidence track, reassign others
        winner = same_jersey_tracks[0]

        for loser_track in same_jersey_tracks[1:]:
            if loser_track["track_id"] == winner["track_id"]:
                continue

            # Try to find alternative jersey number
            team = loser_track.get("team", "")
            team_key = "home" if "home" in team.lower() else "away"

            jersey_nums = loser_track.get("jersey_num", [])
            jersey_confs = loser_track.get("jersey_conf", [])

            if not isinstance(jersey_nums, list):
                jersey_nums = [jersey_nums]
            if not isinstance(jersey_confs, list):
                jersey_confs = [jersey_confs]

            # Sort by confidence and find alternative
            jersey_conf_pairs = sorted(
                zip(jersey_nums, jersey_confs), key=lambda x: x[1], reverse=True
            )

            reallocated = False
            for jersey_num, conf in jersey_conf_pairs:
                if jersey_num == loser_track.get("allocated_jersey"):
                    continue  # Skip the conflicting one

                if jersey_num in valid_jerseys[team_key]:
                    # Check if this alternative also conflicts
                    frames = loser_track.get("frames", [])
                    has_conflict = False

                    for frame in frames:
                        if (team, jersey_num) in frame_team_jersey_tracks[frame]:
                            existing_tracks = frame_team_jersey_tracks[frame][(team, jersey_num)]
                            if any(
                                t["track_id"] != loser_track["track_id"] for t in existing_tracks
                            ):
                                has_conflict = True
                                break

                    if not has_conflict:
                        loser_track["allocated_jersey"] = jersey_num
                        loser_track["allocated_conf"] = conf
                        reallocated = True
                        resolved_count += 1
                        break

            if not reallocated:
                loser_track["allocated_jersey"] = "unsure"
                loser_track["allocated_conf"] = 0.0
                resolved_count += 1

    print(f"✅ Resolved {resolved_count} conflicts")

    # Step 4: Write output
    with open(output_path, "w") as out_f:
        for track in tracks:
            # Create output with allocated jersey as the main jersey_num
            output_track = {
                "track_id": track.get("track_id", ""),
                "team": track.get("team", ""),
                "team_conf": track.get("team_conf", 0.0),
                "jersey_num": track.get("allocated_jersey", "unsure"),
                "jersey_conf": track.get("allocated_conf", 0.0),
                "frame_range": track.get("frame_range", []),
                "frames": track.get("frames", []),
                "projected": track.get("projected", []),
                "bbox_area": track.get("bbox_area", []),
            }
            out_f.write(json.dumps(output_track) + "\n")

    print(f"✅ Jersey number allocation complete. Saved to {output_path}")

    # Print summary statistics
    allocated_counts = defaultdict(int)
    for track in tracks:
        allocated_counts[track.get("allocated_jersey", "unsure")] += 1

    print("\n📊 Allocation Summary:")
    print(allocated_counts)

    # Convert all keys to strings for sorting, with numeric jerseys sorted numerically
    def sort_key(item):
        key = item[0]
        if isinstance(key, int):
            return (0, key)  # Numeric jerseys first, sorted by value
        else:
            return (1, key)  # String categories second, sorted alphabetically

    for jersey, count in sorted(allocated_counts.items(), key=sort_key):
        print(f"   Jersey {jersey}: {count} tracks")


def coarse_postprocessing(
    json_path: str,
    home_jersey_numbers: List[int],
    away_jersey_numbers: List[int],
    field_size: List[int],
    min_track_length: int,
    smoothing_window: int,
    polyorder: int,
    max_merge_gap: int,
    max_merge_overlap_frames: int,
    max_merge_distance: int,
    window_size: int,
    threshold: float,
    endpoint_window_size: int = 61,
    endpoint_speed_factor: float = 2.5,
    endpoint_acceleration_factor: float = 2.5,
    endpoint_direction_threshold: float = 60,
    confidence_threshold: float = 0.99,
    min_accepted_entries: int = 7,
    detector_kwargs=None,
):

    start_load = time.time()
    # Merge and filter tracks
    load_and_split_tracks(
        json_path=json_path,
        output_path=json_path.replace(".jsonl", "_split.jsonl"),
        field_size=field_size,
        min_track_length=min_track_length,
        window_size=window_size,
        threshold=threshold,
    )
    end_load = time.time()
    print(f"✅ Loaded and split tracks in {end_load - start_load:.2f} seconds")
    return "testing"
    determine_track_jersey_number(
        jsonl_path=json_path.replace(".jsonl", "_split.jsonl"),
        output_path=json_path.replace(".jsonl", "_split_with_jersey.jsonl"),
        confidence_threshold=confidence_threshold,
        min_accepted_entries=min_accepted_entries,
    )
    end_jersey = time.time()
    print(f"✅ Determined jersey numbers in {end_jersey - end_load:.2f} seconds")

    hybrid_merge_stream_fixed(
        jsonl_path=json_path.replace(".jsonl", "_split_with_jersey.jsonl"),
        output_path=json_path.replace(".jsonl", "_merged.jsonl"),
        max_merge_gap=max_merge_gap,
        max_merge_overlap_frames=max_merge_overlap_frames,
        max_merge_distance=max_merge_distance,
        smoothing_window=smoothing_window,
        polyorder=polyorder,
    )
    end_merge = time.time()
    print(f"✅ Merged tracks in {end_merge - end_jersey:.2f} seconds")

    trim_track_endpoints_streaming(
        jsonl_path=json_path.replace(".jsonl", "_merged.jsonl"),
        output_jsonl_path=json_path.replace(".jsonl", "_merged_trimmed.jsonl"),
        window_size=endpoint_window_size,
        speed_threshold_factor=endpoint_speed_factor,
        acceleration_threshold_factor=endpoint_acceleration_factor,
        direction_change_threshold=endpoint_direction_threshold,
        check_start=True,
        check_end=True,
        min_track_length=min_track_length,
    )
    end_trim = time.time()
    print(f"✅ Trimmed endpoint anomalies in {end_trim - end_merge:.2f} seconds")

    process_jsonl_detect_replace(
        input_path=json_path.replace(".jsonl", "_merged_trimmed.jsonl"),
        output_path=json_path.replace(".jsonl", "_merged_trimmed_detected.jsonl"),
        detector_kwargs=detector_kwargs,
    )


    remove_tracks_near_boundary_stream(
        jsonl_path=json_path.replace(".jsonl", "_merged_trimmed_detected.jsonl"),
        output_jsonl_path=json_path.replace(".jsonl", "_merged_filtered_near_boundary.jsonl"),
        field_size=field_size,
        margin_meter=30,
    )
    end_boundary = time.time()
    print(f"✅ Removed boundary-only tracks in {end_boundary - end_trim:.2f} seconds")

    remove_static_tracks(
        json_path.replace(".jsonl", "_merged_filtered_near_boundary.jsonl"),
        json_path.replace(".jsonl", "_merged_filtered.jsonl"),
        movement_threshold=30,  # in meters (10 = 1m if 0.1m units)
    )
    end_static_ball = time.time()
    print(f"✅ Removed static ball tracks in {end_static_ball - end_boundary:.2f} seconds")

    return json_path.replace(".jsonl", "_merged_filtered.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Process merged tracks from tracking JSONL")
    parser.add_argument(
        "--json-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the merged tracking JSONL files from different cameras",
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path to the field image")
    parser.add_argument(
        "--home-jersey-numbers",
        type=int,
        nargs="+",
        required=True,
        help="List of home team jersey numbers",
    )
    parser.add_argument(
        "--away-jersey-numbers",
        type=int,
        nargs="+",
        required=True,
        help="List of away team jersey numbers",
    )
    parser.add_argument(
        "--field-size",
        type=int,
        nargs=2,
        default=[1060, 660],
        help="Field size (length, width) as 0.1m",
    )
    parser.add_argument(
        "--min-track-length", type=int, default=10, help="Minimum track length to keep"
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=90,
        help="Savitzky-Golay filter window size",
    )
    parser.add_argument("--polyorder", type=int, default=7, help="Polynomial order for smoothing")
    parser.add_argument(
        "--max-merge-gap",
        type=int,
        default=20,
        help="Max allowed gap (frames) between mergeable tracks",
    )
    parser.add_argument(
        "--max-merge-overlap-frames",
        type=int,
        default=15,
        help="Max allowed overlap for merging",
    )
    parser.add_argument(
        "--max-merge-distance",
        type=int,
        default=50,
        help="Max spatial distance for merging",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Window size for velocity consistency check",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Threshold for velocity consistency",
    )
    parser.add_argument(
        "--endpoint-window-size",
        type=int,
        default=61,
        help="Window size for endpoint checking (frames, default: 61 ≈ 2s @ 30fps)"
    )
    parser.add_argument(
        "--endpoint-speed-factor",
        type=float,
        default=2.5,
        help="Speed threshold factor for endpoint anomalies (default: 2.5 std)"
    )
    parser.add_argument(
        "--endpoint-acceleration-factor",
        type=float,
        default=2.5,
        help="Acceleration threshold factor (default: 2.5 std)"
    )
    parser.add_argument(
        "--endpoint-direction-threshold",
        type=float,
        default=60,
        help="Direction change threshold in degrees (default: 60°)"
    )   
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    # second pass with tighter gates
    DETECTOR = dict(
        window_size=501,
        step=250,
        prominence=2,
        min_wave_len=0,
        max_wave_len=60,
        speed_std_factor=None,
        smooth_window=0,
        savgol_poly=2,
        min_steepness=0.1,
        min_quad_curv=0.9,
        min_monotonic_ratio=0.5,
        max_gap_size=30,
    )
    
    start = time.time()

    # # Stage 1: Process each camera
    processed_files = []
    for json_path in args.json_paths:
        print(f"Processing camera JSONL: {json_path}")
        output = coarse_postprocessing(
            json_path=json_path,
            home_jersey_numbers=args.home_jersey_numbers,
            away_jersey_numbers=args.away_jersey_numbers,
            field_size=tuple(args.field_size),
            min_track_length=args.min_track_length,
            smoothing_window=args.smoothing_window,
            polyorder=args.polyorder,
            max_merge_gap=args.max_merge_gap,
            max_merge_overlap_frames=args.max_merge_overlap_frames,
            max_merge_distance=args.max_merge_distance,
            window_size=args.window_size,
            threshold=args.threshold,
            endpoint_window_size=args.endpoint_window_size,
            endpoint_speed_factor=args.endpoint_speed_factor,
            endpoint_acceleration_factor=args.endpoint_acceleration_factor,
            endpoint_direction_threshold=args.endpoint_direction_threshold,
            detector_kwargs=DETECTOR,
        )
        processed_files.append(output)

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()

# example usage:
# python3 post_processing_player.py \
#  --json-paths "./runs/detect/test_4k_player_640/team_tracking.jsonl" \
#  --image-path "./data/images/mongkok_football_field.png" \
#  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 13 23 25 8 14 17 18 21 24 31 33 34 \
#  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 1 17 22 23 24 28 33 42 43 44 72 88 \
