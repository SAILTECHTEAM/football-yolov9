import argparse
import json
import os
import re
import numpy as np
import random

from collections import defaultdict
from dataclasses import dataclass, field
from fastdtw import fastdtw
from typing import List, Tuple, Dict, Optional, Set

from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import savgol_filter
from tqdm import tqdm

from tools.player_motion_classification import calculate_angle
from fuse_ball_tracks import (
    load_ball_tracks_from_jsonl,
    calibrate_frame_offsets,
    calibrate_coordinate_systems,
)


@dataclass
class CalibrationStats:
    total_pruned_points: int = 0
    tracks_with_pruning: int = 0
    tracks_removed_empty: int = 0
    points_transformed: int = 0


@dataclass
class CalibrationConfig:
    """Holds all hyper-parameters for the calibration pipeline."""
    auto_calibrate: bool = True
    max_search_offset: int = 60
    overlap_threshold: int = 30
    min_overlap_frames: int = 50
    min_confidence: float = 0.3
    max_frame: Optional[int] = None
    verbose: bool = False


@dataclass 
class MatchingConfig:
    """Holds all hyper-parameters for the matching pipeline."""
    min_overlap_frames: int = 10
    max_analysis_frames: int = 150
    max_point_distance: float = 100.0
    max_outlier_ratio: float = 0.3
    direction_threshold: float = 45.0
    direction_frame_stride: int = 3
    min_movement_threshold: float = 0.5
    max_step_distance: float = 50.0
    max_initial_distance: float = 150.0


def load_ball_tracks(ball_jsonl_paths: List[str]) -> List[Dict]:
    """I/O: load ball tracks for each camera."""
    return [load_ball_tracks_from_jsonl(p) for p in ball_jsonl_paths]


def compute_offsets_from_ball_tracks(
    all_ball_tracks: List[Dict],
    n_cameras: int,
    cfg: CalibrationConfig, # New argument
    frame_offsets: Optional[List[int]] = None,
    coord_offsets: Optional[List[np.ndarray]] = None,
) -> Tuple[List[int], List[np.ndarray]]:
    
    if frame_offsets is None:
        if cfg.auto_calibrate:
            # Pass specific fields to the sub-algorithm
            frame_offsets = calibrate_frame_offsets(
                all_ball_tracks,
                max_search_offset=cfg.max_search_offset,
                overlap_threshold=cfg.overlap_threshold,
                verbose=cfg.verbose,
            )
        else:
            frame_offsets = [0] * n_cameras

    if coord_offsets is None:
        if cfg.auto_calibrate:
            coord_offsets = calibrate_coordinate_systems(
                all_ball_tracks,
                min_overlap_frames=cfg.min_overlap_frames,
                min_confidence=cfg.min_confidence,
                verbose=cfg.verbose,
            )
        else:
            coord_offsets = [np.array([0.0, 0.0])] * n_cameras

    return frame_offsets, coord_offsets


def load_player_tracks(player_jsonl_paths: List[str]) -> Dict[int, List[Dict]]:
    """I/O: load player tracks for each camera index."""
    all_tracks_by_file: Dict[int, List[Dict]] = {}
    for src_idx, path in enumerate(player_jsonl_paths):
        all_tracks_by_file[src_idx] = load_tracks_from_jsonl(path)
    return all_tracks_by_file


def infer_global_max_frame(all_tracks_by_file: Dict[int, List[Dict]]) -> int:
    """Logic: infer max frame across cameras before any offsets applied."""
    original_max_frames = []
    for tracks in all_tracks_by_file.values():
        if tracks:
            camera_max_frame = max(
                max(t.get("frames", [0])) if t.get("frames") else 0 for t in tracks
            )
            original_max_frames.append(camera_max_frame)
        else:
            original_max_frames.append(0)
    return max(original_max_frames) if original_max_frames else 0


def apply_frame_offsets_to_tracks(
    all_tracks_by_file: Dict[int, List[Dict]],
    frame_offsets: List[int],
    max_frame: Optional[int] = None,
) -> Tuple[Dict[int, List[Dict]], CalibrationStats]:
    """
    Logic: apply frame offsets and prune frames outside [1, max_frame].
    """
    stats = CalibrationStats()

    if max_frame is None:
        max_frame = infer_global_max_frame(all_tracks_by_file)

    for src_idx, offset in enumerate(frame_offsets):
        if offset == 0:
            continue

        for track in all_tracks_by_file.get(src_idx, []):
            original_length = len(track.get("frames", []))
            frames = track.get("frames", [])
            if not frames:
                track["frame_range"] = [0, 0]
                continue

            offset_frames = [f + offset for f in frames]
            valid_indices = [i for i, f in enumerate(offset_frames) if 1 <= f <= max_frame]

            track["frames"] = [offset_frames[i] for i in valid_indices]

            if "projected" in track and track["projected"]:
                track["projected"] = [track["projected"][i] for i in valid_indices]
            if "bbox_area" in track and track["bbox_area"]:
                track["bbox_area"] = [track["bbox_area"][i] for i in valid_indices]

            if track["frames"]:
                track["frame_range"] = [min(track["frames"]), max(track["frames"])]
            else:
                track["frame_range"] = [0, 0]

            pruned = original_length - len(track["frames"])
            if pruned > 0:
                stats.total_pruned_points += pruned
                stats.tracks_with_pruning += 1

    return all_tracks_by_file, stats


def apply_coord_offsets_to_tracks(
    all_tracks_by_file: Dict[int, List[Dict]],
    coord_offsets: List[np.ndarray],
) -> Tuple[Dict[int, List[Dict]], CalibrationStats]:
    """
    Logic: subtract coordinate offsets from projected points.
    """
    stats = CalibrationStats()

    for src_idx, offset in enumerate(coord_offsets):
        if np.allclose(offset, 0.0):
            continue

        for track in all_tracks_by_file.get(src_idx, []):
            projected = track.get("projected")
            if not projected:
                continue

            transformed = []
            for p in projected:
                if p is None:
                    transformed.append(None)
                    continue
                x, y = p
                transformed.append([x - float(offset[0]), y - float(offset[1])])
                stats.points_transformed += 1

            track["projected"] = transformed

    return all_tracks_by_file, stats


def remove_empty_tracks(all_tracks_by_file: Dict[int, List[Dict]]) -> Tuple[Dict[int, List[Dict]], int]:
    """Logic: remove tracks with no frames."""
    removed = 0
    for src_idx in list(all_tracks_by_file.keys()):
        before = len(all_tracks_by_file[src_idx])
        all_tracks_by_file[src_idx] = [
            t for t in all_tracks_by_file[src_idx] if t.get("frames") and len(t["frames"]) > 0
        ]
        removed += before - len(all_tracks_by_file[src_idx])
    return all_tracks_by_file, removed


def apply_calibration_to_player_tracks(
    player_jsonl_paths: List[str],
    ball_jsonl_paths: List[str],
    # Pre-calculated offsets (optional inputs)
    frame_offsets: Optional[List[int]] = None,
    coord_offsets: Optional[List[np.ndarray]] = None,
    cfg: CalibrationConfig = field(default_factory=CalibrationConfig) 
) -> Tuple[Dict[int, List[Dict]], List[int], List[np.ndarray]]:
    
    if cfg.verbose:
        print("=" * 60)
        print("APPLYING CALIBRATION TO PLAYER TRACKS")
        print("=" * 60)

    # Step 1: Compute Offsets
    # We pass the 'config' object directly to the helper
    all_ball_tracks = load_ball_tracks(ball_jsonl_paths)
    
    frame_offsets, coord_offsets = compute_offsets_from_ball_tracks(
        all_ball_tracks=all_ball_tracks,
        n_cameras=len(player_jsonl_paths),
        frame_offsets=frame_offsets,
        coord_offsets=coord_offsets,
        cfg=cfg
    )

    # Step 2: Load Player Data
    all_tracks_by_file = load_player_tracks(player_jsonl_paths)

    # Step 3: Apply Frame Offsets
    all_tracks_by_file, frame_stats = apply_frame_offsets_to_tracks(
        all_tracks_by_file,
        frame_offsets=frame_offsets,
        max_frame=cfg.max_frame  # Access config attributes directly
    )

    # Step 4: Apply Coord Offsets
    all_tracks_by_file, coord_stats = apply_coord_offsets_to_tracks(
        all_tracks_by_file,
        coord_offsets=coord_offsets,
    )

    # Cleanup
    all_tracks_by_file, removed = remove_empty_tracks(all_tracks_by_file)

    if cfg.verbose:
        print("\n✅ Calibration Offsets:")
        for src_idx in range(len(player_jsonl_paths)):
            print(
                f"  Camera {src_idx}: Frame Offset = {frame_offsets[src_idx]:+d}, "
                f"Coord Offset = ({coord_offsets[src_idx][0]:+.1f}, {coord_offsets[src_idx][1]:+.1f})"
            )
        if frame_stats.total_pruned_points:
            print(
                f"\n  ⚠️  Pruned {frame_stats.total_pruned_points} points from "
                f"{frame_stats.tracks_with_pruning} tracks"
            )
        if removed:
            print(f"  ⚠️  Removed {removed} empty tracks")
        print("=" * 60)

    return all_tracks_by_file, frame_offsets, coord_offsets


def find_overlapping_frames(track1: Dict, track2: Dict) -> List[int]:
    """
    Find frames that exist in both tracks.
    e.g. if track1["frames"] is [1,2,3,4] and track2["frames"] is [3,4,5,6],
    this function will return [3,4].

    Args:
        track1: A dictionary representing the first track with a "frames" key.
        track2: A dictionary representing the second track with a "frames" key.

    Returns:
        List of frame numbers that overlap between the two tracks
    """
    frames1_set = set(track1["frames"])
    frames2_set = set(track2["frames"])
    overlapping = sorted(frames1_set & frames2_set)
    return overlapping


def get_position_at_specific_frame(track: Dict, target_frame: int) -> Optional[np.ndarray]:
    """
    Get the actual observed position for a given frame.
    Returns None if the frame doesn't exist in the track.

    Args:
        track: Track dictionary with 'frames' and 'projected' fields
        target_frame: Frame number to get position for

    Returns:
        [x, y] position as np.ndarray or None if frame not found
    """
    frames = track.get("frames", [])
    projected = track.get("projected", [])

    if target_frame not in frames:
        return None

    try:
        idx = frames.index(target_frame)
        if idx < len(projected):
            return np.array(projected[idx])
    except (ValueError, IndexError):
        pass

    return None


def calculate_pairwise_distances(
    track1: Dict,
    track2: Dict,
    overlapping_frames: List[int],
    max_point_distance: float = 100.0,
    max_outlier_ratio: float = 0.3,
) -> Tuple[List[float], Dict]:
    """
    Calculate per-frame distances between two tracks, filtering outliers.

    Args:
        track1, track2: Track dictionaries
        overlapping_frames: List of frames to compare
        max_point_distance: Maximum allowed distance per frame (outlier threshold)
        max_outlier_ratio: If more than this ratio are outliers, return empty list

    Returns:
        Tuple of (list of valid distances, metadata_dict)
        Returns ([], metadata) if too many outliers or insufficient data
    """

    distances = []
    outlier_frames = []
    skipped_missing = 0

    for frame in overlapping_frames:
        pos1 = get_position_at_specific_frame(track1, frame)
        pos2 = get_position_at_specific_frame(track2, frame)

        # Skip if either position is missing
        if pos1 is None or pos2 is None:
            skipped_missing += 1
            continue

        dist = euclidean(pos1, pos2)

        # Check for outlier
        if dist > max_point_distance:
            outlier_frames.append((frame, dist))
        else:
            distances.append(dist)

    n_valid = len(distances)
    n_outliers = len(outlier_frames)
    total_compared = n_valid + n_outliers
    outlier_ratio = n_outliers / total_compared if total_compared > 0 else 0.0

    metadata = {
        "n_valid": n_valid,  # number of valid distances
        "n_outliers": n_outliers,  # number of outlier distances
        "n_skipped_missing": skipped_missing,  # number of frames skipped due to missing data
        "total_compared": total_compared,  # total number of frames compared (valid + outliers)
        "outlier_ratio": outlier_ratio,  # ratio of outliers to total compared frames
        "max_point_distance": max_point_distance,  # threshold for outlier detection
        "outlier_frames": outlier_frames[:10],  # sample of outlier frames and their distances
    }

    # Reject if too many outliers
    if total_compared > 0 and outlier_ratio > max_outlier_ratio:
        metadata["rejected"] = True
        metadata["reason"] = f"outlier_ratio {outlier_ratio:.2%} > {max_outlier_ratio:.2%}"
        return [], metadata

    metadata["rejected"] = False
    return distances, metadata


def calculate_median_distance(
    track1: Dict,
    track2: Dict,
    overlapping_frames: List[int],
    max_point_distance: float = 100.0,
    max_outlier_ratio: float = 0.3,
) -> Tuple[float, Dict]:
    """
    Calculate median Euclidean distance between tracks.
    Filters out outlier points and rejects if too many outliers.

    Args:
        track1, track2: Track dictionaries
        overlapping_frames: List of frames to compare
        max_distance: Maximum allowed distance per frame (outlier threshold)
        max_outlier_ratio: If more than this ratio are outliers, return inf

    Returns:
        Tuple of (median distance, metadata_dict)
        Returns (inf, metadata) if rejected
    """
    distances, metadata = calculate_pairwise_distances(
        track1,
        track2,
        overlapping_frames,
        max_point_distance=max_point_distance,
        max_outlier_ratio=max_outlier_ratio,
    )

    if metadata["rejected"] or len(distances) == 0:
        return float("inf"), metadata

    median_dist = np.median(distances)
    metadata["median_distance"] = median_dist
    metadata["std_distance"] = np.std(distances)
    metadata["min_distance"] = np.min(distances)
    metadata["max_distance_observed"] = np.max(distances)
    return median_dist, metadata


def calculate_median_velocity_difference(
    track1: Dict,
    track2: Dict,
    overlapping_frames: List[int],
    fps: float,
    window_size: int,
) -> float:
    """
    Calculate median velocity difference between tracks in overlapping region.
    Only uses frame pairs where BOTH tracks have actual observed positions.

    Args:
        track1, track2: Track dictionaries
        overlapping_frames: List of frames to consider
        fps: Frames per second of the video
        window_size: Number of frames for velocity window

    Returns:
        Median velocity difference (units per second)
    """
    velocities = []

    for track in [track1, track2]:
        track_velocities = []
        for i in range(len(overlapping_frames) - window_size):
            frame_start = overlapping_frames[i]
            frame_end = overlapping_frames[i + window_size]

            pos_start = get_position_at_specific_frame(track, frame_start)
            pos_end = get_position_at_specific_frame(track, frame_end)

            # Skip if either position is missing
            if pos_start is None or pos_end is None:
                continue

            dist = euclidean(pos_start, pos_end)
            time_sec = (frame_end - frame_start) / fps
            if time_sec > 0:
                velocity = dist / time_sec
                track_velocities.append(velocity)

        if track_velocities:
            velocities.append(track_velocities)

    if len(velocities) == 2 and velocities[0] and velocities[1]:
        # Match by index (same window positions)
        min_len = min(len(velocities[0]), len(velocities[1]))
        velocity_diffs = [abs(velocities[0][i] - velocities[1][i]) for i in range(min_len)]
        return np.median(velocity_diffs) if velocity_diffs else 0.0

    return 0.0


def calculate_total_distance(
    track: Dict,
    overlapping_frames: List[int],
    max_step_distance: float = 50.0,
    max_outlier_ratio: float = 0.3,
) -> Tuple[float, Dict]:
    """
    Calculate total Euclidean distance travelled by a track in overlapping region.
    Only uses consecutive frames where BOTH positions exist.
    Rejects individual steps that exceed max_step_distance (outliers).

    Args:
        track: Track dictionary
        overlapping_frames: List of frames to analyze
        max_step_distance: Maximum allowed distance per step (outlier threshold)
        max_outlier_ratio: If more than this ratio of steps are outliers, return inf

    Returns:
        Tuple of (total_distance, metadata_dict)
        Returns (inf, metadata) if too many outliers detected
    """
    total_distance = 0.0
    prev_pos = None
    prev_frame = None

    n_valid_steps = 0
    n_outlier_steps = 0
    outlier_frames = []

    for frame in overlapping_frames:
        pos = get_position_at_specific_frame(track, frame)

        if pos is not None:
            if prev_pos is not None:
                # Only add distance if frames are reasonably close
                frame_gap = frame - prev_frame if prev_frame is not None else 1
                if frame_gap <= 5:  # Max gap of 5 frames
                    dist = euclidean(prev_pos, pos)

                    # Check if this step is an outlier
                    if dist > max_step_distance:
                        n_outlier_steps += 1
                        outlier_frames.append((prev_frame, frame, dist))
                        # Skip this step - don't add to total
                    else:
                        total_distance += dist
                        n_valid_steps += 1

            prev_pos = pos
            prev_frame = frame

    total_steps = n_valid_steps + n_outlier_steps
    outlier_ratio = n_outlier_steps / total_steps if total_steps > 0 else 0.0

    metadata = {
        "n_valid_steps": n_valid_steps,
        "n_outlier_steps": n_outlier_steps,
        "total_steps": total_steps,
        "outlier_ratio": outlier_ratio,
        "max_step_distance": max_step_distance,
        "outlier_frames": outlier_frames[:10],  # Limit for readability
    }

    # Reject if too many outliers
    if total_steps > 0 and outlier_ratio > max_outlier_ratio:
        metadata["rejected"] = True
        metadata["reason"] = f"outlier_ratio {outlier_ratio:.2%} > {max_outlier_ratio:.2%}"
        return float("inf"), metadata

    metadata["rejected"] = False
    return total_distance, metadata


def calculate_direction_similarity(
    track1: Dict,
    track2: Dict,
    overlapping_frames: List[int],
    direction_threshold: float = 45.0,
    frame_stride: int = 3,
    min_movement_threshold: float = 0.5,
) -> Tuple[float, Dict]:
    """
    Calculate similarity of movement directions between two tracks.

    Args:
        track1, track2: Track dictionaries
        overlapping_frames: List of frames to compare
        direction_threshold: Angle threshold (degrees) for considering directions similar (default: 45°)
        frame_stride: Stride for sampling frames (1=consecutive, 3=every 3rd frame, etc.)
                     Higher stride = smoother direction, less noise-sensitive
                     At 29.97fps: stride=1 is ~0.033s, stride=3 is ~0.1s, stride=10 is ~0.33s
        min_movement_threshold: Minimum distance (units) to consider as valid movement (default: 0.5)

    Returns:
        Tuple of (similarity_score: float [0-1], metadata: dict)
    """
    if len(overlapping_frames) < frame_stride + 1:
        return 0.0, {
            "reason": "insufficient_frames",
            "valid_comparisons": 0,
            "frame_stride": frame_stride,
        }

    angle_diffs = []
    direction_matches = 0
    skipped_stationary = 0

    # Sample frames with stride
    for i in range(0, len(overlapping_frames) - frame_stride, frame_stride):
        frame_start = overlapping_frames[i]
        frame_end = overlapping_frames[i + frame_stride]

        # Get positions for both tracks
        pos1_start = get_position_at_specific_frame(track1, frame_start)
        pos1_end = get_position_at_specific_frame(track1, frame_end)
        pos2_start = get_position_at_specific_frame(track2, frame_start)
        pos2_end = get_position_at_specific_frame(track2, frame_end)

        if any(p is None for p in [pos1_start, pos1_end, pos2_start, pos2_end]):
            continue

        # Calculate movement vectors
        vec1 = pos1_end - pos1_start
        vec2 = pos2_end - pos2_start

        # Skip stationary movements (below threshold)
        dist1 = np.linalg.norm(vec1)
        dist2 = np.linalg.norm(vec2)
        if dist1 < min_movement_threshold or dist2 < min_movement_threshold:
            skipped_stationary += 1
            continue

        # Calculate angle between vectors using dot product (0-180°)
        angle_diff = calculate_angle(vec1, vec2)
        angle_diffs.append(angle_diff)

        # Count as direction match if within threshold
        if angle_diff < direction_threshold:
            direction_matches += 1

    if len(angle_diffs) < 3:  # Need at least 3 valid movements
        return 0.0, {
            "reason": "insufficient_valid_movements",
            "valid_comparisons": len(angle_diffs),
            "skipped_stationary": skipped_stationary,
            "frame_stride": frame_stride,
        }

    # Calculate metrics
    mean_angle_diff = np.mean(angle_diffs)
    median_angle_diff = np.median(angle_diffs)
    direction_consistency = direction_matches / len(angle_diffs)

    # Similarity score: 0° diff = 1.0, 180° diff = 0.0
    similarity = 1.0 - (mean_angle_diff / 180.0)

    metadata = {
        "valid_comparisons": len(angle_diffs),
        "mean_angle_diff": mean_angle_diff,
        "median_angle_diff": median_angle_diff,
        "direction_consistency": direction_consistency,
        "similar_directions": direction_matches,
        "angle_diff_std": np.std(angle_diffs),
        "skipped_stationary": skipped_stationary,
        "frame_stride": frame_stride,
        "angle_distribution": {
            "very_similar": sum(1 for d in angle_diffs if d < 30),  # 0-30°
            "similar": sum(1 for d in angle_diffs if 30 <= d < 60),  # 30-60°
            "moderate": sum(1 for d in angle_diffs if 60 <= d < 120),  # 60-120°
            "opposite": sum(1 for d in angle_diffs if d >= 120),  # 120-180°
        },
    }

    return similarity, metadata


def get_temporal_overlap(track1: Dict, track2: Dict) -> Tuple[Optional[int], Optional[int]]:
    """
    Get the temporal overlap range between two tracks.

    Returns:
        (overlap_start, overlap_end) or (None, None) if no overlap
    """
    frames1 = set(track1.get("frames", []))
    frames2 = set(track2.get("frames", []))
    overlap = frames1 & frames2

    if not overlap:
        return None, None

    return min(overlap), max(overlap)


def extract_overlapping_segment(
    track: Dict, overlap_start: int, overlap_end: int
) -> Tuple[List[int], np.ndarray]:
    """
    Extract frames and positions from a track within the overlap range.

    Returns:
        (frames, positions) as (List[int], np.ndarray)
    """
    frames = track.get("frames", [])
    projected = track.get("projected", [])

    segment_frames = []
    segment_positions = []

    for i, frame in enumerate(frames):
        if overlap_start <= frame <= overlap_end and i < len(projected):
            pos = projected[i]
            if pos is not None:
                segment_frames.append(frame)
                segment_positions.append(pos)

    return segment_frames, np.array(segment_positions)


def sample_continuous_segment(
    frames: List[int],
    positions: np.ndarray,
    segment_ratio: float,
    min_length: int,
) -> Tuple[List[int], np.ndarray]:
    """
    Sample a continuous segment from the frames/positions.

    Args:
        frames: List of frame numbers
        positions: Corresponding positions
        segment_ratio: Fraction of total length to sample (0-1)
        min_length: Minimum length of segment

    Returns:
        (sampled_frames, sampled_positions)
    """
    total_length = len(frames)
    segment_length = max(int(total_length * segment_ratio), min_length)
    segment_length = min(segment_length, total_length)

    if segment_length >= total_length:
        return frames, positions

    # Random start position
    max_start = total_length - segment_length
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + segment_length

    return frames[start_idx:end_idx], positions[start_idx:end_idx]


def sample_fixed_length_segment(
    frames: List[int],
    positions: np.ndarray,
    segment_length: int,
) -> Tuple[List[int], np.ndarray]:
    total_length = len(frames)
    if total_length <= segment_length:
        return frames, positions

    max_start = total_length - segment_length
    start_idx = random.randint(0, max_start)
    end_idx = start_idx + segment_length
    return frames[start_idx:end_idx], positions[start_idx:end_idx]


def compute_normalized_dtw_distance(
    track1: Dict,
    track2: Dict,
    use_sampling: bool = True,
    num_samples: int = 5,
    segment_ratio: float = 0.5,
    min_length: int = 10,
    segment_length: Optional[int] = 50,
    random_seed: Optional[int] = 42,
) -> Tuple[float, Dict]:
    """
    Compute normalized DTW distance using overlapping temporal region.
    Multiple samples are taken and median distance is returned.

    Args:
        track1, track2: Track dictionaries with 'frames' and 'projected'
        use_sampling: Whether to sample continuous segments
        num_samples: Number of samples to take (if use_sampling=True)
        segment_ratio: Fraction of overlap to sample
        min_length: Minimum points in sampled segment
        random_seed: Random seed for reproducibility

    Returns:
        (median_distance, metadata)
        Returns (inf, metadata) if no overlap
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Find temporal overlap
    overlap_start, overlap_end = get_temporal_overlap(track1, track2)

    if overlap_start is None:
        return float("inf"), {
            "rejected": True,
            "reason": "no_temporal_overlap",
            "has_overlap": False,
        }

    # Extract overlapping segments
    frames1, pos1 = extract_overlapping_segment(track1, overlap_start, overlap_end)
    frames2, pos2 = extract_overlapping_segment(track2, overlap_start, overlap_end)

    if len(pos1) < min_length or len(pos2) < min_length:
        return float("inf"), {
            "rejected": True,
            "reason": "insufficient_overlap_points",
            "has_overlap": True,
            "track1_points": len(pos1),
            "track2_points": len(pos2),
            "min_required": min_length,
        }
    distances = []
    used_lengths = []

    # Take multiple samples if requested
    n_iterations = num_samples if use_sampling else 1

    for sample_idx in range(n_iterations):
        # Sample continuous segments
        if use_sampling:
            if segment_length is not None:
                sample_frames1, sample_pos1 = sample_fixed_length_segment(
                    frames1, pos1, segment_length
                )
                sample_frames2, sample_pos2 = sample_fixed_length_segment(
                    frames2, pos2, segment_length
                )
            else:
                sample_frames1, sample_pos1 = sample_continuous_segment(
                    frames1, pos1, segment_ratio, min_length
                )
                sample_frames2, sample_pos2 = sample_continuous_segment(
                    frames2, pos2, segment_ratio, min_length
                )
        else:
            sample_pos1 = pos1
            sample_pos2 = pos2

        used_lengths.append((len(sample_pos1), len(sample_pos2)))

        # Normalize to remove translation and scale effects
        pos1_centered = sample_pos1 - sample_pos1.mean(axis=0)
        pos2_centered = sample_pos2 - sample_pos2.mean(axis=0)

        scale1 = np.std(pos1_centered)
        scale2 = np.std(pos2_centered)

        pos1_normalized = pos1_centered / scale1 if scale1 > 1e-6 else pos1_centered
        pos2_normalized = pos2_centered / scale2 if scale2 > 1e-6 else pos2_centered

        # Compute DTW
        distance, path = fastdtw(pos1_normalized, pos2_normalized, dist=euclidean)
        # print(f"Sample {sample_idx+1}: DTW raw distance = {distance}, path length = {len(path)}")
        # Normalize by path length
        normalized_distance = distance / len(path) if len(path) > 0 else distance
        distances.append(normalized_distance)

    # Compute statistics
    median_dist = float(np.median(distances))
    print(
        f"DTW distances between track {track1['track_id']} and track {track2['track_id']}: {distances}, median: {median_dist}"
    )
    metadata = {
        "rejected": False,
        "has_overlap": True,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "overlap_duration": overlap_end - overlap_start + 1,
        "track1_points": len(pos1),
        "track2_points": len(pos2),
        "num_samples": n_iterations,
        "segment_length": segment_length,
        "used_lengths": used_lengths[:5],
        "median_distance": median_dist,
        "mean_distance": float(np.mean(distances)),
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "std_distance": float(np.std(distances)),
        "all_distances": [float(d) for d in distances],
    }

    return median_dist, metadata


def calculate_match_score(
    track1: Dict,
    track2: Dict,
    cfg: MatchingConfig = MatchingConfig(),
) -> Tuple[float, Dict]:
    """
    Calculate a composite match score between two tracks.
    Lower score = better match. Returns inf if tracks are incompatible.

    Args:
        track1, track2: Track dictionaries
        min_overlap_frames: Minimum overlapping frames required
        max_analysis_frames: Maximum frames to analyze
        direction_frame_stride: Frame stride for direction calculation
        max_point_distance: Max allowed distance between corresponding points (outlier threshold for pointwise distance)
        max_step_distance: Max allowed distance per movement step (outlier threshold for total distance)
        max_outlier_ratio: If more than this ratio are outliers, reject match

    Returns:
        Tuple of (score: float, metadata: dict)
    """
    # Find overlapping frames
    overlapping_frames = find_overlapping_frames(track1, track2)

    if len(overlapping_frames) > cfg.max_analysis_frames:
        overlapping_frames = overlapping_frames[:cfg.max_analysis_frames]

    metadata = {
        "overlap_count": len(overlapping_frames),
        "track1_id": track1["track_id"],
        "track2_id": track2["track_id"],
    }

    # No overlap = infinite score (incompatible)
    if len(overlapping_frames) < cfg.min_overlap_frames:
        metadata["reason"] = "insufficient_overlap"
        # print(f"Insufficient overlap between track {track1['track_id']} and track {track2['track_id']}: {len(overlapping_frames)} frames")
        return float("inf"), metadata

    # Fast rejection: initial position check
    # Check first 13 overlapping frames (or fewer if overlap is small)
    check_frames = overlapping_frames[: min(13, len(overlapping_frames))]
    initial_distances = []

    for frame in check_frames:
        pos1 = get_position_at_specific_frame(track1, frame)
        pos2 = get_position_at_specific_frame(track2, frame)

        if pos1 is not None and pos2 is not None:
            dist = np.linalg.norm(pos1 - pos2)
            initial_distances.append(dist)

    if initial_distances:
        # Use median of first few frames for robustness
        median_initial_distance = np.median(initial_distances)

        if median_initial_distance >= cfg.max_initial_distance:
            metadata["reason"] = "initial_overlap_positions_too_far"
            metadata["median_initial_distance"] = median_initial_distance
            metadata["initial_distances"] = initial_distances
            metadata["checked_frames"] = check_frames
            return float("inf"), metadata

    # Calculate median distance
    median_distance, median_meta = calculate_median_distance(
        track1,
        track2,
        overlapping_frames,
        max_point_distance=cfg.max_point_distance,
        max_outlier_ratio=cfg.max_outlier_ratio,
    )

    # # Calculate total distances of track 1
    # total_distance_track1, td1_meta = calculate_total_distance(
    #     track1,
    #     overlapping_frames,
    #     max_step_distance=max_step_distance,
    #     max_outlier_ratio=max_outlier_ratio,
    # )

    # # Check if either total distance was rejected due to outliers
    # if total_distance_track1 == float("inf"):
    #     metadata["reason"] = "track1_total_distance_rejected"
    #     metadata["track1_distance_metadata"] = td1_meta
    #     # print(f"Track {track1['track_id']} total distance rejected due to outliers: {td1_meta}")
    #     return float("inf"), metadata

    # # Calculate total distances of track 2
    # total_distance_track2, td2_meta = calculate_total_distance(
    #     track2,
    #     overlapping_frames,
    #     max_step_distance=max_step_distance,
    #     max_outlier_ratio=max_outlier_ratio,
    # )

    # if total_distance_track2 == float("inf"):
    #     metadata["reason"] = "track2_total_distance_rejected"
    #     metadata["track2_distance_metadata"] = td2_meta
    #     # print(f"Track {track2['track_id']} total distance rejected due to outliers: {td2_meta}")
    #     return float("inf"), metadata

    # distance_diff = abs(total_distance_track1 - total_distance_track2)

    velocity_diff = calculate_median_velocity_difference(
        track1, track2, overlapping_frames, fps=30, window_size=5
    )

    direction_sim, direction_meta = calculate_direction_similarity(
        track1,
        track2,
        overlapping_frames,
        direction_threshold=cfg.direction_threshold,
        frame_stride=cfg.direction_frame_stride,
    )
    metadata.update(
        {
            "median_distance": median_distance,
            "median_metadata": median_meta,
            # "distance_diff": distance_diff,
            # "total_distance_track1": total_distance_track1,
            # "total_distance_track2": total_distance_track2,
            # "track1_outliers": td1_meta.get("n_outlier_steps", 0),
            # "track2_outliers": td2_meta.get("n_outlier_steps", 0),
            "point_outliers": median_meta.get("n_outliers", 0),
            "velocity_difference": velocity_diff,
            "direction_similarity": direction_sim,
            "direction_metadata": direction_meta,
        }
    )

    # Composite score: weighted combination (lower = better)
    # Normalize each component to roughly similar scales (this works, but can be tuned)

    # Median distance
    distance_score = median_distance / 10.0

    # # Distance diff
    # distance_diff_score = distance_diff / 10.0

    # Direction: 0-1 (higher is better), invert for score
    direction_score = (1.0 - direction_sim) * 10.0

    # Overlap bonus: more overlap = lower score
    overlap_bonus = -min(len(overlapping_frames) / 500.0, 5.0)  # -5 to 0

    # Velocity diff penalty
    velocity_score = velocity_diff / 20.0

    # Composite score
    score = (
        distance_score * 2.0  # Weight: 2x
        # + distance_diff_score * 1.5  # Weight: 1.5x
        + direction_score * 2.5  # Weight: 2.5x
        + velocity_score * 0.5  # Weight: 0.5x
        + overlap_bonus  # Bonus for overlap
    )

    print(
        f"Calculated match score between track {track1['track_id']} and track {track2['track_id']}: {score:.2f}"
    )
    print(
        f"  Components: distance_score={distance_score:.2f}, "
        #    f"distance_diff_score={distance_diff_score:.2f}, "
        f"direction_score={direction_score:.2f}, "
        f"velocity_score={velocity_score:.2f}, "
        f"overlap_bonus={overlap_bonus:.2f}"
    )

    metadata["composite_score"] = score
    metadata["score_components"] = {
        "distance_score": distance_score,
        # "distance_diff_score": distance_diff_score,
        "direction_score": direction_score,
        "velocity_score": velocity_score,
        "overlap_bonus": overlap_bonus,
    }

    return score, metadata

def prepare_tracks(
    jsonl_paths: List[str],
    frame_offsets: List[int] = None,
    coord_offsets: List[np.ndarray] = None,
    ball_jsonl_paths: List[str] = None,
    cfg: CalibrationConfig = None,
) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
    """
    Load and prepare player tracks from multiple JSONL files.
    Applies ball track calibration if offsets are provided or auto-calibration is enabled.

    Args:
        jsonl_paths: List of JSONL file paths containing player tracks
        frame_offsets: Optional list of frame offsets for calibration
        coord_offsets: Optional list of coordinate offsets for calibration
        ball_jsonl_paths: Optional list of JSONL file paths containing ball tracks
        cfg: CalibrationConfig object for calibration settings
    
    Returns:
        all_tracks: List of all player tracks with metadata
        all_tracks_by_file: Dict mapping source index to list of tracks
    """

    # Apply ball calibration if offsets provided or auto-calibrate enabled
    if frame_offsets is not None or coord_offsets is not None or cfg.auto_calibrate:
        if ball_jsonl_paths is None and cfg.auto_calibrate:
            print("⚠️  Warning: auto_calibrate=True but no ball_jsonl_paths provided")
            print("    Loading player tracks without calibration...")
            all_tracks_by_file = load_player_tracks(jsonl_paths)
        else:
            print("🔧 Applying ball track calibration to player tracks...")
            all_tracks_by_file, frame_offsets, coord_offsets = apply_calibration_to_player_tracks(
                player_jsonl_paths=jsonl_paths,
                ball_jsonl_paths=ball_jsonl_paths or jsonl_paths,  # Fallback
                frame_offsets=frame_offsets,
                coord_offsets=coord_offsets,
                cfg=cfg
            )
    else:
        # Load tracks normally without calibration
        print("📥 Loading player tracks without calibration...")
        all_tracks_by_file = load_player_tracks(jsonl_paths)
    
    # Print summary of loaded tracks
    for src_idx, tracks in all_tracks_by_file.items():
        print(f"  File {src_idx} ({jsonl_paths[src_idx]}): {len(tracks)} tracks loaded")

    # Build all_tracks list with metadata
    all_tracks = []
    for src_idx, tracks in all_tracks_by_file.items():
        for orig_idx, t in enumerate(tracks):
            t["_source_idx"] = src_idx
            t["_orig_idx"] = orig_idx
            t["_global_idx"] = len(all_tracks)
            all_tracks.append(t)

    n_total = len(all_tracks)
    n_files = len(jsonl_paths)
    print(f"Total tracks: {n_total} from {n_files} files")

    return all_tracks, all_tracks_by_file

def greedy_match_tracks(
    all_tracks: List[Dict],
    jsonl_paths: List[str],
    filter_by_team: bool = False,
    filter_by_jersey: bool = False,
    max_score_threshold: float = 50.0,
    temporal_overlap_threshold: float = 0.3,
    cfg: MatchingConfig = MatchingConfig(),
    use_dtw_filter: bool = True,
    dtw_threshold: float = 0.3,
    dtw_num_samples: int = 5,
    dtw_segment_ratio: float = 0.5,
    dtw_min_length: int = 10,
    dtw_segment_length: int = 50,
    save_pairwise_path: str = None,
    verbose: bool = False,
) -> Tuple[Dict, Dict[int, List[Dict]]]:
    """
    Greedy matching: tracks find best matches across multiple files.
    Supports many-to-one matching (multiple short tracks can match one long track)
    as long as the short tracks don't overlap temporally.

    Args:
        all_tracks: List of all track dictionaries
        jsonl_paths: List of JSONL file paths (supports 2+ files)
        min_overlap_frames: Minimum overlapping frames required
        max_analysis_frames: Maximum frames to analyze per comparison
        filter_by_team: Only compare tracks from same team
        filter_by_jersey: Only compare tracks with same jersey number
        direction_frame_stride: Frame stride for direction calculation
        max_score_threshold: Maximum score to accept a match
        temporal_overlap_threshold: Maximum allowed overlap ratio (0-1) between
                                    tracks from same source matching same target
        max_initial_distance: Max initial distance for fast rejection
        max_point_distance: Max allowed distance between corresponding points
                                    (outlier threshold for pointwise distance)
        max_step_distance: Max allowed distance per movement step
                                    (outlier threshold for total distance)
        max_outlier_ratio: If more than this ratio are outliers, reject match
        use_dtw_filter: Whether to apply DTW distance check (default: True)
        dtw_threshold: Maximum DTW distance to accept (default: 0.3)
        dtw_num_samples: Number of samples for DTW (default: 5)
        dtw_segment_ratio: Segment ratio for DTW sampling (default: 0.5)
        dtw_min_length: Minimum segment length for DTW (default: 10)
        verbose: Print detailed progress

    Returns:
        Tuple of (results dict, all_tracks_by_file dict)
    """

    n_total = len(all_tracks)
    n_files = len(jsonl_paths)
    print(f"Total tracks: {n_total} from {n_files} files")

    # Stage 1: Compute all pairwise scores
    print("\n📊 Stage 1/3: Computing pairwise match scores...")
    all_pairs = []
    total_pairs = 0
    skipped_filter = 0
    skipped_incompatible = 0

    for i in tqdm(range(n_total), desc="Tracks processed", unit="track"):
        t1 = all_tracks[i]

        for j in range(i + 1, n_total):
            t2 = all_tracks[j]

            # Skip same file
            if t1["_source_idx"] == t2["_source_idx"]:
                continue

            total_pairs += 1

            # Apply filters
            if filter_by_team and t1.get("team") != t2.get("team"):
                skipped_filter += 1
                continue
            if filter_by_jersey and t1.get("jersey_num") != t2.get("jersey_num"):
                skipped_filter += 1
                continue

            # Calculate score
            score, metadata = calculate_match_score(
                t1,
                t2,
                cfg=cfg,
            )

            if score == float("inf"):

                skipped_incompatible += 1

            all_pairs.append((score, i, j, metadata))

    # Stage 2: Compute DTW scores for valid pairs
    if use_dtw_filter:
        print(f"\n🔍 Stage 2/3: Computing DTW scores for valid pairs...")
        valid_pairs_for_dtw = [
            (score, track1_idx, track2_idx, metadata)
            for score, track1_idx, track2_idx, metadata in all_pairs
            if score != float("inf")
        ]

        dtw_results = {}  # (i, j) -> (dtw_distance, dtw_metadata)
        skipped_dtw = 0

        for score, i, j, metadata in tqdm(valid_pairs_for_dtw, desc="DTW computation"):
            t1 = all_tracks[i]
            t2 = all_tracks[j]

            dtw_dist, dtw_meta = compute_normalized_dtw_distance(
                t1,
                t2,
                use_sampling=True,
                num_samples=dtw_num_samples,
                segment_ratio=dtw_segment_ratio,
                segment_length=dtw_segment_length,
                min_length=dtw_min_length,
                random_seed=42,
            )

            dtw_results[(i, j)] = (dtw_dist, dtw_meta)

            # Count rejections
            if dtw_dist > dtw_threshold or dtw_dist == float("inf"):
                skipped_dtw += 1

        print(f"  Valid pairs: {len(valid_pairs_for_dtw)}")
        print(f"  DTW threshold: {dtw_threshold}")
        print(f"  Rejected by DTW: {skipped_dtw}")
        print(f"  Passing DTW filter: {len(valid_pairs_for_dtw) - skipped_dtw}")

        # Add DTW info to metadata
        for score, i, j, metadata in all_pairs:
            if (i, j) in dtw_results:
                dtw_dist, dtw_meta = dtw_results[(i, j)]
                metadata["dtw_distance"] = dtw_dist
                metadata["dtw_metadata"] = dtw_meta
                metadata["passes_dtw_filter"] = dtw_dist <= dtw_threshold
            else:
                metadata["dtw_distance"] = None
                metadata["dtw_metadata"] = None
                metadata["passes_dtw_filter"] = False
    # ⭐ NEW: Save pairwise scores if requested
    if save_pairwise_path:
        save_pairwise_scores(all_pairs, all_tracks, save_pairwise_path, verbose=verbose)

    # Sort by score (best first) - FILTER OUT inf scores for matching
    # ⭐ Filter valid pairs: must pass both composite score AND DTW (if enabled)
    if use_dtw_filter:
        valid_pairs = [
            (score, track1_idx, track2_idx, metadata)
            for score, track1_idx, track2_idx, metadata in all_pairs
            if score != float("inf") and metadata.get("passes_dtw_filter", False)
        ]
        print(f"\n✅ Final valid pairs after both filters: {len(valid_pairs)}")
    else:
        valid_pairs = [
            (score, track1_idx, track2_idx, metadata)
            for score, track1_idx, track2_idx, metadata in all_pairs
            if score != float("inf")
        ]
    valid_pairs.sort(key=lambda x: x[0])

    print(f"  Skipped (filter): {skipped_filter}")
    print(f"  Skipped (incompatible): {skipped_incompatible}")
    if use_dtw_filter:
        print(f"  Skipped (DTW filter): {skipped_dtw}")

    if verbose and valid_pairs:
        print(f"\n  Top 5 best pairs:")
        for score, track1_idx, track2_idx, _ in valid_pairs[:5]:
            print(
                f"    {all_tracks[track1_idx]['track_id']} <-> {all_tracks[track2_idx]['track_id']}: score={score:.2f}"
            )

    # Track which tracks from each source have matched to a target track
    # matched_to_target[target_idx][source_idx] = list of (matched_idx, frame_start, frame_end)
    matched_to_target = defaultdict(lambda: defaultdict(list))

    # Stage 3: Greedy matching with many-to-one support
    print("🎯 Stage 3/3: Performing greedy matching...")

    matches = []
    groups = defaultdict(set)
    track_to_group = {}
    next_group_id = 0

    def get_match_frame_range(idx1: int, idx2: int) -> Tuple[int, int]:
        """Get the overlapping frame range for a match."""
        t1 = all_tracks[idx1]
        t2 = all_tracks[idx2]

        frames1 = set(t1.get("frames", []))
        frames2 = set(t2.get("frames", []))
        overlap = sorted(frames1 & frames2)

        if overlap:
            return (overlap[0], overlap[-1])

        range1 = t1.get("frame_range", [0, 0])
        range2 = t2.get("frame_range", [0, 0])
        start = max(range1[0], range2[0])
        end = min(range1[1], range2[1])
        return (start, end)

    def get_all_group_members(idx: int) -> Set[int]:
        """Get all tracks in the same group as idx (BFS)."""
        if idx not in track_to_group:
            return {idx}
        group_id = track_to_group[idx]
        return groups[group_id].copy()

    def would_create_source_conflict(
        idx1: int, idx2: int, match_start: int, match_end: int
    ) -> bool:
        """
        Check if matching idx1 and idx2 would create a conflict where:
        - Two tracks from the SAME source end up in the same group
        - With overlapping time ranges

        This checks the ENTIRE connected component, not just direct neighbors.
        """
        # Get all members in both groups (or singleton sets if not yet grouped)
        members1 = get_all_group_members(idx1)
        members2 = get_all_group_members(idx2)

        # The new group would be the union
        future_group = members1 | members2 | {idx1, idx2}

        # Group tracks by source
        tracks_by_source = defaultdict(list)
        for idx in future_group:
            src = all_tracks[idx]["_source_idx"]
            frame_range = all_tracks[idx].get("frame_range", [0, 0])
            tracks_by_source[src].append((idx, frame_range[0], frame_range[1]))

        # Check each source for temporal conflicts
        for src, track_list in tracks_by_source.items():
            if len(track_list) <= 1:
                continue

            # Check all pairs within this source
            for i in range(len(track_list)):
                idx_i, start_i, end_i = track_list[i]
                for j in range(i + 1, len(track_list)):
                    idx_j, start_j, end_j = track_list[j]

                    # Check temporal overlap
                    overlap_start = max(start_i, start_j)
                    overlap_end = min(end_i, end_j)

                    if overlap_start <= overlap_end:
                        # Calculate overlap ratio relative to shorter segment
                        duration_i = end_i - start_i + 1
                        duration_j = end_j - start_j + 1
                        overlap_duration = overlap_end - overlap_start + 1

                        overlap_ratio_i = overlap_duration / duration_i if duration_i > 0 else 1.0
                        overlap_ratio_j = overlap_duration / duration_j if duration_j > 0 else 1.0
                        max_overlap_ratio = max(overlap_ratio_i, overlap_ratio_j)

                        if max_overlap_ratio > temporal_overlap_threshold:
                            if verbose:
                                print(
                                    f"    ⚠️  Would create conflict: {all_tracks[idx_i]['track_id']} "
                                    f"and {all_tracks[idx_j]['track_id']} (both from source {src}) "
                                    f"overlap {max_overlap_ratio:.1%} in frames [{overlap_start}, {overlap_end}]"
                                )
                            return True

        return False

    for score, idx1, idx2, metadata in valid_pairs:
        if score > max_score_threshold:
            continue

        match_start, match_end = get_match_frame_range(idx1, idx2)

        # NEW: Check if this match would create source conflicts in the merged group
        if would_create_source_conflict(idx1, idx2, match_start, match_end):
            if verbose:
                print(
                    f"  Skipping {all_tracks[idx1]['track_id']} <-> {all_tracks[idx2]['track_id']}: "
                    f"would create source conflict in merged group"
                )
            continue

        # Accept this match - record in both directions
        src1 = all_tracks[idx1]["_source_idx"]
        src2 = all_tracks[idx2]["_source_idx"]

        matched_to_target[idx1][src2].append((idx2, match_start, match_end))
        matched_to_target[idx2][src1].append((idx1, match_start, match_end))

        matches.append(
            {
                "i": idx1,
                "j": idx2,
                "score": score,
                "metadata": metadata,
                "frame_range": [match_start, match_end],
            }
        )

        if verbose:
            print(
                f"  ✓ Matched {all_tracks[idx1]['track_id']} <-> {all_tracks[idx2]['track_id']}: "
                f"score={score:.2f}, frames {match_start}-{match_end}"
            )

        # Update groups (union-find)
        group1 = track_to_group.get(idx1)
        group2 = track_to_group.get(idx2)

        if group1 is None and group2 is None:
            groups[next_group_id] = {idx1, idx2}
            track_to_group[idx1] = next_group_id
            track_to_group[idx2] = next_group_id
            next_group_id += 1
        elif group1 is not None and group2 is None:
            groups[group1].add(idx2)
            track_to_group[idx2] = group1
        elif group1 is None and group2 is not None:
            groups[group2].add(idx1)
            track_to_group[idx1] = group2
        elif group1 != group2:
            groups[group1].update(groups[group2])
            for idx in groups[group2]:
                track_to_group[idx] = group1
            del groups[group2]

    print(f"  Matches found: {len(matches)}")
    print(f"  Groups formed: {len(groups)}")

    # Build aggregated results
    aggregated = []
    matched_indices = set()

    for group_id, member_indices in groups.items():
        matched_indices.update(member_indices)

        track_ids = [all_tracks[i]["track_id"] for i in member_indices]
        source_indices = [all_tracks[i]["_source_idx"] for i in member_indices]
        teams = [all_tracks[i].get("team") for i in member_indices]
        jerseys = [all_tracks[i].get("jersey_num") for i in member_indices]
        frame_ranges = [all_tracks[i].get("frame_range", [0, 0]) for i in member_indices]

        # Sort by source_idx first, then by track_id
        # Create list of tuples for sorting
        combined = list(
            zip(source_indices, track_ids, teams, jerseys, frame_ranges, member_indices)
        )

        # Sort by source_idx (primary), then track_id (secondary)
        # Extract numeric part from track_id for sorting (e.g., '12a' -> 12)
        def extract_numeric(track_id):
            """Extract numeric part from track_id like '12a' -> 12"""

            match = re.match(r"(\d+)", str(track_id))
            return int(match.group(1)) if match else 0

        combined.sort(key=lambda x: (x[0], extract_numeric(x[1]), x[1]))

        # Unzip back to separate lists
        source_indices, track_ids, teams, jerseys, frame_ranges, member_indices = zip(*combined)
        source_indices = list(source_indices)
        track_ids = list(track_ids)
        teams = list(teams)
        jerseys = list(jerseys)
        frame_ranges = list(frame_ranges)
        member_list = list(member_indices)

        pairwise_data = {
            "median_distance": [],
            "distance_diff": [],
            "direction_similarity": [],
            "scores": [],
            "pairs": [],
            "frame_ranges": [],
        }

        for a in range(len(member_list)):
            for b in range(a + 1, len(member_list)):
                ia, ib = member_list[a], member_list[b]
                for m in matches:
                    if (m["i"] == ia and m["j"] == ib) or (m["i"] == ib and m["j"] == ia):
                        pairwise_data["median_distance"].append(
                            m["metadata"].get("median_distance")
                        )
                        pairwise_data["distance_diff"].append(m["metadata"].get("distance_diff"))
                        pairwise_data["direction_similarity"].append(
                            m["metadata"].get("direction_similarity")
                        )
                        pairwise_data["scores"].append(m["score"])
                        pairwise_data["pairs"].append(
                            (all_tracks[ia]["track_id"], all_tracks[ib]["track_id"])
                        )
                        pairwise_data["frame_ranges"].append(m.get("frame_range"))

        aggregated.append(
            {
                "track_id": track_ids,
                "source_idx": source_indices,
                "frame_ranges": frame_ranges,  # Now included directly
                "team": teams,
                "jersey_num": jerseys,
                "score(s)": (pairwise_data["scores"] if pairwise_data["scores"] else None),
                "median_distance": pairwise_data["median_distance"],
                "distance_diff": pairwise_data["distance_diff"],
                "direction_similarity": pairwise_data["direction_similarity"],
                "pairs": pairwise_data["pairs"],
                "match_frame_ranges": pairwise_data[
                    "frame_ranges"
                ],  # Renamed to distinguish from track frame_ranges
            }
        )

    # Find unmatched tracks
    unmatched = []
    for i in range(n_total):
        if i not in matched_indices:
            t = all_tracks[i]
            unmatched.append(
                {
                    "track_id": t["track_id"],
                    "team": t.get("team"),
                    "jersey_num": t.get("jersey_num"),
                    "source_idx": t["_source_idx"],
                    "frame_range": t.get("frame_range"),
                }
            )

    # Count many-to-one matches
    many_to_one_count = sum(
        1
        for target_idx in matched_to_target
        for src_idx in matched_to_target[target_idx]
        if len(matched_to_target[target_idx][src_idx]) > 1
    )

    results = {
        "aggregated_groups": aggregated,
        "pairwise_matches": matches,
        "unmatched_tracks": unmatched,
        "stats": {
            "total_tracks": n_total,
            "total_files": n_files,
            "total_candidate_pairs": len(valid_pairs),
            "matches_accepted": len(matches),
            "groups_count": len(groups),
            "matched_tracks": len(matched_indices),
            "unmatched_tracks": len(unmatched),
            "match_rate": len(matched_indices) / n_total if n_total > 0 else 0,
            "many_to_one_matches": many_to_one_count,
        },
    }

    return results


def load_tracks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load all tracks from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        List of track dictionaries
    """
    tracks = []
    with open(jsonl_path, "r") as f:
        for line in f:
            track = json.loads(line.strip())
            tracks.append(track)
    return tracks


def calculate_weighted_average_position(
    positions: List[np.ndarray], weights: List[float]
) -> np.ndarray:
    """
    Calculate weighted average of positions using bbox_area as weights.

    Args:
        positions: List of [x, y] positions
        weights: List of bbox_area values (will be normalized)

    Returns:
        Weighted average position as np.ndarray
    """
    total_weight = sum(weights)
    if total_weight == 0:
        return np.mean(positions, axis=0)

    normalized_weights = [w / total_weight for w in weights]
    weighted_pos = np.zeros(2)
    for pos, w in zip(positions, normalized_weights):
        weighted_pos += np.array(pos) * w

    return weighted_pos


# different from assign_team_by_majority_vote() from post_processing.py
def majority_vote(
    values: List,
    confidences: List[float] = None,
    counts: List = None,
    ignore_values: List = None,
    vote: str = "team",
) -> Optional:
    """
    Return the most common value, ignoring specified values.

    Args:
        values: List of values to vote on
        confidences: List of confidence scores for each value
        counts: List of counts for each value (optional, used only for jersey)
        ignore_values: Values to ignore (e.g., "unsure", list types)

    Returns:
        Most common value or None if no valid values
    """
    if ignore_values is None:
        ignore_values = ["unsure", None]

    filtered = []
    filtered_confidences = []
    filtered_counts = []

    for idx, v in enumerate(values):
        confs = confidences[idx] if confidences else 0.5
        cnt = counts[idx] if counts and idx < len(counts) else 1

        if v in ignore_values:
            continue
        elif isinstance(v, list):
            # Flatten list and add non-ignored values
            if isinstance(confs, list):
                conf_list = confs
            else:
                conf_list = [confs] * len(v)

            if isinstance(cnt, list):
                cnt_list = cnt
            else:
                cnt_list = [cnt] * len(v)

            for item, conf, count in zip(v, conf_list, cnt_list):
                if item not in ignore_values:
                    filtered.append(item)
                    filtered_confidences.append(conf)
                    filtered_counts.append(count)
        else:
            filtered.append(v)
            filtered_confidences.append(confs if not isinstance(confs, list) else confs[0])
            filtered_counts.append(cnt if not isinstance(cnt, list) else cnt[0])

    if not filtered:
        if vote == "team":
            return "unsure", 0.0, 1  # placeholder count for team
        elif vote == "jersey":
            return ["unsure"], [0.0], [1]  # placeholder count for jersey

    # ===== TEAM VOTING =====
    if vote == "team":
        # Group by team and calculate average confidence
        team_data = defaultdict(lambda: {"confs": []})

        for team, conf in zip(filtered, filtered_confidences):
            team_data[team]["confs"].append(conf)

        # Calculate metrics for each team
        team_results = []
        for team, data in team_data.items():
            avg_conf = np.mean(data["confs"])
            occurrence_count = len(data["confs"])  # Count based on occurrences, not "count" field
            team_results.append((team, avg_conf, occurrence_count))

        # Sort by occurrence count (primary), then confidence (secondary)
        team_results.sort(key=lambda x: (x[2], x[1]), reverse=True)

        if team_results:
            return team_results[0]  # (team, avg_conf, occurrence_count) - placeholder count
        else:
            return "unsure", 0.0, 1  # placeholder count

    # ===== JERSEY VOTING =====
    elif vote == "jersey":
        # Group by jersey number and sum counts
        jersey_data = defaultdict(lambda: {"confs": [], "counts": []})

        for jersey, conf, count in zip(filtered, filtered_confidences, filtered_counts):
            jersey_data[jersey]["confs"].append(conf)
            jersey_data[jersey]["counts"].append(count)

        # Calculate metrics for each jersey
        jersey_results = []
        for jersey_num, data in jersey_data.items():
            avg_conf = np.mean(data["confs"])
            total_count = sum(data["counts"])
            jersey_results.append((jersey_num, avg_conf, total_count))

        # Sort by total count (primary), then confidence (secondary)
        jersey_results.sort(key=lambda x: (x[2], x[1]), reverse=True)

        # Separate into three lists
        jersey_nums = [jersey for jersey, _, _ in jersey_results]
        jersey_confs = [conf for _, conf, _ in jersey_results]
        jersey_counts = [count for _, _, count in jersey_results]

        return jersey_nums, jersey_confs, jersey_counts

    else:
        raise ValueError(f"Unknown vote type: {vote}")


def fuse_matched_tracks(
    matched_group: Dict, all_tracks_by_file: Dict[int, List[Dict]], fused_id: int
) -> Dict:
    """
    Fuse multiple matched tracks into a single track.

    Args:
        matched_group: Dict with 'track_id', 'source_idx', etc.
        all_tracks_by_file: Dict mapping source_idx to list of tracks
        fused_id: Integer ID for the fused track

    Returns:
        Fused track dictionary
    """
    # Collect all tracks in this group
    tracks = []
    for track_id, src_idx in zip(matched_group["track_id"], matched_group["source_idx"]):
        for t in all_tracks_by_file[src_idx]:
            if t["track_id"] == track_id:
                tracks.append(t)
                break

    if not tracks:
        return None

    # 1. Majority vote on team and jersey number
    teams = [t.get("team") for t in tracks]
    team_confs = [t.get("team_conf", 0.5) for t in tracks]
    jersey_nums = [t.get("jersey_num") for t in tracks]
    jersey_confs = [t.get("jersey_conf", 0.5) for t in tracks]
    jersey_counts = [t.get("count", 1) for t in tracks]

    fused_team, fused_team_conf, _ = majority_vote(
        teams, team_confs, counts=None, ignore_values=["unsure", None, ""], vote="team"
    )

    fused_jersey_nums, fused_jersey_confs, fused_jersey_counts = majority_vote(
        jersey_nums, jersey_confs, jersey_counts, ignore_values=["unsure", None, ""], vote="jersey"
    )

    # 2. Track ID
    fused_track_id = f"{fused_id}_fused"

    # 3. Frame range - union of all tracks
    all_frames = set()
    for t in tracks:
        all_frames.update(t.get("frames", []))

    all_frames = sorted(all_frames)
    frame_range = [min(all_frames), max(all_frames)] if all_frames else [0, 0]

    # 4. Weighted average of projected positions per frame
    # Build frame -> [(position, bbox_area, track_id)] mapping
    frame_data = defaultdict(list)
    for t in tracks:
        frames = t.get("frames", [])
        projected = t.get("projected", [])
        bbox_areas = t.get("bbox_area", [])

        for i, frame in enumerate(frames):
            if i < len(projected) and i < len(bbox_areas):
                frame_data[frame].append(
                    {
                        "position": np.array(projected[i]),
                        "bbox_area": bbox_areas[i],
                        "track_id": t["track_id"],
                    }
                )

    # Compute weighted average for each frame
    fused_frames = []
    fused_projected = []
    fused_bbox_area = []

    for frame in sorted(frame_data.keys()):
        data_list = frame_data[frame]

        positions = [d["position"] for d in data_list]
        weights = [d["bbox_area"] for d in data_list]

        # Weighted average position
        avg_pos = calculate_weighted_average_position(positions, weights)

        # Sum bbox areas (or use max/mean based on preference)
        avg_bbox = sum(weights) / len(weights)  # Mean bbox area

        fused_frames.append(frame)
        fused_projected.append(avg_pos.tolist())
        fused_bbox_area.append(avg_bbox)

    # 5. Confidence merging - weighted average by track duration
    durations = []
    team_confs_for_merge = []

    for t in tracks:
        duration = len(t.get("frames", [1]))
        durations.append(duration)
        team_confs_for_merge.append(t.get("team_conf", 0.5))

    total_duration = sum(durations)
    if total_duration > 0:
        final_team_conf = (
            sum(tc * d for tc, d in zip(team_confs_for_merge, durations)) / total_duration
        )
    else:
        final_team_conf = np.mean(team_confs_for_merge) if team_confs_for_merge else 0.5

    # Build fused track
    fused_track = {
        "track_id": fused_track_id,
        "team": fused_team,
        "jersey_num": fused_jersey_nums,
        "jersey_conf": fused_jersey_confs,
        "count": fused_jersey_counts,
        "team_conf": final_team_conf,
        "frame_range": frame_range,
        "frames": fused_frames,
        "projected": fused_projected,
        "bbox_area": fused_bbox_area,
        "source_track": matched_group["track_id"],
        "source_idx": matched_group["source_idx"],
        "is_fused": True,
        "is_spatial_merged": False,
    }

    return fused_track


def fuse_all_matched_groups(
    matched_groups: List[Dict],
    unmatched_groups: List[Dict],
    all_tracks_by_file: Dict[int, List[Dict]],
    spatial_distance_threshold: float = 50.0,
) -> List[Dict]:
    """
    Fuse all matched groups into final tracks.

    Args:
        matched_groups: List of matched group dicts from greedy_match_tracks
        all_tracks_by_file: Dict mapping source_idx to list of tracks
        include_singletons: If True, include single-track groups as-is
        spatial_distance_threshold: Threshold for spatial distance to consider tracks as matched

    Returns:
        List of fused track dictionaries
    """
    fused_tracks = []
    fused_id = 1

    # Sort the matched groups by start frame of earliest track
    def group_start_frame(group: Dict) -> int:
        start_frames = []
        for track_id, src_idx in zip(group["track_id"], group["source_idx"]):
            for t in all_tracks_by_file[src_idx]:
                if t["track_id"] == track_id:
                    frange = t.get("frame_range", [0, 0])
                    start_frames.append(frange[0])
                    break
        return min(start_frames) if start_frames else float("inf")

    matched_groups.sort(key=group_start_frame)

    print(f"\n🔀 Fusing {len(matched_groups)} matched groups...")

    for group in matched_groups:
        n_tracks = len(group["track_id"])

        if n_tracks == 0:
            continue
        else:
            # Multiple tracks - fuse them
            fused_track = fuse_matched_tracks(group, all_tracks_by_file, fused_id)
            if fused_track:
                fused_tracks.append(fused_track)

        fused_id += 1

    print(f"  Created {len(fused_tracks)} fused tracks from matched groups")

    # Add unmatched tracks as singletons BEFORE spatial clustering
    print(f"\n➕ Adding {len(unmatched_groups)} unmatched tracks...")
    for unmatched in unmatched_groups:
        track_id = unmatched["track_id"]
        src_idx = unmatched["source_idx"]

        for t in all_tracks_by_file[src_idx]:
            if t["track_id"] == track_id:
                singleton_track = t.copy()
                singleton_track["track_id"] = f"{fused_id}_unmatched"
                singleton_track["source_track"] = [track_id]
                singleton_track["source_idx"] = [src_idx]
                singleton_track["is_fused"] = False
                fused_tracks.append(singleton_track)
                fused_id += 1
                break

    tracks_before_spatial = len(fused_tracks)
    print(f"  Total tracks before spatial clustering: {tracks_before_spatial}")

    # Apply spatial clustering
    print(f"\n🔗 Applying spatial clustering (threshold: {spatial_distance_threshold} units)...")
    fused_tracks = spatial_cluster_tracks(
        fused_tracks,
        distance_threshold=spatial_distance_threshold,
        overlap_threshold=0.5,
    )
    tracks_after_spatial = len(fused_tracks)
    print(f"  Final track count after spatial clustering: {tracks_after_spatial}")

    # Summary statistics
    multi_source = sum(1 for t in fused_tracks if t.get("is_fused", False))
    spatial_merged = sum(1 for t in fused_tracks if t.get("is_spatial_merged", False))
    single_source = len(fused_tracks) - multi_source

    spatial_stats = {
        "tracks_before_spatial": tracks_before_spatial,
        "tracks_after_spatial": tracks_after_spatial,
        "tracks_merged_by_spatial": tracks_before_spatial - tracks_after_spatial,
        "multi_source_fused": multi_source,
        "spatial_merged": spatial_merged,
        "single_source": single_source,
        "spatial_distance_threshold": spatial_distance_threshold,
    }

    print(f"\n📊 Final Track Breakdown:")
    print(f"  Multi-source fused: {multi_source}")
    print(f"  Spatially merged: {spatial_merged}")
    # print(f"  Single-source: {single_source}")
    # print(f"  Total: {len(fused_tracks)}")
    # print(
    #    f"  Reduction from spatial clustering: {tracks_before_spatial - tracks_after_spatial} tracks"
    # )

    return fused_tracks, spatial_stats


def save_fused_tracks(fused_tracks: List[Dict], output_path: str):
    """Save fused tracks to JSONL file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"💾 Saved {len(fused_tracks)} fused tracks to {output_path}")
    with open(output_path, "w") as f:
        for track in fused_tracks:
            # Convert numpy arrays to lists for JSON serialization
            track_copy = track.copy()
            points = np.array(track_copy["projected"])
            polyorder = 7
            smoothing_window = 91
            if len(points) >= smoothing_window:
                xs = savgol_filter(points[:, 0], smoothing_window, polyorder)
                ys = savgol_filter(points[:, 1], smoothing_window, polyorder)
                points = np.stack([xs, ys], axis=1)

            if "unmatched" in track_copy["track_id"]:
                continue  # Skip unmatched tracks if desired
            if "projected" in track_copy:
                track_copy["projected"] = [p if isinstance(p, list) else p.tolist() for p in points]
            if track_copy["jersey_num"] == "NA" and track_copy["team"] != "referee":
                track_copy["jersey_num"] = "unsure"
            f.write(json.dumps(track_copy) + "\n")


def save_matching_results(results: Dict, output_path: str):
    """
    Save matching results to a JSON file.

    Args:
        results: Results dictionary from compare_tracks_between_files
        output_path: Path to save results
    """
    # results now expected to contain 'aggregated_groups'
    summary = {
        "stats": results.get("stats", {}),
        "groups": results.get("aggregated_groups", []),
        "pairwise_matches": results.get("pairwise_matches", []),
        "unmatched_tracks": results.get("unmatched_tracks", []),
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Results saved to {output_path}")


def save_matched_tracks_separately(
    results: Dict, all_tracks_by_file: Dict[int, List[Dict]], output_path: str
):
    """
    Save matched tracks separately (without merging) to JSONL file.
    Each matched group appears as multiple lines (one per source track).

    Args:
        results: Results dictionary from compare_tracks_between_files
        all_tracks_by_file: Dict mapping source_idx -> list of original tracks
        output_path: Output JSONL file path
    """
    groups = results.get("aggregated_groups", [])

    tracks_to_save = []

    print(f"\n[EXPERIMENTAL] Preparing {len(groups)} matched groups for separate output...")

    for group_idx, group in enumerate(groups):
        track_ids = group.get("track_id", [])
        source_indices = group.get("source_idx", [])

        # Pair track_id with their source_idx for unique identification
        if len(source_indices) != len(track_ids):
            print(f"Warning: Group {group_idx} has mismatched track_id and source_idx lengths")
            continue

        for track_id, source_idx in zip(track_ids, source_indices):
            # Find the track in the original data using both track_id and source_idx
            track_found = None
            if source_idx in all_tracks_by_file:
                for track in all_tracks_by_file[source_idx]:
                    if track["track_id"] == track_id:
                        track_found = track.copy()
                        break

            if track_found:
                # Add metadata about the group
                track_found["match_group_id"] = group_idx
                track_found["match_group_size"] = len(track_ids)
                track_found["match_group_tracks"] = track_ids
                track_found["match_group_source_files"] = source_indices
                track_found["is_matched"] = True
                track_found["source_file_index"] = source_idx

                # Remove internal metadata
                track_found.pop("_source_idx", None)
                track_found.pop("_orig_idx", None)

                tracks_to_save.append(track_found)
            else:
                print(
                    f"Warning: Track {track_id} from source {source_idx} not found in original tracks"
                )

    # Save to JSONL
    with open(output_path, "w") as f:
        for track in tracks_to_save:
            f.write(json.dumps(track) + "\n")

    print(f"✅ Saved {len(tracks_to_save)} matched tracks separately to {output_path}")
    print(
        f"   ({len(groups)} groups, avg {len(tracks_to_save)/max(len(groups), 1):.1f} tracks per group)"
    )


def group_tracks_by_player(results: Dict, output_jsonl_path: str):
    """
    Create grouped tracks where each line represents a player with tracks from both cameras.
    Now supports multiple matches - one camera2 track can appear with multiple camera1 tracks.

    Args:
        results: Results dictionary from compare_tracks_between_files
        output_jsonl_path: Path to save grouped tracks
    """
    with open(output_jsonl_path, "w") as f:
        # Changed: Group matches by camera2 track to handle multiple matches
        matches_by_track2 = defaultdict(list)
        for match in results["matches"]:
            track2_id = match["track2"]["track_id"]
            matches_by_track2[track2_id].append(match)

        # Write grouped tracks
        for track2_id, match_list in matches_by_track2.items():
            # Get camera2 track info (same for all matches)
            track2 = match_list[0]["track2"]

            # Changed: camera1 tracks are now a list
            camera1_tracks = [
                {
                    "source": "camera1",
                    "track_id": match["track1"]["track_id"],
                    "frames": match["track1"]["frames"],
                    "projected": match["track1"]["projected"],
                    "frame_range": match["track1"].get("frame_range"),
                    "matching_quality": {
                        "median_distance": match["metadata"]["median_distance"],
                        "average_distance": match["metadata"]["average_distance"],
                        "overlap_frames": match["metadata"]["overlap_count"],
                    },
                }
                for match in match_list
            ]

            grouped_track = {
                "player_id": f"{track2.get('team')}_{track2.get('jersey_num', 'unknown')}",
                "team": track2.get("team"),
                "jersey_num": track2.get("jersey_num"),
                "tracks": {
                    "camera1": camera1_tracks,  # Changed: now a list
                    "camera2": {
                        "source": "camera2",
                        "track_id": track2["track_id"],
                        "frames": track2["frames"],
                        "projected": track2["projected"],
                        "frame_range": track2.get("frame_range"),
                    },
                },
                "match_count": len(camera1_tracks),  # New: number of matches
            }
            f.write(json.dumps(grouped_track) + "\n")

    print(f"✅ Grouped tracks saved to {output_jsonl_path}")


def spatial_cluster_tracks(
    fused_tracks: List[Dict],
    distance_threshold: float = 15.0,
    overlap_threshold: float = 0.5,
) -> List[Dict]:
    """
    Merge spatially overlapping tracks that likely represent the same player.

    Args:
        fused_tracks: List of fused track dictionaries
        distance_threshold: Max median distance to consider for merging (default: 15 units)
        overlap_threshold: Min temporal overlap ratio required (default: 0.5)

    Returns:
        List of spatially clustered tracks
    """

    n = len(fused_tracks)
    if n <= 1:
        return fused_tracks

    # Build distance matrix
    dist_matrix = np.full((n, n), 1e10)  # large initial distances

    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = fused_tracks[i], fused_tracks[j]

            # NEW: Only consider tracks from the SAME team (exact match)
            team1 = t1.get("team")
            team2 = t2.get("team")

            # Skip if teams don't match exactly (including None)
            if team1 != team2 or team1 is None or team2 is None:
                continue

            # Skip if teams are "unsure" or empty
            if team1 in ["unsure", ""] or team2 in ["unsure", ""]:
                continue

            # Check temporal overlap
            frames1 = set(t1["frames"])
            frames2 = set(t2["frames"])
            overlap = frames1 & frames2

            if not overlap:
                continue

            # Require significant overlap
            overlap_ratio = len(overlap) / min(len(frames1), len(frames2))
            if overlap_ratio < overlap_threshold:
                continue

            # Calculate median distance in overlapping region
            distances = []
            for frame in sorted(overlap)[:150]:  # Sample max 150 frames
                try:
                    idx1 = t1["frames"].index(frame)
                    idx2 = t2["frames"].index(frame)
                    pos1 = np.array(t1["projected"][idx1])
                    pos2 = np.array(t2["projected"][idx2])
                    distances.append(euclidean(pos1, pos2))
                except (ValueError, IndexError):
                    continue

            if distances:
                median_dist = np.median(distances)
                dist_matrix[i, j] = median_dist
                dist_matrix[j, i] = median_dist

    # Hierarchical clustering
    valid_pairs = [
        (i, j) for i in range(n) for j in range(i + 1, n) if dist_matrix[i, j] < distance_threshold
    ]

    if not valid_pairs:
        print("  No spatial clusters found")
        return fused_tracks

    # Build condensed distance matrix for scipy
    condensed_dist = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed_dist.append(dist_matrix[i, j])

    # Cluster with threshold
    linkage_matrix = linkage(condensed_dist, method="average")
    cluster_ids = fcluster(linkage_matrix, distance_threshold, criterion="distance")

    # Group tracks by cluster
    clusters = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        clusters[cluster_id].append(idx)

    print(f"  Found {len(clusters)} spatial clusters from {n} tracks")

    # Merge each cluster
    merged_tracks = []
    for cluster_id, track_indices in clusters.items():
        if len(track_indices) == 1:
            merged_tracks.append(fused_tracks[track_indices[0]])
        else:
            # Merge multiple tracks
            # print(
            #     f"    Merging cluster {cluster_id}: {[fused_tracks[i]['track_id'] for i in track_indices]}"
            # )
            merged = merge_track_cluster(
                [fused_tracks[i] for i in track_indices],
                new_id=f"{merged_tracks.__len__() + 1}_spatial_merged",
            )
            merged_tracks.append(merged)

    return merged_tracks


def merge_track_cluster(tracks: List[Dict], new_id: str) -> Dict:
    """
    Merge multiple tracks that represent the same player.
    Similar to fuse_matched_tracks but for spatial clustering.
    """
    # Majority vote on metadata
    teams = [t.get("team") for t in tracks]
    team_confs = [t.get("team_conf", 0.5) for t in tracks]
    jersey_nums = [t.get("jersey_num") for t in tracks]
    jersey_confs = [t.get("jersey_conf", 0.5) for t in tracks]
    jersey_counts = [t.get("count", 1) for t in tracks]

    merged_team, team_conf, _ = majority_vote(  # Ignore returned count (placeholder)
        teams,
        team_confs,
        counts=None,  # Don't pass counts for team
        ignore_values=["unsure", None, ""],
        vote="team",
    )

    merged_jersey_num, jersey_conf, jersey_count = majority_vote(
        jersey_nums, jersey_confs, jersey_counts, ignore_values=["unsure", None, ""], vote="jersey"
    )

    # Union of all frames
    all_frames = set()
    for t in tracks:
        all_frames.update(t.get("frames", []))
    all_frames = sorted(all_frames)

    # Weighted average positions per frame
    frame_data = defaultdict(list)
    for t in tracks:
        for i, frame in enumerate(t.get("frames", [])):
            if i < len(t.get("projected", [])) and i < len(t.get("bbox_area", [])):
                frame_data[frame].append(
                    {
                        "position": np.array(t["projected"][i]),
                        "bbox_area": t["bbox_area"][i],
                        "track_id": t["track_id"],
                    }
                )

    merged_frames = []
    merged_projected = []
    merged_bbox_area = []

    for frame in sorted(frame_data.keys()):
        data_list = frame_data[frame]
        positions = [d["position"] for d in data_list]
        weights = [d["bbox_area"] for d in data_list]

        avg_pos = calculate_weighted_average_position(positions, weights)
        avg_bbox = np.mean(weights)

        merged_frames.append(frame)
        merged_projected.append(avg_pos.tolist())
        merged_bbox_area.append(avg_bbox)

    # Merge source tracks info
    source_tracks = []
    source_indices = []
    for t in tracks:
        if isinstance(t.get("source_track"), list):
            source_tracks.extend(t["source_track"])
        else:
            source_tracks.append(t.get("track_id"))

        if isinstance(t.get("source_idx"), list):
            source_indices.extend(t["source_idx"])
        else:
            source_indices.append(t.get("_source_idx", 0))

    return {
        "track_id": new_id,
        "team": merged_team,
        "jersey_num": merged_jersey_num,
        "jersey_conf": jersey_conf,
        "count": jersey_count,
        "team_conf": team_conf,
        "frame_range": [min(all_frames), max(all_frames)],
        "frames": merged_frames,
        "projected": merged_projected,
        "bbox_area": merged_bbox_area,
        "source_track": source_tracks,
        "source_idx": source_indices,
        "is_fused": False,
        "is_spatial_merged": True,
        "merged_count": len(tracks),
    }


def save_pairwise_scores(
    all_pairs: List[Tuple[float, int, int, Dict]],
    all_tracks: List[Dict],
    output_path: str,
    verbose: bool = False,
):
    """
    Save all pairwise comparison scores to a JSONL file.

    Args:
        all_pairs: List of (score, idx1, idx2, metadata) tuples
        all_tracks: List of all track dictionaries with metadata
        output_path: Path to save JSONL file
        verbose: Print progress
    """
    if verbose:
        print(f"\n💾 Saving {len(all_pairs)} pairwise scores to {output_path}...")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for score, idx1, idx2, metadata in all_pairs:
            t1 = all_tracks[idx1]
            t2 = all_tracks[idx2]

            # Build comprehensive comparison record
            comparison = {
                # Track identifiers
                "track1_id": t1["track_id"],
                "track2_id": t2["track_id"],
                "track1_source": t1["_source_idx"],
                "track2_source": t2["_source_idx"],
                "track1_global_idx": idx1,
                "track2_global_idx": idx2,
                # Track metadata
                "track1_team": t1.get("team"),
                "track2_team": t2.get("team"),
                "track1_jersey": t1.get("jersey_num"),
                "track2_jersey": t2.get("jersey_num"),
                "track1_frame_range": t1.get("frame_range"),
                "track2_frame_range": t2.get("frame_range"),
                # Overall score
                "composite_score": score,
                "is_match_candidate": score != float("inf"),
                # Score components (from metadata)
                "score_components": metadata.get("score_components", {}),
                # Distance metrics
                "median_distance": metadata.get("median_distance"),
                "distance_diff": metadata.get("distance_diff"),
                "total_distance_track1": metadata.get("total_distance_track1"),
                "total_distance_track2": metadata.get("total_distance_track2"),
                # Direction metrics
                "direction_similarity": metadata.get("direction_similarity"),
                "direction_metadata": metadata.get("direction_metadata", {}),
                # Velocity metrics
                "velocity_difference": metadata.get("velocity_difference"),
                # Overlap information
                "overlap_frames": metadata.get("overlap_count"),
                # Outlier information
                "track1_outliers": metadata.get("track1_outliers", 0),
                "track2_outliers": metadata.get("track2_outliers", 0),
                "point_outliers": metadata.get("point_outliers", 0),
                # Detailed distance metadata
                "median_metadata": metadata.get("median_metadata", {}),
                # Rejection reason (if infinite score)
                "rejection_reason": metadata.get("reason") if score == float("inf") else None,
                "initial_distance": metadata.get("median_initial_distance"),
            }

            f.write(json.dumps(comparison) + "\n")

    if verbose:
        print(f"  ✅ Saved pairwise scores to {output_path}")

        # Print statistics
        valid_scores = [s for s, _, _, _ in all_pairs if s != float("inf")]
        if valid_scores:
            print(f"\n  📊 Score Statistics:")
            print(f"    Valid comparisons: {len(valid_scores)}")
            print(f"    Min score: {min(valid_scores):.2f}")
            print(f"    Max score: {max(valid_scores):.2f}")
            print(f"    Mean score: {np.mean(valid_scores):.2f}")
            print(f"    Median score: {np.median(valid_scores):.2f}")

            # Score distribution
            bins = [0, 10, 25, 50, 100, 200, float("inf")]
            labels = ["<10", "10-25", "25-50", "50-100", "100-200", ">200"]
            hist, _ = np.histogram(valid_scores, bins=bins)

            print(f"\n  Score Distribution:")
            for label, count in zip(labels, hist):
                if count > 0:
                    print(f"    {label}: {count} pairs ({count/len(valid_scores)*100:.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare and match tracks across multiple JSONL files from different cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python cluster_tracks_greedy.py camera1_tracks.jsonl camera2_tracks.jsonl -o matched_results.json --max-score 100.0
        """,
    )

    parser.add_argument(
        "--player-jsonl-paths",
        type=str,
        nargs="+",
        metavar="FILE",
        required=True,
        help="Player JSONL track files to compare (minimum 2 files required)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        default="./track_matching_results.json",
        help="Output JSON file path (default: ./track_matching_results.json)",
    )

    # ⭐ NEW: Calibration arguments
    parser.add_argument(
        "--ball-jsonl-paths",
        type=str,
        nargs="+",
        default=None,
        metavar="FILE",
        help="Ball track JSONL files (one per camera, same order as player tracks). "
        "Used for computing frame/coordinate offsets if --auto-calibrate is set.",
    )

    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Automatically compute frame & coordinate offsets from ball tracks",
    )

    parser.add_argument(
        "--frame-offsets",
        type=int,
        nargs="+",
        default=None,
        metavar="OFFSET",
        help="Pre-computed frame offsets (one per camera). " "Example: --frame-offsets 0 15 -10 5",
    )

    parser.add_argument(
        "--coord-offsets",
        type=float,
        nargs="+",
        default=None,
        metavar="OFFSET",
        help="Pre-computed coordinate offsets (using first camera as reference). "
        "Example: --coord-offsets 0.0 0.0 100.0 -50.0 0.0 25.0 25.0 25.0",
    )

    parser.add_argument(
        "--min-overlap-frames",
        type=int,
        default=50,
        metavar="N",
        help="Minimum overlapping frames required for matching (default: 50)",
    )

    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=45.0,
        metavar="S",
        help="Maximum angle difference (degrees) to consider directions as similar (default: 45.0 degrees)",
    )

    parser.add_argument(
        "--direction-frame-stride",
        type=int,
        default=3,
        metavar="N",
        help="Frame stride for direction sampling (default: 3). "
        "Higher values = smoother/less noisy (1=consecutive, 3=every 3rd frame, 10=every 10th frame). "
        "At 29.97fps: stride 1≈33ms, stride 3≈100ms, stride 10≈333ms",
    )

    parser.add_argument(
        "--min-movement-threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="Minimum movement threshold to consider for matching in direction similarity matching (default: 0.5 units)",
    )

    parser.add_argument(
        "--max-analysis-frames",
        type=int,
        default=1500,
        metavar="N",
        help="Maximum frames to analyze per comparison (default: 1500)",
    )

    parser.add_argument(
        "--filter-team",
        action="store_true",
        help="Only compare tracks from the same team in greedy matching",
    )

    parser.add_argument(
        "--filter-jersey",
        action="store_true",
        help="Only compare tracks with the same jersey number in greedy matching",
    )

    parser.add_argument("--verbose", action="store_true", help="Print detailed comparison progress")

    parser.add_argument(
        "--save-separate",
        type=str,
        default=None,
        metavar="FILE",
        help="[EXPERIMENTAL] Save matched tracks separately (unmerged) to this JSONL file. Each matched group will have multiple lines (one per source track).",
    )

    parser.add_argument(
        "--max-score-threshold",
        type=float,
        default=100.0,
        metavar="S",
        help="Maximum acceptable score for greedy matching (default: 10.0, lower = stricter)",
    )

    parser.add_argument(
        "--max-initial-distance",
        type=float,
        default=150.0,
        metavar="D",
        help="Maximum allowed initial distance between tracks (default: 150.0, lower = stricter)",
    )

    parser.add_argument(
        "--max-point-distance",
        type=float,
        default=100.0,
        metavar="D",
        help="Maximum allowed distance between corresponding points (outlier threshold, default: 100.0 units)",
    )

    parser.add_argument(
        "--max-step-distance",
        type=float,
        default=50.0,
        metavar="D",
        help="Maximum allowed distance per movement step (outlier threshold, default: 50.0 units)",
    )

    parser.add_argument(
        "--max-outlier-ratio",
        type=float,
        default=0.3,
        metavar="R",
        help="Maximum ratio of outliers allowed before rejecting match [0-1] (default: 0.3 = 30%%)",
    )

    parser.add_argument(
        "--save-pairwise-path",
        type=str,
        default=None,
        metavar="FILE",
        help="Save all pairwise comparison scores to this JSONL file (default: None, don't save)",
    )

    parser.add_argument(
        "--use-dtw-filter",
        action="store_true",
        default=False,
        help="Enable DTW distance filter (default: disabled for backward compatibility)",
    )

    parser.add_argument(
        "--dtw-threshold",
        type=float,
        default=0.4,
        metavar="D",
        help="Maximum normalized DTW distance to accept match (default: 0.4, lower = stricter)",
    )

    parser.add_argument(
        "--dtw-num-samples",
        type=int,
        default=10,
        metavar="N",
        help="Number of samples for DTW computation (default: 10)",
    )

    parser.add_argument(
        "--dtw-segment-ratio",
        type=float,
        default=0.5,
        metavar="R",
        help="Segment ratio for DTW sampling [0-1] (default: 0.5)",
    )

    parser.add_argument(
        "--dtw-min-length",
        type=int,
        default=10,
        metavar="N",
        help="Minimum segment length for DTW (default: 10 points)",
    )

    parser.add_argument(
        "--dtw-segment-length",
        type=int,
        default=50,
        metavar="N",
        help="Fixed segment length for DTW (overrides --dtw-segment-ratio if set)",
    )

    parser.add_argument(
        "--spatial-distance",
        type=float,
        default=10.0,
        metavar="D",
        help="Distance threshold for spatial merging (default: 10.0 units, ~1.0m)",
    )

    args = parser.parse_args()

    # Validate minimum files
    if len(args.player_jsonl_paths) < 2:
        parser.error("At least 2 JSONL files are required for comparison")

    return args


def main():

    args = parse_args()

    # Load coordinate offsets if provided
    coord_offsets = None
    if args.coord_offsets:
        coord_offsets_raw = args.coord_offsets
        coord_offsets = [
            np.array(coord_offsets_raw[i : i + 2]) for i in range(0, len(coord_offsets_raw), 2)
        ]
        print(f"Loaded coordinate offsets: {coord_offsets_raw}")

    # Validate calibration arguments
    if args.auto_calibrate and not args.ball_jsonl_paths:
        print("⚠️  Warning: --auto-calibrate requires --ball-jsonl-paths")
        print("    Proceeding without calibration...")
        args.auto_calibrate = False

    if args.ball_jsonl_paths:
        assert len(args.ball_jsonl_paths) == len(args.player_jsonl_paths), (
            f"Number of ball track files ({len(args.ball_jsonl_paths)}) must match "
            f"number of player track files ({len(args.player_jsonl_paths)})"
        )

    if args.frame_offsets:
        assert len(args.frame_offsets) == len(args.player_jsonl_paths), (
            f"Number of frame offsets ({len(args.frame_offsets)}) must match "
            f"number of player track files ({len(args.player_jsonl_paths)})"
        )

    # Print configuration
    print("=" * 60)
    print("TRACK MATCHING CONFIGURATION")
    print("=" * 60)
    print(f"Input files: {len(args.player_jsonl_paths)}")
    for i, path in enumerate(args.player_jsonl_paths, 1):
        print(f"  {i}. {path}")
    print(f"Output: {args.output}")
    print(f"Min overlap frames: {args.min_overlap_frames}")
    print(f"Matching mode: GREEDY")
    print(f"Max score threshold: {args.max_score_threshold}")

    print(f"Max analysis frames: {args.max_analysis_frames}")
    print(f"Filter by team: {args.filter_team}")
    print(f"Filter by jersey: {args.filter_jersey}")
    print(f"Direction similarity threshold: {args.direction_threshold}")
    print(
        f"Direction frame stride: {args.direction_frame_stride} frames (~{args.direction_frame_stride/29.97*1000:.1f}ms)"
    )
    print(
        f"Spatial clustering threshold: {args.spatial_distance} units (~{args.spatial_distance/10:.1f}m)"
    )
    if args.save_separate:
        print(f"Save separate (experimental): {args.save_separate}")
    print("=" * 60 + "\n")
    if args.use_dtw_filter:
        print(f"DTW Filter: ENABLED")
        print(f"  DTW threshold: {args.dtw_threshold}")
        print(f"  DTW samples: {args.dtw_num_samples}")
        print(f"  DTW segment ratio: {args.dtw_segment_ratio}")
        print(f"  DTW min length: {args.dtw_min_length}")
    else:
        print(f"DTW Filter: DISABLED (use --use-dtw-filter to enable)")

    # Define calibration config
    calibration_config = CalibrationConfig(
        auto_calibrate=args.auto_calibrate,
        verbose=args.verbose,
    )
    all_tracks, all_tracks_by_file = prepare_tracks(
        jsonl_paths=args.player_jsonl_paths,
        ball_jsonl_paths=args.ball_jsonl_paths,
        cfg=calibration_config,
        frame_offsets=args.frame_offsets,
        coord_offsets=coord_offsets
    )

    matching_config = MatchingConfig(
        min_overlap_frames=args.min_overlap_frames,
        max_analysis_frames=args.max_analysis_frames,
        max_point_distance=args.max_point_distance,
        max_outlier_ratio=args.max_outlier_ratio,
        direction_threshold=args.direction_threshold,
        direction_frame_stride=args.direction_frame_stride,
        min_movement_threshold=args.min_movement_threshold,
        max_step_distance=args.max_step_distance,
        max_initial_distance=args.max_initial_distance
    )

    # Perform greedy matching
    results = greedy_match_tracks(
        all_tracks=all_tracks,
        jsonl_paths=args.player_jsonl_paths,
        filter_by_team=args.filter_team,
        filter_by_jersey=args.filter_jersey,
        max_score_threshold=args.max_score_threshold,
        cfg=matching_config,
        save_pairwise_path=args.save_pairwise_path,
        use_dtw_filter=args.use_dtw_filter,
        dtw_threshold=args.dtw_threshold,
        dtw_num_samples=args.dtw_num_samples,
        dtw_segment_length=args.dtw_segment_length,
        dtw_segment_ratio=args.dtw_segment_ratio,
        dtw_min_length=args.dtw_min_length,
        verbose=args.verbose,
    )

    aggregated_groups = results["aggregated_groups"]
    unmatched_tracks = results["unmatched_tracks"]

    # Fuse matched tracks with spatial clustering
    fused_tracks, spatial_stats = fuse_all_matched_groups(
        aggregated_groups,
        unmatched_tracks,
        all_tracks_by_file,
        spatial_distance_threshold=args.spatial_distance,
    )

    # Save fused tracks
    save_fused_tracks(fused_tracks, args.output)

    metadata_path = args.output.replace(".jsonl", "_metadata.jsonl")
    # UPDATE: Add spatial clustering stats to results
    results["spatial_clustering"] = spatial_stats
    results["fused_tracks_path"] = args.output
    results["fused_tracks_count"] = len(fused_tracks)
    results["stats"]["final_track_count_after_spatial"] = spatial_stats["tracks_after_spatial"]
    results["stats"]["tracks_merged_by_spatial"] = spatial_stats["tracks_merged_by_spatial"]

    save_matching_results(results, metadata_path)

    # Save matched tracks separately if requested (experimental feature)
    if args.save_separate:
        save_matched_tracks_separately(results, all_tracks_by_file, args.save_separate)

    print("\n" + "=" * 60)
    print("MATCHING STATISTICS")
    print("=" * 60)
    print(f"Total tracks: {results['stats'].get('total_tracks')}")

    print(f"Candidate pairs evaluated: {results['stats'].get('total_candidate_pairs')}")
    print(f"Matches accepted: {results['stats'].get('matches_accepted')}")
    print(f"Match rate: {results['stats'].get('match_rate', 0):.1%}")
    print(
        f"Final tracks after spatial clustering: {results['stats'].get('final_track_count_after_spatial', 'NA')}"
    )
    print(
        f"Tracks merged by spatial clustering: {results['stats'].get('tracks_merged_by_spatial', 'NA')}"
    )

    print(f"Groups found: {results['stats'].get('groups_count')}")
    print(f"Unmatched tracks: {len(results.get('unmatched_tracks', []))}")

    # Print a few groups summary
    if results.get("aggregated_groups"):
        print(f"\n🎯 Top {min(5, len(results['aggregated_groups']))} matched groups:")
        for idx, g in enumerate(results["aggregated_groups"][:5], 1):
            print(f"\n  Group {idx}:")
            print(f"    Tracks: {g['track_id']}")
            print(f"    Sources: {g['source_idx']}")
            print(f"    Frame_ranges: {g['frame_ranges']}")
            print(f"    Teams: {set(g['team'])}")
            # flatten jersey numbers and remove None
            flattened_jerseys = []
            for jersey in g["jersey_num"]:
                if isinstance(jersey, list):
                    flattened_jerseys.extend(jersey)
                else:
                    flattened_jerseys.append(jersey)
            print(f"    Jerseys: {set(flattened_jerseys)}")
            print(f"    Pairwise comparisons: {len(g.get('pairs', []))}")
            if g.get("median_distance"):
                print(f"    Avg median distance: {np.mean(g['median_distance']):.2f} units")

    # NEW: Print spatial clustering summary if available
    if results.get("spatial_clustering"):
        sc = results["spatial_clustering"]
        print(f"\n🔗 Spatial Clustering Summary:")
        print(f"  Tracks before spatial clustering: {sc['tracks_before_spatial']}")
        print(
            f"  Tracks after spatial clustering (including unmatched): {sc['tracks_after_spatial']}"
        )
        print(f"  Tracks merged by spatial clustering: {sc['tracks_merged_by_spatial']}")
        print(f"  Distance threshold: {sc['spatial_distance_threshold']} units")
        print(f"  Breakdown:")
        print(f"    - Multi-source fused: {sc['multi_source_fused']}")
        print(f"    - Spatially merged: {sc['spatial_merged']}")
        print(f"    - Single-source (discarded): {sc['single_source']}")

    print("\n" + "=" * 60)
    print(f"✅ Results saved to: {args.output}")
    if results.get("fused_tracks_path"):
        print(f"✅ Fused tracks saved to: {results['fused_tracks_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
