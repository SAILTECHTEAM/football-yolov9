import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.signal import correlate, medfilt

from post_processing_ball import smoothen_fused_ball_tracking


def load_ball_tracks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load ball tracks from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file containing ball tracks

    Returns:
        List of ball track dictionaries
    """
    tracks = []
    with open(jsonl_path, "r") as f:
        for line in f:
            track = json.loads(line.strip())
            if track.get("team") == "ball":
                tracks.append(track)
    return tracks


def estimate_frame_offset_pairwise(
    tracks1: List[Dict],
    tracks2: List[Dict],
    max_search_offset: int = 60,
    overlap_threshold: int = 30,
    velocity_clip_threshold: float = 10.0,
) -> Tuple[int, float]:
    """
    Estimate frame offset between two cameras using cross-correlation on velocity signals.

    Args:
        tracks1: Tracks from camera 1 (reference)
        tracks2: Tracks from camera 2 (to be adjusted)
        max_search_offset: Maximum frame offset to search (±frames)
        overlap_threshold: Minimum overlapping frames needed for comparison
        velocity_clip_threshold: Maximum velocity to prevent high balls from dominating (in 0.1m/frame units)

    Returns:
        (best_offset, confidence_score) where offset means: camera2_frame = camera1_frame + offset
    """

    # Extract velocity signals from both cameras
    def get_velocity_signal(tracks):
        """Convert tracks to a continuous velocity signal indexed by frame."""
        frame_positions = {}
        for track in tracks:
            frames = track.get("frames", [])
            projected = track.get("projected", [])
            for f, p in zip(frames, projected):
                if f not in frame_positions:
                    frame_positions[f] = np.array(p)

        if len(frame_positions) < 2:
            return None, None, None

        # Sort by frame
        sorted_frames = sorted(frame_positions.keys())

        # Downsample positions by a factor of 6 to reduce noise
        sorted_frames = sorted_frames[::6]


        # Calculate velocities (magnitude and direction)
        velocities_x = []
        velocities_y = []
        velocity_frames = []

        for i in range(len(sorted_frames) - 1):
            f1, f2 = sorted_frames[i], sorted_frames[i + 1]
            frame_gap = f2 - f1

            # Only use consecutive or near-consecutive frames
            if frame_gap <= 30:
                pos1 = frame_positions[f1]
                pos2 = frame_positions[f2]

                # Calculate velocity components (preserve direction)
                vel = (pos2 - pos1) / frame_gap

                vel_magnitude = np.linalg.norm(vel)
                if vel_magnitude > velocity_clip_threshold:
                    # print(
                    #     f"    ⚠️ Clipping high velocity {vel_magnitude:.1f} at frames {f1}-{f2}")
                    # Scale down to threshold while preserving direction
                    vel = vel * (velocity_clip_threshold / vel_magnitude)

                velocities_x.append(vel[0])
                velocities_y.append(vel[1])
                velocity_frames.append(f1)

        return np.array(velocity_frames), np.array(velocities_x), np.array(velocities_y)

    frames1, vel1_x, vel1_y = get_velocity_signal(tracks1)
    frames2, vel2_x, vel2_y = get_velocity_signal(tracks2)

    if frames1 is None or frames2 is None:
        return 0, 0.0

    if len(vel1_x) < overlap_threshold or len(vel2_x) < overlap_threshold:
        print(
            f"    ⚠️ Not enough velocity data for offset estimation: cam1={len(vel1_x)} frames, cam2={len(vel2_x)} frames"
        )
        return 0, 0.0

    # Find potential overlapping range (with buffer for offset search)
    min_frame1, max_frame1 = frames1.min(), frames1.max()
    min_frame2, max_frame2 = frames2.min(), frames2.max()

    print(f"min/max frames cam1: {min_frame1}/{max_frame1}, cam2: {min_frame2}/{max_frame2}")

    # Check if there's any potential overlap considering max offset
    if max_frame1 + max_search_offset < min_frame2 or max_frame2 + max_search_offset < min_frame1:
        print(
            f"    ⚠️ No overlap possible even with max offset: cam1[{min_frame1}-{max_frame1}] vs cam2[{min_frame2}-{max_frame2}]"
        )
        return 0, 0.0

    # Create continuous signals over full range
    full_min = min(min_frame1, min_frame2) - max_search_offset
    full_max = max(max_frame1, max_frame2) + max_search_offset
    full_frames = np.arange(full_min, full_max + 1)

    # Interpolate both signals to common frame grid
    signal1_x = np.zeros(len(full_frames))
    signal1_y = np.zeros(len(full_frames))
    signal2_x = np.zeros(len(full_frames))
    signal2_y = np.zeros(len(full_frames))

    for i, frame in enumerate(full_frames):
        if frame in frames1:
            idx = np.where(frames1 == frame)[0][0]
            signal1_x[i] = vel1_x[idx]
            signal1_y[i] = vel1_y[idx]

        if frame in frames2:
            idx = np.where(frames2 == frame)[0][0]
            signal2_x[i] = vel2_x[idx]
            signal2_y[i] = vel2_y[idx]

    # Combine X and Y velocities into magnitude signal
    signal1 = np.sqrt(signal1_x**2 + signal1_y**2)
    signal2 = np.sqrt(signal2_x**2 + signal2_y**2)

    # Normalize signals
    if np.std(signal1) > 0:
        signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    if np.std(signal2) > 0:
        signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)

    # Compute cross-correlation using scipy
    correlation = correlate(signal1, signal2, mode="same")
    lags = np.arange(-len(signal1) // 2, len(signal1) // 2)

    # Only consider lags within our search range
    valid_lag_mask = (lags >= -max_search_offset) & (lags <= max_search_offset)
    valid_lags = lags[valid_lag_mask]
    valid_correlation = correlation[valid_lag_mask]

    if len(valid_correlation) == 0:
        print("    ⚠️ No valid correlation values found within search range")
        return 0, 0.0

    # Find peak correlation
    best_idx = np.argmax(valid_correlation)
    best_offset = valid_lags[best_idx]
    best_correlation_value = valid_correlation[best_idx]

    # Normalize correlation value to [0, 1]
    # Correlation is already somewhat normalized, but scale to confidence
    max_possible_corr = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
    confidence = best_correlation_value / max_possible_corr if max_possible_corr > 0 else 0.0
    confidence = np.clip(confidence, 0.0, 1.0)

    # Verify there's actual overlap after applying offset
    overlap_start = max(min_frame1, min_frame2 + best_offset)
    overlap_end = min(max_frame1, max_frame2 + best_offset)
    overlap_size = overlap_end - overlap_start

    if overlap_size < overlap_threshold:
        print(f"    ⚠️ Insufficient overlap after offset {best_offset}: only {overlap_size} frames")
        return 0, 0.0

    return int(best_offset), float(confidence)


def calibrate_frame_offsets(
    all_tracks_by_camera: List[List[Dict]],
    max_search_offset: int = 30,
    overlap_threshold: int = 30,
    verbose: bool = False,
) -> List[int]:
    """
    Calibrate frame offsets across all cameras using pairwise alignment.

    Args:
        all_tracks_by_camera: List of track lists (one per camera)
        max_search_offset: Maximum frame offset to search
        overlap_threshold: Minimum overlap for valid comparison
        verbose: Print calibration info

    Returns:
        List of frame offsets (one per camera, first camera is reference with offset=0)
    """
    n_cameras = len(all_tracks_by_camera)

    if n_cameras < 2:
        return [0] * n_cameras

    # Use camera 0 as reference
    offsets = [0] * n_cameras

    if verbose:
        print(f"\n🔧 Calibrating frame offsets (using camera 0 as reference)...")

    # Estimate offset for each camera relative to camera 0
    for cam_idx in range(1, n_cameras):
        offset, confidence = estimate_frame_offset_pairwise(
            all_tracks_by_camera[0],
            all_tracks_by_camera[cam_idx],
            max_search_offset=max_search_offset,
            overlap_threshold=overlap_threshold,
        )
        offsets[cam_idx] = offset

        if verbose:
            print(f"  Camera {cam_idx}: offset = {offset:+d} frames (confidence: {confidence:.3f})")
            if offset != 0:
                print(f"    → Adjusted: cam{cam_idx}_frame = cam0_frame + {offset}")

    return offsets


def apply_frame_offsets(
    all_tracks_by_camera: List[List[Dict]],
    offsets: List[int],
) -> List[List[Dict]]:
    """
    Apply frame offsets to align all cameras to the same timeline.

    Args:
        all_tracks_by_camera: List of track lists (one per camera)
        offsets: Frame offsets for each camera

    Returns:
        Aligned track lists
    """
    aligned_tracks = []

    for cam_idx, (tracks, offset) in enumerate(zip(all_tracks_by_camera, offsets)):
        if offset == 0:
            aligned_tracks.append(tracks)
            continue

        # Apply offset to all frames in all tracks
        aligned_camera_tracks = []
        for track in tracks:
            frames = np.array(track.get("frames", []))
            projected = track.get("projected", [])

            # Shift frames
            frames_adjusted = frames + offset

            aligned_track = {
                "track_id": track.get("track_id"),
                "team": "ball",
                "frame_range": [int(frames_adjusted[0]), int(frames_adjusted[-1])],
                "frames": frames_adjusted.tolist(),
                "projected": projected,
            }
            aligned_camera_tracks.append(aligned_track)

        aligned_tracks.append(aligned_camera_tracks)

    return aligned_tracks


def detect_outliers_at_frame(
    positions: List[np.ndarray],
    max_distance_threshold: float = 200.0,
    min_inliers: int = 2,
) -> List[bool]:
    """
    Detect outlier positions at a single frame using pairwise distances.

    Args:
        positions: List of position arrays [x, y] from different cameras
        max_distance_threshold: Maximum allowed distance between cameras
        min_inliers: Minimum number of cameras that must agree

    Returns:
        Boolean list indicating which positions are NOT outliers
    """
    if len(positions) <= 1:
        return [False] * len(positions)

    n = len(positions)
    positions_array = np.array(positions)

    # Calculate pairwise distances
    distances = cdist(positions_array, positions_array)

    # Count how many other cameras each position agrees with
    agreement_counts = []
    for i in range(n):
        # Count cameras within threshold distance
        close_cameras = np.sum(distances[i] < max_distance_threshold) - 1  # Exclude self
        agreement_counts.append(close_cameras)

    # Mark as inlier if enough cameras agree
    is_inlier = [count >= (min_inliers - 1) for count in agreement_counts]

    # If too few inliers, keep all (no consensus)
    if sum(is_inlier) < min_inliers:
        return [False] * len(positions)

    return is_inlier


def calculate_robust_average(
    positions: List[np.ndarray],
    max_distance_threshold: float = 200.0,
    min_inliers: int = 2,
) -> Optional[np.ndarray]:
    """
    Calculate robust average position after removing outliers.

    Args:
        positions: List of position arrays from different cameras
        max_distance_threshold: Maximum allowed distance for outlier detection
        min_inliers: Minimum number of inliers required

    Returns:
        Average position as numpy array, or None if insufficient data
    """
    if not positions:
        return None

    # Detect outliers
    is_inlier = detect_outliers_at_frame(
        positions,
        max_distance_threshold=max_distance_threshold,
        min_inliers=min_inliers,
    )

    # Filter to inliers only
    inlier_positions = [pos for pos, inlier in zip(positions, is_inlier) if inlier]

    if not inlier_positions:
        return None

    # Calculate average
    return np.mean(inlier_positions, axis=0)


def estimate_coordinate_offset(
    tracks1: List[Dict],
    tracks2: List[Dict],
    min_overlap_frames: int = 50,
    position_diff_clip: float = 50.0,  # ✅ NEW: Clip extreme position differences (5m)
    mad_threshold: float = 3.0,  # ✅ NEW: Use MAD (Median Absolute Deviation) for outlier detection
) -> Tuple[np.ndarray, float]:
    """
    Estimate systematic coordinate offset between two cameras.
    Uses overlapping frames to find consistent bias in X and Y coordinates.

    Args:
        tracks1: Tracks from camera 1 (reference)
        tracks2: Tracks from camera 2 (to be adjusted)
        min_overlap_frames: Minimum overlapping frames needed

    Returns:
        (offset_vector, confidence) where offset = cam2 - cam1
    """

    # Get frame-position pairs
    def get_frame_positions(tracks):
        frame_pos = {}
        for track in tracks:
            frames = track.get("frames", [])
            projected = track.get("projected", [])
            for f, p in zip(frames, projected):
                if f not in frame_pos:  # Use first occurrence
                    frame_pos[f] = np.array(p)
        return frame_pos

    fp1 = get_frame_positions(tracks1)
    fp2 = get_frame_positions(tracks2)

    # Find overlapping frames
    common_frames = sorted(set(fp1.keys()) & set(fp2.keys()))

    if len(common_frames) < min_overlap_frames:
        print("Not enough overlapping frames for coordinate offset estimation.")
        return np.array([0.0, 0.0]), 0.0

    # Offset = median of (pos2 - pos1) across all overlapping frames
    position_diffs = []
    for f in common_frames:
        pos_diff = fp2[f] - fp1[f]
        position_diffs.append(pos_diff)

    position_diffs = np.array(position_diffs)

    # MAD is more robust than std for heavy-tailed distributions
    def robust_outlier_filter(diffs, mad_threshold=3.0):
        """Remove outliers using MAD (works better than std for high balls)."""
        # Calculate MAD for each dimension
        median_diff = np.median(diffs, axis=0)
        mad = np.median(np.abs(diffs - median_diff), axis=0)
        
        # Modified Z-score (using MAD instead of std)
        # A common approximation: MAD ≈ 0.6745 * std for normal distribution
        modified_z_scores = np.abs((diffs - median_diff) / (mad / 0.6745 + 1e-6))
        
        # Keep points within threshold MAD units
        inlier_mask = np.all(modified_z_scores < mad_threshold, axis=1)
        
        return diffs[inlier_mask], inlier_mask

    # Filter outliers (high balls, tracking errors, etc.)
    filtered_diffs, inlier_mask = robust_outlier_filter(position_diffs, mad_threshold)
    
    n_outliers = len(position_diffs) - len(filtered_diffs)
    if n_outliers > 0:
        print(f"  Filtered {n_outliers} outliers ({100*n_outliers/len(position_diffs):.1f}%) from coordinate offset estimation")
    
    if len(filtered_diffs) < min_overlap_frames:
        print(f"  Too many outliers removed. Insufficient data ({len(filtered_diffs)} < {min_overlap_frames})")
        return np.array([0.0, 0.0]), 0.0

    median_pos_diff = np.median(filtered_diffs, axis=0)

    # ✅ NEW: Additional safety check - clip extreme offsets
    offset_magnitude = np.linalg.norm(median_pos_diff)
    if offset_magnitude > position_diff_clip:
        print(f"  Clipping large offset: {offset_magnitude:.1f} → {position_diff_clip}")
        median_pos_diff = median_pos_diff * (position_diff_clip / offset_magnitude)
        offset_magnitude = position_diff_clip

    # Calculate consistency (inverse of standard deviation)
    mad_filtered = np.median(np.abs(filtered_diffs - median_pos_diff), axis=0)
    consistency = 1.0 / (
        1.0 + np.mean(mad_filtered) / 100.0
    )  # 100.0 means further than 10m reduces confidence

    # Only apply offset if it's significant and consistent
    offset_magnitude = np.linalg.norm(median_pos_diff)
    if offset_magnitude < 1.0:  # Less than 1.0, ignore
        print("Offset magnitude too small, ignoring offset.")
        return np.array([0.0, 0.0]), 0.0
    if offset_magnitude > 80.0:  # More than 80.0, likely error
        print("Offset magnitude too large, ignoring offset.")
        return np.array([0.0, 0.0]), 0.0

    # Confidence based on consistency and number of samples
    sample_factor = min(len(common_frames) / 100.0, 1.0)
    confidence = consistency * sample_factor

    return median_pos_diff, confidence


def calibrate_coordinate_systems(
    all_tracks_by_camera: List[List[Dict]],
    min_overlap_frames: int = 50,
    min_confidence: float = 0.3,
    verbose: bool = False,
) -> List[np.ndarray]:
    """
    Calibrate coordinate systems across all cameras.
    Finds systematic offsets in X,Y coordinates.

    Args:
        all_tracks_by_camera: List of track lists (one per camera)
        min_overlap_frames: Minimum overlap for valid comparison
        min_confidence: Minimum confidence to apply offset
        verbose: Print calibration info

    Returns:
        List of coordinate offsets (one per camera, first camera is reference)
    """
    n_cameras = len(all_tracks_by_camera)

    if n_cameras < 2:
        return [np.array([0.0, 0.0])] * n_cameras

    # Use camera 0 as reference (no offset)
    offsets = [np.array([0.0, 0.0])] * n_cameras

    if verbose:
        print(f"\n🔧 Calibrating coordinate systems (using camera 0 as reference)...")

    # Estimate offset for each camera relative to camera 0
    for cam_idx in range(1, n_cameras):
        offset, confidence = estimate_coordinate_offset(
            all_tracks_by_camera[0],
            all_tracks_by_camera[cam_idx],
            min_overlap_frames=min_overlap_frames,
        )

        # Only apply if confidence is high enough
        if confidence >= min_confidence:
            offsets[cam_idx] = offset

            if verbose:
                print(
                    f"  Camera {cam_idx}: offset = ({offset[0]:+.1f}, {offset[1]:+.1f}) "
                    f"(confidence: {confidence:.3f})"
                )
                print(f"    → Adjusted: cam{cam_idx}_pos = original_pos - offset")
        else:
            if verbose:
                print(
                    f"  Camera {cam_idx}: offset = (0.0, 0.0) "
                    f"(confidence too low: {confidence:.3f})"
                )

    return offsets


def apply_coordinate_offsets(
    all_tracks_by_camera: List[List[Dict]],
    offsets: List[np.ndarray],
) -> List[List[Dict]]:
    """
    Apply coordinate offsets to align all cameras to the same coordinate system.

    Args:
        all_tracks_by_camera: List of track lists (one per camera)
        offsets: Coordinate offsets for each camera

    Returns:
        Aligned track lists
    """
    aligned_tracks = []

    for cam_idx, (tracks, offset) in enumerate(zip(all_tracks_by_camera, offsets)):
        # If no offset, keep original
        if np.allclose(offset, 0.0):
            aligned_tracks.append(tracks)
            continue

        # Apply offset to all projected coordinates
        aligned_camera_tracks = []
        for track in tracks:
            frames = track.get("frames", [])
            projected = np.array(track.get("projected", []))

            # Subtract offset (to align with reference camera)
            projected_adjusted = projected - offset

            aligned_track = {
                "track_id": track.get("track_id"),
                "team": "ball",
                "frame_range": track.get("frame_range"),
                "frames": frames,
                "projected": projected_adjusted.tolist(),
            }
            aligned_camera_tracks.append(aligned_track)

        aligned_tracks.append(aligned_camera_tracks)

    return aligned_tracks


def print_track_info(all_tracks_by_camera: List[List[Dict]], verbose: bool = False):
    """Print detailed track information for debugging."""
    if not verbose:
        return

    print("\n📊 Track Details:")
    for cam_idx, tracks in enumerate(all_tracks_by_camera):
        if not tracks:
            continue

        for track_idx, track in enumerate(tracks[:3]):  # First 3 tracks per camera
            frames = track.get("frames", [])
            if frames:
                print(
                    f"  Camera {cam_idx}, Track {track_idx}: frames {min(frames)}-{max(frames)} ({len(frames)} points)"
                )


def fuse_ball_tracks(
    ball_jsonl_paths: List[str],
    output_path: str,
    max_gap_frames: int = 30,
    max_distance_threshold: float = 200.0,
    min_cameras: int = 2,
    calibrate_frames: bool = True,
    calibrate_coordinates: bool = True,
    max_search_offset: int = 150,
    overlap_threshold: int = 30,
    verbose: bool = False,
) -> None:
    """
    Fuse ball tracks from multiple camera angles into a single continuous track.

    Args:
        ball_jsonl_paths: List of paths to JSONL files (one per camera)
        output_path: Path to save fused ball tracks
        max_gap_frames: Maximum gap in frames before starting a new track
        max_distance_threshold: Maximum distance for outlier detection (in 0.1m units)
        min_cameras: Minimum number of cameras needed to keep a frame
        calibrate_frames: Whether to calibrate frame offsets between cameras
        calibrate_coordinates: Whether to calibrate coordinate system offsets
        max_search_offset: Maximum frame offset to search (±frames)
        overlap_threshold: Minimum overlapping frames needed for offset estimation
        verbose: Print progress information
    """
    if verbose:
        print(f"🎾 Loading ball tracks from {len(ball_jsonl_paths)} cameras...")

    # Load all ball tracks
    all_tracks_by_camera = []
    for path in ball_jsonl_paths:
        tracks = load_ball_tracks_from_jsonl(path)
        all_tracks_by_camera.append(tracks)
        if verbose:
            print(f"  📹 {Path(path).name}: {len(tracks)} ball tracks")
    # ✅ ADD THIS: Print track details
    print_track_info(all_tracks_by_camera, verbose)
    # Step 1: Calibrate frame offsets (temporal alignment)
    if calibrate_frames and len(all_tracks_by_camera) > 1:
        frame_offsets = calibrate_frame_offsets(
            all_tracks_by_camera,
            max_search_offset=max_search_offset,
            overlap_threshold=overlap_threshold,
            verbose=verbose,
        )

        # Apply frame offsets
        all_tracks_by_camera = apply_frame_offsets(all_tracks_by_camera, frame_offsets)

    # Step 2: Calibrate coordinate systems (spatial alignment)
    if calibrate_coordinates and len(all_tracks_by_camera) > 1:
        coord_offsets = calibrate_coordinate_systems(
            all_tracks_by_camera,
            min_overlap_frames=50,
            min_confidence=0.3,
            verbose=verbose,
        )

        # Apply coordinate offsets
        all_tracks_by_camera = apply_coordinate_offsets(all_tracks_by_camera, coord_offsets)

    # Build frame-to-positions mapping
    frame_positions = defaultdict(list)

    for camera_idx, tracks in enumerate(all_tracks_by_camera):
        for track in tracks:
            frames = track.get("frames", [])
            projected = track.get("projected", [])

            for i, frame in enumerate(frames):
                if i < len(projected):
                    position = np.array(projected[i])
                    frame_positions[frame].append(
                        {
                            "camera_idx": camera_idx,
                            "position": position,
                            "track_id": track.get("track_id"),
                        }
                    )

    if verbose:
        print(f"\n✅ Collected data for {len(frame_positions)} unique frames")

    # Process frames in order and create continuous tracks
    all_frames = sorted(frame_positions.keys())

    if not all_frames:
        print("⚠️ No ball frames found!")
        with open(output_path, "w") as f:
            pass  # Create empty file
        return

    fused_tracks = []
    current_track = {"frames": [], "projected": []}

    prev_frame = None
    total_outliers = 0
    total_points = 0

    for frame in all_frames:
        # Get all positions at this frame
        data_list = frame_positions[frame]
        positions = [d["position"] for d in data_list]

        total_points += len(positions)

        # Calculate robust average (removing outliers)
        avg_position = calculate_robust_average(
            positions,
            max_distance_threshold=max_distance_threshold,
            min_inliers=min_cameras,
        )

        if avg_position is None:
            total_outliers += len(positions)
            continue  # Skip this frame (all outliers or insufficient data)

        # Count outliers
        is_inlier = detect_outliers_at_frame(
            positions,
            max_distance_threshold=max_distance_threshold,
            min_inliers=min_cameras,
        )
        total_outliers += sum(1 for inlier in is_inlier if not inlier)

        # Check if we should start a new track (gap too large)
        if prev_frame is not None and (frame - prev_frame) > max_gap_frames:
            # Save current track if it has data
            if current_track["frames"]:
                track_to_save = {
                    "track_id": len(fused_tracks) + 1,
                    "team": "ball",
                    "frame_range": [
                        current_track["frames"][0],
                        current_track["frames"][-1],
                    ],
                    "frames": current_track["frames"],
                    "projected": current_track["projected"],
                }
                fused_tracks.append(track_to_save)

            # Start new track
            current_track = {"frames": [], "projected": []}

        # Add to current track
        current_track["frames"].append(frame)
        current_track["projected"].append(avg_position.tolist())

        prev_frame = frame

    # Save final track
    if current_track["frames"]:
        track_to_save = {
            "track_id": len(fused_tracks) + 1,
            "team": "ball",
            "frame_range": [current_track["frames"][0], current_track["frames"][-1]],
            "frames": current_track["frames"],
            "projected": current_track["projected"],
        }
        fused_tracks.append(track_to_save)

    # ✅ ADD: Fix fragmentation
    if len(fused_tracks) > 1:
        print(f"\n🔧 Post-processing {len(fused_tracks)} fused tracks...")

        # Step 2: Smooth
        # fused_tracks = smooth_tracks_median_filter(
        #     fused_tracks,
        #     window_size=5,
        #     verbose=verbose
        # )

        # Step 3: Merge with prediction
        fused_tracks = merge_tracks_with_motion_prediction(
            fused_tracks,
            max_merge_gap=100,
            max_prediction_error=200.0,
            min_track_length=10,
            verbose=verbose,
        )

    # Re-assign IDs
    for i, track in enumerate(fused_tracks):
        track["track_id"] = i + 1

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write to output
    with open(output_path, "w") as f:
        for track in fused_tracks:
            json.dump(track, f)
            f.write("\n")

    if verbose:
        print(f"\n✅ Fused ball tracks saved to: {output_path}")
        print(f"📊 Statistics:")
        print(f"  Total input points: {total_points}")
        print(f"  Outliers removed: {total_outliers} ({100*total_outliers/total_points:.1f}%)")
        print(f"  Total frames with data: {len(all_frames)}")
        print(f"  Continuous tracks created: {len(fused_tracks)}")
        for i, track in enumerate(fused_tracks[:20], 1):
            duration = track["frame_range"][1] - track["frame_range"][0] + 1
            coverage = len(track["frames"]) / duration if duration > 0 else 0
            print(
                f"    Track {i}: frames {track['frame_range'][0]}-{track['frame_range'][1]} "
                f"({len(track['frames'])} points, {100*coverage:.1f}% coverage)"
            )
        if len(fused_tracks) > 20:
            print(f"    ... and {len(fused_tracks) - 20} more tracks")


def merge_tracks_with_motion_prediction(
    tracks: List[Dict],
    max_merge_gap: int = 100,
    max_prediction_error: float = 200.0,  # 20m
    min_track_length: int = 10,
    verbose: bool = False,
) -> List[Dict]:
    """
    Merge tracks by predicting where the ball should be based on recent motion.
    This handles cases where the ball continues moving in the same direction.
    """
    if len(tracks) <= 1:
        return tracks

    # Sort by start frame
    sorted_tracks = sorted(tracks, key=lambda t: t["frames"][0])

    merged = [sorted_tracks[0]]
    n_merges = 0

    for current_track in sorted_tracks[1:]:
        last_merged = merged[-1]

        # Get temporal gap
        gap = current_track["frames"][0] - last_merged["frames"][-1]

        if gap > max_merge_gap:
            merged.append(current_track)
            continue

        # ✅ KEY: Predict ball position based on recent motion
        # Use last 5 points to estimate velocity and acceleration
        last_frames = np.array(last_merged["frames"][-5:])
        last_positions = np.array(last_merged["projected"][-5:])

        if len(last_frames) >= 3:
            # Fit polynomial to recent trajectory (captures acceleration)
            # p(t) = a*t^2 + b*t + c
            t = last_frames - last_frames[0]

            # Fit x and y separately
            px = np.polyfit(t, last_positions[:, 0], deg=2)
            py = np.polyfit(t, last_positions[:, 1], deg=2)

            # Predict position at current_track start
            t_predict = current_track["frames"][0] - last_frames[0]
            predicted_x = np.polyval(px, t_predict)
            predicted_y = np.polyval(py, t_predict)
            predicted_pos = np.array([predicted_x, predicted_y])

            # Compare with actual position
            actual_pos = np.array(current_track["projected"][0])
            prediction_error = np.linalg.norm(actual_pos - predicted_pos)

            # Decide if we should merge
            should_merge = prediction_error < max_prediction_error

            if verbose and should_merge:
                print(f"  🔗 Merging tracks (gap={gap}, prediction_error={prediction_error:.1f})")
            elif verbose and gap < 50:  # Only warn for small gaps
                print(
                    f"  ❌ Not merging: prediction error too large ({prediction_error:.1f} > {max_prediction_error})"
                )

            if should_merge:
                # Merge with linear interpolation across gap
                if gap > 1:
                    # Interpolate missing frames
                    missing_frames = list(
                        range(last_merged["frames"][-1] + 1, current_track["frames"][0])
                    )

                    for frame in missing_frames:
                        t_interp = frame - last_frames[0]
                        interp_x = np.polyval(px, t_interp)
                        interp_y = np.polyval(py, t_interp)

                        last_merged["frames"].append(frame)
                        last_merged["projected"].append([interp_x, interp_y])

                # Add current track
                last_merged["frames"].extend(current_track["frames"])
                last_merged["projected"].extend(current_track["projected"])
                last_merged["frame_range"] = [
                    last_merged["frames"][0],
                    last_merged["frames"][-1],
                ]

                n_merges += 1
                continue

        # If we can't predict, fall back to spatial distance check
        spatial_dist = np.linalg.norm(
            np.array(current_track["projected"][0]) - np.array(last_merged["projected"][-1])
        )

        if gap <= 20 and spatial_dist < 100.0:  # Very conservative for short gaps
            last_merged["frames"].extend(current_track["frames"])
            last_merged["projected"].extend(current_track["projected"])
            last_merged["frame_range"] = [
                last_merged["frames"][0],
                last_merged["frames"][-1],
            ]
            n_merges += 1
            if verbose:
                print(f"  🔗 Merging tracks (gap={gap}, spatial_dist={spatial_dist:.1f})")
        else:
            merged.append(current_track)

    # Filter out very short tracks that are isolated
    filtered = []
    for i, track in enumerate(merged):
        if len(track["frames"]) < min_track_length:
            # Check if surrounded by longer tracks
            has_neighbor = False
            if i > 0:
                gap_before = track["frames"][0] - merged[i - 1]["frames"][-1]
                if gap_before < 100:
                    has_neighbor = True
            if i < len(merged) - 1:
                gap_after = merged[i + 1]["frames"][0] - track["frames"][-1]
                if gap_after < 100:
                    has_neighbor = True

            if not has_neighbor:
                if verbose:
                    print(f"  🗑️ Removing isolated short track (length={len(track['frames'])})")
                continue

        filtered.append(track)

    if verbose:
        print(f"\n✅ Track merging: {len(tracks)} → {len(filtered)} tracks ({n_merges} merges)")

    return filtered


def smooth_tracks_median_filter(
    tracks: List[Dict],
    window_size: int = 5,
    verbose: bool = False,
) -> List[Dict]:
    """
    Apply median filter to smooth tracks without Kalman.
    Robust to outliers.
    """

    smoothed_tracks = []

    for track in tracks:
        if len(track["frames"]) < window_size:
            smoothed_tracks.append(track)
            continue

        positions = np.array(track["projected"])

        # Apply median filter separately to x and y
        smoothed_x = medfilt(positions[:, 0], kernel_size=window_size)
        smoothed_y = medfilt(positions[:, 1], kernel_size=window_size)

        smoothed_positions = np.column_stack([smoothed_x, smoothed_y])

        smoothed_track = {
            "track_id": track["track_id"],
            "team": "ball",
            "frame_range": track["frame_range"],
            "frames": track["frames"],
            "projected": smoothed_positions.tolist(),
            "frame_range": track["frame_range"],
        }

        smoothed_tracks.append(smoothed_track)

    if verbose:
        print(f"✅ Smoothed {len(tracks)} tracks with median filter (window={window_size})")

    return smoothed_tracks


def main():

    parser = argparse.ArgumentParser(description="Fuse ball tracks from multiple cameras")
    parser.add_argument(
        "--ball-jsonl-paths", nargs="+", help="Paths to input JSONL files (one per camera)"
    )
    parser.add_argument("--output", "-o", required=True, help="Path to output fused JSONL file")
    parser.add_argument(
        "--max-gap",
        type=int,
        default=30,
        help="Maximum gap in frames before starting new track (default: 30)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=200.0,
        help="Maximum distance threshold for outlier detection (default: 200.0)",
    )
    parser.add_argument(
        "--min-cameras",
        type=int,
        default=2,
        help="Minimum number of cameras that must agree (default: 2)",
    )
    parser.add_argument(
        "--no-calibrate-frames",
        action="store_true",
        help="Skip frame offset calibration between cameras",
    )
    parser.add_argument(
        "--no-calibrate-coordinates",
        action="store_true",
        help="Skip coordinate system calibration between cameras",
    )
    parser.add_argument(
        "--max-search-offset",
        type=int,
        default=60,
        help="Maximum frame offset to search when calibrating (default: 60)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=int,
        default=30,
        help="Minimum overlapping frames needed for offset estimation (default: 30)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    if len(args.ball_jsonl_paths) < 2:
        print("⚠️ Warning: Fusion works best with 2+ camera angles")

    min_cameras = max(1, len(args.ball_jsonl_paths) - 2)

    fuse_ball_tracks(
        ball_jsonl_paths=args.ball_jsonl_paths,
        output_path=args.output,
        max_gap_frames=args.max_gap,
        max_distance_threshold=args.max_distance,
        min_cameras=min_cameras,
        calibrate_frames=not args.no_calibrate_frames,
        calibrate_coordinates=not args.no_calibrate_coordinates,
        max_search_offset=args.max_search_offset,
        overlap_threshold=args.overlap_threshold,
        verbose=not args.quiet,
    )

    smoothen_fused_ball_tracking(
        args.output,
        args.output.replace(".jsonl", "_final.jsonl"),
    )


if __name__ == "__main__":
    main()
