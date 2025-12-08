import json
import os

from click import group
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import argparse

def find_overlapping_frames(track1: Dict, track2: Dict) -> List[int]:
    """
    Find frames that exist in both tracks.
    
    Returns:
        List of frame numbers that overlap between the two tracks
    """
    frames1_set = set(track1['frames'])
    frames2_set = set(track2['frames'])
    overlapping = sorted(frames1_set & frames2_set)
    return overlapping

def get_position_at_frame(track: Dict, target_frame: int) -> Optional[np.ndarray]:
    """
    Get the actual observed position for a given frame (NO interpolation).
    Returns None if the frame doesn't exist in the track.
    
    Args:
        track: Track dictionary with 'frames' and 'projected' fields
        target_frame: Frame number to get position for
        
    Returns:
        [x, y] position as np.ndarray or None if frame not found
    """
    frames = track.get('frames', [])
    projected = track.get('projected', [])
    
    if target_frame not in frames:
        return None
    
    try:
        idx = frames.index(target_frame)
        if idx < len(projected):
            return np.array(projected[idx])
    except (ValueError, IndexError):
        pass
    
    return None

def get_interpolated_position(track: Dict, target_frame: int) -> Optional[np.ndarray]:
    """
    Get interpolated position for a given frame.
    Handles frame alignment errors by interpolating between known positions.
    
    Args:
        track: Track dictionary with 'frames' and 'projected' fields
        target_frame: Frame number to get position for
        
    Returns:
        Interpolated [x, y] position or None if extrapolation would be needed
    """
    frames = np.array(track['frames'])
    projected = np.array(track['projected'])
    
    # Check if target frame is within the range
    if target_frame < frames.min() or target_frame > frames.max():
        return None
    
    # If exact frame exists, return it
    if target_frame in frames:
        idx = np.where(frames == target_frame)[0][0]
        return projected[idx]
    
    # Interpolate
    # Use linear interpolation for x and y separately
    try:
        interp_x = interp1d(frames, projected[:, 0], kind='linear', 
                           bounds_error=True, fill_value='extrapolate')
        interp_y = interp1d(frames, projected[:, 1], kind='linear', 
                           bounds_error=True, fill_value='extrapolate')
        
        return np.array([interp_x(target_frame), interp_y(target_frame)])
    except:
        return None

def calculate_pairwise_distances(track1: Dict, track2: Dict,
                                  overlapping_frames: List[int],
                                  max_distance: float = 100.0,
                                  max_outlier_ratio: float = 0.3) -> Tuple[List[float], Dict]:
    """
    Calculate per-frame distances between two tracks, filtering outliers.
    
    Args:
        track1, track2: Track dictionaries
        overlapping_frames: List of frames to compare
        max_distance: Maximum allowed distance per frame (outlier threshold)
        max_outlier_ratio: If more than this ratio are outliers, return empty list
        
    Returns:
        Tuple of (list of valid distances, metadata_dict)
        Returns ([], metadata) if too many outliers or insufficient data
    """
    distances = []
    outlier_frames = []
    skipped_missing = 0
    
    for frame in overlapping_frames:
        pos1 = get_position_at_frame(track1, frame)
        pos2 = get_position_at_frame(track2, frame)
        
        if pos1 is None or pos2 is None:
            skipped_missing += 1
            continue
        
        dist = euclidean(pos1, pos2)
        
        if dist > max_distance:
            outlier_frames.append((frame, dist))
        else:
            distances.append(dist)
    
    n_valid = len(distances)
    n_outliers = len(outlier_frames)
    total_compared = n_valid + n_outliers
    outlier_ratio = n_outliers / total_compared if total_compared > 0 else 0.0
    
    metadata = {
        'n_valid': n_valid,
        'n_outliers': n_outliers,
        'n_skipped_missing': skipped_missing,
        'total_compared': total_compared,
        'outlier_ratio': outlier_ratio,
        'max_distance': max_distance,
        'outlier_frames': outlier_frames[:10]
    }
    
    # Reject if too many outliers
    if total_compared > 0 and outlier_ratio > max_outlier_ratio:
        metadata['rejected'] = True
        metadata['reason'] = f'outlier_ratio {outlier_ratio:.2%} > {max_outlier_ratio:.2%}'
        return [], metadata
    
    metadata['rejected'] = False
    return distances, metadata

def calculate_average_distance(track1: Dict, track2: Dict, 
                               overlapping_frames: List[int],
                               max_distance: float = 100.0,
                               max_outlier_ratio: float = 0.3) -> Tuple[float, Dict]:
    """
    Calculate average Euclidean distance between tracks in overlapping region.
    Filters out outlier points and rejects if too many outliers.
    
    Args:
        track1, track2: Track dictionaries
        overlapping_frames: List of frames to compare
        max_distance: Maximum allowed distance per frame (outlier threshold)
        max_outlier_ratio: If more than this ratio are outliers, return inf
        
    Returns:
        Tuple of (average distance, metadata_dict)
        Returns (inf, metadata) if rejected
    """
    distances, metadata = calculate_pairwise_distances(
        track1, track2, overlapping_frames,
        max_distance=max_distance,
        max_outlier_ratio=max_outlier_ratio
    )
    
    if metadata['rejected'] or len(distances) == 0:
        return float('inf'), metadata
    
    avg_dist = np.mean(distances)
    metadata['average_distance'] = avg_dist
    return avg_dist, metadata

def calculate_median_distance(track1: Dict, track2: Dict, 
                              overlapping_frames: List[int],
                              max_distance: float = 100.0,
                              max_outlier_ratio: float = 0.3) -> Tuple[float, Dict]:
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
        track1, track2, overlapping_frames,
        max_distance=max_distance,
        max_outlier_ratio=max_outlier_ratio
    )
    
    if metadata['rejected'] or len(distances) == 0:
        return float('inf'), metadata
    
    median_dist = np.median(distances)
    metadata['median_distance'] = median_dist
    metadata['std_distance'] = np.std(distances)
    metadata['min_distance'] = np.min(distances)
    metadata['max_distance_observed'] = np.max(distances)
    return median_dist, metadata

def calculate_median_velocity_difference(track1: Dict, track2: Dict,
                                         overlapping_frames: List[int], 
                                         fps: float, 
                                         window_size: int) -> float:
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
            
            pos_start = get_position_at_frame(track, frame_start)
            pos_end = get_position_at_frame(track, frame_end)
            
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

def calculate_total_distance(track: Dict, overlapping_frames: List[int],
                             max_step_distance: float = 50.0,
                             max_outlier_ratio: float = 0.3) -> Tuple[float, Dict]:
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
        pos = get_position_at_frame(track, frame)
        
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
        'n_valid_steps': n_valid_steps,
        'n_outlier_steps': n_outlier_steps,
        'total_steps': total_steps,
        'outlier_ratio': outlier_ratio,
        'max_step_distance': max_step_distance,
        'outlier_frames': outlier_frames[:10]  # Limit for readability
    }
    
    # Reject if too many outliers
    if total_steps > 0 and outlier_ratio > max_outlier_ratio:
        metadata['rejected'] = True
        metadata['reason'] = f'outlier_ratio {outlier_ratio:.2%} > {max_outlier_ratio:.2%}'
        return float('inf'), metadata
    
    metadata['rejected'] = False
    return total_distance, metadata

def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate angle between two vectors using dot product (more robust than atan2).
    Returns angle in degrees [0, 180].
    
    Args:
        v1, v2: Movement vectors
        
    Returns:
        Angle in degrees between the two vectors
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def calculate_direction_similarity(track1: Dict, track2: Dict, 
                                   overlapping_frames: List[int],
                                   direction_threshold: float = 45.0,
                                   frame_stride: int = 3,
                                   min_movement_threshold: float = 0.5) -> Tuple[float, Dict]:
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
        return 0.0, {'reason': 'insufficient_frames', 'valid_comparisons': 0, 'frame_stride': frame_stride}
    
    angle_diffs = []
    direction_matches = 0
    skipped_stationary = 0
    
    # Sample frames with stride
    for i in range(0, len(overlapping_frames) - frame_stride, frame_stride):
        frame_start = overlapping_frames[i]
        frame_end = overlapping_frames[i + frame_stride]
        
        # Get positions for both tracks
        pos1_start = get_position_at_frame(track1, frame_start)
        pos1_end = get_position_at_frame(track1, frame_end)
        pos2_start = get_position_at_frame(track2, frame_start)
        pos2_end = get_position_at_frame(track2, frame_end)
        
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
            'reason': 'insufficient_valid_movements',
            'valid_comparisons': len(angle_diffs),
            'skipped_stationary': skipped_stationary,
            'frame_stride': frame_stride
        }
    
    # Calculate metrics
    mean_angle_diff = np.mean(angle_diffs)
    median_angle_diff = np.median(angle_diffs)
    direction_consistency = direction_matches / len(angle_diffs)
    
    # Similarity score: 0° diff = 1.0, 180° diff = 0.0
    similarity = 1.0 - (mean_angle_diff / 180.0)
    
    metadata = {
        'valid_comparisons': len(angle_diffs),
        'mean_angle_diff': mean_angle_diff,
        'median_angle_diff': median_angle_diff,
        'direction_consistency': direction_consistency,
        'similar_directions': direction_matches,
        'angle_diff_std': np.std(angle_diffs),
        'skipped_stationary': skipped_stationary,
        'frame_stride': frame_stride,
        'angle_distribution': {
            'very_similar': sum(1 for d in angle_diffs if d < 30),     # 0-30°
            'similar': sum(1 for d in angle_diffs if 30 <= d < 60),    # 30-60°
            'moderate': sum(1 for d in angle_diffs if 60 <= d < 120),  # 60-120°
            'opposite': sum(1 for d in angle_diffs if d >= 120)        # 120-180°
        }
    }
    
    return similarity, metadata    

def calculate_match_score(track1: Dict, track2: Dict,
                          min_overlap_frames: int = 10,
                          max_analysis_frames: int = 150,
                          direction_frame_stride: int = 3,
                          max_point_distance: float = 100.0,
                          max_step_distance: float = 50.0,
                          max_outlier_ratio: float = 0.3) -> Tuple[float, Dict]:
    """
    Calculate a composite match score between two tracks.
    Lower score = better match. Returns inf if tracks are incompatible.
    
    Args:
        track1, track2: Track dictionaries
        min_overlap_frames: Minimum overlapping frames required
        max_analysis_frames: Maximum frames to analyze
        direction_frame_stride: Frame stride for direction calculation
        max_point_distance: Max allowed distance between corresponding points (outlier threshold)
        max_step_distance: Max allowed distance per movement step (outlier threshold)
        max_outlier_ratio: If more than this ratio are outliers, reject match
        
    Returns:
        Tuple of (score: float, metadata: dict)
    """
    # Find overlapping frames
    overlapping_frames = find_overlapping_frames(track1, track2)
    
    if len(overlapping_frames) > max_analysis_frames:
        overlapping_frames = overlapping_frames[:max_analysis_frames]
    
    metadata = {
        'overlap_count': len(overlapping_frames),
        'track1_id': track1['track_id'],
        'track2_id': track2['track_id'],
    }
    
    # No overlap = infinite score (incompatible)
    if len(overlapping_frames) < min_overlap_frames:
        metadata['reason'] = 'insufficient_overlap'
        return float('inf'), metadata
    
    # Fast rejection: initial position check
    if len(track1['projected']) > 0 and len(track2['projected']) > 0:
        initial_pos1 = np.array(track1['projected'][0])
        initial_pos2 = np.array(track2['projected'][0])
        initial_distance = np.linalg.norm(initial_pos1 - initial_pos2)
        if initial_distance >= 150.0:  # Relaxed from 100 to 150
            metadata['reason'] = 'initial_position_too_far'
            metadata['initial_distance'] = initial_distance
            return float('inf'), metadata
    
    # Calculate median distance with outlier filtering
    median_distance, median_meta = calculate_median_distance(
        track1, track2, overlapping_frames,
        max_distance=max_point_distance,
        max_outlier_ratio=max_outlier_ratio
    )
    # # Calculate average distance with outlier filtering
    # avg_distance, avg_meta = calculate_average_distance(
    #     track1, track2, overlapping_frames,
    #     max_distance=max_point_distance,
    #     max_outlier_ratio=max_outlier_ratio
    # )
    
    # Calculate total distances with outlier filtering
    total_distance_track1, td1_meta = calculate_total_distance(
        track1, overlapping_frames,
        max_step_distance=max_step_distance,
        max_outlier_ratio=max_outlier_ratio
    )
    
    total_distance_track2, td2_meta = calculate_total_distance(
        track2, overlapping_frames,
        max_step_distance=max_step_distance,
        max_outlier_ratio=max_outlier_ratio
    )

    # Check if either total distance was rejected
    if total_distance_track1 == float('inf'):
        metadata['reason'] = 'track1_total_distance_rejected'
        metadata['track1_distance_metadata'] = td1_meta
        return float('inf'), metadata
    
    if total_distance_track2 == float('inf'):
        metadata['reason'] = 'track2_total_distance_rejected'
        metadata['track2_distance_metadata'] = td2_meta
        return float('inf'), metadata
    
    distance_diff = abs(total_distance_track1 - total_distance_track2)
    
    velocity_diff = calculate_median_velocity_difference(
        track1, track2, overlapping_frames, fps=30, window_size=5
    )
    
    direction_sim, direction_meta = calculate_direction_similarity(
        track1, track2, overlapping_frames, frame_stride=direction_frame_stride
    )
    
    metadata.update({
        'median_distance': median_distance,
        'median_metadata': median_meta,
        # 'average_distance': avg_distance,
        'distance_diff': distance_diff,
        'total_distance_track1': total_distance_track1,
        'total_distance_track2': total_distance_track2,
        'track1_outliers': td1_meta.get('n_outlier_steps', 0),
        'track2_outliers': td2_meta.get('n_outlier_steps', 0),
        'point_outliers': median_meta.get('n_outliers', 0),
        'velocity_difference': velocity_diff,
        'direction_similarity': direction_sim,
        'direction_metadata': direction_meta,
    })
    
    # Composite score: weighted combination (lower = better)
    # Normalize each component to roughly similar scales
    
    # Median distance
    distance_score = median_distance / 10.0  
    
    # Distance diff
    distance_diff_score = distance_diff / 10.0 
    
    # Direction: 0-1 (higher is better), invert for score
    direction_score = (1.0 - direction_sim) * 10.0 
    
    # Overlap bonus: more overlap = lower score
    overlap_bonus = -min(len(overlapping_frames) / 500.0, 5.0)  # -5 to 0
    
    # Velocity diff penalty
    velocity_score = min(velocity_diff / 5.0, 3.0)  # Cap at 3
    
    # Composite score
    score = (
        distance_score * 2.0 +      # Weight: 2x
        distance_diff_score * 3.0 + # Weight: 3x
        direction_score * 2.5 +     # Weight: 2.5x
        velocity_score * 0.5 +      # Weight: 0.5x
        overlap_bonus               # Bonus for overlap
    )
    
    metadata['composite_score'] = score
    metadata['score_components'] = {
        'distance_score': distance_score,
        'distance_diff_score': distance_diff_score,
        'direction_score': direction_score,
        'velocity_score': velocity_score,
        'overlap_bonus': overlap_bonus,
    }
    
    return score, metadata

def greedy_match_tracks(
    jsonl_paths: List[str],
    min_overlap_frames: int = 10,
    max_analysis_frames: int = 150,
    filter_by_team: bool = True,
    filter_by_jersey: bool = False,
    direction_frame_stride: int = 3,
    max_score_threshold: float = 50.0,
    temporal_overlap_threshold: float = 0.3,
    max_point_distance: float = 100.0,
    max_step_distance: float = 50.0,
    max_outlier_ratio: float = 0.3,
    verbose: bool = False
) -> Tuple[Dict, Dict[int, List[Dict]]]:
    """
    Greedy matching: tracks find best matches across multiple files.
    Supports many-to-one matching (multiple short tracks can match one long track)
    as long as the short tracks don't overlap temporally.
    
    Args:
        jsonl_paths: List of JSONL file paths (supports 2+ files)
        min_overlap_frames: Minimum overlapping frames required
        max_analysis_frames: Maximum frames to analyze per comparison
        filter_by_team: Only compare tracks from same team
        filter_by_jersey: Only compare tracks with same jersey number
        direction_frame_stride: Frame stride for direction calculation
        max_score_threshold: Maximum score to accept a match
        temporal_overlap_threshold: Maximum allowed overlap ratio (0-1) between 
                                    tracks from same source matching same target
        verbose: Print detailed progress
        
    Returns:
        Tuple of (results dict, all_tracks_by_file dict)
    """
    # Load all tracks
    all_tracks = []
    all_tracks_by_file = {}
    
    for src_idx, path in enumerate(jsonl_paths):
        tracks = load_tracks_from_jsonl(path)
        print(f"Loaded {len(tracks)} tracks from {path}")
        all_tracks_by_file[src_idx] = [t.copy() for t in tracks]
        
        for orig_idx, t in enumerate(tracks):
            t['_source_idx'] = src_idx
            t['_orig_idx'] = orig_idx
            t['_global_idx'] = len(all_tracks)
            all_tracks.append(t)
    
    n_total = len(all_tracks)
    n_files = len(jsonl_paths)
    print(f"Total tracks: {n_total} from {n_files} files")
    
    # Step 1: Compute all pairwise scores
    print("\n📊 Computing pairwise match scores...")
    all_pairs = []
    
    total_pairs = 0
    skipped_filter = 0
    skipped_incompatible = 0
    
    for i in range(n_total):
        if (i + 1) % 20 == 0:
            print(f"  Processing track {i + 1}/{n_total}...")
        
        t1 = all_tracks[i]
        
        for j in range(i + 1, n_total):
            t2 = all_tracks[j]
            
            # Skip same file
            if t1['_source_idx'] == t2['_source_idx']:
                continue
            
            total_pairs += 1
            
            # Apply filters
            if filter_by_team and t1.get('team') != t2.get('team'):
                skipped_filter += 1
                continue
            if filter_by_jersey and t1.get('jersey_num') != t2.get('jersey_num'):
                skipped_filter += 1
                continue
            
            # Calculate score
            score, metadata = calculate_match_score(
                t1, t2,
                min_overlap_frames=min_overlap_frames,
                max_analysis_frames=max_analysis_frames,
                direction_frame_stride=direction_frame_stride,
                max_point_distance=max_point_distance,
                max_step_distance=max_step_distance,
                max_outlier_ratio=max_outlier_ratio
            )
            
            if score == float('inf'):
                skipped_incompatible += 1
                continue
            
            all_pairs.append((score, i, j, metadata))
    
    print(f"  Total candidate pairs: {len(all_pairs)}")
    print(f"  Skipped (filter): {skipped_filter}")
    print(f"  Skipped (incompatible): {skipped_incompatible}")
    
    # Step 2: Sort by score (best first)
    all_pairs.sort(key=lambda x: x[0])
    
    if verbose and all_pairs:
        print(f"\n  Top 5 best pairs:")
        for score, i, j, meta in all_pairs[:5]:
            print(f"    {all_tracks[i]['track_id']} <-> {all_tracks[j]['track_id']}: score={score:.2f}")
    
    # Step 3: Greedy matching with many-to-one support
    print("\n🎯 Performing greedy matching...")
    
    # Track which tracks from each source have matched to a target track
    # matched_to_target[target_idx][source_idx] = list of (matched_idx, frame_start, frame_end)
    matched_to_target = defaultdict(lambda: defaultdict(list))
    
    matches = []
    groups = defaultdict(set)
    track_to_group = {}
    next_group_id = 0
    
    def get_match_frame_range(idx1: int, idx2: int) -> Tuple[int, int]:
        """Get the overlapping frame range for a match."""
        t1 = all_tracks[idx1]
        t2 = all_tracks[idx2]
        
        frames1 = set(t1.get('frames', []))
        frames2 = set(t2.get('frames', []))
        overlap = sorted(frames1 & frames2)
        
        if overlap:
            return (overlap[0], overlap[-1])
        
        range1 = t1.get('frame_range', [0, 0])
        range2 = t2.get('frame_range', [0, 0])
        start = max(range1[0], range2[0])
        end = min(range1[1], range2[1])
        return (start, end)
    
    def get_all_group_members(idx: int) -> Set[int]:
        """Get all tracks in the same group as idx (BFS)."""
        if idx not in track_to_group:
            return {idx}
        group_id = track_to_group[idx]
        return groups[group_id].copy()
    
    def would_create_source_conflict(idx1: int, idx2: int,
                                     match_start: int, match_end: int) -> bool:
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
            src = all_tracks[idx]['_source_idx']
            frame_range = all_tracks[idx].get('frame_range', [0, 0])
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
                                print(f"    ⚠️  Would create conflict: {all_tracks[idx_i]['track_id']} "
                                      f"and {all_tracks[idx_j]['track_id']} (both from source {src}) "
                                      f"overlap {max_overlap_ratio:.1%} in frames [{overlap_start}, {overlap_end}]")
                            return True
        
        return False
    
    for score, idx1, idx2, metadata in all_pairs:
        if score > max_score_threshold:
            continue
        
        match_start, match_end = get_match_frame_range(idx1, idx2)
        
        # NEW: Check if this match would create source conflicts in the merged group
        if would_create_source_conflict(idx1, idx2, match_start, match_end):
            if verbose:
                print(f"  Skipping {all_tracks[idx1]['track_id']} <-> {all_tracks[idx2]['track_id']}: "
                      f"would create source conflict in merged group")
            continue
        
        # Accept this match - record in both directions
        src1 = all_tracks[idx1]['_source_idx']
        src2 = all_tracks[idx2]['_source_idx']
        
        matched_to_target[idx1][src2].append((idx2, match_start, match_end))
        matched_to_target[idx2][src1].append((idx1, match_start, match_end))
        
        matches.append({
            'i': idx1,
            'j': idx2,
            'score': score,
            'metadata': metadata,
            'frame_range': [match_start, match_end]
        })
        
        if verbose:
            print(f"  ✓ Matched {all_tracks[idx1]['track_id']} <-> {all_tracks[idx2]['track_id']}: "
                  f"score={score:.2f}, frames {match_start}-{match_end}")
        
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
        
        track_ids = [all_tracks[i]['track_id'] for i in member_indices]
        source_indices = [all_tracks[i]['_source_idx'] for i in member_indices]
        teams = [all_tracks[i].get('team') for i in member_indices]
        jerseys = [all_tracks[i].get('jersey_num') for i in member_indices]
        frame_ranges = [all_tracks[i].get('frame_range', [0, 0]) for i in member_indices]
        
        # Sort by source_idx first, then by track_id
        # Create list of tuples for sorting
        combined = list(zip(source_indices, track_ids, teams, jerseys, frame_ranges, member_indices))
        
        # Sort by source_idx (primary), then track_id (secondary)
        # Extract numeric part from track_id for sorting (e.g., '12a' -> 12)
        def extract_numeric(track_id):
            """Extract numeric part from track_id like '12a' -> 12"""
            import re
            match = re.match(r'(\d+)', str(track_id))
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
            'median_distance': [],
            'distance_diff': [],
            'direction_similarity': [],
            'scores': [],
            'pairs': [],
            'frame_ranges': []
        }
        
        for a in range(len(member_list)):
            for b in range(a + 1, len(member_list)):
                ia, ib = member_list[a], member_list[b]
                for m in matches:
                    if (m['i'] == ia and m['j'] == ib) or (m['i'] == ib and m['j'] == ia):
                        pairwise_data['median_distance'].append(m['metadata'].get('median_distance'))
                        pairwise_data['distance_diff'].append(m['metadata'].get('distance_diff'))
                        pairwise_data['direction_similarity'].append(m['metadata'].get('direction_similarity'))
                        pairwise_data['scores'].append(m['score'])
                        pairwise_data['pairs'].append((all_tracks[ia]['track_id'], all_tracks[ib]['track_id']))
                        pairwise_data['frame_ranges'].append(m.get('frame_range'))
        
        aggregated.append({
            'track_id': track_ids,
            'source_idx': source_indices,
            'frame_ranges': frame_ranges,  # Now included directly
            'team': teams,
            'jersey_num': jerseys,
            'score(s)': pairwise_data['scores'] if pairwise_data['scores'] else None,
            'median_distance': pairwise_data['median_distance'],
            'distance_diff': pairwise_data['distance_diff'],
            'direction_similarity': pairwise_data['direction_similarity'],
            'pairs': pairwise_data['pairs'],
            'match_frame_ranges': pairwise_data['frame_ranges']  # Renamed to distinguish from track frame_ranges
        })
    
    # Find unmatched tracks
    unmatched = []
    for i in range(n_total):
        if i not in matched_indices:
            t = all_tracks[i]
            unmatched.append({
                'track_id': t['track_id'],
                'team': t.get('team'),
                'jersey_num': t.get('jersey_num'),
                'source_idx': t['_source_idx'],
                'frame_range': t.get('frame_range')
            })
    
    # Count many-to-one matches
    many_to_one_count = sum(
        1 for target_idx in matched_to_target
        for src_idx in matched_to_target[target_idx]
        if len(matched_to_target[target_idx][src_idx]) > 1
    )
    
    results = {
        'aggregated_groups': aggregated,
        'pairwise_matches': matches,
        'unmatched_tracks': unmatched,
        'stats': {
            'total_tracks': n_total,
            'total_files': n_files,
            'total_candidate_pairs': len(all_pairs),
            'matches_accepted': len(matches),
            'groups_count': len(groups),
            'matched_tracks': len(matched_indices),
            'unmatched_tracks': len(unmatched),
            'match_rate': len(matched_indices) / n_total if n_total > 0 else 0,
            'many_to_one_matches': many_to_one_count
        }
    }
    
    return results, all_tracks_by_file

def are_tracks_similar(track1: Dict, track2: Dict, 
                       min_overlap_frames: int = 10,
                       median_distance_threshold: float = 50.0,
                       max_analysis_frames: int = 150,
                       total_distance_diff_threshold: float = 15.0,
                       direction_similarity_threshold: float = 0.6,
                       direction_consistency_threshold: float = 0.5,
                       direction_frame_stride: int = 3,
                       max_point_distance: float = 100.0,
                       max_step_distance: float = 50.0,
                       max_outlier_ratio: float = 0.3) -> Tuple[bool, Dict]:
    """
    Compare two tracks for similarity based on overlapping frames.
    Filters outliers and rejects matches with too many outliers.
    
    Args:
        track1, track2: Track dictionaries
        min_overlap_frames: Minimum overlapping frames required
        median_distance_threshold: Maximum median distance for similarity
        max_analysis_frames: Maximum frames to analyze
        total_distance_diff_threshold: Maximum allowed difference in total distance
        direction_similarity_threshold: Minimum direction similarity score (0-1)
        direction_consistency_threshold: Minimum ratio of frames with similar direction
        direction_frame_stride: Frame stride for direction sampling
        max_point_distance: Max allowed distance between corresponding points
        max_step_distance: Max allowed distance per movement step
        max_outlier_ratio: If more than this ratio are outliers, reject match
        
    Returns:
        Tuple of (is_similar: bool, metadata: dict)
    """
    # Fast check 1: Compare initial projected points
    if len(track1['projected']) > 0 and len(track2['projected']) > 0:
        initial_pos1 = np.array(track1['projected'][0])
        initial_pos2 = np.array(track2['projected'][0])
        initial_distance = np.linalg.norm(initial_pos1 - initial_pos2)
        
        if initial_distance >= 100.0:
            return False, {
                'reason': 'initial_position_too_far',
                'initial_distance': initial_distance,
                'track1_id': track1['track_id'],
                'track2_id': track2['track_id'],
                'overlap_count': 0,
                'track1_frames': len(track1['frames']),
                'track2_frames': len(track2['frames']),
            }
    
    overlapping_frames = find_overlapping_frames(track1, track2)
    
    if len(overlapping_frames) > max_analysis_frames:
        overlapping_frames = overlapping_frames[:max_analysis_frames]
    
    metadata = {
        'overlap_count': len(overlapping_frames),
        'track1_id': track1['track_id'],
        'track2_id': track2['track_id'],
        'track1_frames': len(track1['frames']),
        'track2_frames': len(track2['frames']),
    }
    
    if len(overlapping_frames) == 0:
        metadata['reason'] = 'no_overlap'
        metadata['average_distance'] = None
        metadata['median_distance'] = None
        return False, metadata
    
    if len(overlapping_frames) < min_overlap_frames:
        metadata['reason'] = 'insufficient_overlap'
        metadata['average_distance'] = None
        metadata['median_distance'] = None
        return False, metadata
    
    # Calculate distances with outlier filtering
    avg_distance, avg_meta = calculate_average_distance(
        track1, track2, overlapping_frames,
        max_distance=max_point_distance,
        max_outlier_ratio=max_outlier_ratio
    )
    
    median_distance, median_meta = calculate_median_distance(
        track1, track2, overlapping_frames,
        max_distance=max_point_distance,
        max_outlier_ratio=max_outlier_ratio
    )
    
    # Check for rejection due to outliers
    if median_distance == float('inf'):
        metadata['reason'] = median_meta.get('reason', 'too_many_point_outliers')
        metadata['outlier_metadata'] = median_meta
        print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
              f"REJECTED - {metadata['reason']}")
        return False, metadata
    
    # Calculate total distances with outlier filtering
    total_distance_track1, td1_meta = calculate_total_distance(
        track1, overlapping_frames,
        max_step_distance=max_step_distance,
        max_outlier_ratio=max_outlier_ratio
    )
    
    total_distance_track2, td2_meta = calculate_total_distance(
        track2, overlapping_frames,
        max_step_distance=max_step_distance,
        max_outlier_ratio=max_outlier_ratio
    )
    
    if total_distance_track1 == float('inf'):
        metadata['reason'] = 'track1_too_many_step_outliers'
        metadata['track1_outlier_metadata'] = td1_meta
        print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
              f"REJECTED - Track1 has too many step outliers")
        return False, metadata
    
    if total_distance_track2 == float('inf'):
        metadata['reason'] = 'track2_too_many_step_outliers'
        metadata['track2_outlier_metadata'] = td2_meta
        print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
              f"REJECTED - Track2 has too many step outliers")
        return False, metadata
    
    distance_diff = abs(total_distance_track1 - total_distance_track2)
    velocity_diff = calculate_median_velocity_difference(
        track1, track2, overlapping_frames, fps=30, window_size=5
    )
    
    metadata['average_distance'] = avg_distance
    metadata['median_distance'] = median_distance
    metadata['total_distance_track1'] = total_distance_track1
    metadata['total_distance_track2'] = total_distance_track2
    metadata['distance_diff'] = distance_diff
    metadata['velocity_difference'] = velocity_diff
    metadata['point_outliers'] = median_meta.get('n_outliers', 0)
    metadata['track1_step_outliers'] = td1_meta.get('n_outlier_steps', 0)
    metadata['track2_step_outliers'] = td2_meta.get('n_outlier_steps', 0)

    direction_sim, direction_meta = calculate_direction_similarity(
        track1, track2, overlapping_frames, frame_stride=direction_frame_stride
    )
    metadata['direction_similarity'] = direction_sim
    metadata['direction_metadata'] = direction_meta
        
    direction_consistency = direction_meta.get('direction_consistency', 0.0)
    if direction_consistency < direction_consistency_threshold:
        metadata['reason'] = 'direction_mismatch'
        print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
              f"REJECTED - Direction consistency {direction_consistency:.2%} < {direction_consistency_threshold:.2%}")
        return False, metadata
    
    if direction_sim < direction_similarity_threshold:
        metadata['reason'] = 'direction_similarity_low'
        print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
              f"REJECTED - Direction similarity {direction_sim:.2%} < {direction_similarity_threshold:.2%}")
        return False, metadata

    print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
          f"Median Distance = {median_distance:.2f}, Distance Diff = {distance_diff:.2f}, "
          f"Overlap Frames = {len(overlapping_frames)}, "
          f"Outliers: {metadata['point_outliers']} pts, "
          f"{metadata['track1_step_outliers']}+{metadata['track2_step_outliers']} steps", end='')
    
    dir_consistency = metadata['direction_metadata'].get('direction_consistency', 0.0)
    print(f", Direction Sim = {direction_sim:.2%}, Dir Consistency = {dir_consistency:.2%}")

    if median_distance <= median_distance_threshold and distance_diff <= total_distance_diff_threshold:
        metadata['reason'] = 'similar'
        return True, metadata
    else:
        metadata['reason'] = 'different_positions'
        return False, metadata

def load_tracks_from_jsonl(jsonl_path: str) -> List[Dict]:
    """
    Load all tracks from a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of track dictionaries
    """
    tracks = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            track = json.loads(line.strip())
            tracks.append(track)
    return tracks

def compare_tracks_between_files(
    jsonl_paths: List[str],
    min_overlap_frames: int = 10,
    median_distance_threshold: float = 50.0,
    filter_by_team: bool = True,
    filter_by_jersey: bool = False,
    allow_multiple_matches: bool = True,
    max_analysis_frames: int = 150,
    total_distance_diff_threshold: float = 15.0,
    direction_similarity_threshold: float = 0.6,
    direction_consistency_threshold: float = 0.5,
    direction_frame_stride: int = 3
) -> Tuple[Dict[str, List[Dict]], Dict[int, List[Dict]]]:
    """
    Compare all tracks between multiple JSONL files and find matching tracks across files.

    Args:
        jsonl_paths: List of JSONL file paths (must be >= 2)
        min_overlap_frames: Minimum overlapping frames required
        median_distance_threshold: Maximum median distance for similarity
        filter_by_team: Only compare tracks from same team
        filter_by_jersey: Only compare tracks with same jersey number
        allow_multiple_matches: If True, one track can match multiple tracks in other files
        max_analysis_frames: Maximum number of frames to analyze per comparison
        total_distance_diff_threshold: Maximum allowed difference in total distance
        direction_similarity_threshold: Minimum direction similarity score (0-1)
        direction_consistency_threshold: Minimum ratio of frames with similar direction (0-1)
        direction_frame_stride: Frame stride for direction sampling (default: 3 = every 3rd frame)

    Returns:
        Tuple of (results dict with grouped match results and stats, all_tracks_by_file dict)
    """
    # Load all tracks and tag them with source index
    all_tracks = []  # list of (source_idx, original_index, track)
    all_tracks_by_file = {}  # source_idx -> list of original tracks
    file_track_counts = []
    
    for src_idx, path in enumerate(jsonl_paths):
        tracks = load_tracks_from_jsonl(path)
        print(f"Loaded {len(tracks)} tracks from {path}")
        file_track_counts.append(len(tracks))
        
        # Store original tracks by source file
        all_tracks_by_file[src_idx] = [t.copy() for t in tracks]
        
        for orig_idx, t in enumerate(tracks):
            # add metadata about source
            t['_source_idx'] = src_idx
            t['_orig_idx'] = orig_idx
            all_tracks.append(t)

    n_total = len(all_tracks)
    print(f"Total tracks loaded: {n_total} from {len(jsonl_paths)} files")

    matches = []  # pairwise matches across files
    matched_map = defaultdict(list) if allow_multiple_matches else {}

    total_comparisons = 0
    skipped_initial_distance = 0

    # Compare every pair of tracks from different source files
    for i in range(n_total):
        t1 = all_tracks[i]
        # progress print
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_total} tracks...")

        for j in range(i + 1, n_total):
            t2 = all_tracks[j]
            # skip same file
            if t1['_source_idx'] == t2['_source_idx']:
                continue

            # Optional filters by team/jersey (both must match if filtering)
            if filter_by_team and t1.get('team') != t2.get('team'):
                continue
            if filter_by_jersey and t1.get('jersey_num') != t2.get('jersey_num'):
                continue

            # Compare
            is_similar, metadata = are_tracks_similar(
                t1, t2,
                min_overlap_frames=min_overlap_frames,
                median_distance_threshold=median_distance_threshold,
                max_analysis_frames=max_analysis_frames,
                total_distance_diff_threshold=total_distance_diff_threshold,
                direction_similarity_threshold=direction_similarity_threshold,
                direction_consistency_threshold=direction_consistency_threshold,
                direction_frame_stride=direction_frame_stride
            )
            total_comparisons += 1

            if metadata.get('reason') == 'initial_position_too_far':
                skipped_initial_distance += 1

            if is_similar:
                matches.append({'i': i, 'j': j, 'metadata': metadata})
                if allow_multiple_matches:
                    matched_map[i].append(j)
                    matched_map[j].append(i)
                else:
                    # single-match mode: keep first found mapping lists
                    matched_map.setdefault(i, []).append(j)
                    matched_map.setdefault(j, []).append(i)

    # Build connected components (groups) from pairwise matches
    visited = set()
    groups = []
    idx_to_id = lambda idx: all_tracks[idx]['track_id']

    for idx in range(n_total):
        if idx in visited:
            continue
        if idx not in matched_map:
            continue
        # BFS/DFS to collect component
        stack = [idx]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            visited.add(cur)
            for nb in matched_map.get(cur, []):
                if nb not in comp:
                    stack.append(nb)

        if comp:
            groups.append(sorted(list(comp)))

    # For each group build aggregated metrics per user-specified format
    aggregated = []
    for comp in groups:
        track_ids = [idx_to_id(i) for i in comp]
        # Store source_idx to uniquely identify tracks across files
        source_indices = [all_tracks[i].get('_source_idx') for i in comp]
        teams = [all_tracks[i].get('team') for i in comp]
        jerseys = [all_tracks[i].get('jersey_num') for i in comp]

        # totaldistance per track in component (using total_distance over overlapping frames if available)
        totaldist = []
        for i in comp:
            # find any match rows involving i to extract total_distance_track1/2
            # fallback: compute full track travel distance
            td = None
            for m in matches:
                if m['i'] == i:
                    td = m['metadata'].get('total_distance_track1')
                    break
                if m['j'] == i:
                    td = m['metadata'].get('total_distance_track2')
                    break
            if td is None:
                # compute total distance along the entire available frames
                td = calculate_total_distance(all_tracks[i], all_tracks[i].get('frames', []))
            totaldist.append(td)

        # pairwise arrays (distance_diff, velocity_difference, median_distance, overlap_frames)
        pairwise_median = []
        pairwise_distance_diff = []
        pairwise_velocity = []
        pairwise_overlap = []
        pairwise_pairs = []
        for a in range(len(comp)):
            for b in range(a + 1, len(comp)):
                ia = comp[a]
                ib = comp[b]
                # find match entry
                found = None
                for m in matches:
                    if (m['i'] == ia and m['j'] == ib) or (m['i'] == ib and m['j'] == ia):
                        found = m['metadata']
                        break
                if found is not None:
                    pairwise_median.append(found.get('median_distance'))
                    pairwise_distance_diff.append(found.get('distance_diff'))
                    pairwise_velocity.append(found.get('velocity_difference'))
                    pairwise_overlap.append(found.get('overlap_count'))
                    pairwise_pairs.append((idx_to_id(ia), idx_to_id(ib)))

        aggregated.append({
            'track_id': track_ids,
            'source_idx': source_indices,  # Added to uniquely identify tracks
            'team': teams,
            'jersey_num': jerseys,
            'median_distance': pairwise_median,
            'totaldistance_track': totaldist,
            'distance_diff': pairwise_distance_diff,
            'velocity_difference': pairwise_velocity,
            'overlap_frames': pairwise_overlap,
            'pairs': pairwise_pairs
        })

    # Build summary stats
    unmatched = [all_tracks[i] for i in range(n_total) if i not in visited]

    stats = {
        'total_tracks': n_total,
        'total_comparisons': total_comparisons,
        'skipped_initial_distance': skipped_initial_distance,
        'groups_count': len(aggregated)
    }

    results = {
        'aggregated_groups': aggregated,
        'pairwise_matches': matches,
        'unmatched_tracks': [{'track_id': t['track_id'], 'team': t.get('team'), 'jersey_num': t.get('jersey_num')} for t in unmatched],
        'stats': stats
    }
    
    return results, all_tracks_by_file

def weighted_average_position(positions: List[np.ndarray], weights: List[float]) -> np.ndarray:
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


def majority_vote(values: List, ignore_values: List = None) -> Optional:
    """
    Return the most common value, ignoring specified values.
    
    Args:
        values: List of values to vote on
        ignore_values: Values to ignore (e.g., "unsure", list types)
        
    Returns:
        Most common value or None if no valid values
    """
    if ignore_values is None:
        ignore_values = ["unsure", None]
    
    filtered = []
    for v in values:
        if v in ignore_values:
            continue
        elif isinstance(v, list):
            # flatten list and add non-ignored values
            for item in v:
                if item not in ignore_values:
                    filtered.append(item)
        else:
            filtered.append(v)
    
    if not filtered:
        return "unsure"
    
    # Count frequencies
    from collections import Counter
    counts = Counter(filtered)
    
    # Find maximum frequency
    max_count = max(counts.values())
    
    # Get all values with maximum frequency
    most_common = [value for value, count in counts.items() if count == max_count and value not in ignore_values]

    # Return single value if only one, otherwise return list
    if len(most_common) == 1:
        return most_common[0]
    else:
        return most_common  # Return list of tied values


def fuse_matched_tracks(
    matched_group: Dict,
    all_tracks_by_file: Dict[int, List[Dict]],
    fused_id: int
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
    for track_id, src_idx in zip(matched_group['track_id'], matched_group['source_idx']):
        for t in all_tracks_by_file[src_idx]:
            if t['track_id'] == track_id:
                tracks.append(t)
                break
    
    if not tracks:
        return None
    
    # 1. Majority vote on team and jersey number
    teams = [t.get('team') for t in tracks]
    jersey_nums = [t.get('jersey_num') for t in tracks]
    
    fused_team = majority_vote(teams, ignore_values=["unsure", None, ""])
    fused_jersey = majority_vote(jersey_nums, ignore_values=["unsure", None, ""])
    
    # 2. Track ID
    fused_track_id = f"{fused_id}_fused"
    
    # 3. Frame range - union of all tracks
    all_frames = set()
    for t in tracks:
        all_frames.update(t.get('frames', []))
    
    all_frames = sorted(all_frames)
    frame_range = [min(all_frames), max(all_frames)] if all_frames else [0, 0]
    
    # 4. Weighted average of projected positions per frame
    # Build frame -> [(position, bbox_area, track_id)] mapping
    frame_data = defaultdict(list)
    for t in tracks:
        frames = t.get('frames', [])
        projected = t.get('projected', [])
        bbox_areas = t.get('bbox_area', [])
        
        for i, frame in enumerate(frames):
            if i < len(projected) and i < len(bbox_areas):
                frame_data[frame].append({
                    'position': np.array(projected[i]),
                    'bbox_area': bbox_areas[i],
                    'track_id': t['track_id']
                })
    
    # Compute weighted average for each frame
    fused_frames = []
    fused_projected = []
    fused_bbox_area = []
    
    for frame in sorted(frame_data.keys()):
        data_list = frame_data[frame]
        
        positions = [d['position'] for d in data_list]
        weights = [d['bbox_area'] for d in data_list]
        
        # Weighted average position
        avg_pos = weighted_average_position(positions, weights)
        
        # Sum bbox areas (or use max/mean based on preference)
        avg_bbox = sum(weights) / len(weights)  # Mean bbox area
        
        fused_frames.append(frame)
        fused_projected.append(avg_pos.tolist())
        fused_bbox_area.append(avg_bbox)
    
    # 5. Confidence merging - weighted average by track duration
    team_confs = []
    jersey_confs = []
    durations = []
    
    for t in tracks:
        duration = len(t.get('frames', [1]))
        durations.append(duration)
        team_confs.append(t.get('team_conf', 0.5))
        jersey_confs.append(t.get('jersey_conf', 0.5))
    
    total_duration = sum(durations)
    if total_duration > 0:
        fused_team_conf = sum(tc * d for tc, d in zip(team_confs, durations)) / total_duration
        fused_jersey_conf = sum(jc * d for jc, d in zip(jersey_confs, durations)) / total_duration
    else:
        fused_team_conf = np.mean(team_confs)
        fused_jersey_conf = np.mean(jersey_confs)
    
    # Build fused track
    fused_track = {
        'track_id': fused_track_id,
        'team': fused_team,
        'jersey_num': fused_jersey,
        'jersey_conf': fused_jersey_conf,
        'team_conf': fused_team_conf,
        'frame_range': frame_range,
        'frames': fused_frames,
        'projected': fused_projected,
        'bbox_area': fused_bbox_area,
        'source_track': matched_group['track_id'],
        'source_idx': matched_group['source_idx'],
        'is_fused': True
    }
    
    return fused_track


def fuse_all_matched_groups(
    matched_groups: List[Dict],
    unmatched_groups: List[Dict],
    all_tracks_by_file: Dict[int, List[Dict]],
    spatial_distance_threshold: float = 50.0
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
        for track_id, src_idx in zip(group['track_id'], group['source_idx']):
            for t in all_tracks_by_file[src_idx]:
                if t['track_id'] == track_id:
                    frange = t.get('frame_range', [0, 0])
                    start_frames.append(frange[0])
                    break
        return min(start_frames) if start_frames else float('inf')
    
    matched_groups.sort(key=group_start_frame)
    
    print(f"\n🔀 Fusing {len(matched_groups)} matched groups...")

    for group in matched_groups:
        n_tracks = len(group['track_id'])
        
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
        track_id = unmatched['track_id']
        src_idx = unmatched['source_idx']
        
        for t in all_tracks_by_file[src_idx]:
            if t['track_id'] == track_id:
                singleton_track = t.copy()
                singleton_track['track_id'] = f"{fused_id}_unmatched"
                singleton_track['source_track'] = [track_id]
                singleton_track['source_idx'] = [src_idx]
                singleton_track['is_fused'] = False
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
        overlap_threshold=0.5
    )
    tracks_after_spatial = len(fused_tracks)
    print(f"  Final track count after spatial clustering: {tracks_after_spatial}")

    # Summary statistics
    multi_source = sum(1 for t in fused_tracks if t.get('is_fused', False))
    spatial_merged = sum(1 for t in fused_tracks if t.get('is_spatial_merged', False))
    single_source = len(fused_tracks) - multi_source

    spatial_stats = {
        'tracks_before_spatial': tracks_before_spatial,
        'tracks_after_spatial': tracks_after_spatial,
        'tracks_merged_by_spatial': tracks_before_spatial - tracks_after_spatial,
        'multi_source_fused': multi_source,
        'spatial_merged': spatial_merged,
        'single_source': single_source,
        'spatial_distance_threshold': spatial_distance_threshold
    }
    
    print(f"\n📊 Final Track Breakdown:")
    print(f"  Multi-source fused: {multi_source}")
    print(f"  Spatially merged: {spatial_merged}")
    print(f"  Single-source: {single_source}")
    print(f"  Total: {len(fused_tracks)}")
    print(f"  Reduction from spatial clustering: {tracks_before_spatial - tracks_after_spatial} tracks")
    
    return fused_tracks, spatial_stats    


def save_fused_tracks(fused_tracks: List[Dict], output_path: str):
    """Save fused tracks to JSONL file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for track in fused_tracks:
            # Convert numpy arrays to lists for JSON serialization
            track_copy = track.copy()
            if "unmatched" in track_copy['track_id']:
                continue  # Skip unmatched tracks if desired
            if 'projected' in track_copy:
                track_copy['projected'] = [
                    p if isinstance(p, list) else p.tolist() 
                    for p in track_copy['projected']
                ]
            f.write(json.dumps(track_copy) + '\n')
    
    print(f"💾 Saved {len(fused_tracks)} fused tracks to {output_path}")

def save_matching_results(results: Dict, output_path: str):
    """
    Save matching results to a JSON file.
    
    Args:
        results: Results dictionary from compare_tracks_between_files
        output_path: Path to save results
    """
    # results now expected to contain 'aggregated_groups'
    summary = {
        'stats': results.get('stats', {}),
        'groups': results.get('aggregated_groups', []),
        'pairwise_matches': results.get('pairwise_matches', []),
        'unmatched_tracks': results.get('unmatched_tracks', [])
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Results saved to {output_path}")

def save_matched_tracks_separately(
    results: Dict,
    all_tracks_by_file: Dict[int, List[Dict]],
    output_path: str
):
    """
    Save matched tracks separately (without merging) to JSONL file.
    Each matched group appears as multiple lines (one per source track).
    
    Args:
        results: Results dictionary from compare_tracks_between_files
        all_tracks_by_file: Dict mapping source_idx -> list of original tracks
        output_path: Output JSONL file path
    """
    groups = results.get('aggregated_groups', [])
    
    tracks_to_save = []
    
    print(f"\n[EXPERIMENTAL] Preparing {len(groups)} matched groups for separate output...")
    
    for group_idx, group in enumerate(groups):
        track_ids = group.get('track_id', [])
        source_indices = group.get('source_idx', [])
        
        # Pair track_id with their source_idx for unique identification
        if len(source_indices) != len(track_ids):
            print(f"Warning: Group {group_idx} has mismatched track_id and source_idx lengths")
            continue
        
        for track_id, source_idx in zip(track_ids, source_indices):
            # Find the track in the original data using both track_id and source_idx
            track_found = None
            if source_idx in all_tracks_by_file:
                for track in all_tracks_by_file[source_idx]:
                    if track['track_id'] == track_id:
                        track_found = track.copy()
                        break
            
            if track_found:
                # Add metadata about the group
                track_found['match_group_id'] = group_idx
                track_found['match_group_size'] = len(track_ids)
                track_found['match_group_tracks'] = track_ids
                track_found['match_group_source_files'] = source_indices
                track_found['is_matched'] = True
                track_found['source_file_index'] = source_idx
                
                # Remove internal metadata
                track_found.pop('_source_idx', None)
                track_found.pop('_orig_idx', None)
                
                tracks_to_save.append(track_found)
            else:
                print(f"Warning: Track {track_id} from source {source_idx} not found in original tracks")
    
    # Save to JSONL
    with open(output_path, 'w') as f:
        for track in tracks_to_save:
            f.write(json.dumps(track) + '\n')
    
    print(f"✅ Saved {len(tracks_to_save)} matched tracks separately to {output_path}")
    print(f"   ({len(groups)} groups, avg {len(tracks_to_save)/max(len(groups), 1):.1f} tracks per group)")

def group_tracks_by_player(
    results: Dict,
    output_jsonl_path: str
):
    """
    Create grouped tracks where each line represents a player with tracks from both cameras.
    Now supports multiple matches - one camera2 track can appear with multiple camera1 tracks.
    
    Args:
        results: Results dictionary from compare_tracks_between_files
        output_jsonl_path: Path to save grouped tracks
    """
    with open(output_jsonl_path, 'w') as f:
        # Changed: Group matches by camera2 track to handle multiple matches
        matches_by_track2 = defaultdict(list)
        for match in results['matches']:
            track2_id = match['track2']['track_id']
            matches_by_track2[track2_id].append(match)
        
        # Write grouped tracks
        for track2_id, match_list in matches_by_track2.items():
            # Get camera2 track info (same for all matches)
            track2 = match_list[0]['track2']
            
            # Changed: camera1 tracks are now a list
            camera1_tracks = [
                {
                    'source': 'camera1',
                    'track_id': match['track1']['track_id'],
                    'frames': match['track1']['frames'],
                    'projected': match['track1']['projected'],
                    'frame_range': match['track1'].get('frame_range'),
                    'matching_quality': {
                        'median_distance': match['metadata']['median_distance'],
                        'average_distance': match['metadata']['average_distance'],
                        'overlap_frames': match['metadata']['overlap_count']
                    }
                }
                for match in match_list
            ]
            
            grouped_track = {
                'player_id': f"{track2.get('team')}_{track2.get('jersey_num', 'unknown')}",
                'team': track2.get('team'),
                'jersey_num': track2.get('jersey_num'),
                'tracks': {
                    'camera1': camera1_tracks,  # Changed: now a list
                    'camera2': {
                        'source': 'camera2',
                        'track_id': track2['track_id'],
                        'frames': track2['frames'],
                        'projected': track2['projected'],
                        'frame_range': track2.get('frame_range')
                    }
                },
                'match_count': len(camera1_tracks)  # New: number of matches
            }
            f.write(json.dumps(grouped_track) + '\n')
    
    print(f"✅ Grouped tracks saved to {output_jsonl_path}")

def spatial_cluster_tracks(fused_tracks: List[Dict], 
                           distance_threshold: float = 15.0,
                           overlap_threshold: float = 0.5) -> List[Dict]:
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
            team1 = t1.get('team')
            team2 = t2.get('team')
            
            # Skip if teams don't match exactly (including None)
            if team1 != team2 or team1 is None or team2 is None:
                continue
            
            # Skip if teams are "unsure" or empty
            if team1 in ["unsure", ""] or team2 in ["unsure", ""]:
                continue
            
            # Check temporal overlap
            frames1 = set(t1['frames'])
            frames2 = set(t2['frames'])
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
                    idx1 = t1['frames'].index(frame)
                    idx2 = t2['frames'].index(frame)
                    pos1 = np.array(t1['projected'][idx1])
                    pos2 = np.array(t2['projected'][idx2])
                    distances.append(euclidean(pos1, pos2))
                except (ValueError, IndexError):
                    continue
            
            if distances:
                median_dist = np.median(distances)
                dist_matrix[i, j] = median_dist
                dist_matrix[j, i] = median_dist
    
    # Hierarchical clustering
    valid_pairs = [(i, j) for i in range(n) for j in range(i+1, n) 
                   if dist_matrix[i, j] < distance_threshold]
    
    if not valid_pairs:
        print("  No spatial clusters found")
        return fused_tracks
    
    # Build condensed distance matrix for scipy
    condensed_dist = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed_dist.append(dist_matrix[i, j])
    
    # Cluster with threshold
    linkage_matrix = linkage(condensed_dist, method='average')
    cluster_ids = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
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
            print(f"    Merging cluster {cluster_id}: {[fused_tracks[i]['track_id'] for i in track_indices]}")
            merged = merge_track_cluster(
                [fused_tracks[i] for i in track_indices],
                new_id=f"{merged_tracks.__len__() + 1}_spatial_merged"
            )
            merged_tracks.append(merged)
    
    return merged_tracks


def merge_track_cluster(tracks: List[Dict], new_id: str) -> Dict:
    """
    Merge multiple tracks that represent the same player.
    Similar to fuse_matched_tracks but for spatial clustering.
    """
    # Majority vote on metadata
    teams = [t.get('team') for t in tracks]
    jerseys = [t.get('jersey_num') for t in tracks]
    
    merged_team = majority_vote(teams, ignore_values=["unsure", None, ""])
    merged_jersey = majority_vote(jerseys, ignore_values=["unsure", None, ""])
    
    # Union of all frames
    all_frames = set()
    for t in tracks:
        all_frames.update(t.get('frames', []))
    all_frames = sorted(all_frames)
    
    # Weighted average positions per frame
    frame_data = defaultdict(list)
    for t in tracks:
        for i, frame in enumerate(t.get('frames', [])):
            if i < len(t.get('projected', [])) and i < len(t.get('bbox_area', [])):
                frame_data[frame].append({
                    'position': np.array(t['projected'][i]),
                    'bbox_area': t['bbox_area'][i],
                    'track_id': t['track_id']
                })
    
    merged_frames = []
    merged_projected = []
    merged_bbox_area = []
    
    for frame in sorted(frame_data.keys()):
        data_list = frame_data[frame]
        positions = [d['position'] for d in data_list]
        weights = [d['bbox_area'] for d in data_list]
        
        avg_pos = weighted_average_position(positions, weights)
        avg_bbox = np.mean(weights)
        
        merged_frames.append(frame)
        merged_projected.append(avg_pos.tolist())
        merged_bbox_area.append(avg_bbox)
    
    # Merge source tracks info
    source_tracks = []
    source_indices = []
    for t in tracks:
        if isinstance(t.get('source_track'), list):
            source_tracks.extend(t['source_track'])
        else:
            source_tracks.append(t.get('track_id'))
        
        if isinstance(t.get('source_idx'), list):
            source_indices.extend(t['source_idx'])
        else:
            source_indices.append(t.get('_source_idx', 0))
    
    return {
        'track_id': new_id,
        'team': merged_team,
        'jersey_num': merged_jersey,
        'frames': merged_frames,
        'projected': merged_projected,
        'bbox_area': merged_bbox_area,
        'frame_range': [min(all_frames), max(all_frames)],
        'source_track': source_tracks,
        'source_idx': source_indices,
        'is_spatial_merged': True,
        'merged_count': len(tracks)
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare and match tracks across multiple JSONL files from different cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python cluster_tracks.py cam1.jsonl cam2.jsonl
        python cluster_tracks.py cam1.jsonl cam2.jsonl cam3.jsonl --median-distance-threshold 100 --min-overlap 30
        python cluster_tracks.py *.jsonl --output results.json --filter-team --max-frames 200
        """
    )
    
    parser.add_argument(
        'jsonl_files',
        nargs='+',
        metavar='FILE',
        help='JSONL track files to compare (minimum 2 files required)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='./track_matching_results.json',
        help='Output JSON file path (default: ./track_matching_results.json)'
    )
    
    parser.add_argument(
        '--min-overlap',
        type=int,
        default=10,
        metavar='N',
        help='Minimum overlapping frames required for matching (default: 10)'
    )
    
    parser.add_argument(
        '--median-distance-threshold',
        type=float,
        default=50.0,
        metavar='D',
        help='Maximum median distance for similarity in units (default: 50.0, ~5m)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=150,
        metavar='N',
        help='Maximum frames to analyze per comparison (default: 150)'
    )
    
    parser.add_argument(
        '--filter-team',
        action='store_true',
        help='Only compare tracks from the same team'
    )
    
    parser.add_argument(
        '--filter-jersey',
        action='store_true',
        help='Only compare tracks with the same jersey number'
    )
    
    parser.add_argument(
        '--single-match',
        action='store_true',
        help='Allow only single best match per track (default: allow multiple matches)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed comparison progress'
    )
    
    parser.add_argument(
        '--save-separate',
        type=str,
        default=None,
        metavar='FILE',
        help='[EXPERIMENTAL] Save matched tracks separately (unmerged) to this JSONL file. Each matched group will have multiple lines (one per source track).'
    )
    
    parser.add_argument(
        '--total-distance-diff-threshold',
        type=float,
        default=15.0,
        metavar='D',
        help='Maximum allowed difference in distance for matching (default: 15.0 units)'
    )

    parser.add_argument(
        '--direction-threshold',
        type=float,
        default=0.6,
        metavar='S',
        help='Minimum direction similarity score [0-1] (default: 0.6, higher = stricter)'
    )
    
    parser.add_argument(
        '--direction-consistency-threshold',
        type=float,
        default=0.5,
        metavar='C',
        help='Minimum ratio of frames with similar direction [0-1] (default: 0.5)'
    )
    
    parser.add_argument(
        '--direction-frame-stride',
        type=int,
        default=3,
        metavar='N',
        help='Frame stride for direction sampling (default: 3). '
             'Higher values = smoother/less noisy (1=consecutive, 3=every 3rd frame, 10=every 10th frame). '
             'At 29.97fps: stride 1≈33ms, stride 3≈100ms, stride 10≈333ms'
    )

    parser.add_argument(
        '--greedy',
        action='store_true',
        help='Use greedy matching algorithm (ensures maximum coverage)'
    )
    
    parser.add_argument(
        '--max-score',
        type=float,
        default=50.0,
        metavar='S',
        help='Maximum acceptable score for greedy matching (default: 50.0, lower = stricter)'
    )

    parser.add_argument(
        '--max-point-distance',
        type=float,
        default=100.0,
        metavar='D',
        help='Maximum allowed distance between corresponding points (outlier threshold, default: 100.0 units)'
    )
    
    parser.add_argument(
        '--max-step-distance',
        type=float,
        default=50.0,
        metavar='D',
        help='Maximum allowed distance per movement step (outlier threshold, default: 50.0 units)'
    )
    
    parser.add_argument(
        '--max-outlier-ratio',
        type=float,
        default=0.3,
        metavar='R',
        help='Maximum ratio of outliers allowed before rejecting match [0-1] (default: 0.3 = 30%%)'
    )

    parser.add_argument(
        '--spatial-distance',
        type=float,
        default=50.0,
        metavar='D',
        help='Distance threshold for spatial merging (default: 50.0 units, ~5.0m)'
    )

    args = parser.parse_args()
    
    # Validate minimum files
    if len(args.jsonl_files) < 2:
        parser.error("At least 2 JSONL files are required for comparison")
    
    return args


# Example usage:
if __name__ == "__main__":

    args = parse_args()
    
    # Print configuration
    print("="*60)
    print("TRACK MATCHING CONFIGURATION")
    print("="*60)
    print(f"Input files: {len(args.jsonl_files)}")
    for i, path in enumerate(args.jsonl_files, 1):
        print(f"  {i}. {path}")
    print(f"Output: {args.output}")
    print(f"Min overlap frames: {args.min_overlap}")
    print(f"Matching mode: {'GREEDY' if args.greedy else 'THRESHOLD-BASED'}")
    if args.greedy:
        print(f"Max score threshold: {args.max_score}")
    else:
        print(f"Distance threshold: {args.median_distance_threshold} units (~{args.median_distance_threshold/10:.1f}m)")
    print(f"Max analysis frames: {args.max_frames}")
    print(f"Filter by team: {args.filter_team}")
    print(f"Filter by jersey: {args.filter_jersey}")
    print(f"Direction similarity threshold: {args.direction_threshold}")
    print(f"Direction consistency threshold: {args.direction_consistency_threshold}")
    print(f"Direction frame stride: {args.direction_frame_stride} frames (~{args.direction_frame_stride/29.97*1000:.1f}ms)")
    print(f"Spatial clustering threshold: {args.spatial_distance} units (~{args.spatial_distance/10:.1f}m)")
    print(f"Allow multiple matches: {not args.single_match}")
    if args.save_separate:
        print(f"Save separate (experimental): {args.save_separate}")
    print("="*60 + "\n")
    
    # Choose matching algorithm
    if args.greedy:
        results, all_tracks_by_file = greedy_match_tracks(
            jsonl_paths=args.jsonl_files,
            min_overlap_frames=args.min_overlap,
            max_analysis_frames=args.max_frames,
            filter_by_team=args.filter_team,
            filter_by_jersey=args.filter_jersey,
            direction_frame_stride=args.direction_frame_stride,
            max_score_threshold=args.max_score,
            max_point_distance=args.max_point_distance,
            max_step_distance=args.max_step_distance,
            max_outlier_ratio=args.max_outlier_ratio,
            verbose=args.verbose
        )
        
        aggregated_groups = results['aggregated_groups']
        unmatched_tracks = results['unmatched_tracks']

        # Fuse matched tracks with spatial clustering
        fused_tracks, spatial_stats = fuse_all_matched_groups(
            aggregated_groups,
            unmatched_tracks,
            all_tracks_by_file,
            spatial_distance_threshold=args.spatial_distance
        )


        
        # Save fused tracks
        fused_output = args.output.replace('.jsonl', '_fused.jsonl')
        save_fused_tracks(fused_tracks, fused_output)
        
        # UPDATE: Add spatial clustering stats to results
        results['spatial_clustering'] = spatial_stats
        results['fused_tracks_path'] = fused_output
        results['fused_tracks_count'] = len(fused_tracks)
        results['stats']['final_track_count_after_spatial'] = len(fused_tracks)
        results['stats']['tracks_merged_by_spatial'] = spatial_stats['tracks_merged_by_spatial']
    else:
        results, all_tracks_by_file = compare_tracks_between_files(
            jsonl_paths=args.jsonl_files,
            min_overlap_frames=args.min_overlap,
            median_distance_threshold=args.median_distance_threshold,
            filter_by_team=args.filter_team,
            filter_by_jersey=args.filter_jersey,
            allow_multiple_matches=not args.single_match,
            max_analysis_frames=args.max_frames,
            total_distance_diff_threshold=args.total_distance_diff_threshold,
            direction_similarity_threshold=args.direction_threshold,
            direction_consistency_threshold=args.direction_consistency_threshold,
            direction_frame_stride=args.direction_frame_stride
    )

    save_matching_results(results, args.output)
    
    # Save matched tracks separately if requested (experimental feature)
    if args.save_separate:
        save_matched_tracks_separately(results, all_tracks_by_file, args.save_separate)

    print("\n" + "="*60)
    print("MATCHING STATISTICS")
    print("="*60)
    print(f"Total tracks: {results['stats'].get('total_tracks')}")
    if args.greedy:
        print(f"Candidate pairs evaluated: {results['stats'].get('total_candidate_pairs')}")
        print(f"Matches accepted: {results['stats'].get('matches_accepted')}")
        print(f"Match rate: {results['stats'].get('match_rate', 0):.1%}")
        print(f"Final tracks after spatial clustering: {results['stats'].get('final_track_count_after_spatial', 'N/A')}")
        print(f"Tracks merged by spatial clustering: {results['stats'].get('tracks_merged_by_spatial', 'N/A')}")
    else:
        print(f"Total comparisons: {results['stats'].get('total_comparisons')}")
        print(f"Skipped (initial distance > {args.initial_distance_threshold}): {results['stats'].get('skipped_initial_distance')}")
    print(f"Groups found: {results['stats'].get('groups_count')}")
    print(f"Unmatched tracks: {len(results.get('unmatched_tracks', []))}")

    # Print a few groups summary
    if results.get('aggregated_groups'):
        print(f"\n🎯 Top {min(5, len(results['aggregated_groups']))} matched groups:")
        for idx, g in enumerate(results['aggregated_groups'][:5], 1):
            print(f"\n  Group {idx}:")
            print(f"    Tracks: {g['track_id']}")
            print(f"    Sources: {g['source_idx']}")
            print(f"    Frame_ranges: {g['frame_ranges']}")
            print(f"    Teams: {set(g['team'])}")
            # flatten jersey numbers and remove None
            flattened_jerseys = []
            for jersey in g['jersey_num']:
                if isinstance(jersey, list):
                    flattened_jerseys.extend(jersey)
                else:
                    flattened_jerseys.append(jersey)
            print(f"    Jerseys: {set(flattened_jerseys)}")
            print(f"    Pairwise comparisons: {len(g.get('pairs', []))}")
            if g.get('median_distance'):
                print(f"    Avg median distance: {np.mean(g['median_distance']):.2f} units")
    
    # NEW: Print spatial clustering summary if available
    if results.get('spatial_clustering'):
        sc = results['spatial_clustering']
        print(f"\n🔗 Spatial Clustering Summary:")
        print(f"  Tracks before: {sc['tracks_before_spatial']}")
        print(f"  Tracks after: {sc['tracks_after_spatial']}")
        print(f"  Tracks merged: {sc['tracks_merged_by_spatial']}")
        print(f"  Distance threshold: {sc['spatial_distance_threshold']} units")
        print(f"  Breakdown:")
        print(f"    - Multi-source fused: {sc['multi_source_fused']}")
        print(f"    - Spatially merged: {sc['spatial_merged']}")
        print(f"    - Single-source: {sc['single_source']}")

    print("\n" + "="*60)
    print(f"✅ Results saved to: {args.output}")
    if args.greedy and results.get('fused_tracks_path'):
        print(f"✅ Fused tracks saved to: {results['fused_tracks_path']}")
    print("="*60)