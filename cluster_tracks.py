import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
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

def calculate_average_distance(track1: Dict, track2: Dict, 
                               overlapping_frames: List[int]) -> float:
    """
    Calculate average Euclidean distance between tracks in overlapping region.
    Uses interpolation to handle frame misalignment.
    
    Args:
        track1, track2: Track dictionaries
        overlapping_frames: List of frames to compare
        
    Returns:
        Average Euclidean distance across all valid frame pairs
    """
    distances = []
    
    for frame in overlapping_frames:
        pos1 = get_interpolated_position(track1, frame)
        pos2 = get_interpolated_position(track2, frame)
        
        if pos1 is not None and pos2 is not None:
            dist = euclidean(pos1, pos2)
            distances.append(dist)
    
    if len(distances) == 0:
        return float('inf')  # No valid comparisons
    
    return np.mean(distances)

def calculate_median_distance(track1: Dict, track2: Dict, 
                          overlapping_frames: List[int]) -> float:
    """
    Alternative: Calculate median Euclidean distance instead of average.
    Useful for stricter similarity requirements and robust to outliers.
    """
    distances = []
    
    for frame in overlapping_frames:
        pos1 = get_interpolated_position(track1, frame)
        pos2 = get_interpolated_position(track2, frame)
        
        if pos1 is not None and pos2 is not None:
            dist = euclidean(pos1, pos2)
            distances.append(dist)
    
    if len(distances) == 0:
        return float('inf')
    
    return np.median(distances)

def calculate_median_velocity_difference(track1: Dict, track2: Dict,
                            overlapping_frames: List[int], fps: float, window_size: int) -> float:
    """
    Calculate median velocity between tracks in overlapping region.
    
    Args:
        track1, track2: Track dictionaries
        overlapping_frames: List of frames to consider
        fps: Frames per second of the video
    Returns:
        Mean velocity (units per second)
    """
    # Calculate velocity if a track based on slicing windows
    velocities = []
    for track in [track1, track2]:
        track_velocities = []
        for i in range(len(overlapping_frames) - window_size):
            frame_start = overlapping_frames[i]
            frame_end = overlapping_frames[i + window_size]
            
            pos_start = get_interpolated_position(track, frame_start)
            pos_end = get_interpolated_position(track, frame_end)
            
            if pos_start is not None and pos_end is not None:
                dist = euclidean(pos_start, pos_end)
                time_sec = (frame_end - frame_start) / fps
                if time_sec > 0:
                    velocity = dist / time_sec
                    track_velocities.append(velocity)
        
        if track_velocities:
            velocities.append(track_velocities)
    
    if velocities:
        # calculate difference in each corresponding velocity
        velocity_diffs = np.diff(velocities, axis=0).flatten()
        return np.median(np.abs(velocity_diffs))
    else:
        return 0.0

def calculate_total_distance(track1: Dict, overlapping_frames: List[int]) -> float:
    """
    Calculate total Euclidean distance travelled by a track in overlapping region.
    """
    total_distance = 0.0
    for i in range(len(overlapping_frames) - 1):
        frame_start = overlapping_frames[i]
        frame_end = overlapping_frames[i + 1]
        
        pos_start = get_interpolated_position(track1, frame_start)
        pos_end = get_interpolated_position(track1, frame_end)
        
        if pos_start is not None and pos_end is not None:
            dist = euclidean(pos_start, pos_end)
            total_distance += dist
    
    return total_distance

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
        pos1_start = get_interpolated_position(track1, frame_start)
        pos1_end = get_interpolated_position(track1, frame_end)
        pos2_start = get_interpolated_position(track2, frame_start)
        pos2_end = get_interpolated_position(track2, frame_end)
        
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

def are_tracks_similar(track1: Dict, track2: Dict, 
                       min_overlap_frames: int = 10,
                       median_distance_threshold: float = 50.0,
                       max_analysis_frames: int = 150,
                       total_distance_diff_threshold: float = 15.0,
                       direction_similarity_threshold: float = 0.6,
                       direction_consistency_threshold: float = 0.5,
                       direction_frame_stride: int = 3) -> Tuple[bool, Dict]:
    """
    Compare two tracks for similarity based on overlapping frames.
    
    Args:
        track1, track2: Track dictionaries with fields:
            - track_id: string
            - team: string
            - jersey_num: int
            - frame_range: [start, end]
            - frames: list of frame numbers
            - projected: list of [x, y] coordinates
        min_overlap_frames: Minimum number of overlapping frames needed for comparison
        median_distance_threshold: Maximum average distance to consider tracks similar
        max_analysis_frames: Maximum number of frames to analyze (default: 150)
        total_distance_diff_threshold: Maximum allowed difference in total distance        
        direction_similarity_threshold: Minimum direction similarity score (0-1)
        direction_consistency_threshold: Minimum ratio of frames with similar direction (0-1)
        direction_frame_stride: Frame stride for direction sampling (1=consecutive, 3=every 3rd, etc.)

    Returns:
        Tuple of (is_similar: bool, metadata: dict with comparison details)
    """
    # Fast check 1: Compare initial projected points (first frame distance)
    if len(track1['projected']) > 0 and len(track2['projected']) > 0:
        initial_pos1 = np.array(track1['projected'][0])
        initial_pos2 = np.array(track2['projected'][0])
        initial_distance = np.linalg.norm(initial_pos1 - initial_pos2)
        
        if initial_distance >= 100.0:  # 100 units = ~10m
            return False, {
                'reason': 'initial_position_too_far',
                'initial_distance': initial_distance,
                'track1_id': track1['track_id'],
                'track2_id': track2['track_id'],
                'overlap_count': 0,
                'track1_frames': len(track1['frames']),
                'track2_frames': len(track2['frames']),
            }
    
    # Find overlapping frames
    overlapping_frames = find_overlapping_frames(track1, track2)
    
    # Limit analysis to first max_analysis_frames for efficiency
    if len(overlapping_frames) > max_analysis_frames:
        overlapping_frames = overlapping_frames[:max_analysis_frames]
    
    metadata = {
        'overlap_count': len(overlapping_frames),
        'track1_id': track1['track_id'],
        'track2_id': track2['track_id'],
        'track1_frames': len(track1['frames']),
        'track2_frames': len(track2['frames']),
    }
    
    # Case 1: No overlapping frames
    if len(overlapping_frames) == 0:
        metadata['reason'] = 'no_overlap'
        metadata['average_distance'] = None
        metadata['median_distance'] = None
        return False, metadata
    
    # Case 2: Overlapping region less than threshold
    if len(overlapping_frames) < min_overlap_frames:
        metadata['reason'] = 'insufficient_overlap'
        metadata['average_distance'] = None
        metadata['median_distance'] = None
        return False, metadata
    
    # Case 3: Calculate distance in overlapping region
    avg_distance = calculate_average_distance(track1, track2, overlapping_frames)
    median_distance = calculate_median_distance(track1, track2, overlapping_frames)
    velocity_diff = calculate_median_velocity_difference(track1, track2, overlapping_frames, fps=30, window_size=5)
    total_distance_track1 = calculate_total_distance(track1, overlapping_frames)
    total_distance_track2 = calculate_total_distance(track2, overlapping_frames)
    distance_diff = abs(total_distance_track1 - total_distance_track2)
    
    metadata['average_distance'] = avg_distance
    metadata['median_distance'] = median_distance
    metadata['total_distance_track1'] = total_distance_track1
    metadata['total_distance_track2'] = total_distance_track2
    metadata['distance_diff'] = distance_diff
    metadata['velocity_difference'] = velocity_diff

    direction_sim, direction_meta = calculate_direction_similarity(
        track1, track2, overlapping_frames, frame_stride=direction_frame_stride
    )
    metadata['direction_similarity'] = direction_sim
    metadata['direction_metadata'] = direction_meta
        
    # Check direction-based rejection criteria
    direction_consistency = direction_meta.get('direction_consistency', 0.0)
    if direction_consistency < direction_consistency_threshold:
        metadata['reason'] = 'direction_mismatch'
        print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
                f"REJECTED - Direction consistency {direction_consistency:.2%} < {direction_consistency_threshold:.2%}")
        return False, metadata
    
    # Also check overall similarity
    if direction_sim < direction_similarity_threshold:
        metadata['reason'] = 'direction_similarity_low'
        print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
                f"REJECTED - Direction similarity {direction_sim:.2%} < {direction_similarity_threshold:.2%}")
        return False, metadata

    # Print comparison details
    print(f"Comparing Track {track1['track_id']} and Track {track2['track_id']}: "
          f"Median Distance = {median_distance:.2f}, Distance Diff = {distance_diff:.2f}, "
          f"Overlap Frames = {len(overlapping_frames)}", end='')
    

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
            'track_ids': track_ids,
            'source_indices': source_indices,  # Added to uniquely identify tracks
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
        track_ids = group.get('track_ids', [])
        source_indices = group.get('source_indices', [])
        
        # Pair track_ids with their source_indices for unique identification
        if len(source_indices) != len(track_ids):
            print(f"Warning: Group {group_idx} has mismatched track_ids and source_indices lengths")
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
    print(f"Distance threshold: {args.median_distance_threshold} units (~{args.median_distance_threshold/10:.1f}m)")
    print(f"Max analysis frames: {args.max_frames}")
    print(f"Filter by team: {args.filter_team}")
    print(f"Filter by jersey: {args.filter_jersey}")
    print(f"Direction similarity threshold: {args.direction_threshold}")
    print(f"Direction consistency threshold: {args.direction_consistency_threshold}")
    print(f"Direction frame stride: {args.direction_frame_stride} frames (~{args.direction_frame_stride/29.97*1000:.1f}ms)")
    print(f"Allow multiple matches: {not args.single_match}")
    if args.save_separate:
        print(f"Save separate (experimental): {args.save_separate}")
    print("="*60 + "\n")
    
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
    print(f"Total comparisons: {results['stats'].get('total_comparisons')}")
    print(f"Groups found: {results['stats'].get('groups_count')}")
    print(f"Skipped (initial distance > 100): {results['stats'].get('skipped_initial_distance')}")
    print(f"Unmatched tracks: {len(results.get('unmatched_tracks', []))}")

    # Print a few groups summary
    if results.get('aggregated_groups'):
        print(f"\n🎯 Top {min(5, len(results['aggregated_groups']))} matched groups:")
        for idx, g in enumerate(results['aggregated_groups'][:5], 1):
            print(f"\n  Group {idx}:")
            print(f"    Tracks: {g['track_ids']}")
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
    
    print("\n" + "="*60)
    print(f"✅ Results saved to: {args.output}")
    print("="*60)