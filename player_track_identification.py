import json
import numpy as np
from collections import defaultdict
import cv2
import os
import time
from typing import List, Dict, Tuple, Union
from heapq import nsmallest
from tools.remove_track_sharp import process_jsonl_detect_replace
import argparse


def detect_team_size_violations_streaming(
    jsonl_path, save_path, max_team_size=10, allowed_goalkeepers=1, allowed_referees=1
):
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
                elif (
                    not team.endswith("goalkeeper")
                    and team != "referee"
                    and len(tids) > max_team_size
                ):
                    violations[team] = tids

            if violations:
                out_f.write(json.dumps({"frame_id": frame_id, "violations": violations}) + "\n")

    print(f"📄 Saved violations to {save_path}")


def merge_violation_windows_with_track_counts(
    jsonl_path: str, min_gap: int = 1
) -> Dict[str, List[Dict]]:
    """
    Merge consecutive violation frames into windows grouped by team and number of violating tracks.

    Args:
        jsonl_path (str): Path to the input JSONL file with per-frame violations.
        min_gap (int): Allowed gap between frames to merge into the same window.

    Returns:
        Dict[str, List[Dict]]: Dictionary with team as key and list of merged windows as value.
    """
    team_count_to_frames = defaultdict(
        lambda: defaultdict(list)
    )  # team -> count -> list of frame info

    # Read and categorize frames
    with open(jsonl_path, "r") as f:
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
            start, prev_frame, current_ids = (
                frames[0][0],
                frames[0][0],
                frames[0][1].copy(),
            )

            for i in range(1, len(frames)):
                frame_id, ids = frames[i]
                if frame_id <= prev_frame + min_gap:
                    current_ids.update(ids)
                    prev_frame = frame_id
                else:
                    merged.append(
                        {
                            "range": [start, prev_frame],
                            "count": count,
                            "track_ids": sorted(current_ids),
                        }
                    )
                    start = prev_frame = frame_id
                    current_ids = ids.copy()

            # Append the last segment
            merged.append(
                {
                    "range": [start, prev_frame],
                    "count": count,
                    "track_ids": sorted(current_ids),
                }
            )

            merged_result[team].extend(merged)

    return merged_result


def relabel_tracks_by_confidence_and_decrement_windows_streaming(
    track_jsonl_path: str,
    team_windows: Dict[str, List[Dict]],
    output_jsonl_path: str,
    conf_threshold: float = 0.007,
    not_sure_label: str = "unsure",
    force_relabel: bool = True,
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

    # 👇 Deep copy and sort windows by count (descending)
    team_windows = {
        k: sorted([w.copy() for w in v], key=lambda x: x["count"], reverse=True)
        for k, v in team_windows.items()
    }

    team_windows = {k: [w.copy() for w in v] for k, v in team_windows.items()}
    # print(team_windows.items())

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
            # print(f"  Found {candidate_tracks} candidate tracks for relabeling.")

            relabeled_in_window = 0
            for conf, tid, t_start, t_end in candidate_tracks:
                if conf > conf_threshold and not force_relabel:
                    continue
                if tid in relabel_map:
                    continue  # already relabeled

                relabel_map[tid] = not_sure_label
                relabel_count += 1
                relabeled_in_window += 1

                # Decrement other windows that overlap
                new_windows = []
                for other in windows:
                    ow_start, ow_end = other["range"]
                    ow_track_ids = set(other["track_ids"])
                    if tid not in ow_track_ids:
                        new_windows.append(other)
                        continue
                    if ow_end < t_start or ow_start > t_end:
                        new_windows.append(other)
                        continue

                    # Compute overlap span with the relabeled track
                    ov_start = max(ow_start, t_start)
                    ov_end = min(ow_end, t_end)

                    # Left segment (before overlap) - track not present, keep count
                    if ow_start < ov_start:
                        new_windows.append(
                            {
                                "range": [ow_start, ov_start - 1],
                                "count": other["count"],
                                "track_ids": other["track_ids"],
                            }
                        )

                    # Middle segment (overlap) - decrement count and remove tid
                    new_windows.append(
                        {
                            "range": [ov_start, ov_end],
                            "count": other["count"] - 1,
                            "track_ids": [
                                other_tid for other_tid in other["track_ids"] if other_tid != tid
                            ],
                        }
                    )

                    # Right segment (after overlap) - track not present, keep count
                    if ov_end < ow_end:
                        new_windows.append(
                            {
                                "range": [ov_end + 1, ow_end],
                                "count": other["count"],
                                "track_ids": other["track_ids"],
                            }
                        )

                windows = new_windows

            # If still unresolved, re-add current window
            # print(f"  Relabeled {relabeled_in_window} tracks in this window.")
            window["count"] -= relabeled_in_window
            # print(window["count"])
            if window["count"] > allowed_count and relabeled_in_window > 0:
                windows.append(window)

    # Final pass to write relabeled file
    with open(track_jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
        for line in f_in:
            track = json.loads(line.strip())
            tid = track.get("track_id")
            if tid in relabel_map:
                if track.get("team") not in [not_sure_label]:
                    track["team"] = relabel_map[tid]
            json.dump(track, f_out)
            f_out.write("\n")

    print(f"✅ Relabeled {relabel_count} tracks.")
    return relabel_count


def allocate_jersey_numbers_by_count_and_confidence(
    jsonl_path: str,
    output_path: str,
    home_jersey_numbers: List[int],
    away_jersey_numbers: List[int],
):
    """
    Allocate jersey numbers to tracks based on count and confidence scores and handle conflicts.

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
            track["allocated_count"] = 0
            continue

        # Determine team key
        team_key = "home" if "home" in team.lower() else "away" if "away" in team.lower() else None
        if team_key is None:
            track["allocated_jersey"] = "unsure"
            track["allocated_conf"] = 0.0
            track["allocated_count"] = 0
            continue

        jersey_nums = track.get("jersey_num", "unsure")
        jersey_confs = track.get("jersey_conf", 0.0)
        jersey_counts = track.get("count", 1)  # Get counts

        # Handle unsure cases
        if jersey_nums == "unsure" or not jersey_nums:
            track["allocated_jersey"] = "unsure"
            track["allocated_conf"] = 0.0
            track["allocated_count"] = 0
            continue

        # Normalize to lists
        if not isinstance(jersey_nums, list):
            jersey_nums = [jersey_nums]
        if not isinstance(jersey_confs, list):
            jersey_confs = [jersey_confs]
        if not isinstance(jersey_counts, list):
            jersey_counts = [jersey_counts]

        # ✅ Sort by count (descending), then by confidence (descending)
        jersey_data = sorted(
            zip(jersey_nums, jersey_confs, jersey_counts),
            key=lambda x: (x[2], x[1]),  # Sort by count first, then confidence
            reverse=True
        )

        # Find first valid jersey number
        allocated = False
        for jersey_num, conf, count in jersey_data:
            if jersey_num in valid_jerseys[team_key]:
                track["allocated_jersey"] = jersey_num
                track["allocated_conf"] = conf
                track["allocated_count"] = count  # Store count for reference
                allocated = True
                break

        if not allocated:
            track["allocated_jersey"] = "unsure"
            track["allocated_conf"] = 0.0
            track["allocated_count"] = 0

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

        # ✅ Sort by count (descending), then by confidence
        same_jersey_tracks.sort(
            key=lambda t: (t.get("allocated_count", 0), t.get("allocated_conf", 0.0)),
            reverse=True
        )

        # Keep highest count track, reassign others
        winner = same_jersey_tracks[0]

        for loser_track in same_jersey_tracks[1:]:
            if loser_track["track_id"] == winner["track_id"]:
                continue

            # Try to find alternative jersey number
            team = loser_track.get("team", "")
            team_key = "home" if "home" in team.lower() else "away"

            jersey_nums = loser_track.get("jersey_num", [])
            jersey_confs = loser_track.get("jersey_conf", [])
            jersey_counts = loser_track.get("count", [])

            if not isinstance(jersey_nums, list):
                jersey_nums = [jersey_nums]
            if not isinstance(jersey_confs, list):
                jersey_confs = [jersey_confs]
            if not isinstance(jersey_counts, list):
                jersey_counts = [jersey_counts]

            # Sort by count (descending), then by confidence (descending)
            jersey_data = sorted(
                zip(jersey_nums, jersey_confs, jersey_counts),
                key=lambda x: (x[2], x[1]),
                reverse=True
            )

            reallocated = False
            for jersey_num, conf, count in jersey_data:
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
                        loser_track["allocated_count"] = count
                        reallocated = True
                        resolved_count += 1
                        break

            if not reallocated:
                loser_track["allocated_jersey"] = "unsure"
                loser_track["allocated_conf"] = 0.0
                loser_track["allocated_count"] = 0
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


def calculate_track_similarity(
    t1: Dict,
    t2: Dict,
    spatial_threshold: float = 30.0,
    direction_threshold: float = 60.0,
    min_overlap_ratio: float = 0.3,
) -> Tuple[Union[float, None], Dict]:
    """
    Calculate similarity score between two tracks for merging.
    
    Returns:
        Tuple of (similarity_score, metadata) or (None, metadata) if not suitable
        Lower score = more similar
    """
    from tools.player_motion_classification import calculate_angle
    
    # Check temporal overlap
    frames1 = set(t1.get("frames", []))
    frames2 = set(t2.get("frames", []))
    overlap = frames1 & frames2
    
    overlap_ratio = len(overlap) / min(len(frames1), len(frames2)) if frames1 and frames2 else 0
    
    metadata = {
        "overlap_count": len(overlap),
        "overlap_ratio": overlap_ratio,
    }
    
    if overlap_ratio < min_overlap_ratio:
        metadata["reason"] = f"overlap_ratio {overlap_ratio:.2%} < {min_overlap_ratio:.2%}"
        return None, metadata
    
    # Calculate median distance in overlapping frames
    distances = []
    directions_similar = 0
    total_directions = 0
    
    overlapping_frames = sorted(overlap)[:150]  # Sample max 150 frames
    
    for i, frame in enumerate(overlapping_frames):
        try:
            idx1 = t1["frames"].index(frame)
            idx2 = t2["frames"].index(frame)
            pos1 = np.array(t1["projected"][idx1])
            pos2 = np.array(t2["projected"][idx2])
            
            dist = np.linalg.norm(pos1 - pos2)
            distances.append(dist)
            
            # Check direction similarity (using next frame)
            if i < len(overlapping_frames) - 1:
                next_frame = overlapping_frames[i + 1]
                if next_frame in t1["frames"] and next_frame in t2["frames"]:
                    next_idx1 = t1["frames"].index(next_frame)
                    next_idx2 = t2["frames"].index(next_frame)
                    
                    next_pos1 = np.array(t1["projected"][next_idx1])
                    next_pos2 = np.array(t2["projected"][next_idx2])
                    
                    vec1 = next_pos1 - pos1
                    vec2 = next_pos2 - pos2
                    
                    if np.linalg.norm(vec1) > 1e-3 and np.linalg.norm(vec2) > 1e-3:
                        angle = calculate_angle(vec1, vec2)
                        total_directions += 1
                        if angle < direction_threshold:
                            directions_similar += 1
                        
        except (ValueError, IndexError):
            continue
    
    if not distances:
        metadata["reason"] = "no_valid_comparisons"
        return None, metadata
    
    median_distance = np.median(distances)
    direction_consistency = directions_similar / total_directions if total_directions > 0 else 0
    
    metadata.update({
        "median_distance": median_distance,
        "mean_distance": np.mean(distances),
        "std_distance": np.std(distances),
        "direction_consistency": direction_consistency,
        "directions_compared": total_directions,
    })
    
    # Reject if too far apart
    if median_distance > spatial_threshold:
        metadata["reason"] = f"median_distance {median_distance:.1f} > {spatial_threshold}"
        return None, metadata
    
    # Reject if directions don't match
    if direction_consistency < 0.5:
        metadata["reason"] = f"direction_consistency {direction_consistency:.2%} < 50%"
        return None, metadata
    
    # Composite score (lower is better)
    score = median_distance * 2.0 + (1.0 - direction_consistency) * 50.0
    
    return score, metadata


def merge_two_tracks(t1: Dict, t2: Dict) -> Dict:
    """
    Merge two tracks into one by combining their data.
    Uses weighted average for overlapping frames.
    """
    
    # Combine all frames
    all_frames = sorted(set(t1.get("frames", [])) | set(t2.get("frames", [])))
    
    merged_frames = []
    merged_projected = []
    merged_bbox_area = []
    
    for frame in all_frames:
        # Check which tracks have this frame
        in_t1 = frame in t1.get("frames", [])
        in_t2 = frame in t2.get("frames", [])
        
        if in_t1 and in_t2:
            # Both tracks have this frame - use weighted average
            idx1 = t1["frames"].index(frame)
            idx2 = t2["frames"].index(frame)
            
            pos1 = np.array(t1["projected"][idx1])
            pos2 = np.array(t2["projected"][idx2])
            
            area1 = t1.get("bbox_area", [1])[idx1] if idx1 < len(t1.get("bbox_area", [])) else 1
            area2 = t2.get("bbox_area", [1])[idx2] if idx2 < len(t2.get("bbox_area", [])) else 1
            
            # Weighted average by bbox area
            total_area = area1 + area2
            if total_area > 0:
                merged_pos = (pos1 * area1 + pos2 * area2) / total_area
                merged_area = (area1 + area2) / 2
            else:
                merged_pos = (pos1 + pos2) / 2
                merged_area = 1
                
        elif in_t1:
            idx1 = t1["frames"].index(frame)
            merged_pos = np.array(t1["projected"][idx1])
            merged_area = t1.get("bbox_area", [1])[idx1] if idx1 < len(t1.get("bbox_area", [])) else 1
        else:  # in_t2
            idx2 = t2["frames"].index(frame)
            merged_pos = np.array(t2["projected"][idx2])
            merged_area = t2.get("bbox_area", [1])[idx2] if idx2 < len(t2.get("bbox_area", [])) else 1
        
        merged_frames.append(frame)
        merged_projected.append(merged_pos.tolist())
        merged_bbox_area.append(merged_area)
    
    # Merge metadata using majority vote or averaging
    team1 = t1.get("team", "unsure")
    team2 = t2.get("team", "unsure")
    merged_team = team1 if team1 == team2 else "unsure"
    
    conf1 = t1.get("team_conf", 0.5)
    conf2 = t2.get("team_conf", 0.5)
    merged_conf = (conf1 + conf2) / 2
    
    # ===== JERSEY NUMBER MERGING =====
    # Normalize to lists
    jersey1 = t1.get("jersey_num", "unsure")
    jersey2 = t2.get("jersey_num", "unsure")
    
    if not isinstance(jersey1, list):
        jersey1 = [jersey1] if jersey1 != "unsure" else []
    if not isinstance(jersey2, list):
        jersey2 = [jersey2] if jersey2 != "unsure" else []
    
    # Get confidences (normalize to lists)
    conf1_list = t1.get("jersey_conf", 0.5)
    conf2_list = t2.get("jersey_conf", 0.5)
    
    if not isinstance(conf1_list, list):
        conf1_list = [conf1_list] * len(jersey1) if jersey1 else []
    if not isinstance(conf2_list, list):
        conf2_list = [conf2_list] * len(jersey2) if jersey2 else []
    
    # Get counts (normalize to lists)
    count1_list = t1.get("count", 1)
    count2_list = t2.get("count", 1)
    
    if not isinstance(count1_list, list):
        count1_list = [count1_list] * len(jersey1) if jersey1 else []
    if not isinstance(count2_list, list):
        count2_list = [count2_list] * len(jersey2) if jersey2 else []
    
    # Combine all jersey data
    all_jerseys = jersey1 + jersey2
    all_confs = conf1_list + conf2_list
    all_counts = count1_list + count2_list
    
    if not all_jerseys:
        # No valid jersey numbers
        merged_jersey = "unsure"
        merged_jersey_conf = 0.5
        merged_jersey_count = 0
    else:
        # Group by jersey number and sum counts
        jersey_data = defaultdict(lambda: {"confs": [], "counts": []})
        
        for jersey, conf, count in zip(all_jerseys, all_confs, all_counts):
            if jersey != "unsure":
                jersey_data[jersey]["confs"].append(conf)
                jersey_data[jersey]["counts"].append(count)
        
        if not jersey_data:
            # Only "unsure" jerseys
            merged_jersey = "unsure"
            merged_jersey_conf = 0.5
            merged_jersey_count = 0
        else:
            # Calculate metrics for each jersey
            jersey_results = []
            for jersey_num, data in jersey_data.items():
                avg_conf = np.mean(data["confs"])
                total_count = sum(data["counts"])
                jersey_results.append((jersey_num, avg_conf, total_count))
            
            # Sort by total count (primary), then confidence (secondary)
            jersey_results.sort(key=lambda x: (x[2], x[1]), reverse=True)
            
            # Separate into three lists
            merged_jersey = [jersey for jersey, _, _ in jersey_results]
            merged_jersey_conf = [conf for _, conf, _ in jersey_results]
            merged_jersey_count = [count for _, _, count in jersey_results]
            
            # If only one jersey, convert back to scalar
            if len(merged_jersey) == 1:
                merged_jersey = merged_jersey[0]
                merged_jersey_conf = merged_jersey_conf[0]
                merged_jersey_count = merged_jersey_count[0]
    
    return {
        "team": merged_team,
        "jersey_num": merged_jersey,
        "jersey_conf": merged_jersey_conf,
        "count": merged_jersey_count,
        "team_conf": merged_conf,
        "frame_range": [min(merged_frames), max(merged_frames)],
        "frames": merged_frames,
        "projected": merged_projected,
        "bbox_area": merged_bbox_area,
        "is_merged": True,
    }


def merge_tracks_by_team_size_violations(
    jsonl_path: str,
    output_path: str,
    field_size: Tuple[int, int],
    max_team_size: int = 10,
    max_goalkeeper: int = 1,
    spatial_threshold: float = 30.0,
    min_overlap_ratio: float = 0.3,
    direction_threshold: float = 60.0,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Merge tracks based on team size violations by finding similar tracks
    that appear together in violation windows.
    
    This function:
    1. Detects frames where team sizes exceed limits (>10 players or >1 goalkeeper)
    2. Identifies tracks that appear together during violations
    3. Merges similar tracks based on spatial proximity and movement direction
    4. Treats home/homegoalkeeper and away/awaygoalkeeper separately
    
    Args:
        jsonl_path: Input JSONL path with tracks
        output_path: Output path for merged tracks
        field_size: Field dimensions (length, width) in 0.1m units
        max_team_size: Maximum allowed team size (default: 10)
        max_goalkeeper: Maximum allowed goalkeepers (default: 1)
        spatial_threshold: Max distance to consider tracks as mergeable (units)
        min_overlap_ratio: Min overlap ratio to consider merging
        direction_threshold: Max angle difference for direction similarity (degrees)
        verbose: Print progress
        
    Returns:
        Dict with merge statistics
    """
    
    if verbose:
        print("=" * 60)
        print("MERGING TRACKS BY TEAM SIZE VIOLATIONS")
        print("=" * 60)
    
    # Step 1: Detect violations
    violations_path = jsonl_path.replace(".jsonl", "_violations_temp.jsonl")
    detect_team_size_violations_streaming(
        jsonl_path=jsonl_path,
        save_path=violations_path,
        max_team_size=max_team_size,
        allowed_goalkeepers=max_goalkeeper,
        allowed_referees=1,
    )
    
    # Step 2: Merge violation windows
    merged_windows = merge_violation_windows_with_track_counts(
        jsonl_path=violations_path,
        min_gap=3,
    )
    
    if verbose:
        print(f"\n📊 Found violations for teams:")
        for team, windows in merged_windows.items():
            total_frames = sum(w["range"][1] - w["range"][0] + 1 for w in windows)
            print(f"  {team}: {len(windows)} windows ({total_frames} frames)")
    
    # Step 3: Load all tracks
    tracks = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    
    track_map = {t["track_id"]: t for t in tracks}
    
    # Step 4: Process each team's violations
    merge_candidates = []  # List of (track1_id, track2_id, similarity_score, metadata)
    
    for team, windows in merged_windows.items():
        if verbose:
            print(f"\n🔍 Processing violations for {team}...")
        
        # Separate home/homegoalkeeper, away/awaygoalkeeper
        if team.endswith("goalkeeper"):
            base_team = team.replace("goalkeeper", "").strip()
            max_allowed = max_goalkeeper
        elif team == "referee":
            continue  # Skip referee
        else:
            base_team = team
            max_allowed = max_team_size
        
        for window in windows:
            win_start, win_end = window["range"]
            violating_track_ids = window["track_ids"]
            excess_count = window["count"] - max_allowed
            
            if excess_count <= 0:
                continue
            
            # Get tracks involved in this violation
            violation_tracks = []
            for tid in violating_track_ids:
                if tid not in track_map:
                    continue
                t = track_map[tid]
                
                # Check if track's team matches exactly
                track_team = t.get("team", "")
                if track_team != team:
                    continue
                
                # Check temporal overlap with violation window
                track_frames = set(t.get("frames", []))
                window_frames = set(range(win_start, win_end + 1))
                overlap = track_frames & window_frames
                
                if not overlap:
                    continue
                
                overlap_ratio = len(overlap) / len(track_frames)
                if overlap_ratio < 0.1:  # Skip tracks with minimal overlap
                    continue
                
                violation_tracks.append(t)
            
            if len(violation_tracks) < 2:
                continue
            
            if verbose:
                print(f"  Window {win_start}-{win_end}: {len(violation_tracks)} tracks, "
                      f"excess: {excess_count}")
            
            # Step 5: Find merge candidates by comparing tracks pairwise
            for i in range(len(violation_tracks)):
                for j in range(i + 1, len(violation_tracks)):
                    t1, t2 = violation_tracks[i], violation_tracks[j]
                    
                    # Calculate similarity
                    similarity, metadata = calculate_track_similarity(
                        t1, t2,
                        spatial_threshold=spatial_threshold,
                        direction_threshold=direction_threshold,
                        min_overlap_ratio=min_overlap_ratio,
                    )
                    
                    if similarity is not None:
                        merge_candidates.append((
                            t1["track_id"],
                            t2["track_id"],
                            similarity,
                            metadata
                        ))
    
    if verbose:
        print(f"\n✅ Found {len(merge_candidates)} merge candidates")
    
    # Step 6: Sort candidates by similarity (lower is better)
    merge_candidates.sort(key=lambda x: x[2])
    
    if verbose and merge_candidates:
        print(f"\n🏆 Top 5 merge candidates:")
        for tid1, tid2, score, meta in merge_candidates[:5]:
            print(f"  {tid1} <-> {tid2}: score={score:.2f} "
                  f"(dist={meta['median_distance']:.1f}, "
                  f"overlap={meta['overlap_count']})")
    
    # Step 7: Greedy merging
    merged_tracks = {}
    merged_ids = set()
    track_to_merged = {}  # original_id -> merged_id
    
    for tid1, tid2, score, metadata in merge_candidates:
        # Skip if either track already merged
        if tid1 in merged_ids or tid2 in merged_ids:
            continue
        
        # Get tracks
        t1 = track_map[tid1]
        t2 = track_map[tid2]
        
        # Merge tracks
        merged_track = merge_two_tracks(t1, t2)
        merged_id = f"{len(merged_tracks) + 1}_merged"
        merged_track["track_id"] = merged_id
        merged_track["source_tracks"] = [tid1, tid2]
        merged_track["merge_score"] = score
        merged_track["merge_metadata"] = metadata
        
        merged_tracks[merged_id] = merged_track
        merged_ids.update([tid1, tid2])
        track_to_merged[tid1] = merged_id
        track_to_merged[tid2] = merged_id
        
        if verbose:
            print(f"  ✓ Merged {tid1} + {tid2} -> {merged_id}")
    
    # Step 8: Add unmerged tracks
    final_tracks = list(merged_tracks.values())
    for track in tracks:
        tid = track["track_id"]
        if tid not in merged_ids:
            final_tracks.append(track)
    
    # Step 9: Save output
    with open(output_path, 'w') as f:
        for track in final_tracks:
            f.write(json.dumps(track) + '\n')
    
    # Clean up temp file
    if os.path.exists(violations_path):
        os.remove(violations_path)
    
    stats = {
        "total_input_tracks": len(tracks),
        "merge_candidates_found": len(merge_candidates),
        "tracks_merged": len(merged_ids),
        "merged_groups": len(merged_tracks),
        "final_track_count": len(final_tracks),
        "reduction": len(tracks) - len(final_tracks),
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("MERGE STATISTICS")
        print("=" * 60)
        print(f"Input tracks: {stats['total_input_tracks']}")
        print(f"Merge candidates: {stats['merge_candidates_found']}")
        print(f"Tracks merged: {stats['tracks_merged']}")
        print(f"Merged groups created: {stats['merged_groups']}")
        print(f"Final track count: {stats['final_track_count']}")
        print(f"Reduction: {stats['reduction']} tracks")
        print("=" * 60)
    
    return stats


def prepare_background_and_tracks(
    json_path,
    image_path,
    home_jersey_numbers,
    away_jersey_numbers,
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
    detector_firstpass_kwargs=None,
    detector_secondpass_kwargs=None,
):
    # Load and resize background
    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    bg_img = cv2.resize(bg_img, field_size)

    start_load = time.time()

    # We left these later after fusing all matching tracks

    detect_team_size_violations_streaming(
        jsonl_path=json_path,
        save_path=json_path.replace(".jsonl", "_team_size_violations.jsonl"),
        max_team_size=10,
        allowed_goalkeepers=1,
        allowed_referees=1,
    )
    end_violations = time.time()
    print(f"✅ Detected team size violations in {end_violations - start_load:.2f} seconds")

    start_merged = time.time()
    merged_window = merge_violation_windows_with_track_counts(
        jsonl_path=json_path.replace(".jsonl", "_team_size_violations.jsonl"), min_gap=3
    )
    end_merged = time.time()

    for team, windows in merged_window.items():
        print(f"🟢 Team {team} → {len(windows)} merged windows")
    print(f"✅ Merged windows in {end_merged - start_merged:.2f} seconds")

    relabel_count = relabel_tracks_by_confidence_and_decrement_windows_streaming(
        track_jsonl_path=json_path,
        team_windows=merged_window,
        output_jsonl_path=json_path.replace(".jsonl", "_relabeled.jsonl"),
        conf_threshold=0.007,
        not_sure_label="unsure",
    )
    end_relabel = time.time()
    print(f"✅ Relabeled {relabel_count} tracks in {end_relabel - end_merged:.2f} seconds")

    # # Merge tracks based on team size violations
    # start_merge_violations = time.time()
    # merge_stats = merge_tracks_by_team_size_violations(
    #     jsonl_path=json_path,
    #     output_path=json_path.replace(".jsonl", "_merged_violations.jsonl"),
    #     field_size=field_size,
    #     max_team_size=10,
    #     max_goalkeeper=1,
    #     spatial_threshold=30.0,
    #     min_overlap_ratio=0.3,
    #     direction_threshold=60.0,
    #     verbose=True,
    # )
    # end_merge_violations = time.time()
    # print(f"✅ Merged tracks by violations in {end_merge_violations - start_merge_violations:.2f} seconds")

    process_jsonl_detect_replace(
        input_path=json_path.replace(".jsonl", "_relabeled.jsonl"),
        output_path=json_path.replace(".jsonl", "_smoothed.jsonl"),
        detector_kwargs=detector_firstpass_kwargs,
    )

    process_jsonl_detect_replace(
        input_path=json_path.replace(".jsonl", "_smoothed.jsonl"),
        output_path=json_path.replace(".jsonl", "_smoothed_again.jsonl"),
        detector_kwargs=detector_secondpass_kwargs,
    )

    allocate_jersey_numbers_by_count_and_confidence(
        jsonl_path=json_path.replace(".jsonl", "_smoothed_again.jsonl"),
        output_path=json_path.replace(".jsonl", "_final.jsonl"),
        home_jersey_numbers=home_jersey_numbers,
        away_jersey_numbers=away_jersey_numbers,
    )
    end_allocation = time.time()
    print(f"✅ Allocated jersey numbers in {end_allocation - end_relabel:.2f} seconds")

    return bg_img


def process_merged_tracks(
    json_path,
    image_path,
    home_jersey_numbers,
    away_jersey_numbers,
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
    fps=29.97,
    detector_firstpass_kwargs=None,
    detector_secondpass_kwargs=None,
):

    start = time.time()
    # Shared logic
    bg_img = prepare_background_and_tracks(
        json_path,
        image_path,
        home_jersey_numbers,
        away_jersey_numbers,
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
        detector_firstpass_kwargs,
        detector_secondpass_kwargs,
    )
    end = time.time()
    print(f"✅ Processed tracks in {end - start:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Process merged tracks from tracking JSONL")
    parser.add_argument(
        "--json-path",
        type=str,
        required=True,
        help="Path to the merged tracking JSONL file",
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
        "--max-step",
        type=int,
        default=20,
        help="Max distance (in pixels) allowed per frame",
    )
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
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    # start with permissive gates; tighten later
    DETECTOR_FIRSTPASS = dict(
        window_size=2201,
        step=450,
        prominence=2,
        min_wave_len=0,
        max_wave_len=60,
        speed_std_factor=None,  # enable later (e.g., 0.5) if needed
        smooth_window=0,
        savgol_poly=2,
        min_steepness=0.1,
        min_quad_curv=0.9,
        min_monotonic_ratio=0.5,
        max_gap_size=30,
    )
    # second pass with tighter gates
    DETECTOR_SECONDPASS = dict(
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

    process_merged_tracks(
        json_path=args.json_path,
        image_path=args.image_path,
        home_jersey_numbers=args.home_jersey_numbers,
        away_jersey_numbers=args.away_jersey_numbers,
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
        detector_firstpass_kwargs=DETECTOR_FIRSTPASS,
        detector_secondpass_kwargs=DETECTOR_SECONDPASS,
    )

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()

# example usage:
# python3 player_track_identification.py \
#  --json-path "./runs/detect/test_4k_player_640/team_tracking.jsonl" \
#  --image-path "./data/images/mongkok_football_field.png" \
#  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 13 23 25 8 14 17 18 21 24 31 33 34 \
#  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 1 17 22 23 24 28 33 42 43 44 72 88
