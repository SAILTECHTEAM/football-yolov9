import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict, deque
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import cv2
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import ijson.backends.python as ijson_python
from typing import Counter, List, Dict, Any, Tuple, Iterator, Union
from heapq import nsmallest
from tools.remove_track_sharp import process_jsonl_detect_replace
import argparse


def assign_team_by_majority_vote(team_conf_list):
    team_count = defaultdict(float)
    for conf in team_conf_list:
        for k, v in conf.items():
            team_count[k] += v
    return max(team_count, key=team_count.get) if team_count else "ball"


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
    areas_interp = np.interp(all_frames, frames, areas)

    return all_frames, full_points, areas_interp


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
    print(team_windows.items())

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
            print(f"  Found {candidate_tracks} candidate tracks for relabeling.")

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
            print(f"  Relabeled {relabeled_in_window} tracks in this window.")
            window["count"] -= relabeled_in_window
            print(window["count"])
            if window["count"] > allowed_count and relabeled_in_window > 0:
                windows.append(window)

    # Final pass to write relabeled file
    with open(track_jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
        for line in f_in:
            track = json.loads(line.strip())
            tid = track.get("track_id")
            if tid in relabel_map:
                if track.get("team") not in [not_sure_label, "referee"]:
                    track["team"] = relabel_map[tid]
            json.dump(track, f_out)
            f_out.write("\n")

    print(f"✅ Relabeled {relabel_count} tracks.")
    return relabel_count


def resolve_duplicate_jersey_numbers(
    jsonl_path: str,
    output_path: str,
    home_jersey_numbers: list = None,
    away_jersey_numbers: list = None,
):
    """
    Resolves duplicate jersey numbers by identifying and correcting players with the same jersey number on the same team.
    Optimized version with improved data structures and reduced complexity.

    Args:
        jsonl_path: Path to the input JSONL file with track data
        output_path: Path to save the output JSONL with resolved jersey numbers
        home_jersey_numbers: List of valid jersey numbers for the home team
        away_jersey_numbers: List of valid jersey numbers for the away team
    """

    # Read team jersey number lists
    if home_jersey_numbers is None or away_jersey_numbers is None:
        print("No team jersey number lists provided.")
        exit(1)

    team_jerseys = {
        "home": set(home_jersey_numbers),  # Use sets for O(1) lookups
        "away": set(away_jersey_numbers),
    }

    # Read all tracks into memory
    tracks = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))

    # Create mapping from track_id to track and build efficient indexing
    track_map = {}
    frame_to_tracks = defaultdict(list)  # Maps frame -> list of (track_id, team, jersey_num)
    track_jersey_cache = {}  # Cache jersey numbers per track per frame

    print("🔄 Building optimized index structures...")
    for track in tracks:
        track_id = track["track_id"]
        track_map[track_id] = track
        team = track.get("team", "")
        jersey_num = track.get("jersey_num", "unsure")

        # Skip non-player tracks
        if team in ["ball", "referee", "unsure"] or jersey_num == "unsure" or jersey_num == []:
            continue

        frames = track.get("frames", [])
        if not frames:
            continue

        # Build index of frames to tracks
        for i, frame in enumerate(frames):
            # Get jersey number for this frame (either from list or single value)
            current_jersey = (
                jersey_num[i]
                if isinstance(jersey_num, list) and i < len(jersey_num)
                else jersey_num
            )
            if current_jersey == "unsure" or current_jersey == -1:
                continue

            # Store mapping for this frame
            frame_to_tracks[frame].append((track_id, team, current_jersey))

            # Cache jersey confidence
            if isinstance(track.get("jersey_conf", 0.0), list) and i < len(
                track.get("jersey_conf", [])
            ):
                track_jersey_cache[(track_id, frame)] = track.get("jersey_conf", [])[i]
            else:
                track_jersey_cache[(track_id, frame)] = track.get("jersey_conf", 0.0)

    # Find duplicate jersey numbers in each frame
    duplicates_found = defaultdict(list)  # {track_id: [(conflicting_track_id, frame, jersey), ...]}

    print(f"🔄 Checking {len(frame_to_tracks)} frames for duplicate jersey numbers...")
    for frame, track_entries in frame_to_tracks.items():
        # Group tracks by team and jersey number - using dictionary for O(1) lookups
        team_jersey_tracks = defaultdict(lambda: defaultdict(list))

        for track_id, team, jersey_num in track_entries:
            if jersey_num != "unsure":
                team_jersey_tracks[team][jersey_num].append(track_id)

        # Find duplicates within each team and jersey number
        for team, jersey_tracks in team_jersey_tracks.items():
            for jersey_num, track_ids in jersey_tracks.items():
                if len(track_ids) > 1:
                    # Sort by jersey number confidence (highest first)
                    sorted_tracks = sorted(
                        track_ids,
                        key=lambda tid: track_jersey_cache.get((tid, frame), 0.0),
                        reverse=True,
                    )

                    # The first track has highest confidence, others are duplicates
                    main_track_id = sorted_tracks[0]
                    for dup_track_id in sorted_tracks[1:]:
                        duplicates_found[dup_track_id].append((main_track_id, frame, jersey_num))

    # Prepare similar jersey lookup tables
    similar_jersey_cache = {}
    for team, numbers in team_jerseys.items():
        similar_jersey_cache[team] = {}
        for num in numbers:
            similar_jersey_cache[team][num] = find_similar_jersey_numbers(num, list(numbers))

    # Track frame usage of jersey numbers
    frame_team_jersey_usage = defaultdict(lambda: defaultdict(set))
    for frame, track_entries in frame_to_tracks.items():
        for track_id, team, jersey_num in track_entries:
            frame_team_jersey_usage[frame][(team, jersey_num)].add(track_id)

    # Resolve duplicates
    resolved_count = 0

    for track_id, conflicts in duplicates_found.items():
        track = track_map[track_id]
        # Group conflicts by jersey number using Counter for efficient counting
        conflicting_jerseys = Counter(jersey_num for _, _, jersey_num in conflicts)

        # For each conflicting jersey, find alternatives
        team = track.get("team", "")
        if not team or team in ["ball", "referee", "unsure"]:
            continue

        jersey_conf = track.get("jersey_conf", 0.0)
        track_frames = set(track.get("frames", []))

        # Find available alternative jersey numbers
        available_alternatives = set()

        for jersey_num, _ in conflicting_jerseys.items():
            # Get pre-computed similar jersey numbers
            team_key = "home" if "home" in team else "away"
            similar_jerseys = similar_jersey_cache.get(team_key, {}).get(jersey_num, [])

            for similar in similar_jerseys:
                # Check if this similar jersey number is used by anyone in the same frames
                is_used = False

                # Only check frames where this track appears
                for frame in track_frames:
                    if (team, similar) in frame_team_jersey_usage.get(
                        frame, {}
                    ) and track_id not in frame_team_jersey_usage[frame][(team, similar)]:
                        is_used = True
                        break

                if not is_used:
                    available_alternatives.add(similar)

        if available_alternatives:
            # Update the track with alternatives
            track["jersey_num"] = list(available_alternatives)  # Convert set to list
            track["jersey_conf"] = jersey_conf  # Keep same confidence
            resolved_count += 1
        else:
            # No alternatives found, mark as unsure
            track["jersey_num"] = "unsure"
            track["jersey_conf"] = 0.0
            resolved_count += 1

    # Write updated tracks to output file
    with open(output_path, "w") as out_f:
        for track in tracks:
            out_f.write(json.dumps(track) + "\n")

    print(f"✅ Resolved {resolved_count} duplicate jersey numbers. Results saved to {output_path}")


def get_jersey_num_confidence(track, frame_idx=None):
    """
    Get the jersey number confidence for a track, handling both list and scalar values.
    """
    jersey_conf = track.get("jersey_conf", 0.0)
    if isinstance(jersey_conf, list):
        if frame_idx is not None and frame_idx < len(jersey_conf):
            return jersey_conf[frame_idx]
        return sum(jersey_conf) / len(jersey_conf) if jersey_conf else 0.0
    return jersey_conf


def find_similar_jersey_numbers(jersey_num, available_jerseys):
    """
    Find jersey numbers similar to the given one.
    For single-digit jersey (e.g., "7"), look for numbers containing this digit (e.g., "17", "70").
    For multi-digit jerseys (e.g., "28"), look for numbers starting with "2" or ending with "8".

    Args:
        jersey_num: The jersey number to find alternatives for
        available_jerseys: List of available jersey numbers for this team

    Returns:
        List of similar jersey numbers not used by other players
    """
    similar_jerseys = []

    # Convert to string for easier manipulation
    jersey_str = str(jersey_num)

    if len(jersey_str) == 1:
        # Single digit: find all numbers containing this digit
        digit = jersey_str
        for available in available_jerseys:
            if digit in str(available) or available == jersey_num:
                similar_jerseys.append(available)
    else:
        # Multi-digit: find all numbers starting or ending with same digit
        first_digit = jersey_str[0]
        last_digit = jersey_str[-1]

        for available in available_jerseys:
            available_str = str(available)
            if (
                available_str.startswith(first_digit) or available_str.endswith(last_digit)
            ) or available == jersey_num:
                similar_jerseys.append(available)

    return similar_jerseys


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

    allocate_jersey_numbers_by_confidence(
        jsonl_path=json_path.replace(".jsonl", "_smoothed_again.jsonl"),
        output_path=json_path.replace(".jsonl", "_final_1215.jsonl"),
        home_jersey_numbers=home_jersey_numbers,
        away_jersey_numbers=away_jersey_numbers,
    )
    end_allocation = time.time()
    print(f"✅ Allocated jersey numbers in {end_allocation - end_relabel:.2f} seconds")
    # resolve_duplicate_jersey_numbers(
    #     jsonl_path=json_path.replace('.jsonl', '_smoothed_again.jsonl'),
    #     output_path=json_path.replace('.jsonl', '_final.jsonl'),
    #     home_jersey_numbers=home_jersey_numbers,
    #     away_jersey_numbers=away_jersey_numbers,
    # )
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
    output_name,
    fps=29.97,
    detector_firstpass_kwargs=None,
    detector_secondpass_kwargs=None,
):
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(json_path))[0]

    # output_path_video = f"{output_name}.mp4"

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
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Base name of the output file (without extension)",
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
        output_name=args.output_name,
        detector_firstpass_kwargs=DETECTOR_FIRSTPASS,
        detector_secondpass_kwargs=DETECTOR_SECONDPASS,
    )

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()

# example usage:
# python3 post-processing.py \
#  --json-path "./runs/detect/test_4k_player_640/team_tracking.jsonl" \
#  --image-path "./data/images/mongkok_football_field.png" \
#  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 13 23 25 8 14 17 18 21 24 31 33 34 \
#  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 1 17 22 23 24 28 33 42 43 44 72 88 \
#  --output-name './runs/detect/test_4k_player_640/team_tracking_output'
