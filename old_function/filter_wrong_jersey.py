import json
import numpy as np
from collections import defaultdict


kitchee_players_jersey  = [1,2,3,4,7,10,11,16,20,27,30,13,23,25,8,14,17,18,21,24,31,33,34]
eastern_players_jersey = [26,2,6,7,9,16,20,30,36,77,99,1,17,22,23,24,28,33,42,43,44,72,88]

def resolve_duplicate_jersey_numbers(
    jsonl_path: str,
    output_path: str,
    team_jersey_lists: dict = None  # Format: {'eastern': ['1', '2', '7', '8', '10', ...], 'kitchee': [...]}
):
    """
    Resolves duplicate jersey numbers by identifying and correcting players with the same jersey number on the same team.
    
    Args:
        jsonl_path: Path to the input JSONL file with track data
        output_path: Path to save the output JSONL with resolved jersey numbers
        team_jersey_lists: Dictionary mapping team names to lists of valid jersey numbers
    """
    import json
    from collections import defaultdict
    
    # Initialize jersey lists if not provided
    if team_jersey_lists is None:
        team_jersey_lists = defaultdict(list)
    
    # Read all tracks into memory
    tracks = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    
    # Create mapping from track_id to track
    track_map = {track['track_id']: track for track in tracks}
    
    # Find all frames in the dataset
    all_frames = set()
    for track in tracks:
        frames = track.get('frames', [])
        if frames:
            all_frames.update(frames)
    
    # Find duplicate jersey numbers in each frame
    duplicates_found = defaultdict(list)  # {track_id: [(conflicting_track_id, frame), ...]}
    
    print(f"Checking {len(all_frames)} frames for duplicate jersey numbers...")
    for frame in sorted(all_frames):
        # Get all tracks visible in this frame
        frame_tracks = []
        for track in tracks:
            frames = track.get('frames', [])
            if not frames:
                continue
                
            if frame >= track.get('frame_range', [0, 0])[0] and frame <= track.get('frame_range', [0, 0])[1]:
                # Track is visible in this frame
                frame_idx = frames.index(frame) if frame in frames else None
                if frame_idx is not None:
                    frame_tracks.append((track, frame_idx))
        
        # Group tracks by team and jersey number
        team_jersey_tracks = defaultdict(lambda: defaultdict(list))
        for track, frame_idx in frame_tracks:
            team = track.get('team', '')
            if team in ['ball', 'referee', 'unsure']:
                continue
                
            jersey = track.get('jersey', 'unsure')
            if jersey == 'unsure' or jersey == []:
                continue
                
            # Check if jersey is a list (frame-by-frame) or single value
            if isinstance(jersey, list):
                if frame_idx < len(jersey):
                    current_jersey = jersey[frame_idx]
                else:
                    continue
            else:
                current_jersey = jersey
                
            if current_jersey != 'unsure':
                team_jersey_tracks[team][current_jersey].append((track, frame_idx))
        
        # Find duplicates within each team and jersey
        for team, jersey_tracks in team_jersey_tracks.items():
            for jersey_num, track_entries in jersey_tracks.items():
                if len(track_entries) > 1:
                    # Sort by jersey confidence (highest first)
                    track_entries.sort(key=lambda x: get_jersey_confidence(x[0], x[1]), reverse=True)
                    
                    # The first track has highest confidence, others are duplicates
                    main_track, _ = track_entries[0]
                    for dup_track, _ in track_entries[1:]:
                        duplicates_found[dup_track['track_id']].append((main_track['track_id'], frame, jersey_num))
    
    # Resolve duplicates
    print(f"Found {len(duplicates_found)} tracks with duplicate jersey numbers")
    resolved_count = 0
    
    for track_id, conflicts in duplicates_found.items():
        track = track_map[track_id]
        # Group conflicts by jersey number
        conflicting_jerseys = defaultdict(list)
        for _, _, jersey_num in conflicts:
            conflicting_jerseys[jersey_num].append(jersey_num)
        
        # For each conflicting jersey, find alternatives
        team = track.get('team', '')
        if not team or team in ['ball', 'referee', 'unsure']:
            continue
            
        jersey_conf = track.get('jersey_conf', 0.0)
        alternative_jerseys = []
        
        for jersey_num in conflicting_jerseys:
            similar_jerseys = find_similar_jersey_numbers(jersey_num, team_jersey_lists.get(team, []))
            for similar in similar_jerseys:
                # Check if this similar jersey is used by anyone in the same frames
                is_used = False
                for frame in track.get('frames', []):
                    for other_track in tracks:
                        if other_track['track_id'] == track_id:
                            continue
                        if other_track.get('team', '') != team:
                            continue
                        if frame not in other_track.get('frames', []):
                            continue
                        
                        other_jersey = other_track.get('jersey', 'unsure')
                        if other_jersey == similar:
                            is_used = True
                            break
                    if is_used:
                        break
                
                if not is_used:
                    alternative_jerseys.append(similar)
        
        if alternative_jerseys:
            # Update the track with alternatives
            track['jersey'] = alternative_jerseys
            track['jersey_conf'] = jersey_conf  # Keep same confidence
            resolved_count += 1
        else:
            # No alternatives found, mark as unsure
            track['jersey'] = 'unsure'
            track['jersey_conf'] = 0.0
            resolved_count += 1
    
    # Write updated tracks to output file
    with open(output_path, 'w') as out_f:
        for track in tracks:
            out_f.write(json.dumps(track) + '\n')
    
    print(f"Resolved {resolved_count} duplicate jersey numbers. Results saved to {output_path}")

def get_jersey_confidence(track, frame_idx=None):
    """
    Get the jersey confidence for a track, handling both list and scalar values.
    """
    jersey_conf = track.get('jersey_conf', 0.0)
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
            if digit in str(available) and available != jersey_num:
                similar_jerseys.append(available)
    else:
        # Multi-digit: find all numbers starting or ending with same digit
        first_digit = jersey_str[0]
        last_digit = jersey_str[-1]
        
        for available in available_jerseys:
            available_str = str(available)
            if (available_str.startswith(first_digit) or available_str.endswith(last_digit)) and available != jersey_num:
                similar_jerseys.append(available)
    
    return similar_jerseys

def foo():
    kitchee_players_jersey  = [1,2,3,4,7,10,11,16,20,27,30,13,23,25,8,14,17,18,21,24,31,33,34]
    eastern_players_jersey = [26,2,6,7,9,16,20,30,36,77,99,1,17,22,23,24,28,33,42,43,44,72,88]

    track_jsonl_path = "./runs/detect/example_1002/team_tracking_with_jersey_final.jsonl"
    track_map = {}
    with open(track_jsonl_path, "r") as f:
        for line in f:
            track = json.loads(line.strip())
            tid = track.get("track_id")
            team = track.get("team")
            jersey = track.get("jersey")
            if tid is not None:
                track_map[tid] = {
                    "team": team,
                    "jersey": jersey
                }

    for tid, info in track_map.items():
        team = info["team"]
        jersey = info["jersey"]
        if team == "kitchee":
            if jersey not in kitchee_players_jersey:
                print(f"Track {tid} has wrong jersey {jersey} for team {team}.")
                info["jersey"] = "unsure"
        elif team == "eastern":
            if jersey not in eastern_players_jersey:
                print(f"Track {tid} has wrong jersey {jersey} for team {team}.")
                info["jersey"] = "unsure"


team_jersey_lists = {
    'kitchee': kitchee_players_jersey,
    'eastern': eastern_players_jersey
}

resolve_duplicate_jersey_numbers(
    jsonl_path="./runs/detect/example_1002/team_tracking_with_jersey_final.jsonl",
    output_path="./runs/detect/example_1002/team_tracking_with_jersey_final_resolved.jsonl",
    team_jersey_lists=team_jersey_lists
)
