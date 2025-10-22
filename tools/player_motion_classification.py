import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import os


# Helper functions
def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

# Function to track ball possession
def track_ball_possession(
    jsonl_path: str,
    output_path: str,
    window_size: int = 11,
    max_distance_threshold: float = 5.0,
    min_possession_frames: int = 3
):
    """
    Track which player is holding the ball throughout the match.
    
    Parameters:
    -----------
    jsonl_path: str
        Path to the input JSONL file with tracking data
    output_path: str
        Path where the possession data will be saved
    window_size: int
        Size of the sliding window to determine consistent possession (default: 11)
    max_distance_threshold: float
        Maximum distance between player and ball to be considered possession (default: 5.0)
    min_possession_frames: int
        Minimum consecutive frames needed to register a possession event (default: 3)
    
    """
    import json
    import numpy as np
    from collections import defaultdict
    
    # Load all tracks from JSONL
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]
    
    # Separate ball and player tracks
    ball_tracks = [t for t in tracks if t.get("team") == "ball"]
    player_tracks = [t for t in tracks if t.get("team") != "ball" and t.get("team") != "referee"]

    if not ball_tracks or not player_tracks:
        print("Error: No ball track or player tracks found.")
        return {}
    
    # Create player position lookup by frame
    player_by_frame = defaultdict(list)
    for player in player_tracks:
        track_id = player.get("track_id")
        team = player.get("team")
        frames = player.get("frames", [])
        positions = player.get("projected", [])
        
        for i, (frame, pos) in enumerate(zip(frames, positions)):
            if pos is not None and len(pos) >= 2:
                player_by_frame[frame].append({
                    "track_id": track_id,
                    "team": team,
                    "position": pos
                })
    
    # Create ball position lookup by frame
    ball_positions = {}
    ball_frames = ball_tracks[0].get("frames", [])
    ball_coords = ball_tracks[0].get("projected", [])
    
    for frame, pos in zip(ball_frames, ball_coords):
        if pos is not None and len(pos) >= 2:
            ball_positions[frame] = pos
    
    # Find the frame range
    min_frame = min(ball_positions.keys()) if ball_positions else 0
    max_frame = max(ball_positions.keys()) if ball_positions else 0
    
    # Track possession events
    possession_events = []
    current_possession = None
    possession_frames = []
    possession_ball_pos = []
    possession_player_pos = []
    
    # Process each frame
    for frame in range(min_frame, max_frame + 1):
        # Skip if no ball position for this frame
        if frame not in ball_positions:
            continue
            
        # Skip if no players detected in this frame
        if frame not in player_by_frame:
            continue
        
        if (frame % 500) == 0:
            print(f"Processing frame {frame}/{max_frame}")
        ball_pos = np.array(ball_positions[frame])
        closest_player = None
        closest_distance = float('inf')
        
        # Find the closest player to the ball
        for player in player_by_frame[frame]:
            player_pos = np.array(player["position"])
            distance = np.linalg.norm(player_pos - ball_pos)
            
            if distance < closest_distance and distance <= max_distance_threshold:
                closest_distance = distance
                closest_player = player
        
        # Process possession
        if closest_player:
            player_id = closest_player["track_id"]
            team = closest_player["team"]
            
            # New possession starts
            if current_possession is None or current_possession["player_track_id"] != player_id:
                # Save the previous possession if it meets the minimum frame requirement
                if current_possession and len(possession_frames) >= min_possession_frames:
                    possession_events.append({
                        "id": len(possession_events) + 1,
                        "player_track_id": current_possession["player_track_id"],
                        "team": current_possession["team"],
                        "frame_range": [possession_frames[0], possession_frames[-1]],
                        "frames": possession_frames,
                        "ball_pos": possession_ball_pos,
                        "player_pos": possession_player_pos
                    })
                
                # Start new possession
                current_possession = {
                    "player_track_id": player_id,
                    "team": team
                }
                possession_frames = [frame]
                possession_ball_pos = [ball_positions[frame]]
                possession_player_pos = [closest_player["position"]]
            else:
                # Continue current possession
                possession_frames.append(frame)
                possession_ball_pos.append(ball_positions[frame])
                possession_player_pos.append(closest_player["position"])
        else:
            # No player close enough to the ball - potential end of possession
            if current_possession and len(possession_frames) >= min_possession_frames:
                possession_events.append({
                    "id": len(possession_events) + 1,
                    "player_track_id": current_possession["player_track_id"],
                    "team": current_possession["team"],
                    "frame_range": [possession_frames[0], possession_frames[-1]],
                    "frames": possession_frames,
                    "ball_pos": possession_ball_pos,
                    "player_pos": possession_player_pos
                })
                current_possession = None
                possession_frames = []
                possession_ball_pos = []
                possession_player_pos = []
    
    # Add the final possession if it exists
    if current_possession and len(possession_frames) >= min_possession_frames:
        possession_events.append({
            "id": len(possession_events) + 1,
            "player_track_id": current_possession["player_track_id"],
            "team": current_possession["team"],
            "frame_range": [possession_frames[0], possession_frames[-1]],
            "frames": possession_frames,
            "ball_pos": possession_ball_pos,
            "player_pos": possession_player_pos
        })
    
    # Apply sliding window to smooth out noisy detections
    if window_size > 1:
        smoothed_events = []
        for i, event in enumerate(possession_events):
            # Skip very short events that can't be smoothed properly
            if len(event["frames"]) < window_size:
                smoothed_events.append(event)
                continue
                
            # Examine each frame in the event with a sliding window
            new_frames = []
            new_ball_pos = []
            new_player_pos = []
            
            for j in range(len(event["frames"])):
                # Get window boundaries
                start_idx = max(0, j - window_size // 2)
                end_idx = min(len(event["frames"]), j + window_size // 2 + 1)
                
                # If we have enough frames in our window, add this frame
                if end_idx - start_idx >= min_possession_frames:
                    new_frames.append(event["frames"][j])
                    new_ball_pos.append(event["ball_pos"][j])
                    new_player_pos.append(event["player_pos"][j])
            
            if new_frames:  # Only add if we have frames after smoothing
                smoothed_event = event.copy()
                smoothed_event["frames"] = new_frames
                smoothed_event["ball_pos"] = new_ball_pos
                smoothed_event["player_pos"] = new_player_pos
                smoothed_event["frame_range"] = [new_frames[0], new_frames[-1]]
                smoothed_events.append(smoothed_event)
                
        possession_events = smoothed_events
    
    # Save possession data to JSONL
    with open(output_path, 'w') as f:
        for event in possession_events:
            f.write(json.dumps(event) + '\n')
    
    # Return statistics
    stats = {
        "total_possessions": len(possession_events),
        "team_possession": defaultdict(int),
        "player_possession": defaultdict(int)
    }
    
    for event in possession_events:
        frames_count = len(event["frames"])
        stats["team_possession"][event["team"]] += frames_count
        stats["player_possession"][event["player_track_id"]] += frames_count
    
    print(f"✅ Saved possession data to: {output_path}")
    print(f"Total possession events: {stats['total_possessions']}")
    for team, frames in stats["team_possession"].items():
        percentage = frames / sum(stats["team_possession"].values()) * 100
        print(f"Team {team}: {percentage:.1f}% possession ({frames} frames)")
        
    # return stats

# Suspicious Action 1 (completed)
def detect_abnormal_player_movement_direction(
    jsonl_path: str,
    angle_threshold: float = 120,
    velocity_threshold: float = 1e-2,
    min_valid_frames: int = 5,
    conf_threshold: float = 0.5,
    frame_threshold: int = 3,
    distance_threshold: float = 0.5,
    multi_ball_frames: set = None  # NEW: frames to exclude
):
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f]

    ball_tracks = [t for t in tracks if t["team"] == "ball"]
    player_tracks = [t for t in tracks if t["team"] != "ball"]

    # Build ball position lookup
    ball_by_frame: Dict[int, Tuple[float, float]] = {}
    for b in ball_tracks:
        for f, pt in zip(b["frames"], b["projected"]):
            if pt is not None:
                ball_by_frame[f] = pt

    abnormal_tracks = []
    track_confidences = {}
    track_abnormal_frames = {}

    multi_ball_frames = multi_ball_frames or set()

    for t in player_tracks:
        frames = t["frames"]
        points = t["projected"]
        if len(frames) < 2:
            continue

        abnormal_frames = []
        count_opposite = 0
        total_valid = 0

        for i in range(1, len(frames)):
            f_prev, f_curr = frames[i - 1], frames[i]
            pt_prev, pt_curr = points[i - 1], points[i]
            if pt_prev is None or pt_curr is None:
                continue

            if f_prev in multi_ball_frames or f_curr in multi_ball_frames:
                continue

            if f_prev not in ball_by_frame or f_curr not in ball_by_frame:
                continue

            # Get positions
            player_curr = np.array(pt_curr)
            player_prev = np.array(pt_prev)
            ball_curr = np.array(ball_by_frame[f_curr])
            ball_prev = np.array(ball_by_frame[f_prev])

            # Check proximity
            distance_to_ball = np.linalg.norm(player_curr - ball_curr)
            if distance_to_ball > 200: # 20 m in radius?
                # print(f"Skipping frame {f_curr} for track {t['track_id']} due to distance to ball: {distance_to_ball:.2f}")
                continue

            # Compute player movement
            player_vec = player_curr - player_prev
            if np.linalg.norm(player_vec) < velocity_threshold:
                continue

            # Compute ball movement
            ball_vec = ball_curr - ball_prev
            if np.linalg.norm(ball_vec) < velocity_threshold:
                continue

            # Compare directions
            angle = calculate_angle(player_vec, ball_vec)
            total_valid += 1

            if angle > angle_threshold:
                count_opposite += 1
                abnormal_frames.append(f_curr)

        if total_valid >= min_valid_frames:
            conf_score = count_opposite / total_valid

            if conf_score > conf_threshold and len(abnormal_frames) >= frame_threshold:
                f_start = abnormal_frames[0]
                f_end = abnormal_frames[-1]

                try:
                    idx_start = frames.index(f_start)
                    idx_end = frames.index(f_end)
                    pt_start = points[idx_start]
                    pt_end = points[idx_end]
                    ball_start = ball_by_frame.get(f_start)
                    ball_end = ball_by_frame.get(f_end)

                    if None not in [pt_start, pt_end, ball_start, ball_end]:
                        dist_start = np.linalg.norm(np.array(pt_start) - np.array(ball_start))
                        dist_end = np.linalg.norm(np.array(pt_end) - np.array(ball_end))

                        if (dist_end - dist_start) > distance_threshold:
                            abnormal_tracks.append(t["track_id"])
                            track_abnormal_frames[t["track_id"]] = abnormal_frames
                            track_confidences[t["track_id"]] = {
                                "direction_score": conf_score,
                                "dist_start": dist_start,
                                "dist_end": dist_end
                            }
                except ValueError:
                    continue

    suspicious_segments = {
    k: (track_abnormal_frames[k][0], track_abnormal_frames[k][-1])
    for k in abnormal_tracks
    if len(track_abnormal_frames[k]) >= 2
    }

    return suspicious_segments, track_abnormal_frames, track_confidences

# Suspicious Action 2 (completed)
def detect_slow_action(
        jsonl_path: str,
        possession_data_path: str = "../runs/detect/demo_video/possession_data.jsonl",
        distance_threshold: float = 100.0,  # 10 meters proximity to ball (radius)
        velocity_threshold: float = 1.5,     # slow movement threshold, in m/s
        min_valid_frames: int = 300,          # minimum frames to consider as suspicious
        n_closest_players: int = 3,          # number of closest players to track
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[int]], Dict[str, Dict]]:
    """
    Detect players exhibiting suspiciously slow movements in specific game scenarios.
    
    Args:
        jsonl_path: Path to the tracking data
        possession_data_path: Path to possession data
        distance_threshold: Maximum distance to ball to be considered (100 units ≈ 10m)
        velocity_threshold: Minimum velocity to not be considered slow
        min_valid_frames: Minimum frames with slow action to be suspicious
        n_closest_players: Number of closest players to track when opponent has ball
        
    Returns:
        Tuple containing:
        - Dictionary mapping suspicious track IDs to frame ranges
        - Dictionary mapping track IDs to list of abnormal frames
        - Dictionary of confidence scores for each track
    """
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    # Load possession data
    with open(possession_data_path, 'r') as f:
        possession_data = [json.loads(line) for line in f if line.strip()]
    
    # Create possession lookup by frame
    possession_by_frame = {}
    for event in possession_data:
        team = event.get("team", "")
        player_id = event.get("player_track_id", "")
        frames = event.get("frames", [])
        for frame in frames:
            possession_by_frame[frame] = {"team": team, "player_id": player_id}
    
    # Separate ball and player tracks
    ball_tracks = [t for t in tracks if t.get("team") == "ball"]
    player_tracks = [t for t in tracks if t.get("team") != "ball" and t.get("team") != "referee"]
    
    # Extract team names and build team lookup
    teams = set()
    for player in player_tracks:
        team = player.get("team")
        # For testing purposes, ignore "unsure" teams
        if team and team != "unsure":
            # if team name contains "goalkeeper", ignore this team
            if "goalkeeper" in team.lower():
                continue
            teams.add(team)
            if len(teams) == 2:
                # sort the team names alphabetically, we asssume first team as Team A, second as Team B
                teams = sorted(teams)
                break
    teams = list(teams)

    # Build ball position lookup
    ball_by_frame = {}
    for b in ball_tracks:
        for frame, pt in zip(b.get("frames", []), b.get("projected", [])):
            if pt is not None:
                ball_by_frame[frame] = pt
    
    # Build player position lookup, with team info
    player_by_frame = defaultdict(list)
    for player in player_tracks:
        track_id = player.get("track_id")
        team = player.get("team")
        frames = player.get("frames", [])
        points = player.get("projected", [])
        
        for i, (frame, pt) in enumerate(zip(frames, points)):
            if pt is not None and i > 0 and frames[i-1] == frame - 1 and points[i-1] is not None:
                prev_pt = points[i-1]
                # Calculate velocity (distance between consecutive frames)
                displacement = np.linalg.norm(np.array(pt) - np.array(prev_pt))
                # Now 30fps, position unit in 0.1m, so velocity in m/s = displacement * 30 * 0.1
                velocity = displacement * 30 * 0.1

                player_by_frame[frame].append({
                    "track_id": track_id,
                    "team": team,
                    "projected": pt,
                    "velocity": velocity
                })

    # Initialize tracking variables
    suspicious_segments = {}
    track_abnormal_frames = defaultdict(list)
    track_confidences = {}
    teamA_attacking_left = True  # Default assumption
    
    # Get all frames sorted
    all_frames = sorted(ball_by_frame.keys())
    
    for frame in all_frames:
        if frame not in ball_by_frame or frame not in possession_by_frame:
            continue
        
        ball_pos = np.array(ball_by_frame[frame])
        possession_info = possession_by_frame[frame]
        possession_team = possession_info.get("team")
        possession_player_id = possession_info.get("player_id")
        
        # Skip if we don't know which team has possession
        if not possession_team or possession_team == "unsure":
            continue
        
        # Determine which goal Team A and B are attacking from goalkeepers position in this frame
        players_in_frame = player_by_frame[frame]

        # Build goalkeeper position lookup to determine attacking direction
        goalkeeper_positions = {
            team: np.array([p["projected"] for p in players_in_frame if p["team"] == f"{team}goalkeeper"])
            for team in teams
        }
        # {"TeamA": np.array([[100, 200]]), "TeamB": np.array([[900, 400]])}

        # If both detections are present, compare x-coordinates
        if all(len(pos) != 0 for pos in goalkeeper_positions.values()):
            teamA_attacking_left = (goalkeeper_positions[teams[0]] >= goalkeeper_positions[teams[1]])[0][0]

        # Find players to analyze based on the scenario
        players_to_check = []

        # Later refine the logic here. quite messy now.
        for team in teams:
            # Get players from this team in this frame, no goalkeepers
            team_players_in_frame = [p for p in players_in_frame if p["team"] == team]
            
            if not team_players_in_frame:
                continue

            # Skip if own team has the ball
            if possession_team == team:
                continue  

            is_teamA = (possession_team == teams[0]) # We define Team A as the first team in alphabetical order
            
            # Determine which goal the team is attacking
            is_attacking_left = teamA_attacking_left if is_teamA else not teamA_attacking_left

            # Use the middle of the goal as reference point
            goal_position = (0, 330) if is_attacking_left else (1060, 330)

            # Skip if ball is in own half
            if (is_attacking_left and ball_pos[0] >= 530) or (not is_attacking_left and ball_pos[0] < 530):
                continue  

            # Calculate distance to goal for each player
            for player in team_players_in_frame:
                player_pos = np.array(player["projected"])
                dist_to_goal = np.linalg.norm(player_pos - goal_position)
                player["distance_to_goal"] = dist_to_goal

            # Filter players within distance threshold
            close_players = [p for p in team_players_in_frame if p["distance_to_goal"] <= distance_threshold]

            # Sort by distance to goal and take n closest
            close_players.sort(key=lambda x: x["distance_to_goal"])
            players_to_check.extend(close_players[:n_closest_players])
        
        # Check for slow movement
        for player in players_to_check:
            track_id = player["track_id"]
            velocity = player.get("velocity", float('inf'))
            
            if velocity < velocity_threshold:
                track_abnormal_frames[track_id].append(frame)
    
    # Filter tracks with enough abnormal frames
    for track_id, abnormal_frames in track_abnormal_frames.items():
        if len(abnormal_frames) >= min_valid_frames:
            # Sort frames to ensure they're in order
            abnormal_frames.sort()
            
            # Calculate confidence based on number of slow frames
            total_slow_frames = len(abnormal_frames)
            track_confidences[track_id] = {
                "slow_frames": total_slow_frames,
                "avg_velocity": sum(player_by_frame[f][i]["velocity"] 
                                  for f in abnormal_frames 
                                  for i, p in enumerate(player_by_frame[f]) 
                                  if p["track_id"] == track_id) / total_slow_frames
                                  if total_slow_frames > 0 else 0
            }
            
            # Define suspicious segment (start, end)
            suspicious_segments[track_id] = (abnormal_frames[0], abnormal_frames[-1])
    
    return suspicious_segments, track_abnormal_frames, track_confidences

# Suspicious Action 3 (completed)
def detect_stationary_players(
        jsonl_path: str,
        velocity_threshold: float = 1e-2,
        min_valid_frames: int = 5,
        distance_threshold: float = 0.5,
        conf_threshold: float = 0.5,
        frame_threshold: int = 3,
        multi_ball_frames: set = None  # NEW: frames to exclude
):
    """
    Detect players who remain stationary near the ball for extended periods.

    Args:
        jsonl_path: Path to the tracking data
        velocity_threshold: Maximum velocity of player and ball to be considered stationary
        min_valid_frames: Minimum frames with stationary action to be suspicious
        distance_threshold: Maximum displacement of player to be considered stationary
        conf_threshold: Minimum confidence score to flag as suspicious
        frame_threshold: Minimum number of abnormal frames to flag as suspicious
        multi_ball_frames: Set of frames with multiple balls to exclude from analysis
    """
    # Note: same initialization of tracks as Action 1, possible to merge
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f]

    ball_tracks = [t for t in tracks if t["team"] == "ball"]
    player_tracks = [t for t in tracks if t["team"] != "ball"]

    # Build ball position lookup
    ball_by_frame: Dict[int, Tuple[float, float]] = {}
    for b in ball_tracks:
        for f, pt in zip(b["frames"], b["projected"]):
            if pt is not None:
                ball_by_frame[f] = pt

    abnormal_tracks = []
    track_confidences = {}
    track_abnormal_frames = {}

    multi_ball_frames = multi_ball_frames or set()

    # Analyse each player track
    for t in player_tracks:
        frames = t["frames"]
        points = t["projected"]
        if len(frames) < 2:
            continue

        abnormal_frames = []
        count_stationary = 0
        total_valid = 0

        for i in range(1, len(frames)):
            f_prev, f_curr = frames[i - 1], frames[i]
            pt_prev, pt_curr = points[i - 1], points[i]
            if pt_prev is None or pt_curr is None:
                continue

            if f_prev in multi_ball_frames or f_curr in multi_ball_frames:
                continue

            if f_prev not in ball_by_frame or f_curr not in ball_by_frame:
                continue

            # Get positions
            player_curr = np.array(pt_curr)
            player_prev = np.array(pt_prev)
            ball_curr = np.array(ball_by_frame[f_curr])
            ball_prev = np.array(ball_by_frame[f_prev])

            # Check proximity
            distance_to_ball = np.linalg.norm(player_curr - ball_curr)
            if distance_to_ball > 100: # 10 m in radius?
                # print(f"Skipping frame {f_curr} for track {t['track_id']} due to distance to ball: {distance_to_ball:.2f}")
                continue

            # Compute player movement (ignore if player is moving too fast)
            player_vec = player_curr - player_prev
            if np.linalg.norm(player_vec) > velocity_threshold:
                continue

            # Compute ball movement
            ball_vec = ball_curr - ball_prev
            if np.linalg.norm(ball_vec) < velocity_threshold:
                continue

            total_valid += 1

            if np.linalg.norm(player_vec) < distance_threshold:
                count_stationary += 1
                abnormal_frames.append(f_curr)

        if total_valid >= min_valid_frames:
            conf_score = count_stationary / total_valid

            if conf_score > conf_threshold and len(abnormal_frames) >= frame_threshold:
                f_start = abnormal_frames[0]
                f_end = abnormal_frames[-1]

                try:
                    idx_start = frames.index(f_start)
                    idx_end = frames.index(f_end)
                    pt_start = points[idx_start]
                    pt_end = points[idx_end]
                    ball_start = ball_by_frame.get(f_start)
                    ball_end = ball_by_frame.get(f_end)

                    if None not in [pt_start, pt_end, ball_start, ball_end]:
                        dist_start = np.linalg.norm(np.array(pt_start) - np.array(ball_start))
                        dist_end = np.linalg.norm(np.array(pt_end) - np.array(ball_end))

                        if (dist_end - dist_start) > distance_threshold:
                            abnormal_tracks.append(t["track_id"])
                            track_abnormal_frames[t["track_id"]] = abnormal_frames
                            track_confidences[t["track_id"]] = {
                                "stationary_score": conf_score,
                                "dist_start": dist_start,
                                "dist_end": dist_end
                            }
                except ValueError:
                    continue

    suspicious_segments = {
    k: (track_abnormal_frames[k][0], track_abnormal_frames[k][-1])
    for k in abnormal_tracks
    if len(track_abnormal_frames[k]) >= 2
    }

    return suspicious_segments, track_abnormal_frames, track_confidences

# Suspicious Action 4 (completed)
def detect_possession_change_anomalies(
    jsonl_path: str,
    possession_data_path: str = "../runs/detect/demo_video/possession_data.jsonl",
):
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    # Load possession data
    with open(possession_data_path, 'r') as f:
        possession_data = [json.loads(line) for line in f if line.strip()]
    
    # Check possession changes directly from possession data
    possession_changes = {}
    previous_team = None

    for event in possession_data:
        team = event.get("team", "")
        player_id = event.get("player_track_id", "")
        frames = event.get("frames", [])

        if team == "unsure" or not frames:
            continue

        # Replace goalkeeper strings
        if "goalkeeper" in team.lower():
            team = team.replace("goalkeeper", "").strip()

        if previous_team is not None and team != previous_team:
            # Calculate player and ball distance at last 10 frames
            last_10_frames = frames[-10:]
            player_positions = event.get("player_pos", {})[-10:]
            ball_positions = event.get("ball_pos", {})[-10:]
            distances = [np.linalg.norm(np.array(p) - np.array(b)) for p, b in zip(player_positions, ball_positions) if p is not None and b is not None]
            min_index = 0
            actual_change_frame = frames[0]

            # Pick the frame with minimum distance as the actual possession change frame
            if distances:
                min_distance = min(distances)
                min_index = distances.index(min_distance)
                actual_change_frame = last_10_frames[min_index]

            else:
                actual_change_frame = frames[0]

            # Avoid using the last frame of the event as change frame
            if actual_change_frame == frames[-1]:
                min_index = max(0, min_index - 1)
                actual_change_frame = frames[-2]
            player_curr = player_positions[min_index] if min_index < len(player_positions) else None
            player_next = player_positions[min_index + 1] if (min_index + 1) < len(player_positions) else None

            # Log the possession change
            possession_changes[len(possession_changes) + 1] = {
                "possession_team": team,
                "frame_range": (frames[0], frames[-1]),
                "frame": actual_change_frame,
                "player_id": player_id,
                "player_pos_curr": player_curr,
                "player_pos_next": player_next
            }
        previous_team = team
        
    print(f"Detected {len(possession_changes)} possession changes.")

    # Separate ball and player tracks
    ball_tracks = [t for t in tracks if t.get("team") == "ball"]
    player_tracks = [t for t in tracks if t.get("team") != "ball" and t.get("team") != "referee"]

    # Build ball position lookup
    ball_by_frame = {}
    for b in ball_tracks:
        for frame, pt in zip(b.get("frames", []), b.get("projected", [])):
            if pt is not None:
                ball_by_frame[frame] = pt
    
    # Build player position lookup, with team info
    player_by_frame = defaultdict(list)
    for player in player_tracks:
        track_id = player.get("track_id")
        team = player.get("team")
        frames = player.get("frames", [])
        points = player.get("projected", [])

        if "goalkeeper" in team.lower():
            team = team.replace("goalkeeper", "").strip()
        
        for i, (frame, pt) in enumerate(zip(frames, points)):
            if pt is not None and i > 0 and frames[i-1] == frame - 1 and points[i-1] is not None:
                prev_pt = points[i-1]
                # Calculate velocity (distance between consecutive frames)
                displacement = np.linalg.norm(np.array(pt) - np.array(prev_pt))
                # Now 30fps, position unit in 0.1m, so velocity in m/s = displacement * 30 * 0.1
                velocity = displacement * 30 * 0.1

                player_by_frame[frame].append({
                    "track_id": track_id,
                    "team": team,
                    "projected": pt,
                    "velocity": velocity
                })
    # Analyze each possession change for anomalies
    abnormal_tracks = []
    track_abnormal_frames = defaultdict(list)
    track_confidences = {}
    for _ , change in possession_changes.items():
        frame = change["frame"]
        possession_team = change["possession_team"]
        player_id = change["player_id"]
        player_pos_curr = change["player_pos_curr"]
        player_pos_next = change["player_pos_next"]

        if frame not in ball_by_frame or frame not in player_by_frame:
            continue
        
        ball_pos = np.array(ball_by_frame[frame])
        players_in_frame = player_by_frame[frame]

        # Find the player who made the possession change
        changing_player = next((p for p in players_in_frame if p["track_id"] == player_id), None)
        if not changing_player:
            continue

        # Check if the changing player is close to the ball
        changing_player_pos = np.array(changing_player["projected"])
        distance_to_ball = np.linalg.norm(changing_player_pos - ball_pos)
        if distance_to_ball > 100: # 10 m in radius?
            continue

        suspicious = True
        # Check other players from both teams in this frame
        for p in players_in_frame:
            if p["team"] == possession_team:
                continue  # Skip players from the team losing possession

            other_player_pos = np.array(p["projected"])

            # Calculate the angle between the changing player and other player
            changing_player_vec = player_pos_next - changing_player_pos
            other_player_vec = other_player_pos - changing_player_pos
            angle = calculate_angle(changing_player_vec, other_player_vec)
            if angle <= 120 and np.linalg.norm(other_player_vec) < 100: # 10m
                suspicious = False
                # print(f"Frame {frame}: Possession change by player {player_id} from {possession_team} not suspicious due to nearby player {p['track_id']} at distance {np.linalg.norm(other_player_vec)/10:.2f}m and angle {angle:.2f}°")
                break # This change of possession is not considered suspicious
        if suspicious:
            abnormal_tracks.append(player_id)
            # Create a frame range for the abnormal track (now 10 seconds before and after)
            f_start = max(0, frame - 300)
            f_end = frame + 300
            track_abnormal_frames[player_id] = [f_start, f_end]
            track_confidences[player_id] = {
                "possession_team": possession_team,
                "frame": frame,
                "distance_to_ball": distance_to_ball
            }
    suspicious_segments = {
    k: (track_abnormal_frames[k][0], track_abnormal_frames[k][-1])
    for k in abnormal_tracks
    if len(track_abnormal_frames[k]) >= 1
    }
    return suspicious_segments, track_abnormal_frames, track_confidences

# Suspicious Action 5 (completed)
def detect_kicking_outside_the_pitch(
    jsonl_path: str,
    possession_data_path: str = "../runs/detect/demo_video/possession_data.jsonl",
    field_bounds: Tuple[float, float, float, float] = (0, 0, 1060, 680),
    buffer: float = 10.0,  # Buffer window for error in detection
    frame_window: int = 60  # Number of frames to include before and after trigger
):
    """
    Detect players kicking the ball outside the pitch.
    
    Args:
        jsonl_path: Path to the tracking data
        possession_data_path: Path to possession data
        field_bounds: Field boundaries (min_x, min_y, max_x, max_y)
        buffer: Buffer window for error in detection
        frame_window: Number of frames to include before and after trigger
        
    Returns:
        Tuple containing:
        - Dictionary mapping suspicious track IDs to frame ranges
        - Dictionary mapping track IDs to list of abnormal frames
        - Dictionary of confidence scores for each track
    """
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    # Load possession data
    with open(possession_data_path, 'r') as f:
        possession_data = [json.loads(line) for line in f if line.strip()]
    
    # Create possession lookup by frame
    possession_by_frame = {}
    for event in possession_data:
        team = event.get("team", "")
        player_id = event.get("player_track_id", "")
        frames = event.get("frames", [])
        for frame in frames:
            possession_by_frame[frame] = {"team": team, "player_id": player_id}
    
    # Separate ball tracks
    ball_tracks = [t for t in tracks if t.get("team") == "ball"]

    # Build ball position lookup
    ball_by_frame = {}
    for b in ball_tracks:
        for frame, pt in zip(b.get("frames", []), b.get("projected", [])):
            if pt is not None:
                ball_by_frame[frame] = pt
    
    # Define field boundaries with buffer
    min_x = field_bounds[0] + buffer
    min_y = field_bounds[1] + buffer
    max_x = field_bounds[2] - buffer
    max_y = field_bounds[3] - buffer
    
    # Initialize tracking variables
    suspicious_segments = {}
    track_abnormal_frames = defaultdict(list)
    track_confidences = {}
    
    # Get all frames sorted
    all_frames = sorted(ball_by_frame.keys())
    
    # Track when ball goes out of bounds
    out_of_bounds_start = None
    current_possession_player = None
    
    for frame in all_frames:
        if frame not in ball_by_frame:
            continue
        
        ball_pos = ball_by_frame[frame]
        x, y = ball_pos
        
        # Check if ball is out of bounds
        is_out_of_bounds = (x < min_x or x > max_x or y < min_y or y > max_y)
        
        if is_out_of_bounds:
            print(f"Frame {frame}: Ball out of bounds")
            # Get player in possession when ball went out of bounds
            possession_info = possession_by_frame.get(frame, {})

            # Skip if no possession info
            if not possession_info:
                continue
            current_possession_player = possession_info.get("player_id")
            
            # Start a new out-of-bounds sequence if needed
            if out_of_bounds_start is None:
                out_of_bounds_start = frame
                # Use the player who had possession in the last few frames
                for look_back in range(1, 11):  # Look back up to 10 frames
                    prev_frame = frame - look_back
                    if prev_frame in possession_by_frame:
                        prev_possession = possession_by_frame[prev_frame]
                        prev_possession_player = prev_possession.get("player_id")
                        if prev_possession_player:
                            current_possession_player = prev_possession_player
                            break
        else:
            # Ball came back in bounds, record the event if we have a start frame
            if out_of_bounds_start is not None and current_possession_player:
                print(f"Frame {frame}: Ball back in bounds, was out from frame {out_of_bounds_start} by player {current_possession_player}")
                # Include frames before and after the out-of-bounds sequence
                f_start = max(0, out_of_bounds_start - frame_window)
                f_end = frame + frame_window
                
                # Record this event
                track_abnormal_frames[current_possession_player].append(out_of_bounds_start)
                
                # Store confidence data
                if current_possession_player not in track_confidences:
                    print(f"Player {current_possession_player} kicked ball out at frame {out_of_bounds_start}")
                    track_confidences[current_possession_player] = {
                        "out_of_bounds_count": 1,
                        "out_of_bounds_frames": [(out_of_bounds_start, frame)]
                    }
                else:
                    track_confidences[current_possession_player]["out_of_bounds_count"] += 1
                    track_confidences[current_possession_player]["out_of_bounds_frames"].append((out_of_bounds_start, frame))
                
                # Reset tracking variables
                out_of_bounds_start = None
                current_possession_player = None
    
    # Handle case where ball was still out of bounds at the end of the video
    if out_of_bounds_start is not None and current_possession_player:
        f_start = max(0, out_of_bounds_start - frame_window)
        f_end = all_frames[-1] + frame_window
        
        track_abnormal_frames[current_possession_player].append(out_of_bounds_start)
        
        if current_possession_player not in track_confidences:
            track_confidences[current_possession_player] = {
                "out_of_bounds_count": 1,
                "out_of_bounds_frames": [(out_of_bounds_start, all_frames[-1])]
            }
        else:
            track_confidences[current_possession_player]["out_of_bounds_count"] += 1
            track_confidences[current_possession_player]["out_of_bounds_frames"].append((out_of_bounds_start, all_frames[-1]))
    
    # Create suspicious segments from abnormal frames
    for player_id, frames in track_abnormal_frames.items():
        if frames:
            # For each out-of-bounds event, create a segment with the window
            for frame in frames:
                f_start = max(0, frame - frame_window)
                f_end = frame + frame_window
                
                # If this player already has a segment, extend it if they overlap
                if player_id in suspicious_segments:
                    existing_start, existing_end = suspicious_segments[player_id]
                    if f_start <= existing_end and f_end >= existing_start:
                        # Segments overlap, merge them
                        suspicious_segments[player_id] = (
                            min(existing_start, f_start),
                            max(existing_end, f_end)
                        )
                    else:
                        # Non-overlapping segment, use the earliest one
                        # Note: This is a simplification; you might want to handle multiple segments differently
                        suspicious_segments[player_id] = (
                            min(existing_start, f_start),
                            max(existing_end, f_end)
                        )
                else:
                    suspicious_segments[player_id] = (f_start, f_end)
    
    return suspicious_segments, track_abnormal_frames, track_confidences

# Suspicious Action 6 (completed)
def detect_passive_play_in_defense(
  jsonl_path: str,
  possession_data_path: str = "../runs/detect/demo_video/possession_data.jsonl",
  velocity_threshold: float = 1.0,
  min_valid_frames: int = 5
):
    """
    Detect players in defensive positions moving slowly towards their opponents while the opposing team has possession.

    Args:
        jsonl_path: Path to the tracking data
        possession_data_path: Path to possession data
        angle_threshold: Angle threshold in degrees to consider movement towards opponent
        velocity_threshold: Velocity threshold in m/s to consider slow movement
        min_valid_frames: Minimum number of frames to consider a track as suspicious
    
    Returns:
        Tuple containing:
        - Dictionary mapping suspicious track IDs to frame ranges
        - Dictionary mapping track IDs to list of abnormal frames
        - Dictionary of confidence scores for each track
    """
    # Note: same initialization of tracks as Action 2, possible to merge
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    # Load possession data
    with open(possession_data_path, 'r') as f:
        possession_data = [json.loads(line) for line in f if line.strip()]
    
    # Create possession lookup by frame
    possession_by_frame = {}
    for event in possession_data:
        team = event.get("team", "")
        player_id = event.get("player_track_id", "")
        frames = event.get("frames", [])
        for frame in frames:
            possession_by_frame[frame] = {"team": team, "player_id": player_id}
    
    # Separate ball and player tracks
    ball_tracks = [t for t in tracks if t.get("team") == "ball"]
    player_tracks = [t for t in tracks if t.get("team") != "ball" and t.get("team") != "referee"]
    
    # Extract team names and build team lookup
    teams = set()
    for player in player_tracks:
        team = player.get("team")
        # For testing purposes, ignore "unsure" teams
        if team and team != "unsure":
            # if team name contains "goalkeeper", ignore this team
            if "goalkeeper" in team.lower():
                continue
            teams.add(team)
            if len(teams) == 2:
                # sort the team names alphabetically, we asssume first team as Team A, second as Team B
                teams = sorted(teams)
                break
    teams = list(teams)

    # Build ball position lookup
    ball_by_frame = {}
    for b in ball_tracks:
        for frame, pt in zip(b.get("frames", []), b.get("projected", [])):
            if pt is not None:
                ball_by_frame[frame] = pt
    
    # Build player position lookup, with team info
    player_by_frame = defaultdict(list)
    for player in player_tracks:
        track_id = player.get("track_id")
        team = player.get("team")
        frames = player.get("frames", [])
        points = player.get("projected", [])
        
        for i, (frame, pt) in enumerate(zip(frames, points)):
            if pt is not None and i > 0 and frames[i-1] == frame - 1 and points[i-1] is not None:
                prev_pt = points[i-1]
                # Calculate velocity (distance between consecutive frames)
                displacement = np.linalg.norm(np.array(pt) - np.array(prev_pt))
                # Now 30fps, position unit in 0.1m, so velocity in m/s = displacement * 30 * 0.1
                velocity = displacement * 30 * 0.1

                player_by_frame[frame].append({
                    "track_id": track_id,
                    "team": team,
                    "projected": pt,
                    "velocity": velocity
                })

    # Initialize tracking variables
    suspicious_segments = {}
    track_abnormal_frames = defaultdict(list)
    track_confidences = {}
    
    # Get all frames sorted
    all_frames = sorted(ball_by_frame.keys())

    for frame in all_frames:
        if frame not in ball_by_frame or frame not in possession_by_frame:
            continue
        
        ball_pos = np.array(ball_by_frame[frame])
        possession_info = possession_by_frame[frame]
        possession_team = possession_info.get("team")
        possession_player_id = possession_info.get("player_id")
        
        # Skip if we don't know which team has possession
        if not possession_team or possession_team == "unsure":
            continue
        
        # Determine which goal Team A and B are attacking from goalkeepers position in this frame
        players_in_frame = player_by_frame[frame]

        # Find players to analyze based on the scenario
        players_to_check = []

        # Later refine the logic here. quite messy now.
        for team in teams:
            # Get players from this team in this frame, no goalkeepers
            team_players_in_frame = [p for p in players_in_frame if p["team"] == team]
            
            if not team_players_in_frame:
                continue

            # Skip if own team has the ball
            if possession_team == team:
                continue  

            # Calculate distance to ball for each player
            for player in team_players_in_frame:
                player_pos = np.array(player["projected"])
                dist_to_ball = np.linalg.norm(player_pos - ball_pos)
                player["distance_to_ball"] = dist_to_ball
            # Filter players within distance threshold
            players_to_check = [p for p in team_players_in_frame if p["distance_to_ball"] <= 20]

        # Check for slow movement
        for player in players_to_check:
            track_id = player["track_id"]
            velocity = player.get("velocity", float('inf'))
            
            if velocity < velocity_threshold:
                track_abnormal_frames[track_id].append(frame)
    
    # Filter tracks with enough abnormal frames
    for track_id, abnormal_frames in track_abnormal_frames.items():
        if len(abnormal_frames) >= min_valid_frames:
            # Sort frames to ensure they're in order
            abnormal_frames.sort()
            
            # Calculate confidence based on number of slow frames
            total_slow_frames = len(abnormal_frames)
            track_confidences[track_id] = {
                "slow_frames": total_slow_frames,
                "avg_velocity": sum(player_by_frame[f][i]["velocity"] 
                                  for f in abnormal_frames 
                                  for i, p in enumerate(player_by_frame[f]) 
                                  if p["track_id"] == track_id) / total_slow_frames
                                  if total_slow_frames > 0 else 0
            }
            
            # Define suspicious segment (start, end)
            suspicious_segments[track_id] = (abnormal_frames[0], abnormal_frames[-1])
    
    return suspicious_segments, track_abnormal_frames, track_confidences

# Suspicious Action 7 (completed)
def detect_outpaced_player(
    jsonl_path: str,
    velocity_threshold: float = 3.0,      # Minimum velocity difference to consider outpaced
    distance_threshold: float = 100.0,    # Maximum distance to ball to be considered (10 meters)
    min_valid_frames: int = 30,           # Minimum frames to consider a track suspicious
    max_players_to_compare: int = 3       # Number of closest players to compare
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[int]], Dict[str, Dict]]:
    """
    Detect players who are consistently slower than other nearby players when near the ball.
    
    Args:
        jsonl_path: Path to the tracking data
        velocity_threshold: Minimum velocity difference to consider outpaced (m/s)
        distance_threshold: Maximum distance to ball to be considered (100 units ≈ 10m)
        min_valid_frames: Minimum frames with slow relative movement to be suspicious
        max_players_to_compare: Maximum number of nearby players to compare velocities with
        
    Returns:
        Tuple containing:
        - Dictionary mapping suspicious track IDs to frame ranges
        - Dictionary mapping track IDs to list of abnormal frames
        - Dictionary of confidence scores for each track
    """
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]
    
    # Separate ball and player tracks
    ball_tracks = [t for t in tracks if t.get("team") == "ball"]
    player_tracks = [t for t in tracks if t.get("team") != "ball" and t.get("team") != "referee"]
    
    # Extract team names
    teams = set()
    for player in player_tracks:
        team = player.get("team")
        if team and team != "unsure" and "goalkeeper" not in team.lower():
            teams.add(team)
            if len(teams) == 2:
                teams = sorted(teams)
                break
    
    # Build ball position lookup
    ball_by_frame = {}
    for b in ball_tracks:
        for frame, pt in zip(b.get("frames", []), b.get("projected", [])):
            if pt is not None:
                ball_by_frame[frame] = pt
    
    # Build player position lookup, with team info
    player_by_frame = defaultdict(list)
    for player in player_tracks:
        track_id = player.get("track_id")
        team = player.get("team")
        frames = player.get("frames", [])
        points = player.get("projected", [])
        
        for i, (frame, pt) in enumerate(zip(frames, points)):
            if pt is not None and i > 0 and frames[i-1] == frame - 1 and points[i-1] is not None:
                prev_pt = points[i-1]
                # Calculate velocity (distance between consecutive frames)
                displacement = np.linalg.norm(np.array(pt) - np.array(prev_pt))
                # Now 30fps, position unit in 0.1m, so velocity in m/s = displacement * 30 * 0.1
                velocity = displacement * 30 * 0.1

                player_by_frame[frame].append({
                    "track_id": track_id,
                    "team": team,
                    "projected": pt,
                    "velocity": velocity
                })

    # Initialize tracking variables
    suspicious_segments = {}
    track_abnormal_frames = defaultdict(list)
    track_confidences = {}
    
    # Get all frames sorted
    all_frames = sorted(ball_by_frame.keys())
    
    # Process each frame
    for frame in all_frames:
        if frame not in ball_by_frame:
            continue
        
        ball_pos = np.array(ball_by_frame[frame])
        players_in_frame = player_by_frame[frame]
        
        # Skip if no players in this frame
        if not players_in_frame:
            continue
            
        # Calculate distance to ball for each player
        for player in players_in_frame:
            player_pos = np.array(player["projected"])
            dist_to_ball = np.linalg.norm(player_pos - ball_pos)
            player["distance_to_ball"] = dist_to_ball
        
        # Filter players within distance threshold to ball
        nearby_players = [p for p in players_in_frame if p["distance_to_ball"] <= distance_threshold]
        
        # Skip if not enough players nearby
        if len(nearby_players) < 2:
            continue
            
        # Group players by team
        players_by_team = {}
        for p in nearby_players:
            team = p["team"]
            if "goalkeeper" in team.lower():
                team = team.replace("goalkeeper", "").strip()
                
            if team not in players_by_team:
                players_by_team[team] = []
            players_by_team[team].append(p)
        
        # Compare players from the same team
        for team, team_players in players_by_team.items():
            if len(team_players) < 2:
                continue
                
            # Sort by velocity (ascending)
            team_players.sort(key=lambda x: x.get("velocity", 0))
            
            # Get the slowest player
            slowest_player = team_players[0]
            slowest_velocity = slowest_player.get("velocity", 0)
            
            # Compare with fastest players (up to max_players_to_compare)
            for i in range(1, min(max_players_to_compare + 1, len(team_players))):
                faster_player = team_players[-i]
                faster_velocity = faster_player.get("velocity", 0)
                
                # Check if velocity difference exceeds threshold
                if faster_velocity - slowest_velocity > velocity_threshold:
                    track_abnormal_frames[slowest_player["track_id"]].append(frame)
                    break  # Only record once per frame if multiple faster players
    
    # Filter tracks with enough abnormal frames
    for track_id, abnormal_frames in track_abnormal_frames.items():
        if len(abnormal_frames) >= min_valid_frames:
            # Sort frames to ensure they're in order
            abnormal_frames.sort()
            
            # Group consecutive frames
            frame_groups = []
            current_group = [abnormal_frames[0]]
            
            for f in abnormal_frames[1:]:
                if f <= current_group[-1] + 5:  # Allow small gaps (5 frames)
                    current_group.append(f)
                else:
                    if len(current_group) >= min_valid_frames:
                        frame_groups.append(current_group)
                    current_group = [f]
            
            # Add the last group if it's large enough
            if len(current_group) >= min_valid_frames:
                frame_groups.append(current_group)
            
            # If we have any valid groups
            if frame_groups:
                # Find the longest consecutive sequence
                longest_group = max(frame_groups, key=len)
                
                # Calculate average velocity during suspicious frames
                velocities = []
                for f in longest_group:
                    for p in player_by_frame[f]:
                        if p["track_id"] == track_id:
                            velocities.append(p.get("velocity", 0))
                            break
                            
                avg_velocity = sum(velocities) / len(velocities) if velocities else 0
                
                # Store confidence data
                track_confidences[track_id] = {
                    "outpaced_frames": len(longest_group),
                    "avg_velocity": avg_velocity,
                    "consecutive_frames": len(longest_group)
                }
                
                # Define suspicious segment (start, end)
                suspicious_segments[track_id] = (longest_group[0], longest_group[-1])
    
    return suspicious_segments, track_abnormal_frames, track_confidences

# Suspicious Action 8 (completed)
def detect_delay_restart(
    jsonl_path: str,
    possession_data_path: str = "../runs/detect/demo_video/possession_data.jsonl",
    stationary_threshold: float = 3.0,  # Maximum distance ball moves to be considered stationary
    delay_threshold: int = 300,         # Minimum frames (10 seconds at 30fps)
    frame_window: int = 60              # Number of frames to include before/after trigger
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[int]], Dict[str, Dict]]:
    """
    Detect players delaying the restart of play by holding a stationary ball too long.
    
    Args:
        jsonl_path: Path to the tracking data
        possession_data_path: Path to possession data
        stationary_threshold: Maximum movement of ball to be considered stationary (units)
        delay_threshold: Minimum number of frames to consider as delay (300 frames = 10 seconds at 30fps)
        frame_window: Number of frames to include before and after trigger
        
    Returns:
        Tuple containing:
        - Dictionary mapping suspicious track IDs to frame ranges
        - Dictionary mapping track IDs to list of abnormal frames
        - Dictionary of confidence scores for each track
    """
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    # Load possession data
    with open(possession_data_path, 'r') as f:
        possession_data = [json.loads(line) for line in f if line.strip()]
    
    # Create possession lookup by frame
    possession_by_frame = {}
    for event in possession_data:
        team = event.get("team", "")
        player_id = event.get("player_track_id", "")
        frames = event.get("frames", [])
        for frame in frames:
            possession_by_frame[frame] = {"team": team, "player_id": player_id}
    
    # Separate ball tracks
    ball_tracks = [t for t in tracks if t.get("team") == "ball"]

    # Build ball position lookup
    ball_by_frame = {}
    for b in ball_tracks:
        for frame, pt in zip(b.get("frames", []), b.get("projected", [])):
            if pt is not None:
                ball_by_frame[frame] = pt
    
    # Initialize tracking variables
    suspicious_segments = {}
    track_abnormal_frames = defaultdict(list)
    track_confidences = {}
    
    # Get all frames sorted
    all_frames = sorted(ball_by_frame.keys())
    
    # Track stationary ball sequences
    stationary_start = None
    stationary_pos = None
    current_possession_player = None
    current_possession_team = None
    
    for i, frame in enumerate(all_frames):
        if frame not in ball_by_frame or frame not in possession_by_frame:
            continue
        
        ball_pos = np.array(ball_by_frame[frame])
        possession_info = possession_by_frame[frame]
        possession_player = possession_info.get("player_id")
        possession_team = possession_info.get("team")
        
        # Skip frames with unknown possession
        if not possession_player or possession_team == "unsure":
            # If we were tracking a stationary sequence, end it
            if stationary_start is not None:
                stationary_start = None
                stationary_pos = None
                current_possession_player = None
                current_possession_team = None
            continue
        
        # Check if ball is stationary
        if stationary_pos is not None:
            distance_moved = np.linalg.norm(ball_pos - stationary_pos)
            
            # If ball moved too much, reset tracking
            if distance_moved > stationary_threshold:
                stationary_start = None
                stationary_pos = None
                current_possession_player = None
                current_possession_team = None
            # If possession changed, reset tracking
            elif possession_player != current_possession_player:
                stationary_start = None
                stationary_pos = None
                current_possession_player = None
                current_possession_team = None
        
        # Start a new stationary sequence if needed
        if stationary_start is None:
            stationary_start = frame
            stationary_pos = ball_pos
            current_possession_player = possession_player
            current_possession_team = possession_team
        
        # Check if current sequence is long enough to be suspicious
        if stationary_start is not None and (frame - stationary_start) >= delay_threshold:
            # Check if we've already recorded this event
            if not track_abnormal_frames[current_possession_player] or track_abnormal_frames[current_possession_player][-1] < stationary_start:
                # Add this sequence to our tracking
                track_abnormal_frames[current_possession_player].append(stationary_start)
                
                # Store confidence data
                if current_possession_player not in track_confidences:
                    track_confidences[current_possession_player] = {
                        "delay_count": 1,
                        "delay_frames": [(stationary_start, frame)],
                        "team": current_possession_team,
                        "duration": frame - stationary_start
                    }
                else:
                    track_confidences[current_possession_player]["delay_count"] += 1
                    track_confidences[current_possession_player]["delay_frames"].append((stationary_start, frame))
                    track_confidences[current_possession_player]["duration"] += frame - stationary_start
                
                # Reset tracking to avoid double-counting
                stationary_start = None
                stationary_pos = None
                current_possession_player = None
                current_possession_team = None
    
    # Create suspicious segments from abnormal frames
    for player_id, frames in track_abnormal_frames.items():
        if frames:
            segments = []
            
            # For each delay event, create a segment with the window
            for start_frame in frames:
                # Find the end of this delay sequence from confidence data
                end_frames = [end for start, end in track_confidences[player_id]["delay_frames"] 
                             if start == start_frame]
                
                if not end_frames:  # Should never happen, but just in case
                    continue
                    
                end_frame = end_frames[0]
                
                f_start = max(0, start_frame - frame_window)
                f_end = min(end_frame + frame_window, all_frames[-1])
                
                segments.append((f_start, f_end))
            
            # Merge overlapping segments
            if segments:
                segments.sort()
                merged_segments = [segments[0]]
                
                for current in segments[1:]:
                    previous = merged_segments[-1]
                    if current[0] <= previous[1]:
                        # Segments overlap, merge them
                        merged_segments[-1] = (previous[0], max(previous[1], current[1]))
                    else:
                        # Non-overlapping segment
                        merged_segments.append(current)
                
                # Use the earliest segment as the main one
                suspicious_segments[player_id] = merged_segments[0]
    
    return suspicious_segments, track_abnormal_frames, track_confidences

# Suspicious Action 9 (completed)
def count_abnormal_possession_changes_whole_match(
    possession_data_path: str = "../runs/detect/demo_video/possession_data.jsonl",
    max_possession_changes: int = 8
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, List[int]], Dict[str, Dict]]:
    """
    Count players with an unusually high number of possession changes in a match.
    
    Args:
        possession_data_path: Path to possession data
        max_possession_changes: Maximum number of possession changes to be considered normal

    Returns:
        Tuple containing:
        - Dictionary mapping suspicious track IDs to frame ranges
        - Dictionary mapping track IDs to list of abnormal frames
        - Dictionary of confidence scores for each track
    """
    # Load data
    with open(possession_data_path, 'r') as f:
        possession_data = [json.loads(line) for line in f if line.strip()]

    # Count possession changes per player
    possession_changes = defaultdict(int)
    for event in possession_data:
        player_id = event.get("player_track_id", "")
        if player_id:
            possession_changes[player_id] += 1

    # Identify players exceeding the threshold
    suspicious_segments = {}
    track_abnormal_frames = defaultdict(list)
    track_confidences = {}

    for player_id, change_count in possession_changes.items():
        if change_count > max_possession_changes:
            # Mark the frames where the possession changes occurred (60 frames window around each event)
            frames = []
            for event in possession_data:
                if event.get("player_track_id", "") == player_id:
                    frames.extend(event.get("frames", []))
                    break # Only need to find the first event for frame extraction
            track_abnormal_frames[player_id] = frames
            track_confidences[player_id] = {
                "possession_changes": change_count
            }
            if frames:
                suspicious_segments[player_id] = (min(frames), max(frames))
    return suspicious_segments, track_abnormal_frames, track_confidences

# This function calls all suspicious action detectors
def detect_abnormal_tracks_from_jsonl(
    jsonl_path: str,
    angle_threshold: float = 120,
    velocity_threshold: float = 1e-2,
    min_valid_frames: int = 5,
    conf_threshold: float = 0.5,
    frame_threshold: int = 3,
    distance_threshold: float = 0.5,
    multi_ball_frames: set = None  # NEW: frames to exclude
):
    total_abnormal_tracks = {}
    total_track_abnormal_frames = {}
    total_tracking_confidences = {}

    possession_data_path = os.path.join(os.path.dirname(jsonl_path), "possession_data.jsonl")
    track_ball_possession(
        jsonl_path=jsonl_path,
        output_path=possession_data_path,
        window_size=11,
        max_distance_threshold=50,
        min_possession_frames=3
    )
    # Suspicious Action 1: Abnormal Player Movement Relative to Ball
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_abnormal_player_movement_direction(
        jsonl_path,
        angle_threshold=angle_threshold,
        velocity_threshold=velocity_threshold,
        min_valid_frames=min_valid_frames,
        conf_threshold=conf_threshold,
        frame_threshold=frame_threshold,
        distance_threshold=distance_threshold,
        multi_ball_frames=multi_ball_frames
    )
    # Update total results with tag 1
    total_abnormal_tracks.update({1: abnormal_tracks})
    total_track_abnormal_frames.update({1: track_abnormal_frames})
    total_tracking_confidences.update({1: track_confidences})

    # Suspicious Action 2: Slow Action Detection
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_slow_action(
        jsonl_path,
        possession_data_path=possession_data_path,
        distance_threshold=100.0,  # 10 meters proximity to ball
        velocity_threshold=1.5,     # slow movement threshold, in m/s
        min_valid_frames=200,          # minimum frames to consider as suspicious
        n_closest_players=3,          # number of closest players to track
    )
    # Update total results with tag 2
    total_abnormal_tracks.update({2: abnormal_tracks})
    total_track_abnormal_frames.update({2: track_abnormal_frames})
    total_tracking_confidences.update({2: track_confidences})

    # Suspicious Action 3: Stationary Player Detection
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_stationary_players(
        jsonl_path,
        velocity_threshold=1e-2,
        min_valid_frames=5,
        conf_threshold=0.5,
        frame_threshold=3,
        distance_threshold=0.5,  # 0.5 meters proximity to ball
        multi_ball_frames=None
    )
    # Update total results with tag 3
    total_abnormal_tracks.update({3: abnormal_tracks})
    total_track_abnormal_frames.update({3: track_abnormal_frames})
    total_tracking_confidences.update({3: track_confidences})

    # Suspicious Action 4: Possession Change Detection
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_possession_change_anomalies(
        jsonl_path=jsonl_path,
        possession_data_path=possession_data_path
    )
    # Update total results with tag 4
    total_abnormal_tracks.update({4: abnormal_tracks})
    total_track_abnormal_frames.update({4: track_abnormal_frames})
    total_tracking_confidences.update({4: track_confidences})

    # Suspicious Action 5: Kicking Outside the Pitch
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_kicking_outside_the_pitch(
        jsonl_path=jsonl_path,
        possession_data_path=possession_data_path,
        field_bounds=(0, 0, 1060, 680),
        buffer=40.0,  # Buffer window for error in detection
        frame_window=0  # Number of frames to include before and after trigger
    )
    # Update total results with tag 5
    total_abnormal_tracks.update({5: abnormal_tracks})
    total_track_abnormal_frames.update({5: track_abnormal_frames})
    total_tracking_confidences.update({5: track_confidences})

    # Suspicious Action 6: Passive Play in Defense
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_passive_play_in_defense(
        jsonl_path=jsonl_path,
        possession_data_path=possession_data_path,
        velocity_threshold=1.0,
        min_valid_frames=60
    )
    # Update total results with tag 6
    total_abnormal_tracks.update({6: abnormal_tracks})
    total_track_abnormal_frames.update({6: track_abnormal_frames})
    total_tracking_confidences.update({6: track_confidences})

    # Suspicious Action 7: Outpaced Player Detection
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_outpaced_player(
        jsonl_path=jsonl_path,
        velocity_threshold = 3.0,
        distance_threshold = 100.0,
        min_valid_frames = 60,
        max_players_to_compare = 3
    )
    # Update total results with tag 7
    total_abnormal_tracks.update({7: abnormal_tracks})
    total_track_abnormal_frames.update({7: track_abnormal_frames})
    total_tracking_confidences.update({7: track_confidences})

    # Suspicious Action 8: Delay Restart Detection
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_delay_restart(
        jsonl_path=jsonl_path,
        possession_data_path=possession_data_path,
        stationary_threshold=3.0,
        delay_threshold=300,
        frame_window=0
    )
    # Update total results with tag 8
    total_abnormal_tracks.update({8: abnormal_tracks})
    total_track_abnormal_frames.update({8: track_abnormal_frames})
    total_tracking_confidences.update({8: track_confidences})

    # Suspicious Action 9: Abnormal Possession Changes
    suspicious_segments, track_abnormal_frames, track_confidences = count_abnormal_possession_changes_whole_match(
        possession_data_path=possession_data_path,
        max_possession_changes=8
    )
    # Update total results with tag 9
    total_abnormal_tracks.update({9: suspicious_segments})
    total_track_abnormal_frames.update({9: track_abnormal_frames})
    total_tracking_confidences.update({9: track_confidences})

    return total_abnormal_tracks, total_track_abnormal_frames, total_tracking_confidences

    
def get_frames_with_multiple_balls(jsonl_path: str) -> set:
    counter = defaultdict(int)
    with open(jsonl_path) as f:
        for line in f:
            t = json.loads(line)
            if t.get("team") != "ball":
                continue
            frames = t.get("frames", [])
            projected = t.get("projected", [])
            for f_id, pt in zip(frames, projected):
                if pt is not None:
                    counter[f_id] += 1
    return {f for f, count in counter.items() if count > 1}


if __name__ == "__main__":
    # Load JSONL
    jsonl_path = "./runs/detect/test_4k/team_tracking_relabeled.jsonl"
    angle_threshold = 120
    velocity_threshold = 1e-2
    min_valid_frames = 5
    conf_threshold = 0.7
    frame_threshold = 3
    distance_threshold = 50

    multi_ball_frames = get_frames_with_multiple_balls(jsonl_path)
    print("Multi-ball frames detected:", len(multi_ball_frames))

    # Detect abnormal tracks
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_abnormal_tracks_from_jsonl(
        jsonl_path,
        angle_threshold=angle_threshold,
        velocity_threshold=velocity_threshold,
        min_valid_frames=min_valid_frames,
        conf_threshold=conf_threshold,
        frame_threshold=frame_threshold,
        distance_threshold=distance_threshold,
        multi_ball_frames=multi_ball_frames
    )
    
    # Output
    print("Track Abnormal Frames (start and end only):", abnormal_tracks)
    # print("Abnormal Track Scores:", {k: track_confidences[k] for k in abnormal_tracks[:5]})
    # print("Total Abnormal Tracks:", len(abnormal_tracks))
    # print("Track Count (analyzed):", len(track_confidences))