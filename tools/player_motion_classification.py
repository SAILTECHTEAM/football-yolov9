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
    
    Returns:
    --------
    Dict with possession statistics
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
        
    return stats

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

# Suspicious Action 2 (accelerate motion not yet checked, gate)
def detect_slow_action(
        jsonl_path: str,
        possession_data_path: str = "../runs/detect/demo_video/possession_data.jsonl",
        distance_threshold: float = 100.0,  # 10 meters proximity to ball (radius)
        velocity_threshold: float = 1.5,     # slow movement threshold, in m/s
        min_valid_frames: int = 5,          # minimum frames to consider as suspicious
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

# Suspicious Action 3 (not completed)
def detect_stationary_players(
        jsonl_path: str,
):
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]
    
    # Separate ball and player tracks
    ball_tracks = [t for t in tracks if t.get("team") == "ball"]
    player_tracks = [t for t in tracks if t.get("team") != "ball" and t.get("team") != "referee"]
    
    # Extract team names and build team lookup
    teams = list()
    for player in player_tracks:
        team = player.get("team")
        # For testing purposes, ignore "unsure" teams
        if team and team != "unsure":
            # if team name contains "goalkeeper", remove that part
            if "goalkeeper" in team.lower():
                team = team.lower().replace("goalkeeper", "").strip()
            teams.append(team)
            if len(teams) == 2:
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

# Suspicious Action 4
def detect_possession_change_anomalies(
    jsonl_path: str,
    possession_data_path: str = "../runs/detect/demo_video/possession_data.jsonl",
):
    pass

# Suspicious Action 5
def detect_kicking_outside_the_pitch(
        
):
    pass

# The function which calls all suspicious action detectors
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
    total_abnormal_tracks.update(abnormal_tracks)
    total_track_abnormal_frames.update(track_abnormal_frames)
    total_tracking_confidences.update(track_confidences)

    abnormal_tracks, track_abnormal_frames, track_confidences = detect_slow_action(
        jsonl_path,
        possession_data_path="../runs/detect/demo_video/possession_data.jsonl",
        distance_threshold=100.0,  # 10 meters proximity to ball
        velocity_threshold=1.5,     # slow movement threshold, in m/s
        min_valid_frames=5,          # minimum frames to consider as suspicious
        n_closest_players=3,          # number of closest players to track
    )

    total_abnormal_tracks.update(abnormal_tracks)
    total_track_abnormal_frames.update(track_abnormal_frames)
    total_tracking_confidences.update(track_confidences)

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