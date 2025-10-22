import os
import numpy as np
import pandas as pd
import json
import argparse


import os
import numpy as np
import pandas as pd
import json
import argparse


def calculate_player_statistics(
    jsonl_path: str,
    frame_interval: int = 30,
    fps: int = 29.97,
    comparison_split: float = None
):
    """
    Calculate player statistics and optionally analyze performance changes by comparing
    the first k% of data to the remaining (1-k)% data.
    
    Args:
        jsonl_path: Path to the tracking data JSON file
        frame_interval: Interval between frames to use for calculations (default: 30)
        fps: Frames per second of the video
        comparison_split: Value between 0 and 1 indicating the split point (e.g., 0.7 for first 70%)
        
    Returns:
        DataFrame containing player statistics or tuple of DataFrames if comparison_split is specified
    """
    # Load data
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f if line.strip()]

    # Separate player tracks and extract metadata
    player_tracks = []
    for track in tracks:
        if track.get("team") != "ball" and track.get("team") != "referee":
            # Skip tracks with unsure jersey numbers or jersey numbers as lists
            
            team = track.get("team", "")
            if "goalkeeper" in team.lower():
                track["jersey_num"] = "GK"
            if isinstance(track.get("jersey_num"), list):
                track["jersey_num"] = "/".join(map(str, track["jersey_num"]))
            jersey_num = track.get("jersey_num", "unsure")
            if jersey_num == "unsure":
                continue
            
            player_tracks.append(track)

    # Calculate statistics
    player_stats = []
    fatigue_stats = [] if comparison_split else None
    
    for player in player_tracks:
        track_id = player.get("track_id")
        team = player.get("team", "")
        jersey_num = player.get("jersey_num")
        points = player.get("projected", [])
        frames = player.get("frames", [])
        
        if not points or len(points) < 2:
            continue  # Skip tracks with insufficient data
            
        # Calculate total distance using the specified frame interval
        total_distance = 0.0
        speeds = []
        
        # Store point-by-point data for potential split analysis
        point_data = []
        
        # Use frame_interval to sample frames
        for i in range(frame_interval, len(points), frame_interval):
            prev_idx = i - frame_interval
            if points[i] is not None and points[prev_idx] is not None:
                # Calculate displacement between points at interval
                displacement = np.linalg.norm(np.array(points[i]) - np.array(points[prev_idx]))
                
                # Convert to meters (0.1m per unit)
                distance_meters = displacement * 0.1
                total_distance += distance_meters
                
                # Calculate time between frames
                dt = (frames[i] - frames[prev_idx]) / fps  # Time in seconds
                if dt > 0:
                    # Calculate speed in km/h
                    speed = (distance_meters / dt) * 3.6
                    speeds.append(speed)
                    
                    # Store data point for split analysis if needed
                    if comparison_split:
                        point_data.append({
                            'index': i,  # Store index for later splitting
                            'distance': distance_meters,
                            'speed': speed
                        })

        # Calculate speed statistics
        if speeds:
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            min_speed = np.quantile(speeds, 0.1)  # 10th percentile to avoid outliers
        else:
            avg_speed = max_speed = min_speed = 0.0
            

        
        # If fatigue analysis is requested and we have enough data points
        if comparison_split and len(point_data) >= 4:  # At least 4 data points for meaningful comparison
            # Calculate split point index
            split_idx = int(len(point_data) * comparison_split)
            
            if split_idx > 0 and split_idx < len(point_data):
                # Split the data
                first_part = point_data[:split_idx]
                second_part = point_data[split_idx:]
                
                # Calculate metrics for each part
                first_dist = sum(p['distance'] for p in first_part)
                second_dist = sum(p['distance'] for p in second_part)
                
                first_speeds = [p['speed'] for p in first_part]
                second_speeds = [p['speed'] for p in second_part]
                
                if first_speeds and second_speeds:
                    first_avg_speed = np.mean(first_speeds)
                    second_avg_speed = np.mean(second_speeds)
                    
                    first_max_speed = np.max(first_speeds)
                    second_max_speed = np.max(second_speeds)
                    
                    first_min_speed = np.quantile(first_speeds, 0.1)
                    second_min_speed = np.quantile(second_speeds, 0.1)
                    
                    # Calculate ratios (second part / first part)
                    dist_ratio = second_dist / first_dist if first_dist > 0 else 0
                    avg_speed_ratio = second_avg_speed / first_avg_speed if first_avg_speed > 0 else 0
                    max_speed_ratio = second_max_speed / first_max_speed if first_max_speed > 0 else 0
                    min_speed_ratio = second_min_speed / first_min_speed if first_min_speed > 0 else 0
                    
                    # Adjust ratios based on segment length for fair comparison
                    # A smaller segment should be adjusted proportionally
                    length_adjustment = len(first_part) / len(second_part)
                    adjusted_dist_ratio = dist_ratio * length_adjustment
                            
                    # Add to player stats
                    player_stats.append({
                        'track_id': track_id,
                        'team': team,
                        'jersey_num': jersey_num,
                        'total_distance (m)': round(total_distance, 2),
                        'avg_speed (kmh)': round(avg_speed, 2),
                        'max_speed (kmh)': round(max_speed, 2),
                        'min_speed (kmh)': round(min_speed, 2),
                        'distance_ratio': round(dist_ratio, 2),
                        'distance_ratio': round(adjusted_dist_ratio, 2),
                        'avg_speed_ratio': round(avg_speed_ratio, 2),
                        'max_speed_ratio': round(max_speed_ratio, 2),
                        'min_speed_ratio': round(min_speed_ratio, 2),
                        # 'first_segment_points': len(first_part),
                        # 'second_segment_points': len(second_part),
                    })
    
    # Convert to DataFrame for easy CSV creation
    stats_df = pd.DataFrame(player_stats)
    
    # Group by team and jersey_num if needed
    # This will combine stats for the same player across multiple tracks
    if len(stats_df) > 0:
        grouped_stats = stats_df.groupby(['team', 'jersey_num']).agg({
            'total_distance (m)': 'sum',
            'avg_speed (kmh)': 'mean',
            'max_speed (kmh)': 'max',
            'min_speed (kmh)': 'min',
            'distance_ratio': 'mean',
            'avg_speed_ratio': 'mean',
            'max_speed_ratio': 'mean',
            'min_speed_ratio': 'mean',
        }).reset_index()

        # Save to CSV without error handling
        output_csv_path = os.path.dirname(jsonl_path)
        output_csv_path = os.path.join(output_csv_path, "player_statistics.csv")
        grouped_stats.to_csv(output_csv_path, index=False)
        print(f"✅ Player statistics saved to {output_csv_path}")
        
        return grouped_stats
    else:
        print("No valid player statistics found")
        return None


def parse_opts():
    parser = argparse.ArgumentParser(description="Calculate player total distance travelled, speed related statistics for comparison.")
    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--frame-interval", type=int, default=30, help="Interval between frames to use for calculations (default: 30 = every 30th frame).")
    parser.add_argument("--fps", type=float, default=29.97, help="Video frames per second.")
    parser.add_argument("--comparison-split", type=float, help="Value between 0 and 1 to compare first k%% vs remaining (1-k)%% of data")
    args = parser.parse_args()

    return args


def main():
    args = parse_opts()
    result = calculate_player_statistics(
        jsonl_path=args.jsonl_path,
        frame_interval=args.frame_interval,
        fps=args.fps,
        comparison_split=args.comparison_split
    )
    
    if result is not None:
        if isinstance(result, tuple) and len(result) == 2:
            player_stats, comparison_stats = result
            print("\nRegular Statistics:")
            print(player_stats.head())
            print("\nPerformance Comparison (Ratios):")
            print(comparison_stats.head())
        else:
            print(result.head())


if __name__ == "__main__":
    main()