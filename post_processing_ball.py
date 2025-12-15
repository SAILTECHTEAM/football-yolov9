import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict, deque
from scipy.signal import savgol_filter
import cv2
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import ijson.backends.python as ijson_python
from typing import List, Dict, Any, Tuple, Iterator, Union, Optional
from heapq import nsmallest
from tools.remove_track_sharp import process_jsonl_detect_replace

import argparse

from scipy import interpolate
from scipy.signal import find_peaks

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph, KDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor

from scipy.linalg import expm
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from tqdm import tqdm


# --- Ball tracking specific functions --#


def convert_ball_tracking_format(json_path, output_path, max_frame_gap=10):
    """
    Convert input ball tracking JSONL with per-frame data to a consolidated track format.

    Args:
        json_path (str): Path to input JSONL file with per-frame ball detections
        output_path (str): Path to output JSONL file with consolidated track format

    The input format is:
    {
        "frame_id": 1,
        "projected": [[x, y], ...]
    }

    The output format is:
    {
        "track_id": 0,  # 0 for ball
        "team": "ball",
        "frame_range": [start_frame, end_frame],
        "frames": [frame_ids],
        "projected": [[x, y], ...]
    }
    """
    # Load and sort all frames
    frame_data = []

    with open(json_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    frame_data.append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line")

    # Extract frames and projected points
    frames = []
    projected = []

    for frame in frame_data:
        frame_id = frame.get("frame_id")
        proj_points = frame.get("projected", [])

        # Skip frames without data
        if not frame_id or not proj_points:
            continue

        # In case there are multiple ball detections in a frame, take the first one
        # You could implement more complex logic here if needed
        if proj_points and len(proj_points) > 0:
            frames.append(frame_id)
            projected.append(proj_points)

    # Generate output track
    if frames:
        output_track = {
            "track_id": 0,  # Fixed for ball
            "team": "ball",  # Fixed for ball
            "frames": frames,
            "projected": projected,
        }

        # Write to output file
        with open(output_path, "w") as out_f:
            out_f.write(json.dumps(output_track) + "\n")

        # print(f"✅ Converted {len(frames)} frames to ball track in {output_path}")
    else:
        print("⚠️ No valid frames found to convert")


def convert_ball_tracking_json_to_numpy(json_file):
    """
    Convert ball tracking data from JSONL format to a sorted NumPy array.

    Args:
        json_file (str): Path to the input JSONL file containing ball tracking data
        output_file (str): Path to save the output .npy file

    Returns:
        np.ndarray: The sorted array with format [[frame_id, x, y], ...]
    """
    # List to store the ball positions
    ball_positions = []

    # Read the JSONL file
    with open(json_file, "r") as file:
        for line in file:
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    if "frame_id" in data:  # Format in {frame_id, x, y}
                        frame_id = data.get("frame_id")

                        # Extract the projected coordinates (x, y)
                        if "projected" in data:
                            x, y = data["projected"]

                            # Add to our list as [frame_id, x, y]
                            ball_positions.append([frame_id, x, y])
                    elif (
                        "frames" in data
                    ):  # Format in {track_id, team, frame_range, frames, projected}
                        frames = data.get("frames", [])
                        projected = data.get("projected", [])
                        for f, p in zip(frames, projected):
                            if p is not None and len(p) == 2:
                                ball_positions.append([f, p[0], p[1]])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
                except Exception as e:
                    print(f"Error processing line: {e}")

    # Convert to NumPy array
    ball_positions_array = np.array(ball_positions)

    # Sort by frame_id
    sorted_indices = np.argsort(ball_positions_array[:, 0])
    sorted_ball_positions = ball_positions_array[sorted_indices]

    return sorted_ball_positions


def convert_ball_tracking_numpy_to_json(numpy_arr, output_file):
    """
    Convert ball tracking data from NumPy format back to JSONL format.

    Args:
        numpy_arr (np.ndarray): Array containing ball tracking data [[frame_id, x, y], ...]
        output_file (str): Path to save the output JSONL file

    Returns:
        None
    """

    # Write to JSONL file
    with open(output_file, "w") as file:
        for frame_id, x, y in numpy_arr:
            json_line = json.dumps({"frame_id": int(frame_id), "projected": [float(x), float(y)]})
            file.write(json_line + "\n")

    # print(f"Converted ball positions from NumPy array to JSONL format")
    # print(f"Saved to {output_file}")


def filter_multiple_detections(ball_xy, max_speed=150, static_threshold=5, window_size=5, fps=30):
    """
    Filter out false ball detections when multiple candidates exist in the same frame.

    Parameters:
    -----------
    ball_xy : np.ndarray
        Array with shape (n, 3) where each row is [frame_idx, x, y]
    max_speed : float
        Maximum plausible speed of the ball (pixels/frame)
    static_threshold : float
        Minimum movement required to not be considered static
    window_size : int
        Size of window to analyze for trajectory consistency
    fps : int
        Frames per second of the video (for speed calculations)

    Returns:
    --------
    np.ndarray
        Filtered ball_xy with only one detection per frame
    """
    # Group detections by frame
    frame_to_detections = defaultdict(list)
    for i in range(len(ball_xy)):
        frame_id = int(ball_xy[i, 0])
        frame_to_detections[frame_id].append((i, ball_xy[i, 1:]))

    # Initialize Kalman filter for trajectory prediction
    kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
    dt = 1 / fps  # Time step based on fps

    # State transition matrix (constant velocity model)
    kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Measurement matrix (we only observe position, not velocity)
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    # Initial state uncertainty
    kf.P *= 100

    # Measurement noise
    kf.R = np.diag([50, 50])

    # Process noise
    kf.Q = np.eye(4) * 10

    # Track frame-to-frame movement for each detection path
    movements = defaultdict(list)

    # Create a mask for points to keep
    keep_indices = []
    last_position = None

    # Process frames in order
    sorted_frames = sorted(frame_to_detections.keys())

    # Initialize with the first frame
    if sorted_frames:
        first_frame = sorted_frames[0]
        # If only one detection in first frame, use it
        if len(frame_to_detections[first_frame]) == 1:
            idx, pos = frame_to_detections[first_frame][0]
            keep_indices.append(idx)
            last_position = pos
            # Initialize Kalman filter
            kf.x = np.array([[pos[0]], [pos[1]], [0], [0]])
        # If multiple detections, use the one closest to the center
        else:
            center = np.array(
                [530, 330]
            )  # Approximate field center based on size of field [1060, 660]
            dists = [np.linalg.norm(pos - center) for _, pos in frame_to_detections[first_frame]]
            best_idx = np.argmin(dists)
            idx, pos = frame_to_detections[first_frame][best_idx]
            keep_indices.append(idx)
            last_position = pos
            # Initialize Kalman filter
            kf.x = np.array([[pos[0]], [pos[1]], [0], [0]])

    # Process remaining frames
    for i in range(1, len(sorted_frames)):
        current_frame = sorted_frames[i]
        last_frame = sorted_frames[i - 1]

        dt = (current_frame - last_frame) / fps
        # Update F matrix with actual dt
        kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

        # In case of frame gaps, increase process noise
        if dt > 1 / fps:
            # Update process noise based on frame gap
            kf.Q = np.eye(4) * 10 * min(1 + dt, 5)  # Cap the growth for very large gaps
        else:
            kf.Q = np.eye(4) * 10

        detections = frame_to_detections[current_frame]

        # If only one detection in this frame, easy case
        if len(detections) == 1:
            idx, pos = detections[0]

            # Check if this is a plausible continuation
            if last_position is not None:
                distance = np.linalg.norm(pos - last_position)
                if distance > max_speed:
                    # Skip implausibly fast movements
                    print(
                        f"Skipping detection at frame {current_frame} due to high speed: {distance:.1f} pixels"
                    )
                    continue

            keep_indices.append(idx)
            last_position = pos

            # Update Kalman filter
            kf.predict()
            kf.update(pos)

        # If multiple detections, choose the most likely one
        else:
            # Predict next position using Kalman filter
            kf.predict()
            predicted_pos = kf.x[:2].flatten()

            # Score each detection based on multiple criteria
            scores = []
            for idx, pos in detections:
                # 1. Distance from predicted position
                pred_distance = np.linalg.norm(pos - predicted_pos)

                # 2. Check if this is a static detection point
                static_score = 0
                for prev_frame in range(max(0, current_frame - window_size), current_frame):
                    if prev_frame in frame_to_detections:
                        for _, prev_pos in frame_to_detections[prev_frame]:
                            if np.linalg.norm(pos - prev_pos) < static_threshold:
                                static_score += 1

                # 3. Distance from previous position
                if last_position is not None:
                    prev_distance = np.linalg.norm(pos - last_position)
                    # Heavily penalize positions too far from previous
                    if prev_distance > max_speed:
                        prev_distance_score = 1000  # Large penalty
                    else:
                        prev_distance_score = prev_distance
                else:
                    prev_distance_score = 0

                # Combined score (lower is better)
                score = pred_distance + (static_score * 50) + prev_distance_score
                scores.append((idx, score))

            # Choose detection with lowest score
            scores.sort(key=lambda x: x[1])
            best_idx, _ = scores[0]
            best_pos = ball_xy[best_idx, 1:]

            keep_indices.append(best_idx)
            last_position = best_pos

            # Update Kalman filter with chosen position
            kf.update(best_pos)

    # Return the filtered array
    return ball_xy[keep_indices]


def remove_detections_near_high_density_region(
    ball_xy,
    field_size=(1060, 660),
    bin_size=(5, 5),
    primary_threshold=500,
    secondary_threshold=100,
    neighbor_radius=10,
):
    """
    Density-based filter with multi-level thresholding and spatial awareness.
    Divide the football field into grids, allocate all detections to their respective grids,
    identify high-density grids, and remove points in and around these grids.

    Parameters:
    -----------
    ball_xy : np.ndarray
        Array with shape (n, 3) where each row is [frame_idx, x, y]
    field_size : list
        Size of the football field [width, height]
    bin_size : tuple
        Size of histogram bins (x, y)
    primary_threshold : int
        Primary threshold for high-density regions
    secondary_threshold : int
        Secondary threshold for neighboring regions
    neighbor_radius : int
        Radius for considering neighboring bins

    Returns:
    --------
    filtered_ball_xy : np.ndarray
        Filtered ball tracking data with high-density points removed
    """
    # Create a 2D histogram as before
    x_bins = np.linspace(0, field_size[0], int(field_size[0] / bin_size[0]) + 1)
    y_bins = np.linspace(0, field_size[1], int(field_size[1] / bin_size[1]) + 1)

    hist, x_edges, y_edges = np.histogram2d(ball_xy[:, 1], ball_xy[:, 2], bins=[x_bins, y_bins])

    # Find the bin for each data point
    x_indices = np.digitize(ball_xy[:, 1], x_edges) - 1
    y_indices = np.digitize(ball_xy[:, 2], y_edges) - 1

    # Bound indices to valid range
    x_indices = np.clip(x_indices, 0, len(x_edges) - 2)
    y_indices = np.clip(y_indices, 0, len(y_edges) - 2)

    # Create masks for primary and extended regions
    primary_mask = np.zeros(len(ball_xy), dtype=bool)
    extended_mask = np.zeros(len(ball_xy), dtype=bool)

    # Check density for each point and its neighborhood
    for i in range(len(ball_xy)):
        xi, yi = x_indices[i], y_indices[i]

        # Check if this point is in a high-density bin
        if hist[xi, yi] >= primary_threshold:
            primary_mask[i] = True
            continue

        # Check neighborhood for secondary threshold
        x_min, x_max = max(0, xi - neighbor_radius), min(hist.shape[0] - 1, xi + neighbor_radius)
        y_min, y_max = max(0, yi - neighbor_radius), min(hist.shape[1] - 1, yi + neighbor_radius)

        neighborhood = hist[x_min : x_max + 1, y_min : y_max + 1]
        if np.any(neighborhood >= primary_threshold) and hist[xi, yi] >= secondary_threshold:
            extended_mask[i] = True

    # Combine masks for complete filtering
    remove_mask = primary_mask | extended_mask

    # # Visualize before and after
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.scatter(ball_xy[:, 1], ball_xy[:, 2], c=ball_xy[:, 0], s=2, cmap='viridis')
    # plt.scatter(ball_xy[remove_mask, 1], ball_xy[remove_mask, 2],
    #            c='red', s=10, alpha=0.5, marker='x')
    # plt.title(f'Original with Removed Points ({np.sum(remove_mask)} points)')
    # plt.axis('equal')

    # plt.subplot(1, 2, 2)
    # keep_mask = ~remove_mask
    # plt.scatter(ball_xy[keep_mask, 1], ball_xy[keep_mask, 2],
    #            c=ball_xy[keep_mask, 0], s=2, cmap='viridis')
    # plt.title(f'Filtered ({np.sum(keep_mask)} points)')
    # plt.axis('equal')

    # plt.tight_layout()
    # plt.show()

    return ball_xy[~remove_mask]


def combine_close_detections(ball_xy, distance_threshold=70):
    """
    Combine multiple close ball detections in the same frame into a single detection.

    Parameters:
    ball_xy : np.ndarray
        Array with shape (n, 3) where each row is [frame_idx, x, y]
    distance_threshold : float
        Maximum distance between detections to consider them for combination

    Return:
    combined_ball_xy : np.ndarray
        Ball tracking data with close detections combined
    """

    combined_indices = []
    combined_positions = []

    i = 0
    while i < len(ball_xy):
        current_frame = ball_xy[i, 0]
        current_pos = ball_xy[i, 1:3]

        # Find all detections in the same frame
        same_frame_indices = np.where(ball_xy[:, 0] == current_frame)[0]
        same_frame_positions = ball_xy[same_frame_indices, 1:3]

        # Calculate distances from the current position
        distances = np.linalg.norm(same_frame_positions - current_pos, axis=1)

        # Find indices of detections within the distance threshold
        close_indices = same_frame_indices[distances < distance_threshold]

        if len(close_indices) > 1:
            # Combine close detections by weighted average (using inverse distance as weights)
            close_positions = ball_xy[close_indices, 1:3]
            close_distances = distances[distances < distance_threshold]
            weights = 1 / (close_distances + 1e-6)  # Avoid division by zero
            weights /= weights.sum()  # Normalize weights

            combined_pos = np.sum(close_positions * weights[:, np.newaxis], axis=0)
            combined_indices.append(close_indices[0])  # Keep the first index for reference
            combined_positions.append(combined_pos)

            # Skip all close detections in the next iteration
            i += len(close_indices)
        else:
            combined_indices.append(i)
            combined_positions.append(current_pos)
            i += 1

    combined_ball_xy = np.array(
        [
            [ball_xy[idx, 0], pos[0], pos[1]]
            for idx, pos in zip(combined_indices, combined_positions)
        ]
    )

    return combined_ball_xy


def filter_static_detections(
    json_path,
    save_path,
    field_size=(1060, 660),  # Standard football field size in 0.1m units
    bin_size=(5, 5),  # smaller bins for finer resolution
    primary_threshold=1000,
    secondary_threshold=500,
    neighbor_radius=10,
):
    """
    Comprehensive filtering of static ball detections.

    Parameters:
    json_path : str
        Path to the input JSONL file containing ball tracking data
    save_path : str
        Path to save the filtered ball tracking data
    field_size : tuple
        Size of the football field (width, height)
    bin_size : tuple
        Size of histogram bins (x, y)
    primary_threshold : int
        Primary threshold for high-density regions
    secondary_threshold : int
        Secondary threshold for neighboring regions
    neighbor_radius : int
        Radius for considering neighboring bins
    max_speed : float
        Maximum plausible speed of the ball (pixels/frame)
    static_threshold : float
        Minimum movement required to not be considered static
    window_size : int
        Look back this many frames to check for static patterns
    """
    ball_xy = convert_ball_tracking_json_to_numpy(json_path)

    # sort ball_xy
    ball_xy = ball_xy[np.argsort(ball_xy[:, 0])]

    # Filter out elements with x < 0 or x > field_size[0]
    ball_xy = ball_xy[(ball_xy[:, 1] >= 0) & (ball_xy[:, 1] <= field_size[0])]
    # Filter out elements with y < 0 or y > field_size[1]
    ball_xy = ball_xy[(ball_xy[:, 2] >= 0) & (ball_xy[:, 2] <= field_size[1])]

    filtered_ball_xy = remove_detections_near_high_density_region(
        ball_xy,
        field_size=field_size,
        bin_size=bin_size,
        primary_threshold=primary_threshold,
        secondary_threshold=secondary_threshold,
        neighbor_radius=neighbor_radius,
    )

    # Combine close detections first
    combined_ball_xy = combine_close_detections(filtered_ball_xy, distance_threshold=70)

    # Remove detections near high density region again, but with lower thresholds and fewer data each time
    filtered_ball_xy_second_pass = []

    # Separate combined_ball_xy into 10 equal segments
    segment_total = 10
    total_points = len(combined_ball_xy)
    segment_size = total_points // segment_total

    # Process each segment individually (remove static clusters)
    for i in range(segment_total):
        # Calculate segment indices (last segment gets any remainder)
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < segment_total - 1 else total_points

        # Extract segment
        segment = combined_ball_xy[start_idx:end_idx]

        if len(segment) == 0:
            continue

        # Calculate adjusted thresholds - gradually decrease for each segment
        segment_primary_threshold = primary_threshold * 0.5
        segment_secondary_threshold = secondary_threshold * 0.5

        # Process this segment with lower thresholds
        filtered_segment = remove_detections_near_high_density_region(
            segment,
            field_size=field_size,
            bin_size=bin_size,
            primary_threshold=int(segment_primary_threshold),
            secondary_threshold=int(segment_secondary_threshold),
            neighbor_radius=neighbor_radius,
        )

        # Append filtered segment to the result
        filtered_ball_xy_second_pass.append(filtered_segment)

    # Concatenate all filtered segments
    if filtered_ball_xy_second_pass:
        filtered_ball_xy_second_pass = np.vstack(filtered_ball_xy_second_pass)
    else:
        filtered_ball_xy_second_pass = combined_ball_xy.copy()  # Fallback if filtering failed

    convert_ball_tracking_numpy_to_json(filtered_ball_xy_second_pass, save_path)
    print(f"Done filtering static balls.")


def separate_ball_tracks(ball_xy, max_separate_distance=70, min_track_length=5):
    """
    Separate mixed detections into distinct tracks based on spatial proximity.

    Parameters:
    -----------
    ball_xy : np.ndarray
        Array with shape (n, 3) where each row is [frame_idx, x, y]
    max_separate_distance : float
        Maximum distance between consecutive points to be considered part of the same track
    min_track_length : int
        Minimum number of detections required for a valid track

    Returns:
    --------
    list of np.ndarray
        List where each element is a separate track as a numpy array with [frame_idx, x, y]
    """
    # Sort by frame number
    sorted_data = ball_xy[ball_xy[:, 0].argsort()]

    # Group detections by frame
    frame_to_detections = {}
    for detection in sorted_data:
        frame = int(detection[0])
        if frame not in frame_to_detections:
            frame_to_detections[frame] = []
        frame_to_detections[frame].append(detection)

    # Initialize tracks
    tracks = []
    active_tracks = []  # (last_frame, last_x, last_y, track_points)

    # Process each frame in order
    for frame in sorted(frame_to_detections.keys()):
        detections = frame_to_detections[frame]

        # For each detection in this frame, find the best matching active track
        unmatched_detections = []
        matched_track_indices = set()

        for detection in detections:
            x, y = detection[1], detection[2]
            best_track_idx = None
            best_distance = float("inf")

            # Find closest active track
            for i, (last_frame, last_x, last_y, _) in enumerate(active_tracks):
                if i in matched_track_indices:  # Skip already matched tracks
                    continue

                # Calculate distance to last point in track
                distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

                # If distance is acceptable and better than current best
                if distance < max_separate_distance and distance < best_distance:
                    best_distance = distance
                    best_track_idx = i

            if best_track_idx is not None:
                # Add to existing track
                active_tracks[best_track_idx][0] = frame
                active_tracks[best_track_idx][1] = x
                active_tracks[best_track_idx][2] = y
                active_tracks[best_track_idx][3].append(detection)
                matched_track_indices.add(best_track_idx)
            else:
                # No matching track found
                unmatched_detections.append(detection)

        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            x, y = detection[1], detection[2]
            active_tracks.append([frame, x, y, [detection]])

        # Check for inactive tracks (not updated in recent frames)
        still_active = []
        for track_data in active_tracks:
            last_frame, _, _, track_points = track_data
            if frame - last_frame > 5:  # Track is stale after 50 frames without updates
                if len(track_points) >= min_track_length:
                    tracks.append(np.array(track_points))
            else:
                still_active.append(track_data)

        active_tracks = still_active

    # Add remaining active tracks
    for _, _, _, track_points in active_tracks:
        if len(track_points) >= min_track_length:
            tracks.append(np.array(track_points))

    # Sort tracks by length (descending)
    tracks.sort(key=lambda x: len(x), reverse=True)

    return tracks


# Helper functions for track connection
def track_to_dataframe(track: Dict) -> pd.DataFrame:
    """Convert a track to a pandas DataFrame with frame and position."""
    frames = track[:, 0].astype(int)
    positions = track[:, 1:3]

    df = pd.DataFrame(
        {
            "frame": frames,
            "x": [pos[0] for pos in positions],
            "y": [pos[1] for pos in positions],
        }
    )
    return df


# Helper functions for track connection
def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate velocity (both magnitude and direction) between consecutive frames."""
    # Sort by frame to ensure correct velocity calculation
    df = df.sort_values("frame")

    # Calculate delta x, y, and frame
    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dt"] = df["frame"].diff()

    # Calculate velocity components (pixels per frame)
    df["vx"] = df["dx"] / df["dt"].replace(0, np.nan)  # Avoid division by zero
    df["vy"] = df["dy"] / df["dt"].replace(0, np.nan)

    # Calculate velocity magnitude
    df["v_mag"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)

    # Calculate velocity direction in radians
    df["v_dir"] = np.arctan2(df["vy"], df["vx"])

    return df


# Helper functions for track connection
def is_ball_tracks_connectable(
    track1: pd.DataFrame,
    track2: pd.DataFrame,
    max_frame_gap: int = 120,
    max_connect_distance: float = 100.0,
    max_velocity_change: float = 50.0,
    max_direction_change: float = np.pi / 2,
) -> bool:
    """
    Determine if two tracks can be connected based on various criteria.

    Parameters:
    - track1, track2: DataFrames containing the tracks
    - max_frame_gap: Maximum allowed gap in frames between tracks
    - max_connect_distance: Maximum spatial distance allowed between end of track1 and start of track2
    - max_velocity_change: Maximum allowed change in velocity magnitude
    - max_direction_change: Maximum allowed change in velocity direction (in radians)

    Returns:
    - Boolean indicating if the tracks can be connected
    """
    # Check if the tracks are sequential in time
    track1_end_frame = track1["frame"].max()
    track2_start_frame = track2["frame"].min()

    if not (track1_end_frame < track2_start_frame):
        return False  # Track2 should start after track1 ends

    frame_gap = track2_start_frame - track1_end_frame
    if frame_gap > max_frame_gap:
        return False  # Gap is too large

    # Get the last position of track1 and first position of track2
    last_pos1 = track1.loc[track1["frame"] == track1_end_frame, ["x", "y"]].values[0]
    first_pos2 = track2.loc[track2["frame"] == track2_start_frame, ["x", "y"]].values[0]

    # Calculate distance
    dist = np.sqrt(((last_pos1[0] - first_pos2[0]) ** 2) + ((last_pos1[1] - first_pos2[1]) ** 2))
    if dist > max_connect_distance:
        return False  # Distance is too large

    # Get average velocity at the end of track1 (if available)
    last_few_frames1 = track1.nlargest(3, "frame")
    if len(last_few_frames1) >= 2:
        last_vel1_mag = last_few_frames1["v_mag"].mean()
        last_vel1_dir = last_few_frames1["v_dir"].mean()
    else:
        return True  # Not enough frames to calculate velocity, default to connectable

    # Get average velocity at the beginning of track2 (if available)
    first_few_frames2 = track2.nsmallest(3, "frame")
    if len(first_few_frames2) >= 2:
        first_vel2_mag = first_few_frames2["v_mag"].mean()
        first_vel2_dir = first_few_frames2["v_dir"].mean()
    else:
        return True  # Not enough frames to calculate velocity, default to connectable

    # Check velocity magnitude change
    vel_mag_change = abs(last_vel1_mag - first_vel2_mag)
    if vel_mag_change > max_velocity_change:
        return False  # Velocity magnitude change is too large

    # Check velocity direction change (consider angle wrapping)
    dir_change = abs(last_vel1_dir - first_vel2_dir)
    dir_change = min(dir_change, 2 * np.pi - dir_change)  # Handle angle wrapping
    if dir_change > max_direction_change:
        return False  # Velocity direction change is too large

    # If we've passed all checks, the tracks can be connected
    return True


def merge_ball_tracks(track1: pd.DataFrame, track2: pd.DataFrame) -> pd.DataFrame:
    """Merge two tracks into a single track."""
    # Combine the frames and sort by frame number
    merged_track = pd.concat([track1, track2])
    merged_track = merged_track.sort_values("frame").reset_index(drop=True)

    # Recalculate velocities for the merged track
    merged_track = calculate_velocity(merged_track)

    return merged_track


def connect_ball_tracks(
    tracks: List[Dict],
    max_frame_gap: int = 30,
    max_connect_distance: float = 100.0,
    max_velocity_change: float = 50.0,
    max_direction_change: float = np.pi / 2,
) -> List[pd.DataFrame]:
    """
    Connect tracks that are likely from the same trajectory.

    Parameters:
    - tracks: List of track dictionaries
    - max_frame_gap: Maximum allowed gap in frames between tracks
    - max_connect_distance: Maximum spatial distance allowed between end of one track and start of another
    - max_velocity_change: Maximum allowed change in velocity magnitude
    - max_direction_change: Maximum allowed change in velocity direction (in radians)

    Returns:
    - List of merged track DataFrames
    """
    # Convert tracks to DataFrames and calculate velocities
    track_dfs = []
    for track in tracks:
        df = track_to_dataframe(track)
        df = calculate_velocity(df)
        track_dfs.append(df)

    # Sort tracks by their starting frame
    track_dfs.sort(key=lambda df: df["frame"].min())

    # Initialize with the first track
    merged_tracks = [track_dfs[0]]

    # Try to connect each subsequent track
    for i in range(1, len(track_dfs)):
        current_track = track_dfs[i]
        connected = False

        # Try to connect with any of the existing merged tracks
        for j in range(len(merged_tracks)):
            if is_ball_tracks_connectable(
                merged_tracks[j],
                current_track,
                max_frame_gap,
                max_connect_distance,
                max_velocity_change,
                max_direction_change,
            ):
                merged_tracks[j] = merge_ball_tracks(merged_tracks[j], current_track)
                connected = True
                break

        # If couldn't connect, add as a new track
        if not connected:
            merged_tracks.append(current_track)

    return merged_tracks


def reconstruct_ball_track_from_detections(
    json_path,
    save_path,
    max_separate_distance=70,
    min_track_length=5,
    max_frame_gap=120,
    max_connect_distance=100.0,
    max_velocity_change=50.0,
    max_direction_change=np.pi / 2,
):
    """
    Reconstruct ball track from separated detections.

    Original format: {frame_id, projected: [x, y]} (by frame)
    After processing: {track_id, team, frame_range, frames, projected} (by track)

    Parameters:
    -----------
    json_path : str
        Path to the input JSONL file containing ball tracking data.
    save_path : str
        Path to save the reconstructed ball track JSONL file.
    max_distance : float
        Maximum distance between consecutive detections to be considered part of the same track.
    min_track_length : int
        Minimum number of frames for a track to be considered valid.
    Returns:
    --------
    None
    """
    ball_xy = convert_ball_tracking_json_to_numpy(json_path)

    # Separate into distinct tracks
    tracks = separate_ball_tracks(ball_xy, max_separate_distance, min_track_length)

    if len(tracks) == 0:
        raise ValueError("No valid ball tracks found.")
    tracks.sort(key=lambda x: x[0, 0])

    print(f"✅ Constructed {len(tracks)} separate ball tracks.")

    # Connect tracks based on motion continuity
    merged_tracks = connect_ball_tracks(
        tracks,
        max_frame_gap=max_frame_gap,
        max_connect_distance=max_connect_distance,
        max_velocity_change=max_velocity_change,
        max_direction_change=max_direction_change,
    )
    print(f"✅ Merged {len(tracks)} ball tracks into {len(merged_tracks)} ball tracks.")

    with open(save_path, "w") as f:
        for i, track in enumerate(merged_tracks):
            # Convert numpy int64 to standard Python int for JSON serialization
            track_dict = {
                "track_id": int(i),
                "team": "ball",
                "frame_range": [int(track["frame"].min()), int(track["frame"].max())],
                "frames": [int(x) for x in track["frame"].tolist()],
                "projected": [
                    (float(x), float(y)) for x, y in zip(track["x"].tolist(), track["y"].tolist())
                ],
            }
            f.write(json.dumps(track_dict) + "\n")


def remove_duplicate_detections(input_file, output_file):
    """
    Process ball tracking data to remove duplicate detections in frames.
    Keeps detections from longer tracks when duplicates are found.

    Args:
        input_file: Path to the input JSONL file
        output_file: Path to save the output JSONL file
    """
    # Read all tracks from the input file
    tracks = []
    with open(input_file, "r") as f:
        for line in f:
            track = json.loads(line)
            tracks.append(track)

    # Calculate the actual length of each track (number of frames)
    track_lengths = {track["track_id"]: len(track["frames"]) for track in tracks}

    # Map frames to their detections (track_id, track_length)
    frames_to_detections = defaultdict(list)
    for track in tracks:
        track_id = track["track_id"]
        for frame in track["frames"]:
            frames_to_detections[frame].append((track_id, track_lengths[track_id]))

    # Find frames with multiple detections
    duplicate_frames = {
        frame: detections
        for frame, detections in frames_to_detections.items()
        if len(detections) > 1
    }

    print(f"🔍 Found {len(duplicate_frames)} frames with duplicate detections.")

    # For each duplicate frame, determine which track's detection to keep
    # (the one from the longest track)
    frames_to_remove = defaultdict(set)
    for frame, detections in duplicate_frames.items():
        # Sort detections by track length in descending order
        sorted_detections = sorted(detections, key=lambda x: x[1], reverse=True)
        # Keep the detection from the longest track
        track_to_keep = sorted_detections[0][0]

        # Mark all other detections for removal
        for track_id, _ in sorted_detections[1:]:
            frames_to_remove[track_id].add(frame)

    # Create new tracks with duplicates removed
    processed_tracks = []
    for track in tracks:
        track_id = track["track_id"]
        # Remove the frames marked for removal for this track
        frames_to_remove_for_track = frames_to_remove.get(track_id, set())
        new_frames = [frame for frame in track["frames"] if frame not in frames_to_remove_for_track]

        if new_frames:  # Only keep tracks with remaining frames
            # Create a new track with updated frames and projected points
            new_track = {
                "track_id": track["track_id"],
                "team": track["team"],
                "frame_range": [new_frames[0], new_frames[-1]],
                "frames": new_frames,
                "projected": [],
            }

            # Keep only the projected points for the remaining frames
            for i, frame in enumerate(track["frames"]):
                if frame in new_frames:
                    new_track["projected"].append(track["projected"][i])

            processed_tracks.append(new_track)

    # Write the processed tracks to the output file
    with open(output_file, "w") as f:
        for track in processed_tracks:
            f.write(json.dumps(track) + "\n")

    # Return summary statistics
    total_duplicates = sum(len(frames) for frames in frames_to_remove.values())

    print(f"✅ Removed {total_duplicates} duplicate detections")


def remove_static_ball_tracks(
    jsonl_path,
    output_jsonl_path,
    movement_threshold=20,  # in meters (10 = 1m if 0.1m units)
):
    """
    Remove ball tracks that don't move significantly.

    Args:
        jsonl_path (str): Input path to .jsonl file.
        output_jsonl_path (str): Output path to write filtered tracks.
        movement_threshold (float): Minimum total movement (Euclidean) to keep.
    """
    with open(jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
        for line in f_in:
            track = json.loads(line)
            if track.get("team") != "ball":
                f_out.write(json.dumps(track) + "\n")
                continue

            points = np.array(track.get("projected", []))
            if len(points) < 2:
                continue  # skip too short

            # Compute total movement
            deltas = np.diff(points, axis=0)
            distances = np.linalg.norm(deltas, axis=1)
            total_distance = distances.sum()

            if total_distance >= movement_threshold:
                f_out.write(json.dumps(track) + "\n")


def fuse_overlapping_tracks(jsonl_path, output_jsonl_path):
    """
    Fuse overlapping ball tracks into single tracks.

    Args:
        jsonl_path (str): Input path to .jsonl file.
        output_jsonl_path (str): Output path to write fused tracks.
    """
    tracks = []
    fuse_count = 0
    with open(jsonl_path, "r") as f_in, open(output_jsonl_path, "w") as f_out:
        for line in f_in:
            track = json.loads(line)
            # if nothing in tracks, just append
            if not tracks:
                tracks.append(track)
                continue

            # Check for overlap with the last track
            last_track = tracks[-1]
            if track["frame_range"][0] <= last_track["frame_range"][1]:
                # Merge tracks
                last_track["frame_range"][1] = max(
                    last_track["frame_range"][1], track["frame_range"][1]
                )
                # Combine frames and projected points in ascending order of frames
                combined_frames = np.concatenate(
                    [np.array(last_track["frames"]), np.array(track["frames"])]
                )
                combined_projected = np.concatenate(
                    [np.array(last_track["projected"]), np.array(track["projected"])]
                )
                sort_indices = np.argsort(combined_frames)
                combined_frames = combined_frames[sort_indices].tolist()
                combined_projected = combined_projected[sort_indices].tolist()
                last_track["frames"] = combined_frames
                last_track["projected"] = combined_projected
                # print(f"Merged track {track['track_id']} into track {last_track['track_id']}")
                fuse_count += 1
            else:
                saved_track = tracks.pop()
                f_out.write(json.dumps(saved_track) + "\n")
                tracks.append(track)
        # Write the last track
        if tracks:
            f_out.write(json.dumps(tracks[-1]) + "\n")
    print(f"Fused {fuse_count} overlapping tracks.")


def merge_tracks_with_larger_gap(
    json_path,
    output_path,
    max_frame_gap=500,
    max_connect_distance=150.0,
    max_velocity_change=60.0,
    max_direction_change=np.pi / 1.5,
):
    """
    Directly merge existing tracks with larger gap tolerance.

    Parameters:
    -----------
    json_path : str
        Path to input tracks (after fuse_overlapping_tracks)
    output_path : str
        Path to save merged tracks
    max_frame_gap : int
        Maximum frame gap to consider for merging
    max_connect_distance : float
        Maximum spatial distance
    max_velocity_change : float
        Maximum velocity change
    max_direction_change : float
        Maximum direction change (radians)
    """
    # Load all tracks
    tracks = []
    with open(json_path, "r") as f:
        for line in f:
            track = json.loads(line)
            if track.get("team") == "ball":
                # Convert to numpy array format for connect_ball_tracks
                frames = np.array(track["frames"])
                positions = np.array(track["projected"])
                track_array = np.column_stack([frames, positions])
                tracks.append(track_array)

    print(f"📊 Loaded {len(tracks)} tracks for second-pass merging")

    # Use existing connect_ball_tracks function
    merged_tracks = connect_ball_tracks(
        tracks,
        max_frame_gap=max_frame_gap,
        max_connect_distance=max_connect_distance,
        max_velocity_change=max_velocity_change,
        max_direction_change=max_direction_change,
    )

    print(f"✅ Merged into {len(merged_tracks)} tracks")

    # Save merged tracks
    with open(output_path, "w") as f:
        for i, track in enumerate(merged_tracks):
            track_dict = {
                "track_id": int(i),
                "team": "ball",
                "frame_range": [int(track["frame"].min()), int(track["frame"].max())],
                "frames": [int(x) for x in track["frame"].tolist()],
                "projected": [
                    (float(x), float(y)) for x, y in zip(track["x"].tolist(), track["y"].tolist())
                ],
            }
            f.write(json.dumps(track_dict) + "\n")


def remove_ball_false_detection(json_path, save_path, field_size=[1060, 660]):
    """
    Remove false ball detections using Agglomerative Clustering.

    Parameters:
    -----------
    json_path : str
        Path to the input JSONL file containing ball tracking data.
    save_path : str
        Path to save the filtered JSONL file.
    field_size : list
        Size of the football field [width, height]

    Returns:
    --------
    None
    """
    ball_xy = convert_ball_tracking_json_to_numpy(json_path)

    fs = 30  # frame rate
    Xraw = ball_xy.copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(Xraw)
    X[:, 0] *= 6  # weight of temporal dimension, tune this if needed
    tree = KDTree(X)
    dist, ind = tree.query(X, k=max(fs // 3, 10))
    thresh = 2 * np.quantile(dist[:, -1], 0.5)

    knn_graph = kneighbors_graph(X[:, 0].reshape(-1, 1), 5, include_self=False)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        linkage="single",
        connectivity=knn_graph,
        distance_threshold=thresh,
    )
    clustering.fit(X)

    labs, n_in_labs = np.unique(clustering.labels_, return_counts=True)
    outs = np.array([])
    for lab, n_in in zip(labs, n_in_labs):
        if n_in < 3:
            outs = np.append(outs, np.where(clustering.labels_ == lab)[0])
    outs = np.sort(outs).astype(int)

    # plt.scatter(ball_xy[:,1], ball_xy[:,2], s=0.5)
    # plt.plot(ball_xy[outs,1], ball_xy[outs,2],'r.')
    # plt.legend(['detections','outliers'])
    # plt.axis('equal')
    # plt.axis('off')
    # plt.show()

    ball_xy_val = np.delete(ball_xy, outs, axis=0)

    convert_ball_tracking_numpy_to_json(ball_xy_val, save_path)
    print(f"Done removing false detections.")


def remove_static_clusters(
    ball_xy, time_window=30, spatial_threshold=10, min_points=5, max_displacement=20
):
    """
    Remove static clusters from ball tracking data.

    Parameters:
    -----------
    filtered_ball_xy : np.ndarray
        Array with shape (n, 3) where each row is [frame_idx, x, y]
    time_window : int
        Time window (in frames) to consider for static cluster detection
    spatial_threshold : float
        Maximum radius for points to be considered in the same spatial cluster
    min_points : int
        Minimum number of points required to form a static cluster
    max_displacement : float
        Maximum allowed displacement within a cluster to be considered "static"

    Returns:
    --------
    np.ndarray
        Filtered filtered_ball_xy array with static clusters removed
    """
    import numpy as np
    from sklearn.cluster import DBSCAN

    # Create a copy to avoid modifying the original
    filtered_ball_xy = ball_xy.copy()

    # Create a mask for points to keep
    keep_mask = np.ones(len(filtered_ball_xy), dtype=bool)

    # Get unique frame numbers
    frame_nums = np.unique(filtered_ball_xy[:, 0])

    # Process each time window
    for start_frame in range(0, int(frame_nums[-1]), time_window // 2):  # Overlapping windows
        end_frame = start_frame + time_window

        # Get points in this time window
        window_mask = (filtered_ball_xy[:, 0] >= start_frame) & (filtered_ball_xy[:, 0] < end_frame)
        window_points = filtered_ball_xy[window_mask]

        if len(window_points) < min_points:
            continue

        # Cluster based on spatial coordinates only
        spatial_clustering = DBSCAN(eps=spatial_threshold, min_samples=min_points).fit(
            window_points[:, 1:3]
        )

        # For each spatial cluster, check if it's static
        for cluster_id in np.unique(spatial_clustering.labels_):
            if cluster_id == -1:  # Skip noise points
                continue

            # Get points in this spatial cluster
            cluster_mask = spatial_clustering.labels_ == cluster_id
            cluster_points = window_points[cluster_mask]

            # Check if this cluster is static (small displacement over time)
            if len(cluster_points) >= min_points:
                # Calculate the maximum displacement within the cluster
                x_range = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
                y_range = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
                displacement = np.sqrt(x_range**2 + y_range**2)

                # Calculate frame span
                frame_span = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])

                # If displacement is small and cluster spans significant time, it's likely static
                if displacement < max_displacement and frame_span > min_points:
                    # Find these points in the original array
                    for point in cluster_points:
                        # Find matching points in the original array
                        matches = np.where(
                            (filtered_ball_xy[:, 0] == point[0])
                            & (filtered_ball_xy[:, 1] == point[1])
                            & (filtered_ball_xy[:, 2] == point[2])
                        )[0]
                        keep_mask[matches] = False

    # Return the filtered data
    return filtered_ball_xy[keep_mask]


def smoothen_fused_ball_tracking(json_path, save_path):
    """
    Enhanced version that uses Kalman-filtered acceleration instead of raw numerical derivatives.
    Recommended if you have noisy data.

    Use RANSAC to find linear segments and smooth ball tracking data.
    Then use find_peaks on acceleration to identify key points for interpolation.

    Processes each track independently from the input JSONL file.

    Args:
        json_path: Path to input JSONL file (should be after Kalman smoothing)
        save_path: Path to save final interpolated tracks
    """
    processed_tracks = []

    print(f"📊 Processing tracks from {json_path}")

    with open(json_path, "r") as f_in:
        line_count = sum(1 for _ in f_in)
        f_in.seek(0)

        for line_idx, line in tqdm(enumerate(f_in), total=line_count):
            if not line.strip():
                continue

            track = json.loads(line)

            if track.get("team") != "ball":
                continue

            frames = np.array(track["frames"])
            positions = np.array(track["projected"])

            if len(frames) < 10:
                # print(f"  ⚠️ Skipping track {track.get('track_id', line_idx)}: too short")
                continue

            # print(f"\n  Processing track {track.get('track_id', line_idx)}: {len(frames)} frames")

            # ===== STEP 1: Apply Kalman filter to get smooth acceleration =====
            # print("    🔧 Applying Kalman filter for smooth derivatives...")
            smoother = BallKalmanSmoother(measurement_noise=10.0, process_noise=10.0)

            try:
                smoothed_positions, velocities, accelerations = smoother.filter_track(
                    frames=frames, measurements=positions, max_iterations=5, verbose=False
                )

                # Use Kalman acceleration magnitude
                acc_magnitude = np.linalg.norm(accelerations, axis=1)
                # print(f"    ✓ Kalman filtering completed")

            except Exception as e:
                # print(f"    ⚠️ Kalman filtering failed: {e}, using numerical derivatives")
                # Fallback to numerical derivatives
                dt = np.diff(frames)
                dt = np.where(dt == 0, 1, dt)

                dx = np.diff(positions[:, 0])
                dy = np.diff(positions[:, 1])
                vx = dx / dt
                vy = dy / dt

                dvx = np.diff(vx)
                dvy = np.diff(vy)
                dt_acc = dt[:-1]
                dt_acc = np.where(dt_acc == 0, 1, dt_acc)

                ax = dvx / dt_acc
                ay = dvy / dt_acc
                acc_magnitude = np.sqrt(ax**2 + ay**2)
                acc_magnitude = np.concatenate(
                    [[acc_magnitude[0]], acc_magnitude, [acc_magnitude[-1]]]
                )

                smoothed_positions = positions  # Use original positions

            # ===== STEP 2: RANSAC segmentation =====
            # print("    🔍 Running RANSAC segmentation...")
            all_ins, ins = process_trajectory_in_chunks(
                smoothed_positions,
                frames,
                chunk_size=200,
                overlap=50,
                segment_threshold=8,
            )

            if not ins:
                debs = np.array([frames[0]])
                ends = np.array([frames[-1]])
            else:
                debs = np.sort(np.array([seg[0] for seg in ins]))
                ends = np.sort(np.array([seg[-1] for seg in ins]))

            # ===== STEP 3: Find peaks =====
            # print("    🔝 Finding acceleration peaks...")
            fs = 29.97  # frame rate
            peaks, _ = find_peaks(acc_magnitude, distance=int(0.8 * fs), prominence=1)
            peaks_frames = frames[peaks]

            peaks_in = []
            for end, deb in zip(ends[:-1], debs[1:]):
                in_between = np.logical_and(peaks_frames >= end + 8, peaks_frames <= deb - 8)
                peaks_in.append(peaks_frames[in_between])

            new_pts = np.concatenate(peaks_in) if peaks_in else np.array([])

            # ===== STEP 4: Interpolate =====
            vertices = np.concatenate([debs, ends, new_pts, [frames[0], frames[-1]]])
            vertices = np.sort(np.unique(vertices)).astype(int)
            vertices_indices = np.searchsorted(frames, vertices)
            vertices_indices = np.clip(vertices_indices, 0, len(frames) - 1)

            if len(vertices) < 2:
                interpolated_frames = frames
                interpolated_positions = smoothed_positions
            else:
                try:
                    f_interp = interpolate.interp1d(
                        vertices,
                        smoothed_positions[vertices_indices],
                        axis=0,
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    interpolated_frames = np.arange(frames[0], frames[-1] + 1)
                    interpolated_positions = f_interp(interpolated_frames)
                except:
                    interpolated_frames = frames
                    interpolated_positions = smoothed_positions

            # ===== STEP 5: Save =====
            processed_track = {
                "track_id": track.get("track_id", line_idx),
                "team": "ball",
                "frame_range": [int(interpolated_frames[0]), int(interpolated_frames[-1])],
                "frames": interpolated_frames.tolist(),
                "frame_range": [int(interpolated_frames[0]), int(interpolated_frames[-1])],
                "projected": interpolated_positions.tolist(),
            }
            processed_tracks.append(processed_track)

    with open(save_path, "w") as f_out:
        for track in processed_tracks:
            f_out.write(json.dumps(track) + "\n")

    print(f"\n💾 Saved {len(processed_tracks)} processed tracks to {save_path}")


class BallKalmanSmoother:
    """
    Custom 2D Kalman filter with RTS smoother for ball tracking.
    This smoother supports variable time steps and adaptive process noise.

    State vector: [x, vx, ax, jx, y, vy, ay, jy]
    - x, y: position
    - vx, vy: velocity
    - ax, ay: acceleration
    - jx, jy: jerk (acceleration derivative)
    """

    def __init__(
        self,
        measurement_noise: float = 10.0,
        process_noise: float = 10.0,
        initial_state_uncertainty: Optional[np.ndarray] = None,
    ):
        """
        Initialize the Kalman filter.

        Args:
            measurement_noise: Standard deviation of measurement noise (pixels)
            process_noise: Standard deviation of process noise
            initial_state_uncertainty: Custom P matrix (if None, uses defaults)
        """
        self.dim_x = 8  # State dimension
        self.dim_z = 2  # Measurement dimension

        # State transition matrix (continuous time)
        self.A = np.zeros((self.dim_x, self.dim_x))
        self.A[0, 1] = 1  # x' = vx
        self.A[1, 2] = 1  # vx' = ax
        self.A[2, 3] = 1  # ax' = jx
        self.A[4, 5] = 1  # y' = vy
        self.A[5, 6] = 1  # vy' = ay
        self.A[6, 7] = 1  # ay' = jy

        # Create Kalman filter
        self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)

        # Measurement matrix (observe position only)
        self.kf.H = np.zeros((self.dim_z, self.dim_x))
        self.kf.H[0, 0] = 1  # Measure x
        self.kf.H[1, 4] = 1  # Measure y

        # Initial state uncertainty
        if initial_state_uncertainty is not None:
            self.kf.P = initial_state_uncertainty
        else:
            self.kf.P = np.diag([100**2, 50**2, 20**2, 10**2, 100**2, 50**2, 20**2, 10**2])

        # Measurement noise
        self.kf.R = np.diag([measurement_noise**2, measurement_noise**2])

        # Process noise (base value, will be adapted)
        self.base_process_noise = process_noise

        # Storage for filtering/smoothing results
        self.filtered_means = []
        self.filtered_covs = []
        self.smoothed_means = None

    def initialize_state(self, first_measurement: np.ndarray):
        """Initialize filter state with first measurement."""
        self.kf.x = np.zeros((self.dim_x, 1))
        self.kf.x[0, 0] = first_measurement[0]  # x
        self.kf.x[4, 0] = first_measurement[1]  # y

    def forward_pass(
        self, measurements: np.ndarray, frame_diffs: np.ndarray, verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward Kalman filtering pass.

        Args:
            measurements: (N, 2) array of [x, y] measurements
            frame_diffs: (N,) array of time differences between frames
            verbose: Print progress

        Returns:
            filtered_means: (N, 8) array of filtered state estimates
            filtered_covs: (N, 8, 8) array of covariance matrices
        """
        if verbose:
            print("  Running forward Kalman filter...")

        self.filtered_means = []
        self.filtered_covs = []

        # Initialize with first measurement
        self.initialize_state(measurements[0])

        for i in range(len(measurements)):
            dt = frame_diffs[i]

            # Update state transition matrix for this time step
            self.kf.F = expm(self.A * dt)

            # Adaptive process noise (higher for larger dt)
            self.kf.Q = Q_discrete_white_noise(
                dim=4, dt=dt, var=self.base_process_noise**2, block_size=2
            )

            # Predict and update
            self.kf.predict()
            self.kf.update(measurements[i])

            # Store results
            self.filtered_means.append(self.kf.x.copy())
            self.filtered_covs.append(self.kf.P.copy())

        # Convert to arrays
        filtered_means = np.array(self.filtered_means).squeeze()
        filtered_covs = np.array(self.filtered_covs)

        return filtered_means, filtered_covs

    def backward_pass(
        self,
        filtered_means: np.ndarray,
        filtered_covs: np.ndarray,
        frame_diffs: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        RTS smoother backward pass.

        Args:
            filtered_means: Output from forward_pass
            filtered_covs: Output from forward_pass
            frame_diffs: Time differences between frames
            verbose: Print progress

        Returns:
            smoothed_means: (N, 8) array of smoothed state estimates
        """
        if verbose:
            print("  Running RTS smoother (backward pass)...")

        N = len(filtered_means)
        smoothed_means = np.zeros_like(filtered_means)
        smoothed_means[-1] = filtered_means[-1]

        for i in range(N - 2, -1, -1):
            dt = frame_diffs[i + 1]
            F = expm(self.A * dt)
            Q = Q_discrete_white_noise(dim=4, dt=dt, var=self.base_process_noise**2, block_size=2)

            # Predict forward
            x_pred = F @ filtered_means[i].reshape(-1, 1)
            P_pred = F @ filtered_covs[i] @ F.T + Q
            P_pred += 1e-6 * np.eye(self.dim_x)  # Regularization

            # Smoother gain
            K = filtered_covs[i] @ F.T @ np.linalg.pinv(P_pred)

            # Smoothed estimate
            smoothed_means[i] = (
                filtered_means[i].reshape(-1, 1)
                + K @ (smoothed_means[i + 1].reshape(-1, 1) - x_pred)
            ).flatten()

        self.smoothed_means = smoothed_means
        return smoothed_means

    def iterative_outlier_removal(
        self,
        measurements: np.ndarray,
        frame_diffs: np.ndarray,
        max_iterations: int = 10,
        convergence_tolerance: float = 0.05,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iteratively refilter data, replacing outliers with smoothed estimates.

        Args:
            measurements: (N, 2) array of [x, y] measurements
            frame_diffs: (N,) array of time differences
            max_iterations: Maximum refinement iterations
            convergence_tolerance: Stop when threshold change < this
            verbose: Print progress

        Returns:
            smoothed_trajectory: (N, 2) final smoothed [x, y] positions
            velocities: (N, 2) smoothed [vx, vy]
            accelerations: (N, 2) smoothed [ax, ay]
        """
        if verbose:
            # print("  Starting iterative outlier removal...")
            pass

        cleaned_measurements = measurements.copy()
        prev_thresholds = np.array([np.inf, np.inf])

        for iteration in range(max_iterations):
            # Re-run filter with cleaned data
            filtered_means, filtered_covs = self.forward_pass(
                cleaned_measurements, frame_diffs, verbose=False
            )
            smoothed_means = self.backward_pass(
                filtered_means, filtered_covs, frame_diffs, verbose=False
            )

            # Extract positions
            smoothed_positions = smoothed_means[:, [0, 4]]

            # Calculate outlier thresholds
            deltas = np.abs(smoothed_positions - cleaned_measurements)
            thresholds = 5 * np.median(deltas, axis=0)

            # Check convergence
            if iteration > 0:
                rel_change = np.abs(thresholds - prev_thresholds) / (prev_thresholds + 1e-6)
                max_change = np.max(rel_change)

                if verbose:
                    # print(f"    Iteration {iteration}: threshold change = {max_change:.4f}")
                    pass

                if max_change < convergence_tolerance:
                    if verbose:
                        # print(f"    ✓ Converged after {iteration + 1} iterations")
                        pass
                    break

            prev_thresholds = thresholds.copy()

            # Replace outliers with smoothed estimates
            for dim in range(2):
                outlier_mask = deltas[:, dim] >= thresholds[dim]
                cleaned_measurements[outlier_mask, dim] = smoothed_positions[outlier_mask, dim]

        if iteration == max_iterations - 1 and verbose:
            # print(f"    ⚠️ Reached max iterations ({max_iterations})")
            pass

        # Extract final results
        smoothed_trajectory = smoothed_means[:, [0, 4]]
        velocities = smoothed_means[:, [1, 5]]
        accelerations = smoothed_means[:, [2, 6]]

        return smoothed_trajectory, velocities, accelerations

    def filter_track(
        self,
        frames: np.ndarray,
        measurements: np.ndarray,
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete filtering pipeline for a single track.

        Args:
            frames: (N,) array of frame numbers
            measurements: (N, 2) array of [x, y] positions
            max_iterations: Max outlier removal iterations
            verbose: Print progress

        Returns:
            smoothed_positions: (N, 2) smoothed [x, y]
            velocities: (N, 2) estimated [vx, vy]
            accelerations: (N, 2) estimated [ax, ay]
        """
        # Calculate frame differences
        frame_diffs = np.diff(frames)
        frame_diffs = np.insert(frame_diffs, 0, 1)

        # Run complete pipeline
        smoothed_positions, velocities, accelerations = self.iterative_outlier_removal(
            measurements, frame_diffs, max_iterations=max_iterations, verbose=verbose
        )

        return smoothed_positions, velocities, accelerations


def apply_kalman_filtering_to_tracks(json_path, save_path):
    """
    Apply Kalman filtering with RTS smoother to ball tracks using BallKalmanSmoother class.

    Args:
        json_path: Path to input JSONL with ball tracks
        save_path: Path to save final processed tracks
        field_size: Field dimensions [width, height]
        detector_kwargs: Parameters for sharp change detection
    """

    # Load ball tracking data
    tracks = []
    with open(json_path, "r") as f_in:
        for line in f_in:
            track = json.loads(line)
            tracks.append(track)

    if not tracks:
        raise ValueError("No tracks found in input file")

    # Process each track
    all_processed_tracks = []

    for track_idx, track in enumerate(tracks):
        frames = np.array(track["frames"])
        positions = np.array(track["projected"])

        if len(frames) < 10:
            # all_processed_tracks.append(track)
            print(f"⚠️ Skipping track {track_idx}: too short ({len(frames)} frames)")
            continue

        # print(f"\n📊 Processing track {track_idx}: {len(frames)} frames [{frames[0]}-{frames[-1]}]")

        # ✅ Create Kalman smoother instance
        smoother = BallKalmanSmoother(
            measurement_noise=10.0,
            process_noise=10.0,
        )

        # ✅ Run filtering pipeline
        smoothed_positions, velocities, accelerations = smoother.filter_track(
            frames=frames, measurements=positions, max_iterations=10, verbose=True
        )

        # Create output track
        output_track = {
            "track_id": track["track_id"],
            "team": "ball",
            "frame_range": [int(frames[0]), int(frames[-1])],
            "frames": frames.tolist(),
            "projected": smoothed_positions.tolist(),
            # Optionally store velocities/accelerations
            # 'velocities': velocities.tolist(),
            # 'accelerations': accelerations.tolist(),
        }
        all_processed_tracks.append(output_track)

    # Save all processed tracks to output JSONL
    with open(save_path, "w") as f_out:
        for processed_track in all_processed_tracks:
            f_out.write(json.dumps(processed_track) + "\n")

    print(f"\n✅ Saved {len(all_processed_tracks)} processed tracks to {save_path}")


def process_trajectory_in_chunks(
    traj,
    frames,
    chunk_size=2000,
    overlap=200,
    segment_threshold=8,
):
    """
    Process a long trajectory by breaking it into overlapping chunks.

    Parameters:
    -----------
    traj : np.ndarray
        Array with shape (n, 2) containing x,y coordinates
    frames : np.ndarray
        Array with shape (n,) containing frame indices
    chunk_size : int
        Number of points to process in each chunk
    overlap : int
        Number of points to overlap between consecutive chunks
    segment_threshold : float
        Residual threshold for RANSAC

    Returns:
    --------
    np.ndarray
        Combined array of frame indices where line segments were detected
    """
    all_segments = []
    n_points = len(traj)

    # Process data in chunks with overlap
    for start_idx in range(0, n_points, chunk_size - overlap):
        end_idx = min(start_idx + chunk_size, n_points)

        # Extract chunk
        chunk_traj = traj[start_idx:end_idx]
        chunk_frames = frames[start_idx:end_idx]

        # print(f"Processing chunk {start_idx}-{end_idx} ({len(chunk_traj)} points)")

        # Skip if chunk is too small
        if len(chunk_traj) < 10:  # Minimum size for meaningful processing
            continue

        # Process this chunk using your existing algorithm
        chunk_segments = process_single_chunk(chunk_traj, chunk_frames, segment_threshold)

        if len(chunk_segments) > 0:
            # Map segment frames back to original frame indices if needed
            all_segments.extend(chunk_segments)

    # Combine and remove duplicates
    if all_segments:
        combined = np.concatenate(all_segments)
        combined = np.unique(combined)
        return combined, all_segments
    else:
        return np.array([]), []


def process_single_chunk(traj_chunk, frames_chunk, segment_threshold=8):
    """
    Process a single chunk to find line segments.
    This is your existing algorithm, adapted to work on a chunk.
    """
    chunk_segments = []

    traj_reg = traj_chunk.copy()
    reg_frames = frames_chunk.copy()
    i = 0

    while len(traj_reg) > 2:
        # Check if enough points remain
        if len(traj_reg) < 3:
            break

        # Try x -> y regression
        reg1 = RANSACRegressor(random_state=i, residual_threshold=segment_threshold)
        reg1.fit(traj_reg[:, 0].reshape(-1, 1), traj_reg[:, 1])
        frames_in_reg_1 = reg_frames[reg1.inlier_mask_]

        # Check if any inliers were found
        if len(frames_in_reg_1) == 0:
            break

        # Cluster inliers to find continuous segments
        clustering1 = AgglomerativeClustering(
            n_clusters=None, linkage="single", distance_threshold=15
        )
        clustering1.fit(frames_in_reg_1.reshape(-1, 1))
        labs1, n_in_labs1 = np.unique(clustering1.labels_, return_counts=True)

        # Try y -> x regression
        reg2 = RANSACRegressor(random_state=i, residual_threshold=segment_threshold)
        reg2.fit(traj_reg[:, 1].reshape(-1, 1), traj_reg[:, 0])
        frames_in_reg_2 = reg_frames[reg2.inlier_mask_]

        # Check if any inliers were found
        if len(frames_in_reg_2) == 0:
            break

        # Cluster these inliers
        clustering2 = AgglomerativeClustering(
            n_clusters=None, linkage="single", distance_threshold=15
        )
        clustering2.fit(frames_in_reg_2.reshape(-1, 1))
        labs2, n_in_labs2 = np.unique(clustering2.labels_, return_counts=True)

        # Find maximum segment length
        n_max = max(
            n_in_labs1.max() if len(n_in_labs1) > 0 else 0,
            n_in_labs2.max() if len(n_in_labs2) > 0 else 0,
        )

        # Check if any significant segment was found
        if n_max <= 25:
            break

        # Select the best segment
        if (
            len(n_in_labs1) > 0 and len(n_in_labs2) > 0 and n_in_labs1.max() > n_in_labs2.max()
        ) or len(n_in_labs2) == 0:
            new_seg = frames_in_reg_1[clustering1.labels_ == np.argmax(n_in_labs1)].astype(int)
        else:
            new_seg = frames_in_reg_2[clustering2.labels_ == np.argmax(n_in_labs2)].astype(int)

        # Add to results
        chunk_segments.append(new_seg)

        # Remove detected segment from the data
        indexes = np.searchsorted(reg_frames, new_seg)
        reg_frames = np.delete(reg_frames, indexes)
        traj_reg = np.delete(traj_reg, indexes, axis=0)
        i += 1

    return chunk_segments


def prepare_background_and_tracks(
    json_path,
    image_path,
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
    detector_kwargs=None,
):
    # Load and resize background
    bg_img = cv2.imread(image_path)
    if bg_img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    bg_img = cv2.resize(bg_img, field_size)

    filter_static_detections(
        json_path,
        json_path.replace(".jsonl", "_filtered.jsonl"),
        field_size=field_size,
        bin_size=(5, 5),  # smaller bins for finer resolution
        primary_threshold=1000,
        secondary_threshold=500,
        neighbor_radius=10,
    )

    reconstruct_ball_track_from_detections(
        json_path.replace(".jsonl", "_filtered.jsonl"),
        json_path.replace(".jsonl", "_merged.jsonl"),
        max_separate_distance=70,
        min_track_length=5,
        max_frame_gap=120,
        max_connect_distance=100.0,
        max_velocity_change=50.0,
        max_direction_change=np.pi / 2,
    )

    remove_static_ball_tracks(
        jsonl_path=json_path.replace(".jsonl", "_merged.jsonl"),
        output_jsonl_path=json_path.replace(".jsonl", "_merged_filtered.jsonl"),
        movement_threshold=30,  # unit in 0.1m
    )

    remove_duplicate_detections(
        json_path.replace(".jsonl", "_merged_filtered.jsonl"),
        json_path.replace(".jsonl", "_deduplicated.jsonl"),
    )

    fuse_overlapping_tracks(
        json_path.replace(".jsonl", "_deduplicated.jsonl"),
        json_path.replace(".jsonl", "_fused.jsonl"),
    )

    merge_tracks_with_larger_gap(
        json_path.replace(".jsonl", "_fused.jsonl"),
        json_path.replace(".jsonl", "_remerged.jsonl"),
        max_frame_gap=500,
        max_connect_distance=150.0,
        max_velocity_change=60.0,
        max_direction_change=np.pi / 1.5,
    )

    apply_kalman_filtering_to_tracks(
        json_path.replace(".jsonl", "_remerged.jsonl"),
        json_path.replace(".jsonl", "_smoothed.jsonl"),
    )

    # Sharp change detection and correction
    process_jsonl_detect_replace(
        json_path.replace(".jsonl", "_smoothed.jsonl"),
        json_path.replace(".jsonl", "_processed.jsonl"),
        detector_kwargs=detector_kwargs,
        overwrite_projected=True,
    )

    return bg_img


def process_merged_tracks(
    json_path,
    image_path,
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
    detector_kwargs=None,
):
    if output_name is None:
        output_name = os.path.splitext(os.path.basename(json_path))[0]

    # output_path_video = f"{output_name}.mp4"

    if detector_kwargs is None:
        detector_kwargs = dict(
            window_size=501,
            step=250,
            prominence=7,
            min_wave_len=10,
            max_wave_len=60,
            speed_std_factor=0.7,
            smooth_window=7,
            savgol_poly=2,
            min_steepness=0.2,
            min_quad_curv=0.7,
            min_monotonic_ratio=0.7,
            max_gap_size=5,
        )

    start = time.time()
    # Shared logic
    bg_img = prepare_background_and_tracks(
        json_path,
        image_path,
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
        detector_kwargs,
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
        required=False,
        help="Base name of the output file (without extension)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    BALL_DETECTOR = dict(
        window_size=501,
        step=250,
        prominence=7,
        min_wave_len=10,
        max_wave_len=60,
        speed_std_factor=0.7,
        smooth_window=7,
        savgol_poly=2,
        min_steepness=0.2,
        min_quad_curv=0.7,
        min_monotonic_ratio=0.7,
        max_gap_size=5,
    )

    process_merged_tracks(
        json_path=args.json_path,
        image_path=args.image_path,
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
        detector_kwargs=BALL_DETECTOR,
    )

    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()

# example usage:
# python3 post-processing-ball.py --json-path "./runs/detect/test_4k-2h-crop/team_tracking.jsonl" --image-path "./data/images/mongkok_football_field.png" --output-name './runs/detect/test_4k-2h-crop/team_tracking_output'
