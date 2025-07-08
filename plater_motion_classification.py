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

def detect_abnormal_tracks_from_jsonl(
    jsonl_path: str,
    angle_threshold: float = 120,
    velocity_threshold: float = 1e-2,
    min_valid_frames: int = 5,
    conf_threshold: float = 0.5,
    frame_threshold: int = 3,
    distance_threshold: float = 0.5
):
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f]

    # Split ball and player
    ball_tracks = [t for t in tracks if t["team"] == "ball"]
    player_tracks = [t for t in tracks if t["team"] != "ball"]

    # Build ball position lookup per frame
    ball_by_frame: Dict[int, Tuple[float, float]] = {}
    for b in ball_tracks:
        for f, pt in zip(b["frames"], b["projected"]):
            if pt is not None:
                ball_by_frame[f] = pt

    abnormal_tracks = []
    track_confidences = {}
    track_abnormal_frames = {}

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

            player_vec = np.array(pt_curr) - np.array(pt_prev)
            if np.linalg.norm(player_vec) < velocity_threshold:
                continue

            if f_prev not in ball_by_frame or f_curr not in ball_by_frame:
                continue

            ball_vec = np.array(ball_by_frame[f_curr]) - np.array(ball_by_frame[f_prev])
            if np.linalg.norm(ball_vec) < velocity_threshold:
                continue

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

    return abnormal_tracks, track_abnormal_frames, track_confidences

if __name__ == "__main__":
    # Load JSONL
    jsonl_path = "./runs/detect/test_4k-2h/team_tracking_relabeled.jsonl"
    angle_threshold = 120
    velocity_threshold = 1e-2
    min_valid_frames = 5
    conf_threshold = 0.7
    frame_threshold = 3
    distance_threshold = 20
    # Detect abnormal tracks
    abnormal_tracks, track_abnormal_frames, track_confidences = detect_abnormal_tracks_from_jsonl(
        jsonl_path,
        angle_threshold=angle_threshold,
        velocity_threshold=velocity_threshold,
        min_valid_frames=min_valid_frames,
        conf_threshold=conf_threshold,
        frame_threshold=frame_threshold,
        distance_threshold=distance_threshold
    )
    
    # Output
    print("Abnormal Tracks:", abnormal_tracks[:5])
    print("Track Abnormal Frames:", {k: track_abnormal_frames[k] for k in abnormal_tracks[:5]})
    print("Abnormal Track Scores:", {k: track_confidences[k] for k in abnormal_tracks[:5]})
    print("Total Abnormal Tracks:", len(abnormal_tracks))

    print("Track Count (analyzed):", len(track_confidences))