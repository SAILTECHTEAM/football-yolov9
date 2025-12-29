import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

from tqdm import tqdm
from player_motion_classification import get_frames_with_multiple_balls, detect_abnormal_tracks_from_jsonl, calculate_angle
from align_video_times import convert_game_frame_to_video_frame

def frame_to_time(f: int, fps=29.97, no_colon=False) -> str:
    seconds = f / fps
    m, s = divmod(int(seconds), 60)
    if no_colon:
        return f"{m:03}{s:02}"
    return f"{m:03}:{s:02}"

def bgr_to_rgb(bgr):
    return (bgr[2], bgr[1], bgr[0])  # reverse the tuple

def reverse_homography(point, homography_matrix):
    """
    Apply reverse homography to transform a point from 2D pitch coordinates back to video coordinates.
    
    Args:
        point: The 2D point [x, y] in pitch coordinates
        homography_matrix: The 3x3 homography matrix that maps from video to pitch
    
    Returns:
        np.array: The transformed point [x, y] in video coordinates, or None if transformation fails
    """
    try:
        # Invert the homography matrix
        inv_h = np.linalg.inv(homography_matrix)
        
        # Convert point to homogeneous coordinates
        p_homogeneous = np.array([point[0], point[1], 1.0])
        
        # Apply inverse transformation
        transformed = np.dot(inv_h, p_homogeneous)
        
        # Check if the homogeneous coordinate is too close to zero
        if abs(transformed[2]) < 1e-6:
            return None
        
        # Convert back to 2D coordinates
        transformed = transformed / transformed[2]
        
        return np.array([transformed[0], transformed[1]])
    except np.linalg.LinAlgError:
        return None


def is_point_in_frame(point, frame_width, frame_height, margin=0):
    """
    Check if a point is within the frame boundaries.
    
    Args:
        point: [x, y] coordinates
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        margin: Optional margin to consider point out of bounds (default: 0)
    
    Returns:
        bool: True if point is within bounds, False otherwise
    """
    if point is None:
        return False
    
    x, y = point
    return (margin <= x < frame_width - margin and 
            margin <= y < frame_height - margin)


def draw_triangle(img, center, size=10, color=(0, 0, 255), thickness=2):
    """
    Draw a triangle with its tip facing upward.
    
    Args:
        img: The image to draw on
        center: The center point [x, y] where the triangle should be placed
        size: Size of the triangle (height)
        color: BGR color tuple
        thickness: Line thickness (-1 for filled)
    """
    x, y = int(center[0]), int(center[1])
    
    # Define triangle points (tip at the top)
    pts = np.array([
        [x, y - size],           # Top point
        [x - size//2, y + size//2],  # Bottom left
        [x + size//2, y + size//2],  # Bottom right
    ], np.int32)
    
    # Reshape to a format OpenCV expects
    pts = pts.reshape((-1, 1, 2))
    
    # Draw the triangle
    # cv2.polylines(img, [pts], True, color, thickness)
    
    # Fill the triangle if thickness is -1
    if thickness == -1:
        cv2.fillPoly(img, [pts], color)

def crop_video_frames(video_path: str, start_frame: int, end_frame: int, output_path: str, fps: float):
    """
    Crop a segment from a video between start_frame and end_frame (inclusive), and save it.

    Args:
        video_path (str): Input video file path.
        start_frame (int): Start frame index.
        end_frame (int): End frame index.
        output_path (str): Path to save the cropped clip.
        fps (float): Frames per second.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_id in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Failed to read frame {frame_id}, stopping early.")
            break
        writer.write(frame)

    cap.release()
    writer.release()

def render_segments_to_images_and_videos(
    suspicious_segments: dict,
    video_paths: list,
    game_time: list,
    jsonl_path: str,
    bg_img,
    field_size: tuple,
    output_dir: str,
    fps: float = 29.97,
    buffer: int = 75,
    homography_matrices: list = None,
    flip_vertically: bool = False,
):
    """
    Render suspicious segments to images and videos.
    
    Args:
        suspicious_segments (dict): Dictionary of suspicious segments {class_tag: {track_id: (start_frame, end_frame), ...}, ...}
        video_paths (list): List of video file paths.
        game_time (list): List of tuples for game time alignment [(game_start, half_end, second_start, game_end), ...]
        jsonl_path (str): Path to the JSONL file with tracking data.
        bg_img: Background image (numpy array).
        field_size (tuple): Size of the field (width, height).
        output_dir (str): Directory to save output images and videos.
        fps (float): Frames per second of the videos.
        buffer (int): Number of frames to add as buffer before and after each segment.
        homography (np.array): Homography matrix for coordinate transformation.
        flip_vertically (bool): Whether to flip the y-axis vertically.
    """
    os.makedirs(output_dir, exist_ok=True)
    # resize background image to match field size
    bg_img = cv2.resize(bg_img, (field_size[0], field_size[1]))

    if homography_matrices is None:
        homography_matrices = [np.eye(3) for _ in video_paths] 

    if len(homography_matrices) != len(video_paths):
        raise ValueError(
            f"Number of homography matrices ({len(homography_matrices)}) must match "
            f"number of video paths ({len(video_paths)})"
        )   

    if not isinstance(suspicious_segments, dict):
            raise TypeError(f"suspicious_segments should be a dict, got {type(suspicious_segments).__name__}")

    # Load all tracks once
    with open(jsonl_path, 'r') as f:
        tracks = [json.loads(line) for line in f]
    
    # Index by frame
    frame_to_objects = defaultdict(list)
    for t in tracks:
        tid = t["track_id"]
        team = t.get("team", "unsure")
        jersey_num = t.get("jersey_num", "unsure")
        if isinstance(jersey_num, list):
            # [12,13,14] -> "12/13/14"
            # jersey_num = "/".join(map(str, jersey_num))
            jersey_num = "U"  # mark as unsure if multiple jersey numbers

        if jersey_num == "unsure":
            jersey_num = "U"
        if "goalkeeper" in team.lower():
            jersey_num = "GK"
        if team == "referee":
            jersey_num = "REF"
        if team == "ball":
            jersey_num = "Ball"

        frames = t["frames"]
        points = t.get("projected", t.get("points", []))
        for i, f_id in enumerate(frames):
            if i >= len(points):
                continue
            pt = points[i]
            if pt is not None:
                frame_to_objects[f_id].append((pt, tid, jersey_num, team))

    team_colors_cv = {
        'home': (255, 192, 203),
        'homegoalkeeper': (0, 165, 255),
        'away': (255, 0, 0),
        'awaygoalkeeper': (0, 255, 0),
        'referee': (0, 255, 255),
        'ball': (0, 0, 0),
        'unsure': (128, 128, 128),
    }

    team_colors_plot = {
        k: tuple(np.array(bgr_to_rgb(v)) / 255.0)
        for k, v in team_colors_cv.items()
    }

    total_segments = sum(len(v) for v in suspicious_segments.values())
    current_segment = 0
    print(f"🚩 Total suspicious segments to render: {total_segments}")
    # Note: format of suspicious_segments: {1: {'start_frame': int, 'end_frame': int}, ...}
    for tag, segment in suspicious_segments.items():
        if not isinstance(segment, dict):
            raise TypeError(f"Each segment should be a dict, got {type(segment).__name__} for tag {tag}")
        print(f"🔍 Processing Class {tag} with {len(segment)} segments...")

        for track_id, (start_f, end_f) in segment.items():
            current_segment += 1
            # give a larger window for context
            start_f = max(0, start_f - buffer)
            end_f = min(end_f + buffer, max(frame_to_objects.keys()))
            # Find team and jersey number for the track using the first valid frame
            suspicious_player_team = "unsure"
            suspicious_jersey_num = "U"
            # Search within the first 10 frames
            for f in range(start_f, start_f + 10):
                for pt, tid, jnum, t in frame_to_objects.get(f, []):
                    if tid == track_id:
                        suspicious_player_team = t
                        suspicious_jersey_num = jnum
                        break
                if suspicious_player_team != "unsure":
                    break
                if suspicious_jersey_num != "U":
                    break
            
            if suspicious_jersey_num == "unsure" or "/" in str(suspicious_jersey_num) or isinstance(suspicious_jersey_num, list):
                suspicious_jersey_num = track_id  # fallback to track_id
            if "goalkeeper" in suspicious_player_team.lower():
                suspicious_jersey_num = "GK"
            if suspicious_player_team == "referee":
                continue  # skip referees
            print(f"🎯 Rendering track {track_id}, team {suspicious_player_team}, player {suspicious_jersey_num} from {start_f} to {end_f}...")
            video_start_frame = convert_game_frame_to_video_frame(start_f, game_time[0], fps)
            video_end_frame = convert_game_frame_to_video_frame(end_f, game_time[0], fps)

            # Create output subfolder
            track_name = f"{suspicious_player_team}_{suspicious_jersey_num}"
            track_output_dir = os.path.join(output_dir, track_name, f"class_{tag}")
            os.makedirs(track_output_dir, exist_ok=True)

            # === 🖼️ Render Image ===
            fig, ax = plt.subplots(figsize=(12, 7))
            bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            ax.imshow(bg_rgb, extent=[0, field_size[0], field_size[1], 0])

            # Top left corner is the origin
            ax.set_xlim([0, field_size[0]])
            if flip_vertically:
                ax.set_ylim([0, field_size[1]])
            else:
                ax.set_ylim([field_size[1], 0])

            for t in tracks:
                tid = t["track_id"]
                team = t.get("team", "unsure")
                jersey_num = t.get("jersey_num", "U")
                if isinstance(jersey_num, list):
                    # [12,13,14] -> "12/13/14"
                    jersey_num = "/".join(map(str, jersey_num))

                if jersey_num == "unsure":
                    jersey_num = "U"
                if "goalkeeper" in team.lower():
                    jersey_num = "GK"
                if team == "referee":
                    jersey_num = "REF"
                    
                frames = t.get("frames", [])
                points = np.array(t.get("projected", t.get("points", [])))
                if len(frames) != len(points):
                    continue

                filtered = [(f, pt) for f, pt in zip(frames, points)
                            if pt is not None and start_f <= f <= end_f]
                if not filtered:
                    continue
                f_used, pts = zip(*filtered)
                pts = np.array(pts)
                xs, ys = pts[:, 0], pts[:, 1]
                color = team_colors_plot.get(t.get("team", "unsure"), (0.5, 0.5, 0.5))

                ax.plot(xs, ys, color=color, alpha=0.8)
                ax.scatter(xs[-1], ys[-1], color=color)
                label_color = 'red' if tid == track_id else 'black'
                ax.text(xs[-1], ys[-1], str(jersey_num), fontsize=8, color=label_color)
            ax.set_title(f"Track {track_id} [{frame_to_time(start_f)} → {frame_to_time(end_f)}]")
            plt.tight_layout()
            image_path = os.path.join(track_output_dir, f"{track_id}_{suspicious_player_team}_{suspicious_jersey_num}_{frame_to_time(video_start_frame, no_colon=True)}_{frame_to_time(video_end_frame, no_colon=True)}.png")
            plt.savefig(image_path, dpi=300)
            plt.close()
            print(f"🖼️ Saved image: {image_path}")

            # === 🎥 Render Video ===
            height, width, _ = bg_img.shape
            video_path = os.path.join(track_output_dir, f"{track_id}_{suspicious_player_team}_{suspicious_jersey_num}_{frame_to_time(video_start_frame, no_colon=True)}_{frame_to_time(video_end_frame, no_colon=True)}.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

            for f in range(start_f, end_f + 1):
                frame_img = bg_img.copy()
                for pt, tid, jersey_num, team in frame_to_objects.get(f, []):
                    if flip_vertically:
                        pt = (pt[0], field_size[1] - pt[1])
                    x, y = int(pt[0]), int(pt[1])
                    color = team_colors_cv.get(team, (128, 128, 128))
                    text_color = (0, 0, 255) if tid == track_id else (0, 0, 0)
                    cv2.circle(frame_img, (x, y), 5, color, -1)
                    cv2.putText(frame_img, str(jersey_num), (x + 6, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
                writer.write(frame_img)

            writer.release()
            print(f"🎞️ Saved video: {video_path}")
            # === 🎥 Crop raw clips from each video angle ===
            for idx, (video_file, align_time, homography) in enumerate(zip(video_paths, game_time, homography_matrices)):
                video_start_frame = convert_game_frame_to_video_frame(start_f, align_time, fps)
                video_end_frame = convert_game_frame_to_video_frame(end_f, align_time, fps)

                # raw_clip_path = os.path.join(track_output_dir, f"{track_id}_cam{idx}_{frame_to_time(video_start_frame, fps, no_colon=True)}_{frame_to_time(video_end_frame, fps, no_colon=True)}.mp4")

                # crop_video_frames(video_file, video_start_frame, video_end_frame, raw_clip_path, fps)
                # print(f"🎥 Saved raw clip for camera {idx} → {raw_clip_path}")
                
                # Create annotated version of the clip
                annotated_clip_path = os.path.join(track_output_dir, f"{track_id}_{suspicious_player_team}_{suspicious_jersey_num}_cam{idx}_{frame_to_time(video_start_frame, fps, no_colon=True)}_{frame_to_time(video_end_frame, fps, no_colon=True)}.mp4")
                
                # Process and annotate video
                cap = cv2.VideoCapture(video_file)
                if not cap.isOpened():
                    print(f"⚠️ Cannot open video: {video_file}")
                    continue
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                writer = cv2.VideoWriter(annotated_clip_path, fourcc, fps, (width, height))
                
                # Set position to start frame and annotate
                cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)
                for f in tqdm(range(start_f, end_f + 1), desc=f"🎥 Video {current_segment}/{total_segments}: team {suspicious_player_team}, player {suspicious_jersey_num}, cam {idx}"):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"⚠️ Failed to read, stopping early.")
                        break
                    # ⭐ MODIFIED: Check if point is within frame bounds before drawing
                    for pt, tid, jersey_num, team in frame_to_objects.get(f, []):
                        if tid != track_id:
                            continue
                        
                        pitch_point = (float(pt[0]), float(pt[1]))
                        video_point = reverse_homography(pitch_point, homography)
                        
                        # ⭐ NEW: Only draw if point is valid and within frame bounds
                        if video_point is not None and is_point_in_frame(video_point, width, height):
                            draw_triangle(frame, video_point, size=30, color=(0, 255, 255), thickness=-1)
                        # else:
                        #     print(f"⚠️ Skipping point at frame {f}: point {video_point} is out of bounds ({frame_width}x{frame_height})")
                writer.release()
                print(f"🎥 Saved annotated clip → {annotated_clip_path}")
                cap.release()

    print("✅ All segments rendered successfully.")

def main(args):
    # Load JSONL
    jsonl_path = args.jsonl_path
    video_path = args.video_paths
    game_time = args.game_time
    angle_threshold = args.angle_threshold
    velocity_threshold = args.velocity_threshold
    min_valid_frames = args.min_valid_frames
    conf_threshold = args.conf_threshold
    frame_threshold = args.frame_threshold
    distance_threshold = args.distance_threshold
    output_dir = args.output_dir
    bg_img_path = args.bg_img_path
    field_size = tuple(map(int, args.field_size.split(',')))
    fps = args.fps
    homography_paths = args.homography
    homography_matrices = []

    for h_path in homography_paths:
        if os.path.isfile(h_path):
            try:
                H = np.load(h_path)
                homography_matrices.append(H)
                print(f"✅ Loaded homography from: {h_path}")
            except Exception as e:
                print(f"⚠️ Failed to load homography from {h_path}: {e}")
                homography_matrices.append(np.eye(3))
        else:
            print(f"⚠️ Homography file not found: {h_path}, using identity matrix")
            homography_matrices.append(np.eye(3))
    
    # ⭐ NEW: Validate number of homographies matches videos
    if len(homography_matrices) != len(video_path):
        raise ValueError(
            f"Number of homography matrices ({len(homography_matrices)}) must match "
            f"number of video paths ({len(video_path)})"
        )
    # Assume these functions are defined elsewhere
    multi_ball_frames = get_frames_with_multiple_balls(jsonl_path)
    print("Multi-ball frames detected:", len(multi_ball_frames))
    # print(multi_ball_frames)  # Show first 5 for debugging

    suspicious_tracks, track_abnormal_frames, track_confidences = detect_abnormal_tracks_from_jsonl(
        jsonl_path,
        angle_threshold=angle_threshold,
        velocity_threshold=velocity_threshold,
        min_valid_frames=min_valid_frames,
        conf_threshold=conf_threshold,
        frame_threshold=frame_threshold,
        distance_threshold=distance_threshold,
        multi_ball_frames=multi_ball_frames
    )

    render_segments_to_images_and_videos(
        suspicious_segments=suspicious_tracks,
        video_paths=video_path,
        game_time=game_time,
        jsonl_path=jsonl_path,
        bg_img=cv2.imread(bg_img_path),
        field_size=field_size,
        output_dir=output_dir,
        fps=fps,
        homography_matrices=homography_matrices,  
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect abnormal tracks and render suspicious segments.")

    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--video-paths", nargs="+", required=True, help="Paths to start and end videos.")
    parser.add_argument('--game-time', nargs='+', type=int, required=True,
                        help="Game time in frames, e.g. '0 2700 3600 6300' for one game or '0 2700 3600 6300 100 2800 3700 6400' for two games. Each game should have four integers: first_half_start_second, first_half_end_second, second_half_start_second, second_half_end_second.")
    parser.add_argument("--bg-img-path", type=str, required=True, help="Path to background image.")
    parser.add_argument("--field-size", type=str, default="1060,660", help="Field size as 'width,height'.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output.")
    parser.add_argument("--fps", type=float, default=29.97, help="Video frames per second.")

    # Parameters for abnormal track detection
    parser.add_argument("--angle-threshold", type=float, default=120, help="Angle threshold for detection.")
    parser.add_argument("--velocity-threshold", type=float, default=1e-2, help="Velocity threshold.")
    parser.add_argument("--min-valid-frames", type=int, default=5, help="Minimum number of valid frames.")
    parser.add_argument("--conf-threshold", type=float, default=0.7, help="Confidence threshold.")
    parser.add_argument("--frame-threshold", type=int, default=3, help="Frame count threshold.")
    parser.add_argument("--distance-threshold", type=float, default=50, help="Distance threshold in pixels or meters.")
    parser.add_argument("--homography", nargs="+", required=True, 
                        help="Paths to homography matrix files (.npy), one per video.")

    args = parser.parse_args()
    # 1. Validate that total game_time values is divisible by 4
    if len(args.game_time) % 4 != 0:
        raise ValueError("game_time should be a list of integers with length multiple of 4.")
    # 2. Reshape game_time to a list of 4-value tuples
    args.game_time = [
        tuple(args.game_time[i:i + 4])
        for i in range(0, len(args.game_time), 4)
    ]
    # 3. Now check length match
    if len(args.video_paths) != len(args.game_time):
        raise ValueError(
            f"video_paths should have the same length as game_time, got {len(args.video_paths)} and {len(args.game_time)} respectively."
        )
    # 4. Validate homography files count
    if len(args.homography) != len(args.video_paths):
        raise ValueError(
            f"Number of homography files ({len(args.homography)}) must match "
            f"number of video paths ({len(args.video_paths)})"
        )
    main(args)


# python3 render_classification_output.py --jsonl-path ./runs/detect/test_4k-2h-crop/team_tracking_relabeled.jsonl --video-paths ./data/video/test_sample/C0478.MP4 --bg-img-path ./data/images/mongkok_football_field.png --output-dir ./runs/detect/test_4k-2h-crop/suspicious-output --game-time 317 3085 3982 6809 --homography ./homography_cam1.npy ./homography_cam2.npy