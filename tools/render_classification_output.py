import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
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
    fps: float = 29.97
):
    os.makedirs(output_dir, exist_ok=True)
    # resize background image to match field size
    bg_img = cv2.resize(bg_img, (field_size[0], field_size[1]))
    
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
        frames = t["frames"]
        points = t.get("projected", t.get("points", []))
        for i, f_id in enumerate(frames):
            if i >= len(points):
                continue
            pt = points[i]
            if pt is not None:
                frame_to_objects[f_id].append((pt, tid, team))

    team_colors_cv = {
        'eastern': (255, 0, 0),
        'easterngoalkeeper': (0, 255, 0),
        'kitchee': (255, 192, 203),
        'kitcheegoalkeeper': (0, 165, 255),
        'referee': (0, 255, 255),
        'ball': (0, 0, 0),
        'unsure': (128, 128, 128),
    }

    team_colors_plot = {
        k: tuple(np.array(bgr_to_rgb(v)) / 255.0)
        for k, v in team_colors_cv.items()
    }
    # Note: format of suspicious_segments: {1: {'start_frame': int, 'end_frame': int}, ...}
    for tag, segment in suspicious_segments.items():
        if not isinstance(segment, dict):
            raise TypeError(f"Each segment should be a dict, got {type(segment).__name__} for tag {tag}")
        print(f"🔍 Processing Class {tag} with {len(segment)} segments...")

        for track_id, (start_f, end_f) in segment.items():
            print(f"🎯 Rendering track {track_id} from {start_f} to {end_f}...")
            video_start_frame = convert_game_frame_to_video_frame(start_f, game_time[0], fps)
            video_end_frame = convert_game_frame_to_video_frame(end_f, game_time[0], fps)

            # Create output subfolder
            track_output_dir = os.path.join(output_dir, track_id, f"class_{tag}")
            os.makedirs(track_output_dir, exist_ok=True)

            # === 🖼️ Render Image ===
            fig, ax = plt.subplots(figsize=(12, 7))
            bg_rgb = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            ax.imshow(bg_rgb, extent=[0, field_size[0], 0, field_size[1]])

            for t in tracks:
                tid = t["track_id"]
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
                ax.text(xs[-1], ys[-1], str(tid), fontsize=8, color=label_color)

            ax.set_xlim(0, field_size[0])
            ax.set_ylim(0, field_size[1])
            ax.set_title(f"Track {track_id} [{frame_to_time(start_f)} → {frame_to_time(end_f)}]")
            plt.tight_layout()
            image_path = os.path.join(track_output_dir, f"{track_id}_{frame_to_time(video_start_frame, no_colon=True)}_{frame_to_time(video_end_frame, no_colon=True)}.png")
            plt.savefig(image_path, dpi=300)
            plt.close()
            print(f"🖼️ Saved image: {image_path}")

            # === 🎥 Crop raw clips from each video angle ===
            for idx, (video_file, align_time) in enumerate(zip(video_paths, game_time)):
                video_start_frame = convert_game_frame_to_video_frame(start_f, align_time, fps)
                video_end_frame = convert_game_frame_to_video_frame(end_f, align_time, fps)

                raw_clip_path = os.path.join(track_output_dir, f"{track_id}_cam{idx}_{frame_to_time(video_start_frame, fps, no_colon=True)}_{frame_to_time(video_end_frame, fps, no_colon=True)}.mp4")

                crop_video_frames(video_file, video_start_frame, video_end_frame, raw_clip_path, fps)
                print(f"🎥 Saved raw clip for camera {idx} → {raw_clip_path}")

            # === 🎥 Render Video ===
            height, width, _ = bg_img.shape
            video_path = os.path.join(track_output_dir, f"{track_id}_{frame_to_time(video_start_frame, no_colon=True)}_{frame_to_time(video_end_frame, no_colon=True)}.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

            for f in range(start_f, end_f + 1):
                frame_img = bg_img.copy()
                for pt, tid, team in frame_to_objects.get(f, []):
                    x, y = int(pt[0]), field_size[1] - int(pt[1])
                    color = team_colors_cv.get(team, (128, 128, 128))
                    text_color = (0, 0, 255) if tid == track_id else (0, 0, 0)
                    cv2.circle(frame_img, (x, y), 5, color, -1)
                    cv2.putText(frame_img, str(tid), (x + 6, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
                writer.write(frame_img)

            writer.release()
            print(f"🎞️ Saved video: {video_path}")
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
        fps=fps
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect abnormal tracks and render suspicious segments.")

    parser.add_argument("--jsonl-path", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--video-paths", nargs="+", required=True, help="Paths to start and end videos.")
    parser.add_argument('--game-time', nargs='+', type=int, required=True,
                        help="Game time in frames, e.g. '0 2700 3600 6300' for one game or '0 2700 3600 6300 100 2800 3700 6400' for two games. Each game should have four integers: start_frame, first_half_end_frame, second_half_start_frame, final_end_frame.")
    parser.add_argument("--bg-img-path", type=str, required=True, help="Path to background image.")
    parser.add_argument("--field-size", type=str, default="1060,660", help="Field size as 'width,height'.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output.")
    parser.add_argument("--fps", type=float, default=29.97, help="Video frames per second.")

    # Parameters for abnormal track detection
    parser.add_argument("--angle-threshold", type=float, default=90, help="Angle threshold for detection.")
    parser.add_argument("--velocity-threshold", type=float, default=1e-1, help="Velocity threshold.")
    parser.add_argument("--min-valid-frames", type=int, default=5, help="Minimum number of valid frames.")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--frame-threshold", type=int, default=3, help="Frame count threshold.")
    parser.add_argument("--distance-threshold", type=float, default=0.5, help="Distance threshold in pixels or meters.")

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
    main(args)


# python3 render_classification_output.py --jsonl-path ./runs/detect/test_4k-2h-crop/team_tracking_relabeled.jsonl --video-paths ./data/video/test_sample/C0478.MP4 --bg-img-path ./data/images/mongkok_football_field.png --output-dir ./runs/detect/test_4k-2h-crop/suspicious-output --game-time 317 3085 3982 6809 