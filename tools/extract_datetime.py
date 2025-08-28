import subprocess
import json
from datetime import datetime, timedelta ,timezone


def get_video_start_time_and_fps(video_path):
    """
    Use 'mediainfo' to extract the Encoded_Date (start time in UTC) 
    and FrameRate (FPS) of the given video file.
    Returns (start_time_utc, fps) or (None, None) if unavailable.
    """
    cmd = ["mediainfo", "--Output=JSON", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error running mediainfo: {result.stderr}")
        return None, None

    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return None, None

    # Navigate JSON structure (some versions: metadata["media"]["track"])
    if "media" not in metadata or "track" not in metadata["media"]:
        return None, None

    tracks = metadata["media"]["track"]
    start_time = None
    fps = None

    for track in tracks:
        if track.get("@type") == "General":
            encoded_date = track.get("Encoded_Date")
            frame_rate = track.get("FrameRate")
            if encoded_date:
                # Example: "UTC 2025-02-09 07:39:53"
                encoded_date = encoded_date.replace("UTC ", "").strip()
                try:
                    # Mark it as UTC time
                    start_time = datetime.strptime(encoded_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            if frame_rate:
                try:
                    fps = float(frame_rate)
                except ValueError:
                    pass

    return start_time, fps

def calculate_real_timestamp(video_start_time, base_frame_idx, current_frame_idx, fps):
    """
    Convert the current frame index into a real-world timestamp, considering the UTC offset.
    """
    if not video_start_time or not fps:
        return None, None

    # Convert from global frame index to local frame index for this video
    local_frame_idx = current_frame_idx - base_frame_idx
    elapsed_seconds = local_frame_idx / fps

    # Apply elapsed time (in unix time) to the video start time
    real_time_utc = video_start_time + timedelta(seconds=elapsed_seconds)


    return real_time_utc.timestamp(), elapsed_seconds


if __name__ == "__main__":
    # Example Usage
    video_file = "./data/video/GX040011.mp4"
    frame_num = 10000  # Replace with desired frame number

    # Unpack the tuple returned by get_video_start_time_and_fps
    video_start_time, fps = get_video_start_time_and_fps(video_file)
    print(f"Video start time : {video_start_time}")
    print(f"FPS : {fps}")

    if video_start_time and fps:
        real_time, video_time = calculate_real_timestamp(video_start_time, 0, frame_num, fps)
        print(f"Frame real time : {real_time}")
        print(f"Frame video time : {video_time}")
    else:
        print("Failed to get video start time or FPS.")
