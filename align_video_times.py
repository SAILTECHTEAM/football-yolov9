import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

def parse_time_str(time_str: str) -> float:
    """Convert hh:mm:ss string to seconds."""
    h, m, s = map(int, time_str.strip().split(":"))
    return h * 3600 + m * 60 + s


def align_videos_times(
    start_times_str,           # List of start times in string for each video
    first_half_end_str,        # string (video0 timeline)
    second_half_start_str,     # string (video0 timeline)
    final_end_str              # string (video0 timeline)
):
    # check input
    if not isinstance(start_times_str, list):
        raise ValueError("start_times_str must be a list of strings")

    start_times = [parse_time_str(t) for t in start_times_str]
    first_half_end = parse_time_str(first_half_end_str)
    second_half_start = parse_time_str(second_half_start_str)
    final_end = parse_time_str(final_end_str)

    # Compute game segment durations based on video0's clock
    first_half_duration = first_half_end - start_times[0]
    second_half_duration = final_end - second_half_start

    print(f"🎥 Aligning and cropping videos:")
    print(f"  - First half duration: {first_half_duration // 60}m {first_half_duration % 60}s")
    print(f"  - Second half duration: {second_half_duration // 60}m {second_half_duration % 60}s")

    # For each video, compute the absolute time ranges
    aligned_times = []
    for i, t in enumerate(start_times):
        fh_start = t
        fh_end = t + first_half_duration
        sh_start = t + (second_half_start - start_times[0])
        sh_end = sh_start + second_half_duration
        aligned_times.append([fh_start, fh_end, sh_start, sh_end])

    return aligned_times


def convert_game_frame_to_video_frame(frame_in_game: int, aligned_time: list, fps: float) -> int:
    """
    Convert a frame index (in game timeline) to original video timeline frame index.

    Args:
        frame_in_game (int): Frame index based on game timeline (starting at game start).
        aligned_time (list): List of [fh_start, fh_end, sh_start, sh_end] in seconds.
        fps (float): Frames per second of the video.

    Returns:
        int: Corresponding frame index in the original video.
    """
    fh_start, fh_end, sh_start, sh_end = aligned_time
    seconds_in_game = frame_in_game / fps

    first_half_duration = fh_end - fh_start
    if seconds_in_game <= first_half_duration:
        # It's in first half
        original_time_sec = fh_start + seconds_in_game
    else:
        # It's in second half
        seconds_into_second_half = seconds_in_game - first_half_duration
        original_time_sec = sh_start + seconds_into_second_half

    return int(original_time_sec * fps)


def main(args):
    start_times_str = args.start_times.split(',')
    first_half_end_str = args.first_half_end
    second_half_start_str = args.second_half_start
    final_end_str = args.final_end

    aligned_times = align_videos_times(
        start_times_str,
        first_half_end_str,
        second_half_start_str,
        final_end_str,
    )

    print("Aligned times for each video (in seconds):", aligned_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align video start and end times based on time strings.")

    parser.add_argument("--start_times", type=str, required=True,
                        help="Comma-separated list of start times in HH:MM:SS format, e.g. '00:05:17,00:06:18'")
    parser.add_argument("--first_half_end", type=str, required=True,
                        help="Time string for first half end, e.g. '00:51:25'")
    parser.add_argument("--second_half_start", type=str, required=True,
                        help="Time string for second half start, e.g. '01:06:22'")
    parser.add_argument("--final_end", type=str, required=True,
                        help="Time string for final end, e.g. '01:53:29'")

    args = parser.parse_args()
    main(args)
    # python align_video_times.py --start_times 00:05:17,00:06:18,00:07:19,00:08:20 --first_half_end 00:51:25 --second_half_start 01:06:22 --final_end 01:53:29



    # aligned_time = [317, 3085, 3982, 6809]  # in seconds
    # fps = 29.97
    # game_frame_index = 48011  # frame from game time
    # video_frame_index = convert_game_frame_to_video_frame(game_frame_index, aligned_time, fps)
    # print(f"🎯 Video frame index: {video_frame_index}")