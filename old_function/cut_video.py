from moviepy.editor import VideoFileClip

def cut_video_by_user_input_with_seconds(input_path, output_path):
    """
    Load a video, print its duration, ask user for start/end time 
    in minutes and seconds, then cut and save the video segment.
    
    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the cut video.
    """
    clip = VideoFileClip(input_path)
    duration_sec = clip.duration
    duration_min = duration_sec / 60

    print(f"📼 Video duration: {int(duration_min)} minutes {int(duration_sec % 60)} seconds.")

    try:
        # Get start time
        start_min = int(input("Enter start time - minutes: "))
        start_sec = int(input("Enter start time - seconds: "))
        # Get end time
        end_min = int(input("Enter end time - minutes: "))
        end_sec = int(input("Enter end time - seconds: "))

        # Convert to total seconds
        start_total = start_min * 60 + start_sec
        end_total = end_min * 60 + end_sec

        if start_total < 0 or end_total > duration_sec or start_total >= end_total:
            print("❌ Invalid time range.")
            return

        subclip = clip.subclip(start_total, end_total)
        subclip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print(f"✅ Output saved to {output_path}")

    except ValueError:
        print("❌ Invalid input. Please enter integers for minutes and seconds.")
    
    finally:
        clip.close()

if __name__ == "__main__": 
    input_path = input("Enter the path to the input video file: ")
    output_path = input("Enter the path to save the cut video: ")
    cut_video_by_user_input_with_seconds(input_path, output_path)
    # Example usage:
    # cut_video_by_user_input_with_seconds("input_video.mp4", "output_video.mp4")
