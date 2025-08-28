from render_classification_output import render_segments_to_images_and_videos
import cv2

def render_period_with_correct_timing(
    start_frame: int,
    end_frame: int,
    jsonl_path: str,
    bg_img_path: str,
    video_paths: list,
    game_time: list,  # (game_start, half_end, second_start, game_end) in seconds
    output_dir: str,
    field_size: tuple,
):
    """
    Render time period with correct game_time format for video cropping
    """
    bg_img = cv2.imread(bg_img_path)
    
    # Create one "suspicious segment" for the time period
    suspicious_segments = {
        f"period_{start_frame}_{end_frame}": (start_frame, end_frame)
    }

    render_segments_to_images_and_videos(
        suspicious_segments=suspicious_segments,
        video_paths=video_paths,
        game_time=game_time,
        jsonl_path=jsonl_path,
        bg_img=bg_img,
        field_size=field_size,
        output_dir=output_dir
    )
    
    print("✅ Rendering complete!")
    print("📁 Output includes:")
    print("   📊 Field visualization showing all tracks in the period")
    print("   🎥 Animation of all tracks moving on the field") 
    print("   ✂️ Cropped video segment from your camera for this time period")


if __name__ == "__main__":

    render_period_with_correct_timing(
        start_frame=57873,
        end_frame=58952,
        jsonl_path="./runs/detect/test_4k_converted/team_tracking_relabeled.jsonl",
        bg_img_path="./data/images/mongkok_football_field.png",
        video_paths=['./data/video/test_sample/C0478.MP4'],
        game_time=[(317, 3085, 3982, 6809)],  # (game_start, half_end, second_start, game_end) in seconds
        output_dir='./runs/detect/',
        field_size=(1060, 660),
    )
    