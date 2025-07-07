from render_output import render_to_video_from_jsonl
import cv2

if __name__ == "__main__":
    # Specify the path to the JSONL file and the output video file
    jsonl_file_path = "./runs/detect/test_4k-2h/team_tracking_relabeled.jsonl"
    background_image_path = "./data/images/mongkok_football_field.png"  # Optional background image
    field_size = (1060, 660) 
    output_video_path = "./runs/detect/test_4k-2h/team_tracking_output.mp4"
    bg_img = cv2.imread(background_image_path)
    bg_img = cv2.resize(bg_img, field_size)
    # Call the function to render the video
    render_to_video_from_jsonl(jsonl_file_path, bg_img, field_size, output_video_path)
    print(f"Video rendered and saved to {output_video_path}")
