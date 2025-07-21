from yolov9.proprcess import render_to_image_from_jsonl
import cv2

if __name__ == "__main__":
    # Specify the path to the JSONL file and the output image file
    jsonl_file_path = "./runs/detect/test_4k-2h/team_tracking_relabeled.jsonl"
    background_image_path = "./data/images/mongkok_football_field.png"  # Optional background image
    field_size = (1060, 660) 
    output_image_path = "./runs/detect/test_4k-2h/team_tracking_output.png"
    start_frame = 48011
    end_frame = 48800
    bg_img = cv2.imread(background_image_path)
    bg_img = cv2.resize(bg_img, field_size)
    highlight_ids = ["7628a"]
    
    # Call the function to render the image
    render_to_image_from_jsonl(jsonl_file_path, bg_img, field_size, output_image_path, start_frame, end_frame, highlight_ids=highlight_ids)
    print(f"Image rendered and saved to {output_image_path}")
    