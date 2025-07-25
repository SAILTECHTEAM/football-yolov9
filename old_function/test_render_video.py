import cv2
import json
import numpy as np
from collections import defaultdict

def render_to_video_from_jsonl(jsonl_path, bg_img, field_size, output_path, fps=29.97):
    height, width, _ = bg_img.shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    team_colors = {
        'eastern': (255, 0, 0),
        'easterngoalkeeper': (0, 255, 0),
        'kitchee': (255, 192, 203),
        'kitcheegoalkeeper': (0, 165, 255),
        'referee': (0, 255, 255),
        'ball': (0, 0, 0),
        'unsure': (0, 0, 255),  # For relabeled tracks
    }

    tracks = [json.loads(line) for line in open(jsonl_path)]
    min_frame = 0
    max_frame = max(max(t["frames"]) for t in tracks)

    frame_to_objects = defaultdict(list)
    for t in tracks:
        for i, f in enumerate(t["frames"]):
            if i >= len(t["projected"]):
                continue
            pt = t["projected"][i]
            if pt is None:
                continue
            frame_to_objects[f].append((pt, t["track_id"], t["team"]))

    for f in range(min_frame, max_frame + 1):
        frame_img = bg_img.copy()
        for pt, tid, team in frame_to_objects.get(f, []):
            x, y = int(pt[0]), field_size[1] - int(pt[1])
            color = team_colors.get(team, (128, 128, 128))
            cv2.circle(frame_img, (x, y), 5, color, -1)
            cv2.putText(frame_img, str(tid), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        writer.write(frame_img)

    writer.release()
    print(f"✅ Saved video to: {output_path}")

if __name__ == "__main__":
    # Specify the path to the JSONL file and the output video file
    jsonl_file_path = "./runs/detect/test_4k_converted/team_tracking_relabeled.jsonl"
    background_image_path = "./data/images/mongkok_football_field.png"  # Optional background image
    field_size = (1060, 660) 
    output_video_path = "./runs/detect/test_4k_converted/team_tracking_output.mp4"

    bg_img = cv2.imread(background_image_path)
    bg_img = cv2.resize(bg_img, field_size)
    # Call the function to render the video
    render_to_video_from_jsonl(jsonl_file_path, bg_img, field_size, output_video_path, 29.97)
    print(f"Video rendered and saved to {output_video_path}")
