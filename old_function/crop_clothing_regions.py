import json
import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def crop_players_from_video(jsonl_path, video_path, output_folder, 
                            top_ratio=0.15, bottom_ratio=0.55, 
                            left_ratio=0.3, right_ratio=0.7):
    """
    Crop player regions from video frames based on JSONL tracking data.
    
    Args:
        jsonl_path: Path to the JSONL file containing player tracking data
        video_path: Path to the video file
        output_folder: Directory to save cropped images
        padding: Optional padding to add around bounding boxes (pixels)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the JSONL file and collect frame data
    tracks = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            track = json.loads(line)
            tracks.append(track)
    
    # Organize frame indices and corresponding bounding boxes
    frame_to_crops = {}
    for track in tracks:
        track_id = track['track_id']
        frame_ids = track['frame_id']
        bboxes = track['bbox']
        
        for i, frame_id in enumerate(frame_ids):
            if frame_id not in frame_to_crops:
                frame_to_crops[frame_id] = []
            
            # Each bbox is [x1, y1, x2, y2, confidence]
            bbox = bboxes[i][:4]  # Just take the coordinates, not the confidence
            frame_to_crops[frame_id].append((track_id, bbox))
    
    # Get all frames we need to extract
    all_frames = sorted(frame_to_crops.keys())
    if not all_frames:
        print("No frames found to crop!")
        return
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video has {total_frames} frames at {fps} FPS")
    print(f"Will extract {len(all_frames)} unique frames with {sum(len(crops) for crops in frame_to_crops.values())} crops")
    
    # Process the video
    current_frame_idx = 0
    
    with tqdm(total=len(all_frames), desc="Processing frames") as pbar:
        for target_frame in all_frames:
            # Skip ahead to the target frame
            if target_frame < current_frame_idx:
                print(f"Warning: Frames must be processed in order. Skipping frame {target_frame}")
                continue
            
            # Fast-forward to the target frame
            while current_frame_idx < target_frame:
                success = cap.grab()
                if not success:
                    print(f"Error: Could not grab frame {current_frame_idx}")
                    break
                current_frame_idx += 1
            
            # Read the target frame
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {target_frame}")
                break
            
            current_frame_idx += 1
            
            # Process all crops for this frame
            for track_id, bbox in frame_to_crops[target_frame]:
                try:
                    x1, y1, x2, y2 = map(int, bbox)

                    w = x2 - x1
                    h = y2 - y1

                    # Vertical bounds
                    new_y1 = y1 + int(h * top_ratio)
                    new_y2 = y1 + int(h * bottom_ratio)

                    # Horizontal bounds
                    new_x1 = x1 + int(w * left_ratio)
                    new_x2 = x1 + int(w * right_ratio)

                    # Clip to image bounds
                    new_x1 = max(new_x1, 0)
                    new_x2 = min(new_x2, frame.shape[1])
                    new_y1 = max(new_y1, 0)
                    new_y2 = min(new_y2, frame.shape[0])
                    
                    # Crop the region
                    crop = frame[new_y1:new_y2, new_x1:new_x2]
                    
                    # Skip empty crops
                    if crop.size == 0:
                        print(f"Warning: Empty crop for track {track_id} at frame {target_frame}, bbox: {bbox}")
                        continue
                    
                    # Save the crop
                    output_filename = f"{track_id}_{target_frame}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, crop)
                    
                except Exception as e:
                    print(f"Error processing track {track_id} at frame {target_frame}: {e}")
            
            pbar.update(1)
    
    cap.release()
    print(f"Done! Extracted crops saved to {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Crop player regions from video using JSONL tracking data')
    parser.add_argument('--jsonl', required=True, help='Path to JSONL file with tracking data')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', default='crops', help='Output directory for cropped images')
    parser.add_argument('--padding', type=int, default=0, help='Padding around bounding boxes in pixels')
    
    args = parser.parse_args()
    
    crop_players_from_video(args.jsonl, args.video, args.output, args.padding)

if __name__ == "__main__":
    main()

# example usage: python crop_clothing_regions.py --jsonl ./runs/detect/test_4k_player_640/team_tracking.jsonl --video ./data/video/C0478.MP4 --output ./runs/detect/test_4k_player_640/cloth_cropping --padding 10