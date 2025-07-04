import cv2
import sys

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Total frames in video: {total_frames}")

    cap.release()

if __name__ == "__main__":
    # Replace this path or pass from CLI
    video_path = "./data/video/test_sample/C0478.MP4"

    # Optional: allow passing path from command line
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    count_frames(video_path)