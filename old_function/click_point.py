import cv2

def click_points_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to read the first frame.")
        return

    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"📍 Clicked at: ({x}, {y})")
            points.append((x, y))
            cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.imshow("Click to select points", frame)

    cv2.imshow("Click to select points", frame)
    cv2.setMouseCallback("Click to select points", click_event)

    print("🖱️ Click on the image to collect points. Press 'q' to quit.")

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("✅ Collected points:", points)
    return points

if __name__ == "__main__":
    points = click_points_on_video("../runs/detect/demo_video/clips/0025/GX010025_clip_001.mp4")
