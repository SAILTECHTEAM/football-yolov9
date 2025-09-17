# Inference on player detection and tracking
python3 mini_patch_detect_v1_for_video.py --source './data/video/test_sample/period_57873_58952_cam0.mp4' --game-time 0 0 0 36 --img 640 --device 0 --weights './weight/yolov9-s-converted.pt' --name test_4k_player_640 --classes 0 --clothes-folder-path ./data/histograms/0525/ --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 --homography-dst-points 530 0 530 660 1060 660 1060 0 --nosave

# Run post-processing on player tracks
python3 post-processing.py --json-path "./runs/detect/test_4k_player_640/team_tracking.jsonl" --image-path "./data/images/mongkok_football_field.png" --output-name './runs/detect/test_4k_player_640/team_tracking_output'

# Inference on ball detection
python3 mini_patch_detect_ball_for_video.py --source './data/video/test_sample/C0478.MP4' --game-time 317 3085 3982 6809 --img 640 --device 0 --weights './weight/yolov9-s-converted_ball_detection_1280_20250822.pt' --name test_4k_ball_640_whole_match_20250905 --classes 0 --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 --homography-dst-points 530 0 530 660 1060 660 1060 0 --nosave

# Run post-processing on ball tracks
python3 post-processing-ball.py --json-path "./runs/detect/test_4k_ball_640/ball_tracking.jsonl" --image-path "./data/images/mongkok_football_field.png" --output-name './runs/detect/test_4k_ball_640/ball_tracking_output'

# Render classification output
python render_classification_output.py --jsonl-path "../runs/detect/test_4k_player_640/team_ball_tracking_final.jsonl" --video-paths '../data/video/test_sample/C0478.MP4' --video-paths ../data/video/test_sample/C0478.MP4 --bg-img-path ../data/images/mongkok_football_field.png --output-dir "../runs/detect/test_4k_player_640_numba_20250829_final/suspicious-output" --game-time 0 0 0 36

# Goalkeper
python3 detect_goal_v2.py --weights "./weight/yolov9-s-converted.pt" --source "./data/video/GX010025_clips/" --name "demo_video" --nosave --radar_data_path ./data/excel/PR_20250208_1739_session.csv --homography_path ./runs/detect/demo_video/homography_matrix.npy

python3 detections_to_tracks_and_scores.py --clips-root ./runs/detect/demo_video/clips/0025/ --hist ./data/histograms/gk01.npy

python3 render_track_on_video.py --root ./runs/detect/demo_video_22cm/clips/0025/

python3 analyze_goalkeeper_behavior.py --root ./runs/detect/demo_video_22cm/clips/0025/ --speeds 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45 --out-dir ./runs/detect/demo_video_22cm/analysis --homography ./runs/detect/demo_video_22cm/homography_matrix.npy

# Render specific time period (For checking tracking quality in specific time period)
# python3 render_specific_time_period.py