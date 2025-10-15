# Inference on player detection and tracking
python3 mini_patch_detect_v1_for_video.py \
  --source './data/video/test_sample/C0478.MP4' \
  --game-time 317 3085 3982 6809 \
  --img 640 \
  --device 0 \
  --weights './weight/yolov9-s-converted.pt' \
  --name test_4k_player_640 \
  --classes 0 \
  --clothes-folder-path ./data/histograms/0525/ \
  --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 \
  --homography-dst-points 530 0 530 660 1060 660 1060 0 \
  --jersey-weights ./weights/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt \
  --nosave

# Inference on ball detection
python3 mini_patch_detect_ball_for_video.py \
  --source './data/video/test_sample/C0478.MP4' \
  --game-time 317 3085 3982 6809 \
  --img 640 \
  --device 0 \
  --weights './weight/yolov9-s-converted_ball_detection_1280_20250822.pt' \
  --name test_4k_ball_640 \
  --classes 0 \
  --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 \
  --homography-dst-points 530 0 530 660 1060 660 1060 0 \
  --nosave

# Run post-processing on player tracks
python3 post-processing.py \
  --json-path "./runs/detect/test_4k_player_640/team_tracking.jsonl" \
  --image-path "./data/images/mongkok_football_field.png" \
  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 13 23 25 8 14 17 18 21 24 31 33 34 \
  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 1 17 22 23 24 28 33 42 43 44 72 88 \
  --output-name './runs/detect/test_4k_player_640/team_tracking_output'


# Run post-processing on ball tracks
python3 post-processing-ball.py \
  --json-path "./runs/detect/test_4k_ball_640/ball_tracking.jsonl" \
  --image-path "./data/images/mongkok_football_field.png" \
  --output-name './runs/detect/test_4k_ball_640/ball_tracking_output'

# Combine player and ball tracking results
python3 combine_team_ball_tracks.py \
  --player-jsonl "./runs/detect/test_4k_player_640/team_tracking_final.jsonl" \
  --ball-jsonl "./runs/detect/test_4k_ball_640/ball_tracking_final.jsonl" \
  --output-jsonl "./runs/detect/test_4k_player_640/team_ball_tracking_final.jsonl"

# Compute homography matrix for rendering
python3 ./tools/extract_homography_matrix.py \
  --src "172,1104 2101,895 3800,1021 3458,2057" \
  --dst "530,0 530,660 1060,660 1060,0" \
  --out "./weight/homography_matrix_whole_match.npy"

# Render classification output
python3 ./tools/render_classification_output.py \
  --jsonl-path "./runs/detect/test_4k_player_640/team_ball_tracking_final.jsonl" \
  --video-paths './data/video/test_sample/C0478.MP4' \
  --bg-img-path ./data/images/mongkok_football_field.png \
  --output-dir './runs/detect/test_4k_player_640/suspicious-output' \
  --game-time 317 3085 3982 6809 \
  --homography './weight/homography_matrix_whole_match.npy'

# Goalkeper
python3 ./tools/extract_homography_matrix.py \
  --src "86,242 1658,258 1644,766 100,771" \
  --dst "0,0 640,0 640,213 0,213" \
  --out ./runs/detect/demo_video/homography_matrix.npy

# Detect goal clips
python3 detect_goal_v2.py \
  --weights "./weight/yolov9-s-converted.pt" \
  --source "./data/video/GX010025_clips/" \
  --name "demo_video" \
  --nosave \
  --radar_data_path ./data/excel/PR_20250208_1739_session.csv \
  --homography_path ./runs/detect/demo_video/homography_matrix.npy \
  --use-tqdm


python3 render_track_on_video.py \
  --root ./runs/detect/demo_video/clips/0025/

python3 analyze_goalkeeper_behavior.py \
  --root ./runs/detect/demo_video/clips/0025/ \
  --speeds 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45 \
  --out-dir ./runs/detect/demo_video/analysis \
  --homography ./runs/detect/demo_video/homography_matrix.npy

# Render specific time period (For checking tracking quality in specific time period)
# python3 render_specific_time_period.py

# Side note: second half starts at around frame 82960