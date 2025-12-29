# imgsz =  640
# cam0 172 1104 2101 895 3800 1021 3458 2057
python mini_patch_detect_v2_for_video.py   \
  --source './data/video/whole_match/C0478.MP4'   \
  --game-time 317 3085 3982 6809   \
  --img 640   \
  --device 0   \
  --weights './weights/yolov9-s-converted-player-ball-25112025-50eps.pt'   \
  --name 28112025_whole_match_0525_cam0   \
  --classes 0 1   \
  --clothes-folder-path ./data/histograms/0525/   \
  --homography-src-points 172 1104 2101 895 3800 1021 3458 2057   \
  --homography-dst-points 530 0 530 660 1060 660 1060 0   \
  --jersey-weights ./weights/parseq_epoch\=24-step\=2575-val_accuracy\=95.6044-val_NED\=96.3255_new.ckpt \
  --half \
  --nosave

# cam1 366 1910 0 860 1770 748 3780 1000 (bottom left top left top middle bottom middle)
python mini_patch_detect_v2_for_video.py   \
  --source './data/video/whole_match/C0475.MP4'   \
  --game-time 450 3218 4115 6942   \
  --img 640   \
  --device 0   \
  --weights './weights/yolov9-s-converted-player-ball-25112025-50eps.pt'   \
  --name 28112025_whole_match_0525_cam1   \
  --classes 0 1   \
  --clothes-folder-path ./data/histograms/0525/   \
  --homography-src-points 366 1910 0 860 1770 748 3780 1000   \
  --homography-dst-points 0 0 0 660 530 660 530 0   \
  --jersey-weights ./weights/parseq_epoch\=24-step\=2575-val_accuracy\=95.6044-val_NED\=96.3255_new.ckpt \
  --half \
  --nosave

# cam2 141 721 2064 613 3735 826 3225 1641 (top right bottom right bottom middle top middle)
python mini_patch_detect_v2_for_video.py   \
  --source './data/video/whole_match/C0034.MP4'   \
  --game-time 798 3563 4460 7287   \
  --img 640   \
  --device 0   \
  --weights './weights/yolov9-s-converted-player-ball-25112025-50eps.pt'  \
  --name 28112025_whole_match_0525_cam2   \
  --classes 0 1   \
  --clothes-folder-path ./data/histograms/0525/   \
  --homography-src-points 141 721 2064 613 3735 826 3225 1641   \
  --homography-dst-points 1060 660 1060 0 530 0 530 660   \
  --jersey-weights ./weights/parseq_epoch\=24-step\=2575-val_accuracy\=95.6044-val_NED\=96.3255_new.ckpt \
  --half \
  --nosave


# cam3 266 1776 119 866 1799 872 3737 1231 (top middle bottom middle bottom left top left)
python mini_patch_detect_v2_for_video.py   \
  --source './data/video/whole_match/C0477.MP4'   \
  --game-time 872 3640 4537 7364   \
  --img 640   \
  --device 0   \
  --weights './weights/yolov9-s-converted-player-ball-25112025-50eps.pt'   \
  --name 28112025_whole_match_0525_cam3   \
  --classes 0 1   \
  --clothes-folder-path ./data/histograms/0525/   \
  --homography-src-points 266 1776 119 866 1799 872 3737 1231   \
  --homography-dst-points 530 660 530 0 0 0 0 660   \
  --jersey-weights ./weights/parseq_epoch\=24-step\=2575-val_accuracy\=95.6044-val_NED\=96.3255_new.ckpt \
  --half \
  --nosave

python3 post_processing.py   \
  --json-path "./runs/detect/28112025_whole_match_0525_cam3/team_tracking.jsonl"   \
  --image-path "./data/images/mongkok_football_field.png"   \
  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 13 23 25 8 14 17 18 21 24 31 33 34   \
  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 1 17 22 23 24 28 33 42 43 44 72 88   \
  --output-name './runs/detect/28112025_whole_match_0525_cam3/team_tracking_output'


python cluster_tracks.py \
  "./runs/detect/1125_test_4k_player_640_half_cam0_50eps/team_tracking_final.jsonl" \
  "./runs/detect/1125_test_4k_player_640_half_cam1_50eps/team_tracking_final.jsonl"  \
  -o ./runs/detect/1203_track_matching/track_matching_results_1203_cam0_1.json \
  --min-overlap 10 \
  --median-distance-threshold 50 \
  --total-distance-diff-threshold 50 \
  --direction-threshold 0.3 \
  --direction-consistency-threshold 0.6 \
  --max-frames 1500 \
  --direction-frame-stride 3

python cluster_tracks_greedy.py "./runs/detect/1205_track_matching_merged_filtered/cam0/team_tracking_cam0_merged_filtered.jsonl" "./runs/detect/1205_track_matching_merged_filtered/cam1/team_tracking_cam1_merged_filtered.jsonl"  -o ./runs/detect/1205_track_matching_merged_filtered/track_matching_results_1205_cam0_1_no_interpolate.json --min-overlap 10 --median-distance-threshold 50 --total-distance-diff-threshold 50 --direction-threshold 0.3 --direction-consistency-threshold 0.6 --max-frames 1500 --direction-frame-stride 3 --greedy --max-score 30

# 20251208 (this parameter set works best so far)
python cluster_tracks.py \
  --player-jsonl-paths \
  "./runs/detect/1205_track_matching_merged_filtered/cam0/team_tracking_cam0_merged_filtered.jsonl" \
  "./runs/detect/1205_track_matching_merged_filtered/cam1/team_tracking_cam1_merged_filtered.jsonl" \
  "./runs/detect/1205_track_matching_merged_filtered/cam2/team_tracking_cam2_merged_filtered.jsonl" \
  "./runs/detect/1205_track_matching_merged_filtered/cam3/team_tracking_cam3_merged_filtered.jsonl" \
  --ball-jsonl-paths \
  "./runs/detect/28112025_whole_match_0525_cam0/ball_tracking_fused_processed.jsonl" \
  "./runs/detect/28112025_whole_match_0525_cam1/ball_tracking_fused_processed.jsonl" \
  "./runs/detect/28112025_whole_match_0525_cam2/ball_tracking_fused_processed.jsonl" \
  "./runs/detect/28112025_whole_match_0525_cam3/ball_tracking_fused_processed.jsonl" \
  --auto-calibrate \
  --output ./runs/detect/1208_track_matching_merged_filtered/track_matching_results_fused.jsonl \
  --min-overlap 50 \
  --median-distance-threshold 50 \
  --total-distance-diff-threshold 50 \
  --direction-threshold 0.3 \
  --direction-consistency-threshold 0.6 \
  --max-frames 1500 \
  --direction-frame-stride 3 \
  --max-score 10 \
  --spatial-distance 40
python3 post_processing_player.py     --json-paths "./runs/detect/28112025_whole_match_0525_cam0/team_tracking.jsonl" "./runs/detect/28112025_whole_match_0525_cam1/team_tracking.jsonl" "./runs/detect/28112025_whole_match_0525_cam2/team_tracking.jsonl" "./runs/detect/28112025_whole_match_0525_cam3/team_tracking.jsonl"     --image-path "./data/images/mongkok_football_field.png"     --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 13 23 25 8 14 17 18 21 24 31 33 34     --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 1 17 22 23 24 28 33 42 43 44 72 88     --output-name './runs/detect/1205_track_matching_merged_filtered/team_tracking_output'


python3 player_track_identification.py  --json-path "./runs/detect/1215_track_matching_whole_match/track_matching_results_1215_cam0_1_2_3_savgol_fused.jsonl"  --image-path "./data/images/mongkok_football_field.png"     --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 33 31 14 17 24     --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 22 33 17 28 42     --output-name './runs/detect/1205_track_matching_merged_filtered/team_tracking_output'
python fuse_ball_tracks.py \
  "./runs/detect/28112025_whole_match_0525_cam0/ball_tracking_fused_processed.jsonl" \
  "./runs/detect/28112025_whole_match_0525_cam1/ball_tracking_fused_processed.jsonl" \
  "./runs/detect/28112025_whole_match_0525_cam2/ball_tracking_fused_processed.jsonl" \
  "./runs/detect/28112025_whole_match_0525_cam3/ball_tracking_fused_processed.jsonl" \
  -o "./runs/detect/1212_ball_four_in_one_whole_match/ball_tracking_final_cam_0_to_3_fused_processed_median.jsonl" 

python cluster_tracks.py --player-jsonl-paths "./runs/detect/28112025_whole_match_0525_cam0/team_tracking_merged_filtered.jsonl" "./runs/detect/28112025_whole_match_0525_cam1/team_tracking_merged_filtered.jsonl" "./runs/detect/28112025_whole_match_0525_cam2/team_tracking_merged_filtered.jsonl" "./runs/detect/28112025_whole_match_0525_cam3/team_tracking_merged_filtered.jsonl" --ball-jsonl-paths "./runs/detect/28112025_whole_match_0525_cam0/ball_tracking_fused_processed.jsonl" "./runs/detect/28112025_whole_match_0525_cam1/ball_tracking_fused_processed.jsonl" "./runs/detect/28112025_whole_match_0525_cam2/ball_tracking_fused_processed.jsonl" "./runs/detect/28112025_whole_match_0525_cam3/ball_tracking_fused_processed.jsonl" --auto-calibrate --min-overlap 50 --median-distance-threshold 50 --total-distance-diff-threshold 50 --direction-threshold 0.3 --direction-consistency-threshold 0.6 --max-frames 1500 --direction-frame-stride 3 --max-score 100 --spatial-distance 15 -o "./runs/detect/1219_testing/team_tracking_4in1_fixed_init_pos_error.jsonl


# python3 post_processing-ball.py   \
#   --json-path "./runs/detect/1124_test_4k_player_1280_half_cam3/ball_tracking.jsonl"   \
#   --image-path "./data/images/mongkok_football_field.png"   \
#   --output-name './runs/detect/1124_test_4k_player_1280_half_cam3/ball_tracking_output'
# python3 ./tools/combine_team_ball_tracks.py   \
#   --player-jsonl "./runs/detect/1124_test_4k_player_1280_half_cam3/team_tracking_final.jsonl"   \
#   --ball-jsonl "./runs/detect/1124_test_4k_player_1280_half_cam3/ball_tracking_final.jsonl"   \
#   --output-jsonl "./runs/detect/1124_test_4k_player_1280_half_cam3/team_ball_tracking_final.jsonl"
# # imgsz = 640
# # 172 1104 2101 895 3800 1021 3458 2057
# python mini_patch_detect_v2_for_video.py   --source './data/video/period_57873_58952_cam0.mp4'   --game-time 0 0 0 36   --img 640   --device 0   --weights './weights/yolov9-s-ball-player-24112025-converted.pt'   --name 1124_test_4k_player_640_half_cam0   --classes 0 1   --clothes-folder-path ./data/histograms/0525/   --homography-src-points 172 1104 2101 895 3800 1021 3458 2057   --homography-dst-points 530 0 530 660 1060 660 1060 0   --jersey-weights ./weights/parseq_epoch\=24-step\=2575-val_accuracy\=95.6044-val_NED\=96.3255_new.ckpt --half
# # 366 1910 0 860 1770 748 3780 1000 (bottom left top left top middle bottom middle)
# python mini_patch_detect_v2_for_video.py   --source './data/video/period_57873_58952_cam1.mp4'   --game-time 0 0 0 36   --img 640   --device 0   --weights './weights/yolov9-s-ball-player-24112025-converted.pt'   --name 1124_test_4k_player_640_half_cam1   --classes 0 1   --clothes-folder-path ./data/histograms/0525/   --homography-src-points 366 1910 0 860 1770 748 3780 1000   --homography-dst-points 0 0 0 660 530 660 530 0   --jersey-weights ./weights/parseq_epoch\=24-step\=2575-val_accuracy\=95.6044-val_NED\=96.3255_new.ckpt --half
# # 141 721 2064 613 3735 826 3225 1641 (top right bottom right bottom middle top middle)
# python mini_patch_detect_v2_for_video.py   --source './data/video/period_57873_58952_cam2.mp4'   --game-time 0 0 0 36   --img 640   --device 0   --weights './weights/yolov9-s-ball-player-24112025-converted.pt'   --name 1124_test_4k_player_640_half_cam2   --classes 0 1   --clothes-folder-path ./data/histograms/0525/   --homography-src-points 141 721 2064 613 3735 826 3225 1641   --homography-dst-points 1060 660 1060 0 530 0 530 660   --jersey-weights ./weights/parseq_epoch\=24-step\=2575-val_accuracy\=95.6044-val_NED\=96.3255_new.ckpt --half
# # 266 1776 119 872 1799 872 3737 1231 (top middle bottom middle bottom left top left)
# python mini_patch_detect_v2_for_video.py   --source './data/video/period_57873_58952_cam3.mp4'   --game-time 0 0 0 36   --img 640   --device 0   --weights './weights/yolov9-s-ball-player-24112025-converted.pt'   --name 1124_test_4k_player_640_half_cam3   --classes 0 1   --clothes-folder-path ./data/histograms/0525/   --homography-src-points 266 1776 119 872 1799 872 3737 1231   --homography-dst-points 530 660 530 0 0 0 0 660   --jersey-weights ./weights/parseq_epoch\=24-step\=2575-val_accuracy\=95.6044-val_NED\=96.3255_new.ckpt --half