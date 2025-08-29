# Example usage:
# python3 mini_patch_detect_v1_for_video.py --source './data/video/test_sample/C0478.MP4' --game-time 317 3085 3982 6809 --img 640 --device 0 --weights './weight/yolov9-s-converted.pt' --name test_4k --classes 0 32 --clothes-folder-path ./data/histograms/0525/ --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 --homography-dst-points 530 0 530 660 1060 660 1060 0 --nosave

# yolov9-s weights trained on football player detection dataset (not used as pretrained coco weight is better)

# python3 mini_patch_detect_v1_for_video.py --source '../videos/period_57873_58952_cam0.mp4'  --img 1280 --device 0 --weights '../checkpoint/yolov9-s-converted_ball_detection_1280_20250822.pt' --name test_4k  --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 --homography-dst-points 530 0 530 660 1060 660 1060 0 --nosave
# output: Processing game-time frames:  13%|▉      | 4/30 [00:07<00:45,  1.73s/frame, pre=0.00s, inf=1.57s, nms=0.00s, proc=0.01s, trk=0.00s, reid=0.00s, json=0.00s, draw=0.06s, total=1.64s (time calculated according to dt[i])

# Inference on player detection and tracking
python3 mini_patch_detect_v1_for_video_original.py --source './data/video/test_sample/period_57873_58952_cam0.mp4' --game-time 0 0 0 36 --img 640 --device 0 --weights './weight/yolov9-s-converted.pt' --name test_4k_player_640 --classes 0 --clothes-folder-path ./data/histograms/0525/ --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 --homography-dst-points 530 0 530 660 1060 660 1060 0 --nosave

# Run post-processing on player tracks
python3 post-processing.py --json-path "./runs/detect/test_4k_player_640_numba_20250829_final/team_tracking.jsonl" --image-path "./data/images/mongkok_football_field.png" --output-name './runs/detect/test_4k_player_640_numba_20250829_final/team_tracking_output'

python3 render_specific_time_period.py
### Changelog
# person_tracker sv.ByteTrack
# sv.InferenceSlicer to replace get_images_patches, crop_image_with_overlap, simple_global_nms 
# (differences in patches: sv.InferenceSlicer gets the image patches on the edges by padding to the slice_size)
# (original method collects the patches near the edge by adjusting the last offest to fit the edge of the high resolution image)