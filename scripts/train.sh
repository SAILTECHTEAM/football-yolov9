#!/bin/bash
# Player detection model we directly use pretrained weights from COCO
# YOLO9-s model summary: 1219 layers, 9745688 parameters, 9745656 gradients, 39.6 GFLOPs (with PGI)
#python train_dual.py --workers 8 --device 0 --batch 4 --data datasets/football-players-detection-10/data.yaml --img 1280 --cfg '' --weights pretrained/yolov9-s.pt --name yolov9-s --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

#python val.py --data datasets/football-players-detection-10/data.yaml --img 1280 --batch 8 --conf 0.001 --iou 0.7 --device 0 --weights './yolov9-s-converted.pt' --save-json --name yolov9_s_c_1280_val

#python detect.py --source './videos/C0478.MP4' --img 1280 --device 0 --weights './yolov9-s-converted.pt' --name yolov9_s_c_1280_detect

# Inference with slicer
#python detect.py --source './videos/C0478.MP4' --img 1280 --device 0 --weights './checkpoint/yolov9-s-converted_player_detection_1280_20250821.pt' --use-slicer --slice-size 1280 1280 --nms-threshold 0.1 --name yolov9_s_c_1280_slicer_detect


# Ball detection
python train_dual.py --workers 8 --device 0 --batch 8 --data datasets/football-ball-detection-2/data.yaml --img 1280 --cfg '' --weights pretrained/yolov9-s.pt --name yolov9-s-ball-detection-1280 --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15

# Inference with slicer
python detect.py --source './videos/C0478.MP4' --img 1280 --device 0 --weights './checkpoint/yolov9-s-converted_ball_detection_1280.pt' --use-slicer --slice-size 1280 1280 --nms-threshold 0.1 --name yolov9_s_c_1280_slicer_detect_


