# YOLO-v9 project for Football pipeline 

## Installation
We provide the `Dockerfile` and `docker-compose.yml` for building the docker image used for this project.
```{shell}
docker-compose up --build
docker exec -it yolov9_football bash
```

## Training
We use the codes from [WongKinYu/yolov9](https://github.com/WongKinYiu/yolov9/tree/main) to train our ball detection model. For player detection model, we directly use the model weight trained on MS COCO provided by the above repo. Both model architecture are based on YOLOv9-S.

## Whole Match Pipeline
This pipeline detects players and ball in the field, postprocess player tracks and ball tracks, and analyse players behaviour based on tracking data.

We include a script for the whole match pipeline. See `scripts/inference.sh` for more details.
```{shell}
bash scripts/inference.sh
```

### 1️⃣ Inference
Run detection on the whole match clip, and output detection results in JSONL format. Currently 1 JSONL file for player detection results, and 1 JSONL file for ball detection results. The jersey model weight is downloaded from [here](https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view). In case the checkpoint does not match the model state_dict, run this command:
```{shell}
python3 ./old_function/convert_parseq_weight.py \
  --old_ckpt "./weights/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt" \
  --new_ckpt "./weights/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255_new.ckpt"
```

```{shell}
# Player detection
python3 mini_patch_detect_v1_for_video.py \
  --source './data/video/test_sample/C0478.MP4' \
  --game-time 317 3085 3982 6809 \
  --img 640 \
  --device 0 \
  --weights './weight/yolov9-s-converted.pt' \
  --name test_4k_player_640 \
  --classes 0 32 \
  --clothes-folder-path ./data/histograms/0525/ \
  --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 \
  --homography-dst-points 530 0 530 660 1060 660 1060 0 \
  --jersey-weights ./weights/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt \
  --nosave
```

```{shell}
# Ball detection
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
```

### 2️⃣ Postprocess of Tracks
The raw detection results are processed in `post-processing.py` and `post-processing-ball.py` sequentially. Both scripts would output several intermediate JSONL files. The final JSONL file to use is named `team_tracking_final.jsonl`.

```{shell}
python3 post-processing.py \
  --json-path "./runs/detect/test_4k_player_640/team_tracking.jsonl" \
  --image-path "./data/images/mongkok_football_field.png" \
  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 13 23 25 8 14 17 18 21 24 31 33 34 \
  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 1 17 22 23 24 28 33 42 43 44 72 88 \
  --output-name './runs/detect/test_4k_player_640/team_tracking_output'
```

```{shell}
python3 post-processing-ball.py \
  --json-path "./runs/detect/test_4k_ball_640/ball_tracking.jsonl" \
  --image-path "./data/images/mongkok_football_field.png" \
  --output-name './runs/detect/test_4k_ball_640/ball_tracking_output'
```
After that run this command to combine both JSONL files into one. The final JSONL file to use is named `team_tracking_final.jsonl`.
```{shell}
python3 combine_team_ball_tracks.py \
  --player-jsonl "./runs/detect/test_4k_player_640/team_tracking_final.jsonl" \
  --ball-jsonl "./runs/detect/test_4k_ball_640/ball_tracking_final.jsonl" \
  --output-jsonl "./runs/detect/test_4k_player_640/team_ball_tracking_final.jsonl"
```

### 3️⃣ Analyse and Visualise Detection and Tracking Result
Compute the .npy file required for the homographic transformation:

```{shell}
python3 ./tools/extract_homography_matrix.py \
  --src "172,1104 2101,895 3800,1021 3458,2057" \
  --dst "530,0 530,660 1060,660 1060,0" \
  --out "./weight/homography_matrix_whole_match.npy"
```

Prepare your jsonl file from previous step and run this command:
```{shell}
python3 ./tools/render_classification_output.py \
  --jsonl-path "./runs/detect/test_4k_player_640/team_ball_tracking_final.jsonl" \
  --video-paths './data/video/test_sample/C0478.MP4' \
  --bg-img-path ./data/images/mongkok_football_field.png \
  --output-dir './runs/detect/test_4k_player_640/suspicious-output' \
  --game-time 317 3085 3982 6809 \
  --homography './weight/homography_matrix_whole_match.npy'
```
The output contains a radar view image, a radar view video of the suspicious player plotted on 2D field, and the footage of the specific time period of suspicious action. Each suspicious output is stored in a folder named after the track id of the player (e.g. 16a).

```
Classify players' behaviour near the ball into these cases:
- Class 0: Normal. Not classified as any below classes, or not near the ball
- Class 1: Angle between player track and ball track exceeds threshold (in degree) for a period of time
- Class 2: Velocity of player movement is smaller than a threshold, while the ball is possessed by opposing team
- Class 3: Coordinates of player on the field remain the same with buffer for a period of time
- Class 4: Change of team possession of the ball from this player to opponent with a distance threshold in an instance
- Class 5: The coordinates of the ball are outside the boundaries of the field, when the ball is possessed by this player
- Class 6: The player not in possession is in a close distance to the ball possessed by opponent team, and the distance between player and opponent in possession does not decrease for a period of time
- Class 7: Average velocity of the player is lower than a threshold based on the average velocity of other players, for a period of time
- Class 8: Ball position remains unchagned with buffer while possessed by the player, for a long period of time
- Class 9: Change of team possession of the ball from this player to opponent more than a number of times

```

For the statistics of each player, run the following command:

```{shell}
python3 ./tools/calculate_player_statistics.py 
  --jsonl-path "./runs/detect/test_4k_player_640/team_ball_tracking_final.jsonl" \
  --frame-interval 30 \
  --fps 29.97 \
  --comparison-split 0.85
```
This will output a csv file called `player_statstics.csv` to the same directory of the jsonl file.

Player statistics include team, jersey_num, total_distance (m), avg_speed (kmh), max_speed (kmh), min_speed (kmh), distance_ratio, avg_speed_ratio, max_speed_ratio, min_speed_ratio.


`comparison-split`: for example, setting to $0.85$ means that the ratios are calculated as follows:

$$ \text{average speed first part} = \text{average spped of first } 85\% \text{ of the tracking data}$$
$$ \text{average speed second part } = \text{average speed of the remaining } 15\% \text{ of the tracking data}$$
$$ \text{average speed ratio} = \frac{\text{average speed first part}}{\text{average speed second part}}$$

To generate position heatmap of a player, run the following command:

```{shell}
python3 ./tools/generate_player_heatmap.py 
  --jsonl-path "./runs/detect/test_4k_player_640/team_ball_tracking_final.jsonl" \
  --bg-img-path "./data/images/mongkok_football_field.png" \
  --jersey_number "16" \
  --team "home" # home, homegoalkeeper, away, awaygoalkeeper      
```
Change the value of `jersey-number` and `team` for selecting the specific player. This outputs a png file named after the jersey number and team, e.g. `home_16_heatmap.png`, on the same directory of the jsonl file.

### Summary
- Input Clips → mini_patch_detect_v1_for_video.py → player detections and tracks (JSONL)
- Player detections and tracks → post-processing.py → refined player detections and tracks (JSONL)

- Input Clips → mini_patch_detect_ball_for_video.py → ball detections (JSONL)
- Ball detections → post-processing.py → refined ball detections and tracks (JSONL)

- Player JSONL + Ball JSONL → combine_team_ball_tracks.py → whole match tracks (JSONL)

- Whole match JSONL → render_classification_output.py → folder of suspicious-output

- Whole match JSONL → calculate_player_statistics.py → player_statistics.csv

- Whole match JSONL → generate_player_heatmap.py → heatmap.png

Here let say player 16a with jersey number 8 from home team has been detected suspicious on Class 1 and 7. For the naming format, the first 3 digits combined is the minute, the last 2 digits combined is the second (00016 is 000:16, 10545 is 105:45).
```text
.
├── suspicious-output/
│   ├── home_8
│   │   ├── class_1/
│   │   │   ├── 16a_home_8_00016_00036.mp4
│   │   │   ├── 16a_home_8_00016_00036.png
│   │   │   └── 16a_home_8_cam0_00016_00036.mp4
│   │   ├── class_7/
│   │   │   ├── 16a_home_8_10511_10545.mp4
│   │   │   ├── 16a_home_8_10511_10545.png
│   │   │   └── 16a_home_8_cam0_10511_10545.mp4
│   ├── away_30/
│   │   ├── ...
```



## Goalkeeper Pipeline

🧤 Goalkeeper Behavior Classification Pipeline

This pipeline extracts goalkeeper clips, tracks players and the ball, and classifies goalkeeper behaviors using homography and radar speed data.


Prepare Input Videos

Select all clips of the goalkeeper (e.g., first half or second half).

⚠️ GoPro automatically splits recordings into multiple clips (GX010025.mp4, GX020025.mp4, …). Put all clips belonging to the same session into a single folder. The file names must be exactly the same as this format.

1️⃣ Compute Homography Matrix

Before running detection, compute the homography matrix that maps video coordinates to real-world goal coordinates (top-left, top-right, bottom-right, bottom-left).

```{shell}
python3 ./tools/extract_homography_matrix.py \
  --src "86,242 1658,258 1644,766 100,771" \
  --dst "0,0 640,0 640,213 0,213" \
  --out ./runs/detect/demo_video/homography_matrix.npy
```

This will print and save a file like:

Homography Matrix:
 [[ 2.45e+00 -1.20e-02 -2.11e+02]
  [ 2.13e-02  1.60e+00 -3.50e+02]
  [ 2.70e-05 -1.20e-05  1.00e+00]]
Saved matrix to ./runs/detect/demo_video/homography_matrix.npy

2️⃣ Detect Goal Clips

Run detection on all goalkeeper video clips (e.g., GoPro splits them into GX010025, GX020025, etc.).
The script will filter goal clips and output detection results in JSONL format.
```{shell}
python3 detect_goal_v2.py \
  --weights "./weight/yolov9-s-converted.pt" \
  --source "./data/video/GX010025_clips/" \
  --name "demo_video" \
  --nosave \
  --radar_data_path ./data/excel/PR_20250208_1739_session.csv \
  --homography_path ./runs/detect/demo_video/homography_matrix.npy \
  --use-tqdm
```
3️⃣ Track Players and Ball

Before running the next codes, remember to download the pretrained weight of ViTPose from the official repo ([link](https://1drv.ms/u/s!AimBgYV7JjTlgSbHyN2mjh2n2LyG?e=y0FgMK))

Next, run `detections_to_tracks_and_scores.py` on the folder of detection JSONLs.

Uses ByteTrack to track all detected players.

Tracks the ball by nearest detection (fused track).

Assigns team scores using goalkeeper histogram .npy.

Outputs:

xxxxx_clip_00x.cls_0.tracks.jsonl → tracked players (with skeleton + team score).

xxxxx_clip_00x.cls_32.tracks.jsonl → tracked ball.
```{shell}
python3 detections_to_tracks_and_scores.py \
  --clips-root ./runs/detect/demo_video/clips/0025/ \
  --hist ./data/histograms/gk01.npy
```
4️⃣ Render Tracks on Video

Use render_track_on_video.py to visualize results.

Draws skeletons on the goalkeeper.

Draws bounding boxes for other players and the ball.

Works with a root folder containing all clips and JSONLs, or a single video + its JSONLs.

Example:
```{shell}
python3 render_track_on_video.py \
  --root ./runs/detect/demo_video/clips/0025/
```
5️⃣ Classify Goalkeeper Behaviors

Finally, classify goalkeeper behaviors using the tracks, radar speed, and homography matrix.
The script processes all clips in a folder and prints tags or saves per-clip JSON results.
```{shell}
python3 analyze_goalkeeper_behavior.py \
  --root ./runs/detect/demo_video/clips/0025/ \
  --speeds 30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45 \
  --out-dir ./runs/detect/demo_video/analysis \
  --homography ./runs/detect/demo_video/homography_matrix.npy
```

Output example (XXX_clip_001.analysis.json):

{
  "warped": true,
  "gk_tid": 1,
  "ball_tid": -1,
  "fps": 29.97,
  "ball_speed": 30.0,
  "speed_units": "km/h",
  "tags": [
    0
  ],
  "clip_id": "GX010025_clip_001",
  "people_tracks_jsonl": "runs/detect/demo_video/clips/0025/GX010025_clip_001.cls_0.tracks.jsonl",
  "ball_tracks_jsonl": "runs/detect/demo_video/clips/0025/GX010025_clip_001.cls_32.tracks.jsonl"
}


    Classifies goalkeeper behavior for three cases:
    - Class 0 : No any below class detected.
    - Class 1 & 2 & 11: Ball is far from skeleton, and goalkeeper's movement is limited.
    - Class 3: Goalkeeper's last 5-frame average center is farther from the ball than the first 5-frame average center.
    - Class 4 & 12 : Ball above the skeleton ear point but low or not jump to catch the ball.
    - Class 5: Ball above the skeleton ear point but low or not raise elbow above ear to catch the ball.
    - Class 6: Elbow angle is below the threshold (degrees) when the ball is in the shoulder-centered area.
    - Class 7: Elbow angle is decreasing over time when the ball is in the shoulder-centered area.
    - Class 8: Ball speed is below the threshold (km/h).
    - Class 9: low speed & ball is in the shoulder-centered area.
    - Class 10: Ball is within the area formed by skeleton points.


👉 Summary of Workflow

Input Clips → detect_goal_v2.py → detections (JSONL)

Detections → detections_to_tracks_and_scores.py → tracks (players + ball)

Tracks → render_track_on_video.py → video with overlays

Tracks + Radar Speeds → analyze_goalkeeper_behavior.py → behavior tags
