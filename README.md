# YOLO-v9 project for Football pipeline 

## Installation
We provide the `Dockerfile` and `docker-compose.yml` for building the docker image used for this project.
```{shell}
docker-compose up --build
docker exec -it yolov9_football bash
```

## Training
We use the codes from [WongKinYu/yolov9](https://github.com/WongKinYiu/yolov9/tree/main) to train our detection model. See `scripts/train.sh` for more details. The dataset is downloaded from [Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg). We use yolov9-s pretrained weight for training.

```{shell}
python train_dual.py \
  --workers 8 \
  --device 0 \
  --batch 16 \
  --data datasets/football-ball-detection-2/data.yaml \
  --img 1280 \
  --cfg '' \
  --weights "./weights/yolov9-s.pt" \
  --name yolov9-s_player_ball_detection_1280 \
  --hyp hyp.scratch-high.yaml \ 
  --min-items 0  \
  --epochs 50  \
  --close-mosaic 15
```

Note: for the `./data/hyps/hyp.scratch-high.yaml`, change the value of `copy_paste` into 0.

Reparameterize the trained model weight for inference.

```{shell}
python ./reparam-yolov9.py \
  --config ./models/detect/gelan-s.yaml \
  --checkpoint "./weights/yolov9-s_player_ball_detection_1280.pt" \
  --classes 1 \
  --output "./weights/yolov9-s-converted_player_ball_detection_1280.pt"
```

## Whole Match Pipeline
This pipeline detects players and ball in the field, postprocess player tracks and ball tracks, and analyse players behaviour based on tracking data.

We include a script for the whole match pipeline. See `scripts/inference.sh` for more details.
```{shell}
bash scripts/inference.sh
```

### 1️⃣ Inference
The jersey model weight is downloaded from [here](https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view), provided by [
mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline). Put this weight under `./weights` folder. In case the checkpoint does not match the model state_dict, run this command:
```{shell}
python3 ./tools/convert_parseq_weight.py \
  "./weights/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt" \
  "./weights/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255_new.ckpt"
```

Prepare the clothes histogram data for all teams in the match, including away, awaygoalkeeper, home, homegoalkeeper, and referee team.

Note: The file name must be in the format of `teamA_01.npy`, and `teamA` will be the team name in process. You may add more clothes data of the same team, named after `teamA_xx.npy`.

```{shell}
python3 identify_player_team.py \
  --image ./data/images/0525/homegoalkeeper_01.jpg \
  --histogram_save_path ./data/histograms/0525/homegoalkeeper_01.npy
```

The folder storing the histogram data should look like this:

```text
histograms
├── 0525/
│   ├── home_01.npy
│   ├── homegoalkeeper_01.npy
│   ├── away_01.npy
│   ├── awaygoalkeeper_01.npy
│   ├── referee_01.npy
│   ├── ...
```


Run detection on the whole match clip, and output detection results in JSONL format. The output would be two JSONL files, `team_tracking.jsonl` and `ball_tracking.jsonl` in the output folder.
```{shell}
# Player and Ball Detection
python3 mini_patch_detect_v2_for_video.py \
  --source './data/video/test_sample/C0478.MP4' \
  --game-time 317 3085 3982 6809 \
  --img 640 \
  --device 0 1 \
  --weights './weights/yolov9-s-converted_player_ball_detection_1280.pt' \
  --name test_4k_640_cam0 \
  --classes 0 \
  --clothes-folder-path ./data/histograms/0525/ \
  --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 \
  --homography-dst-points 530 0 530 660 1060 660 1060 0 \
  --jersey-weights ./weights/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt \
  --nosave
  --half
```

- Note: the input order of the homography-src-points and homography-dst-points matter, and wrong order can lead to wrong homographic projection.
- For multiple camera processing, run the above command with changes on the `source, game-time, name, homography-src-points, homography-dst-points`.

The folder storing the otuput jsonl data should look like this:

```text
./runs
├── detect/
│   ├── test_4k_640_cam0
│   │   ├── team_tracking.jsonl
│   │   └── ball_tracking.jsonl
│   │
│   ├── test_4k_640_cam1
│   │   ├── team_tracking.jsonl
│   │   └── ball_tracking.jsonl
│   │
│   ├── test_4k_640_cam2
│   │   ├── team_tracking.jsonl
│   │   └── ball_tracking.jsonl
│   │
│   ├── test_4k_640_cam3
│   │   ├── team_tracking.jsonl
│   │   └── ball_tracking.jsonl
```

### 2️⃣ Postprocess of Tracks
The raw detection results (JSONL files) are processed in `post_processing_player.py` and `post_processing_ball.py`. Both scripts would output several intermediate JSONL files. The final JSONL file to use is named `team_tracking_final.jsonl` and `ball_tracking_final.jsonl`.

#### Ball tracks

Step 1: Run this program for all ball tracks jsonl separately.
```{shell}
python3 post_processing_ball.py \
  --json-paths \
    "./runs/detect/test_4k_640_cam0/ball_tracking.jsonl" \
    "./runs/detect/test_4k_640_cam1/ball_tracking.jsonl" \
    "./runs/detect/test_4k_640_cam2/ball_tracking.jsonl" \
    "./runs/detect/test_4k_640_cam3/ball_tracking.jsonl" \
  --image-path "./data/images/mongkok_football_field.png" \
```

Step 2: Run this once to apply the fusion of ball tracks.
```{shell}
python fuse_ball_tracks.py \
  --ball-jsonl-paths \
    "./runs/detect/test_4k_640_cam0/ball_tracking_processed.jsonl" \
    "./runs/detect/test_4k_640_cam1/ball_tracking_processed.jsonl" \
    "./runs/detect/test_4k_640_cam2/ball_tracking_processed.jsonl" \
    "./runs/detect/test_4k_640_cam3/ball_tracking_processed.jsonl" \
  --output "./runs/detect/test_4k_640/ball_tracking_fused.jsonl"
```

Note: after creating the fused ball tracks, we apply some postprocessing in the same program and the final output ball tracks jsonl to be used is `ball_tracking_fused_final.jsonl`

#### Player tracks
Step 1: Run this program for all player tracks jsonl.
```{shell}
python3 post_processing_player.py \
  --json-paths \
    "./runs/detect/test_4k_640_cam0/team_tracking.jsonl" \
    "./runs/detect/test_4k_640_cam1/team_tracking.jsonl" \
    "./runs/detect/test_4k_640_cam2/team_tracking.jsonl" \
    "./runs/detect/test_4k_640_cam3/team_tracking.jsonl" \
  --image-path "./data/images/mongkok_football_field.png" \
  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 33 31 14 17 24  \
  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 22 33 17 28 42
```

Step 2: Run this once to apply the fusion of player tracks. The input file name of the player track jsonl should be `team_tracking_merged_filtered.jsonl`, while the input file name of the ball track jsonl should be `ball_tracking_processed.jsonl`.

```{shell}
python cluster_player_tracks.py \
  --player-jsonl-paths \
    "./runs/detect/test_4k_640_cam0/team_tracking_merged_filtered.jsonl" \
    "./runs/detect/test_4k_640_cam1/team_tracking_merged_filtered.jsonl" \
    "./runs/detect/test_4k_640_cam2/team_tracking_merged_filtered.jsonl" \
    "./runs/detect/test_4k_640_cam3/team_tracking_merged_filtered.jsonl" \
  --ball-jsonl-paths \
    "./runs/detect/test_4k_640_cam0/ball_tracking_processed.jsonl" \
    "./runs/detect/test_4k_640_cam1/ball_tracking_processed.jsonl" \
    "./runs/detect/test_4k_640_cam2/ball_tracking_processed.jsonl" \
    "./runs/detect/test_4k_640_cam3/ball_tracking_processed.jsonl" \
  --auto-calibrate \
  --use-dtw-filter \
  --output "./runs/detect/test_4k_640/team_tracking_fused.jsonl" \
```

Step 3: Run this once to postporcess the fused player tracks.
```{shell}
python3 player_track_identification.py  \
  --json-path "./runs/detect/test_4k_640/team_tracking_fused.jsonl"  \
  --image-path "./data/images/mongkok_football_field.png"     \
  --home-jersey-numbers 1 2 3 4 7 10 11 16 20 27 30 33 31 14 17 24  \
  --away-jersey-numbers 26 2 6 7 9 16 20 30 36 77 99 22 33 17 28 42

```
This will give the `team_tracking_fused_final.jsonl`.

After that run this command to combine both JSONL files into one. The final JSONL file to use is named `team_ball_tracking_final.jsonl`.
```{shell}
python3 ./tools/combine_team_ball_tracks.py \
  --player-jsonl "./runs/detect/test_4k_640/team_tracking_fused_final.jsonl" \
  --ball-jsonl "./runs/detect/test_4k_640/ball_tracking_fused_final.jsonl" \
  --output-jsonl "./runs/detect/test_4k_640/team_ball_tracking_final.jsonl"
```

### 3️⃣ Analyse and Visualise Detection and Tracking Result
Compute the .npy file required for the homographic transformation:

```{shell}
python3 ./tools/extract_homography_matrix.py \
  --src "172,1104 2101,895 3800,1021 3458,2057" \
  --dst "530,0 530,660 1060,660 1060,0" \
  --out "./weights/homography_cam0.npy"
```

Prepare your jsonl file from previous step and run this command:
```{shell}
python3 ./tools/render_classification_output.py \
  --jsonl-path "./runs/detect/test_4k_640/team_ball_tracking_final.jsonl" \
  --video-paths \
    './data/video/test_sample/cam0.mp4' \
    './data/video/test_sample/cam1.mp4' \
    './data/video/test_sample/cam2.mp4' \
    './data/video/test_sample/cam3.mp4' \
  --bg-img-path ./data/images/mongkok_football_field.png \
  --output-dir './runs/detect/test_4k_640/suspicious-output' \
  --game-time 317 3085 3982 6809 450 3218 4115 6942 798 3563 4460 7287 872 3640 4537 7364 \
  --homography \
    "./weights/homography_cam0.npy" \
    "./weights/homography_cam1.npy" \
    "./weights/homography_cam2.npy" \
    "./weights/homography_cam3.npy"
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
  --jsonl-path "./runs/detect/test_4k_640/team_ball_tracking_final.jsonl" \
  --frame-interval 30 \
  --fps 29.97 \
  --comparison-split 0.85
```
This will output a csv file called `player_statstics.csv` to the same directory of the jsonl file.

Player statistics include team, jersey_num, total_distance (m), avg_speed (kmh), max_speed (kmh), min_speed (kmh), distance_ratio, avg_speed_ratio, max_speed_ratio, min_speed_ratio.


`comparison-split`: for example, setting to $0.85$ means that the ratios are calculated as follows:

$$ \text{average speed first part} = \text{average spped of first 85 percent of the tracking data}$$
$$ \text{average speed second part } = \text{average speed of the remaining 15 percent of the tracking data}$$
$$ \text{average speed ratio} = \frac{\text{average speed first part}}{\text{average speed second part}}$$

To generate position heatmap of a player, run the following command:

```{shell}
python3 ./tools/generate_player_heatmap.py 
  --jsonl-path "./runs/detect/test_4k_640/team_ball_tracking_final.jsonl" \
  --bg-img-path "./data/images/mongkok_football_field.png" \
  --jersey_number "16" \
  --team "home" # home, homegoalkeeper, away, awaygoalkeeper      
```
Change the value of `jersey-number` and `team` for selecting the specific player. This outputs a png file named after the jersey number and team, e.g. `home_16_heatmap.png`, on the same directory of the jsonl file.

### Summary
- Input Clips → mini_patch_detect_v2_for_video.py → player detections and tracks (JSONL) + ball detections and tracks (JSONL) 

- Ball detections and tracks → post_processing_ball.py → refined ball detections and tracks (JSONL)
- 4 ball JSONL → fuse_ball_tracks.py → final ball JSONL

- Player detections and tracks → post_processing_player.py → refined player detections and tracks (JSONL)
- 4 player JSONL + 4 ball JSONL → cluster_player_tracks.py → 1 player JSONL → player_track_identification.py → final player JSONL

- Final player JSONL + Final ball JSONL → combine_team_ball_tracks.py → whole match tracks (JSONL)

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

## TODO:
- [ ] Train a YOLO model for detecting both players and balls
- [ ] Combine four camera views into one overall JSONL file
