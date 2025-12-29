import argparse
import os
import string
import sys
from pathlib import Path
import numpy as np
import supervision as sv
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import json
import time
from tqdm import tqdm
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend

from utils.general import (
    Profile,
    check_img_size,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
)
from utils.torch_utils import select_device, smart_inference_mode
from collections import defaultdict
from numba import njit
from tools.extract_homography_matrix import (
    compute_homography,
    apply_homography_to_point,
)
from tools.identify_player_team import (
    extract_color_histogram_with_specific_background_color,
    match_histograms_to_teams,
    load_team_histograms_from_folder,
)

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


def crop_clothing_region(
    image, bbox, top_ratio=0.25, bottom_ratio=0.45, left_ratio=0.3, right_ratio=0.7
):
    """
    Crop only the clothing region (centered shirt area) from a person bounding box.

    Args:
        image: input image (NumPy array)
        bbox: (x1, y1, x2, y2)
        top_ratio: vertical start (0 = top, 1 = bottom)
        bottom_ratio: vertical end
        left_ratio: horizontal start (0 = left, 1 = right)
        right_ratio: horizontal end

    Returns:
        cropped_image: central shirt region
    """
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
    new_x2 = min(new_x2, image.shape[1])
    new_y1 = max(new_y1, 0)
    new_y2 = min(new_y2, image.shape[0])

    cropped = image[new_y1:new_y2, new_x1:new_x2]
    return cropped


# ----------------------
@njit
def _remove_enclosed_numba(dets, area_thresh, containment_thresh):
    N = dets.shape[0]
    keep = np.ones(N, dtype=np.bool_)

    for i in range(N):
        if not keep[i]:
            continue
        box_i = dets[i, :4]
        cls_i = int(dets[i, 5])
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])

        for j in range(N):
            if i == j or not keep[j]:
                continue
            box_j = dets[j, :4]
            cls_j = int(dets[j, 5])
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            if cls_i != cls_j:
                continue

            # Determine small and large
            if area_i < area_j:
                small_idx, large_idx = i, j
            else:
                small_idx, large_idx = j, i

            box_small = dets[small_idx, :4]
            box_large = dets[large_idx, :4]
            small_area = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])
            large_area = (box_large[2] - box_large[0]) * (box_large[3] - box_large[1])

            xA = max(box_small[0], box_large[0])
            yA = max(box_small[1], box_large[1])
            xB = min(box_small[2], box_large[2])
            yB = min(box_small[3], box_large[3])
            inter_w = max(0, xB - xA)
            inter_h = max(0, yB - yA)
            inter_area = inter_w * inter_h

            containment = inter_area / (small_area + 1e-6)
            area_ratio = small_area / (large_area + 1e-6)

            if containment >= containment_thresh and area_ratio <= area_thresh:
                keep[small_idx] = False

    return keep


def remove_boxes_with_numba(
    detections: torch.Tensor, area_ratio_thresh=0.6, containment_thresh=0.9
) -> torch.Tensor:
    if len(detections) < 2:
        return detections

    det_np = detections.cpu().numpy()
    keep_mask = _remove_enclosed_numba(det_np, area_ratio_thresh, containment_thresh)
    return detections[keep_mask]


def draw_detections(image, detections, class_names, color=(0, 255, 0)):
    for det in detections:
        if len(det) == 6:
            x1, y1, x2, y2, conf, cls = det
            track_id = None
        else:
            x1, y1, x2, y2, conf, cls, track_id, projected_position = det

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        label = f"{class_names[cls]} {conf:.2f}"
        if track_id is not None:
            label += f" ID:{int(track_id)}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def parse_time_str(time_str: str) -> float:
    """Convert hh:mm:ss string to seconds."""
    h, m, s = map(int, time_str.strip().split(":"))
    return h * 3600 + m * 60 + s


class FrameJsonlStreamer:
    """
    Streams per-frame data to disk as JSONL.
    Each frame with ball detection is written as a single line.
    """

    def __init__(self, out_path: str, flush_interval: int = 50):
        self.out_path = Path(out_path)
        self.flush_interval = flush_interval
        self.buffer = []
        self.fh = self.out_path.open("w")

    def update(self, frame_idx, bbox, proj_pt):
        # Create a record for this specific frame
        record = {"frame_id": frame_idx, "bbox": bbox, "projected": proj_pt}
        self.buffer.append(record)

    def maybe_flush(self, frame_idx):
        if frame_idx % self.flush_interval == 0 and self.buffer:
            self._write_buffer()

    def _write_buffer(self):
        for record in self.buffer:
            json.dump(record, self.fh, ensure_ascii=False)
            self.fh.write("\n")  # write as JSONL
        self.buffer = []

    def close(self):
        if self.buffer:
            self._write_buffer()
        self.fh.close()


class TrackJsonlStreamer:
    """
    Streams per-track data to disk as JSONL.
    Each track is written as a single line once it becomes stale.
    More efficient and streaming-friendly than JSON array.
    """

    def __init__(self, out_path: str, flush_interval: int = 500, lost_thresh: int = 100):
        self.out_path = Path(out_path)
        self.flush_interval = flush_interval
        self.lost_thresh = lost_thresh
        self.records = defaultdict(
            lambda: {
                "track_id": None,
                "frame_id": [],
                "team_conf": [],
                "jersey_num": [],
                "jersey_conf": [],
                "bbox": [],
                "projected": [],
            }
        )
        self.last_seen = {}
        self.fh = self.out_path.open("w")

    def update(self, tid, frame_idx, bbox, team_conf, proj_pt, jersey_num, jersey_conf):
        rec = self.records[tid]
        if rec["track_id"] is None:
            rec["track_id"] = tid
        rec["frame_id"].append(frame_idx)
        rec["team_conf"].append(team_conf)
        rec["jersey_num"].append(jersey_num)
        rec["jersey_conf"].append(jersey_conf)
        rec["bbox"].append(bbox)
        rec["projected"].append(proj_pt)
        self.last_seen[tid] = frame_idx

    def maybe_flush(self, frame_idx):
        if frame_idx % self.flush_interval != 0:
            return

        stale = [tid for tid, last in self.last_seen.items() if frame_idx - last > self.lost_thresh]

        for tid in stale:
            self._write_record(self.records.pop(tid))
            self.last_seen.pop(tid, None)

    def _write_record(self, rec):
        json.dump(rec, self.fh, ensure_ascii=False)
        self.fh.write("\n")  # write as JSONL

    def close(self):
        for rec in self.records.values():
            self._write_record(rec)
        self.fh.close()


@smart_inference_mode()
def run(
    weights=ROOT / "yolo.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    game_time=[
        0,
        2700,
        3600,
        6300,
    ],  # start and end time of first and second half (seconds)
    data=ROOT / "data/coco.yaml",  # dataset.yaml path
    clothes_folder_path=ROOT / "",  # path to clothing features
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    homography_src_points=None,  # image coordinate system for homography
    homography_dst_points=None,  # destination points for homography
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    ema_alpha=0.5,  # EMA smoothing factor for bottom center
    slice_size=(640, 640),  # slice width and height
    nms_threshold=0.45,  # NMS threshold for slicer
    jersey_weights=ROOT / "jersey_net.pt",  # path to jersey number recognition model
):

    # Determine valid frame ranges (as set)
    valid_frame_ids = set()
    cap = cv2.VideoCapture(source)
    fps = (
        cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 29.97
    )  # default to 29.97 FPS if not available

    if fps > 0:
        fh_start = int((game_time[0]) * fps)
        fh_end = int((game_time[1]) * fps)
        sh_start = int((game_time[2]) * fps)
        sh_end = int((game_time[3]) * fps)

        valid_frame_ids.update(range(fh_start, fh_end + 1))
        valid_frame_ids.update(range(sh_start, sh_end + 1))
    else:
        raise ValueError("FPS cannot be 0")

    # check homography points
    if homography_src_points is None or homography_dst_points is None:
        raise ValueError("Both homography source and destination points must be provided.")
    if len(homography_src_points) != 4 or len(homography_dst_points) != 4:
        raise ValueError("Homography points must be lists of 4 tuples.")
    # Convert points to homography matrix
    H = compute_homography(
        np.array(homography_src_points, dtype=np.float32),
        np.array(homography_dst_points, dtype=np.float32),
    )[0]

    source = str(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = str(
        Path(save_dir) / ("after_globalNMS_overlap_remove_annotated_" + Path(source).name)
    )

    # Load jersey number recognition model
    if jersey_weights and os.path.exists(jersey_weights):
        charset_test = string.digits
        # print(charset_test) # 0123456789
        kwargs = {"charset_test": charset_test}
        jersey_model = load_from_checkpoint(jersey_weights, **kwargs).eval().to(device)
        hp = jersey_model.hparams
        if hp is not None:
            jersey_img_size = hp.img_size
    else:
        jersey_model = None
        print("⚠️ Jersey number recognition model not found or path not provided.")

    # init json
    output_player_json_path = str(Path(save_dir) / "team_tracking.jsonl")
    output_ball_json_path = str(Path(save_dir) / "ball_tracking.jsonl")

    player_json_streamer = TrackJsonlStreamer(
        output_player_json_path, flush_interval=500, lost_thresh=100
    )  # tune as needed
    ball_json_streamer = FrameJsonlStreamer(
        output_ball_json_path, flush_interval=500
    )  # tune as needed, 50 and 10 for 30 seconds video testing

    if not nosave:
        print(f"🔄 Saving video to: {output_path}")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    # Initialize tracker
    player_tracker = sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=100,
        minimum_matching_threshold=0.8,
        frame_rate=fps,
    )

    # load the team reference features
    if clothes_folder_path and os.path.exists(clothes_folder_path):
        # print(f"🔍 Loading clothing features from: {clothes_folder_path}")
        team_histograms = load_team_histograms_from_folder(clothes_folder_path)
        # print(f"✅ Loaded {len(team_histograms)} team histograms.")

    video_frame_idx = 0  # original frame index (matches video)
    processed_frame_idx = 0  # game-time processed frame index
    bs = 1  # batch_size, only 1 supported for slicing inference

    # Create slicer callback dynamically to access the model, for sv.InferenceSlicer()
    def slicer_callback(image_slice: np.ndarray):
        with torch.no_grad():
            h, w = image_slice.shape[:2]
            img_size = imgsz[0]  # 640/1280
            # Check if dimensions need padding to be same as expected slice size (640*640/ 1280*1280)
            need_padding = (h != img_size) or (w != img_size)
            # print(f"Before padding: {image_slice.shape}")
            if need_padding:
                # Calculate padding needed
                pad_h = (img_size - h) if h < img_size else 0
                pad_w = (img_size - w) if w < img_size else 0

                # Apply padding (right and bottom)
                padded_slice = cv2.copyMakeBorder(
                    image_slice,
                    0,
                    pad_h,
                    0,
                    pad_w,
                    cv2.BORDER_CONSTANT,
                    value=(114, 114, 114),  # Using gray color common in YOLO
                )
                # Use the padded image for further processing
                image_slice = padded_slice
                # print(f"After padding: {image_slice.shape}") # [640, 640, 3]

            # Convert image to tensor (similar to the original preprocessing)
            img = torch.from_numpy(image_slice.transpose(2, 0, 1)).to(model.device)
            img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
            img /= 255  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # Run inference (similar to original inference)
            pred = model(img, augment=augment)

        # Apply NMS (similar to original NMS)
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )

        # Convert to supervision Detections format if no detections
        if len(pred[0]) == 0:
            return sv.Detections.empty()

        # Process and resize boxes to the slice coordinate system
        boxes = pred[0][:, :4].cpu().numpy()  # xyxy format
        confidences = pred[0][:, 4].cpu().numpy()
        class_ids = pred[0][:, 5].cpu().numpy().astype(int)

        # If we padded the image, we need to filter out detections in the padded area
        if need_padding:
            # Keep only boxes that are mainly in the original image area
            valid_indices = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                # Calculate box center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # Check if center is in the original image
                if center_x < w and center_y < h:
                    valid_indices.append(i)

            # Filter boxes, confidences, and class_ids
            if valid_indices:
                boxes = boxes[valid_indices]
                confidences = confidences[valid_indices]
                class_ids = class_ids[valid_indices]
            else:
                return sv.Detections.empty()

        return sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)

    # Create slicer on-demand for each frame
    overlap_ratio = (0.2, 0.2)  # overlap ratio for the slicer
    overlap_wh = (slice_size[0] * overlap_ratio[0], slice_size[1] * overlap_ratio[1])
    slicer = sv.InferenceSlicer(
        callback=slicer_callback,
        slice_wh=slice_size,
        overlap_ratio_wh=None,
        overlap_wh=overlap_wh,
    )
    # Warmup for more stable inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    pbar = tqdm(
        total=len(valid_frame_ids),
        desc="Processing game-time frames",
        unit="frame",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    )

    # Use this to skip beginning frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, fh_start)
    video_frame_idx = fh_start - 1  # set to one before the first valid frame
    while cap.isOpened():
        ret, high_resolution_image = cap.read()
        if not ret:
            break

        video_frame_idx += 1  # match actual video frame number

        if video_frame_idx % vid_stride != 0:
            continue

        # Skip frames between halves
        if video_frame_idx > fh_end and video_frame_idx < sh_start:
            video_frame_idx = sh_start - 1  # jump to second half
            cap.set(cv2.CAP_PROP_POS_FRAMES, sh_start)
            continue

        if video_frame_idx not in valid_frame_ids:
            continue  # ⛔ skip frames not in game time

        processed_frame_idx += 1  # ✅ only increase if frame is actually used

        # print(f"🔍 Processing frame {processed_frame_idx}")

        seen, windows, dt = 0, [], [Profile() for _ in range(6)]
        # Inference + NMS
        with dt[0]:
            # ✅ Convert once at the start
            high_resolution_image_rgb = cv2.cvtColor(high_resolution_image, cv2.COLOR_BGR2RGB)

            # v2 update: now both player and ball detection are done in one model forward pass
            all_dets = slicer(high_resolution_image_rgb).with_nms(threshold=nms_threshold)

        # proprocess predictions
        with dt[1]:

            # start = time.time()  # reset timer
            # Make a copy of the original 4K image for drawing
            annotated_image = high_resolution_image.copy()
            all_dets_np = np.hstack(
                (
                    all_dets.xyxy,
                    all_dets.confidence[:, np.newaxis],
                    all_dets.class_id[:, np.newaxis],
                )
            )
            all_dets_np = torch.from_numpy(all_dets_np).to(device)
            if all_dets_np.shape[0] > 0:
                all_dets_np = remove_boxes_with_numba(
                    all_dets_np, area_ratio_thresh=0.6, containment_thresh=0.9
                )

        with dt[2]:
            # the shape of final is [N, 6] where N is the number of detections, the 6 columns are [x1, y1, x2, y2, conf, cls]

            player_dets_np = all_dets_np[all_dets_np[:, 5] == 0]  # Class 0 for player
            ball_dets_np = all_dets_np[all_dets_np[:, 5] == 1]  # Class 1 for ball
            player_dets = (
                player_dets_np[:, :5].cpu().numpy() if player_dets_np.numel() else np.empty((0, 5))
            )
            ball_dets = ball_dets_np.cpu().numpy() if ball_dets_np.numel() else np.empty((0, 5))
            online_players = player_tracker.update_with_tensors(player_dets)
            online_balls = ball_dets  # No tracking for ball

            # print("Online players after tracking:", len(online_players))
            # print("Online balls:", len(online_balls))

            # Store all crop tensors and track info for matching
            crop_hists = []
            crop_track_ids = []

            jersey_numbers = []
            jersey_confs = []

            # Format detections with track ID
            final_detections = []
            jersey_crops = []
            for t in online_players:
                tlbr = t.tlbr  # (x1, y1, x2, y2)
                track_id = t.external_track_id
                conf = t.score
                x1, y1, x2, y2 = tlbr
                # Compute bbox properties
                curr_cx = (x1 + x2) / 2
                curr_h = y2 - y1

                # Smooth bbox height and center x
                if not hasattr(t, "smooth_h"):
                    t.smooth_h = curr_h
                    t.smooth_cx = curr_cx
                else:
                    t.smooth_h = ema_alpha * curr_h + (1 - ema_alpha) * t.smooth_h
                    t.smooth_cx = ema_alpha * curr_cx + (1 - ema_alpha) * t.smooth_cx

                # Compute smoothed bottom center
                smoothed_cx = t.smooth_cx
                smoothed_y2 = y1 + t.smooth_h
                cx, cy = smoothed_cx, smoothed_y2  # use these instead of raw values

                projected_position = apply_homography_to_point(
                    (cx, cy), H
                )  # (x, y) projected point

                cls = 0  # Class 0 for player
                final_detections.append(
                    (
                        tlbr[0],
                        tlbr[1],
                        tlbr[2],
                        tlbr[3],
                        conf,
                        cls,
                        track_id,
                        projected_position,
                    )
                )

                # Crop the clothing region
                x1, y1, x2, y2 = map(int, [tlbr[0], tlbr[1], tlbr[2], tlbr[3]])
                crop_img = crop_clothing_region(
                    high_resolution_image,
                    (x1, y1, x2, y2),
                    top_ratio=0.2,
                    bottom_ratio=0.5,
                    left_ratio=0.25,
                    right_ratio=0.75,
                )
                if crop_img.size == 0:
                    continue  # skip empty crops
                crop_hist = extract_color_histogram_with_specific_background_color(crop_img)
                crop_hists.append(crop_hist)
                crop_track_ids.append(
                    track_id
                )  # may not be required, cause the index of crop_tensors is first N element of final_detections

                # Crop the clothing region with jersey number, looser crop
                crop_img_jersey = crop_clothing_region(
                    high_resolution_image,
                    (x1, y1, x2, y2),
                    top_ratio=0.1,
                    bottom_ratio=0.6,
                    left_ratio=0.25,
                    right_ratio=0.75,
                )
                if crop_img_jersey.size == 0:
                    continue  # skip empty crops
                crop_img_jersey_rgb = cv2.cvtColor(crop_img_jersey, cv2.COLOR_BGR2RGB)
                jersey_crops.append(Image.fromarray(crop_img_jersey_rgb))

            # Run jersey number recognition if model is available
            if jersey_model and jersey_crops:
                transform = SceneTextDataModule.get_transform(jersey_img_size)
                valid_crops = [c for c in jersey_crops if c is not None]
                if valid_crops:
                    batch_imgs = torch.stack([transform(c) for c in valid_crops]).to(
                        jersey_model.device
                    )
                    with torch.no_grad():
                        logits = jersey_model.forward(batch_imgs)  # Batch forward
                        probs_full = logits[:, :3, :11].softmax(-1)
                        preds, probs = jersey_model.tokenizer.decode(probs_full)

                    # Map back to detections
                    jersey_numbers = [p for p in preds]  # e.g. ['10', '23', ...]
                    jersey_confs = [
                        p.cpu().numpy().squeeze().tolist() for p in probs
                    ]  # e.g. [['0.9999', '0.999', '1.0'], [...], ...]

            for t in online_balls:
                tlbr = t[:4]
                conf = t[4]
                x1, y1, x2, y2 = tlbr
                # Compute bbox properties
                cx = (x1 + x2) / 2
                cy = y2

                projected_position = apply_homography_to_point(
                    (cx, cy), H
                )  # (x, y) projected point

                cls = 1  # Class 1 for ball
                track_id = None
                final_detections.append(
                    (
                        tlbr[0],
                        tlbr[1],
                        tlbr[2],
                        tlbr[3],
                        conf,
                        cls,
                        track_id,
                        projected_position,
                    )
                )

        with dt[3]:
            if crop_hists:  # crop_images = list of color histograms for each person
                # Load team histograms from file (or define in code)
                if team_histograms:
                    team_scores = match_histograms_to_teams(
                        crop_hists, team_histograms
                    )  # white mask example
                else:
                    team_scores = [{} for _ in crop_hists]
                    print("⚠️ No team histograms found, using empty scores.")
                    break

            # Update tracking JSON records
            # Track current feature index to sync with crop detections
            feature_index_jersey = 0
            feature_index_team = 0

            for det in final_detections:
                x1, y1, x2, y2, conf, cls, track_id, projected_position = det
                bbox_out = [int(x1), int(y1), int(x2), int(y2), float(conf)]

                if cls == 0:  # Class 0 for player
                    if feature_index_jersey < len(jersey_numbers):
                        jersey_str = jersey_numbers[feature_index_jersey]
                        jersey_num = int(jersey_str) if jersey_str.isdigit() else -1
                        jersey_confidence = jersey_confs[feature_index_jersey]
                        feature_index_jersey += 1
                    else:
                        jersey_num = -1
                        jersey_confidence = 0.0

                    if feature_index_team < len(team_scores):
                        team_conf = {
                            k: float(v) for k, v in team_scores[feature_index_team].items()
                        }
                        feature_index_team += 1
                    else:
                        team_conf = {}
                    # ⭐ update the streamer
                    player_json_streamer.update(
                        track_id,
                        processed_frame_idx,
                        bbox_out,
                        team_conf,
                        projected_position,
                        jersey_num,
                        jersey_confidence,
                    )

                elif cls == 1:  # Class 1 for ball
                    team_conf = {}
                    # ⭐ update the streamer
                    ball_json_streamer.update(processed_frame_idx, bbox_out, projected_position)
                else:  # unknown class
                    raise ValueError(f"Expected 0 (player) or 1 (ball). Got {cls} instead.")

        # save json every N frames
        with dt[4]:
            # flush every N frames
            player_json_streamer.maybe_flush(processed_frame_idx)
            ball_json_streamer.maybe_flush(processed_frame_idx)

        # Draw results
        with dt[5]:
            if not nosave or view_img:
                # print("team_scores: ",team_scores, )
                draw_detections(annotated_image, final_detections, names)
            if not nosave:
                # Save results
                out.write(annotated_image)

        total_time = sum(dt[i].dt for i in range(len(dt)))
        # print(f"Frame {processed_frame_idx} total use: {total_time} (preprocessed: {dt[0].dt:.2f}s, inference: {dt[1].dt:.2f}s, NMS: {dt[2].dt:.2f}s, proprocess: {dt[3].dt:.2f}s, tracking & crop patches: {dt[4].dt:.2f}s, ReID: {dt[5].dt:.2f}s, Json: {dt[6].dt:.2f}s, Draw: {dt[7].dt:.2f}s)")
        pbar.set_postfix(
            {
                "inf+nms": f"{dt[0].dt:.2f}s",
                "proc": f"{dt[1].dt:.2f}s",
                "trk": f"{dt[2].dt:.2f}s",
                "reid": f"{dt[3].dt:.2f}s",
                "json": f"{dt[4].dt:.2f}s",
                "draw": f"{dt[5].dt:.2f}s",
                "total": f"{total_time:.2f}s",
            }
        )
        pbar.update(1)

        if view_img:
            cv2.imshow("YOLO Detection", annotated_image)
            if cv2.waitKey(1) == ord("q"):
                break
    if not nosave:
        cap.release()
        out.release()
        if view_img:
            cv2.destroyAllWindows()
        pbar.close()
        print(f"✅ Saved video to: {output_path}")

    # close streamer → writes remaining tracks & final ‘]’
    player_json_streamer.close()
    ball_json_streamer.close()
    print(f"✅ Streamed team tracking JSON to {output_player_json_path}")
    print(f"✅ Streamed ball tracking JSON to {output_ball_json_path}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolo.pt",
        help="model path or triton URL",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob/screen/0(webcam)",
    )
    parser.add_argument(
        "--game-time",
        type=int,
        nargs=4,
        default=[0, 2700, 3600, 6300],
        help="Game time in seconds in source video: first_half_start_second, first_half_end_second, second_half_start_second, second_half_end_second.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/coco128.yaml",
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--clothes-folder-path",
        type=str,
        default=ROOT / "",
        help="path to clothing features for assigning team IDs",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument(
        "--homography-src-points",
        type=int,
        nargs=8,
        default=[0, 0, 1, 0, 1, 1, 0, 1],
        help="source points for homography transformation (x1, y1, x2, y2, x3, y3, x4, y4)",
    )
    parser.add_argument(
        "--homography-dst-points",
        type=int,
        nargs=8,
        default=[0, 0, 1, 0, 1, 1, 0, 1],
        help="destination points for homography transformation (x1, y1, x2, y2, x3, y3, x4, y4)",
    )
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.5,
        help="EMA smoothing factor for bottom center",
    )
    parser.add_argument(
        "--slice-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="slice width and height",
    )
    parser.add_argument(
        "--nms-threshold", type=float, default=0.45, help="NMS threshold for slicer"
    )
    parser.add_argument(
        "--jersey-weights",
        type=str,
        default=ROOT / "jersey_net.pt",
        help="path to jersey number recognition model",
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.game_time = np.array(opt.game_time, dtype=np.int32).reshape(4)  # reshape to 1D array
    opt.homography_src_points = np.array(opt.homography_src_points, dtype=np.float32).reshape(4, 2)
    opt.homography_dst_points = np.array(opt.homography_dst_points, dtype=np.float32).reshape(4, 2)
    opt.homography_src_points = opt.homography_src_points.tolist()
    opt.homography_dst_points = opt.homography_dst_points.tolist()
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    start_time = time.time()
    main(opt)
    print("Finished processing script for video patch detection and tracking.")
    end_time = time.time()
    print(
        f"Total execution time(HH:MM:SS): {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}"
    )
# Example usage:
# python3 mini_patch_detect_v2_for_video.py --source './data/video/test_sample/C0478.MP4' --game-time 317 3085 3982 6809 --img 640 --device 0 --weights './weight/yolov9-s-converted.pt' --name test_4k --classes 0 1 --clothes-folder-path ./data/histograms/0525/ --homography-src-points 172 1104 2101 895 3800 1021 3458 2057 --homography-dst-points 530 0 530 660 1060 660 1060 0 --jersey-weights ./weights/parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt --nosave
