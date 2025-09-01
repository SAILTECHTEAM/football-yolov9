import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import numpy as np
import onepose
from identify_goalkeeper import extract_color_histogram_with_specific_background_color, extract_color_histogram_from_rotated_skelton, compare_histograms, load_histogram
from goalkeeper_motion_classification import classify_goalkeeper_behavior
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
    increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode




def setup_output_dir(project, name, exist_ok, save_txt):
    """
    Create the output directory and returns its path.
    """
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # e.g. runs/detect/exp
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def load_yolo_model(weights, device, dnn, data, half):
    """
    Load the YOLO model (DetectMultiBackend) and return:
    (model, stride, names, pt).
    """
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    return model, stride, names, pt


def load_pose_model():
    """
    Load the OnePose model (or other pose model).
    """
    pose_model = onepose.create_model('ViTPose_huge_simple_coco').to("cuda")
    return pose_model


def create_dataloader(source, imgsz, stride, pt, vid_stride):
    """
    Create a dataset/dataloader for images, video, or webcam.
    Returns (dataset, batch_size, is_webcam_mode).
    """
    source_str = str(source)
    is_file = Path(source_str).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source_str.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source_str.isnumeric() or source_str.endswith('.txt') or (is_url and not is_file)
    screenshot = source_str.lower().startswith('screen')

    if is_url and is_file:
        # For example: download if it's a direct URL
        source_str = check_file(source_str)

    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source_str, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source_str, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    else:
        dataset = LoadImages(source_str, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1

    return dataset, bs, webcam


def warmup_yolo_model(model, pt, bs, imgsz):
    """
    Warm up model with a simple forward pass. 
    """
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))


def infer_on_dataset(
    dataset,
    model,
    pose_model,
    names,
    pt,
    imgsz,
    conf_thres,
    iou_thres,
    max_det,
    classes,
    agnostic_nms,
    augment,
    visualize,
    goalkeeper_clothes_colors_histogram,  # If not None, we pick best skeleton among persons
    ball_speed,
    save_img,
    save_txt,
    save_conf,
    save_crop,
    line_thickness,
    hide_labels,
    hide_conf,
    view_img,
    save_dir,
    distance_threshold,
    movement_threshold,
    speed_threshold,
    jump_threshold,
    elbow_angle_threshold,
    use_tqdm,
):
    vid_path, vid_writer = [None] * 1, [None] * 1
    seen, windows = 0, []
    dt = (Profile(), Profile(), Profile())
    video_detections = []  # To store detections per video

    current_video_detections = []  # Detections for current video
    current_video_path = None      # Track which video we're processing

    dataset = list(dataset)

    # --- Compute total unique videos (if possible) ---
    if len(dataset) > 0:
        all_video_paths = [item[0] for item in dataset]
        total_video_count = len(set(all_video_paths))
    else:
        total_video_count = 0

    video_index = 0  # current video counter

    # Define target range parameters: process frames 90 to 149 (i.e. 60 frames)
    target_start = 90                
    target_total = 60                
    target_end = target_start + target_total  # non-inclusive

    # total_frames = len(dataset) if hasattr(dataset, '__len__') else None
    pbar = None

    for path, im, im0s, vid_cap, s in dataset:
        # New video detection: reset per-video counters
        if current_video_path is None or current_video_path != path:
            if pbar:
                pbar.close()
                pbar=None

            if current_video_detections:
                video_detections.append((current_video_path, current_video_detections))
                current_video_detections = []

            current_video_path = path
            video_index += 1
            video_frame_counter = 0  # reset per-video frame counter
            pbar = None

        video_frame_counter += 1

        # Process only frames within target range
        if video_frame_counter < target_start or video_frame_counter >= target_end:
            continue


        # Re-index processed frames to start at 1 (so frame 90 becomes 1, 149 becomes 60)
        processed_frame = video_frame_counter - target_start + 1

        # --- Preprocessing & Inference ---
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]  # add batch dimension

        with dt[1]:
            visualize_path = increment_path(save_dir / Path(current_video_path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_path)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # --- Process Predictions ---
        for i, det in enumerate(pred):
            seen += 1

            # For multi-camera inputs, assume path is a list.
            # Use channel-specific variables but log only for the first channel.
            if isinstance(path, list):
                p_i = Path(path[i])
                im0_i = im0s[i].copy()
            else:
                p_i = Path(path)
                im0_i = im0s.copy()

            # Only log once per frame using the first channel (i == 0)
            if not (isinstance(path, list) and i > 0):
                frame = processed_frame
                # Use p_i from the first channel for logging
                save_path = str(save_dir / p_i.name)
                ext = p_i.suffix.lower()  # e.g. ".mp4", ".jpg", etc.
                image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

                if ext in image_exts:
                    # We'll treat this as an image file => no frame number
                    txt_path = str(save_dir / 'labels' / p_i.stem)
                    mode = 'image'
                else:
                    # We'll treat this as a video => append frame number
                    txt_path = str(save_dir / 'labels' / p_i.stem) + f'_{frame}'
                    mode = 'video'

                im0_warped = im0_i

                # If detections exist, rescale boxes
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0_i.shape).round()

                # Build detection summary (assume class 0 represents "person")
                person_count = 0
                if len(det):
                    person_count = sum(1 for *_, conf, cls_id in det if int(cls_id) == 0)
                person_str = f"{person_count} person" if person_count == 1 else f"{person_count} persons"



            # --- Process detections (for each channel) ---
            all_detections_for_frame = []
            if len(det):
                for *xyxy, conf, cls_id in reversed(det):
                    detection_result = process_single_detection(
                        im0s,
                        xyxy,
                        conf,
                        cls_id,
                        pose_model,
                        save_crop,
                        hide_labels,
                        hide_conf,
                        names,
                        goalkeeper_clothes_colors_histogram
                    )
                    if detection_result:
                        all_detections_for_frame.append(detection_result)

            current_video_detections.append(all_detections_for_frame)

            # --- (Optional) Post-processing for best skeleton selection ---
            if goalkeeper_clothes_colors_histogram is not None:
                best_score = -1
                best_idx = -1
                for idx, detres in enumerate(all_detections_for_frame):
                    if detres['cls'] == 0 and detres['score'] > best_score:
                        best_score = detres['score']
                        best_idx = idx
                for idx, detres in enumerate(all_detections_for_frame):
                    if detres['cls'] == 0 and idx != best_idx:
                        detres['keypoints'] = None
                        detres['keypoints_conf'] = None
                        detres['score'] = 0.0

            # --- Save and draw ---
            if save_txt and len(all_detections_for_frame) > 0:
                save_all_detections_txt(
                    all_detections_for_frame,
                    txt_path,
                    save_conf
                )

            draw_all_detections(
                im0_i,
                all_detections_for_frame,
                pose_model,
                line_thickness
            )

            if view_img:
                handle_view_img(p_i, windows, im0_i)

            if save_img:
                handle_save_results(
                    mode,
                    i,
                    im0_i,
                    save_path,
                    vid_path,
                    vid_writer
                )
        
        # Create a new progress bar for this video, from 1..60
        if use_tqdm and pbar is None:
            pbar = tqdm(
                total=60,
                desc=f"Video {video_index}/{total_video_count}",
                leave=True
            )

        if pbar:
            pbar.update(1)
        else:
            log_str = (
                f"video {video_index}/{total_video_count} "
                f"({frame}/60) {p_i}: {person_str}, "
                f"{im0_i.shape[0]}x{im0_i.shape[1]} {dt[1].dt * 1E3:.1f}ms"
            )
            LOGGER.info(log_str)

    if pbar:
        pbar.close()

    if current_video_detections:
        video_detections.append((current_video_path, current_video_detections))

    behavior_dict = {}
    if goalkeeper_clothes_colors_histogram is not None:
        for video_path, video_dets in video_detections:
            if video_dets:
                behavior_dict[video_path] = classify_goalkeeper_behavior(
                    video_dets,
                    ball_speed,
                    distance_threshold,
                    movement_threshold,
                    speed_threshold,
                    jump_threshold,
                    elbow_angle_threshold
                )
            else:
                behavior_dict[video_path] = "No detections"

    return seen, windows, dt, behavior_dict


def process_single_detection(
    im0s,
    xyxy,
    conf,
    cls,
    pose_model,
    save_crop,
    hide_labels,
    hide_conf,
    names,
    clothes_colors_histogram
):
    """
    Return detection data (incl. bounding box, skeleton, label, 'score').
    If clothes_colors is not None, we do color matching on the person's torso.
    """
    x1, y1, x2, y2 = map(int, xyxy)
    c = int(cls)

    # Initialize detection dictionary
    detection_result = {
        'cls': c,
        'conf': float(conf),
        'bbox_src': [x1, y1, x2, y2],  # bounding box in original coords
        'bbox_warp': None,            # bounding box in warped coords
        'label_str': None,            # class + conf text
        'keypoints': None,
        'keypoints_conf': None,
        'save_crop': save_crop,
        'class_name': names[c] if c < len(names) else f"class_{c}",
        'score': 0.0                  # color-matching score
    }

    # Build label text
    if not hide_labels:
        if hide_conf:
            detection_result['label_str'] = f'{names[c]}'
        else:
            detection_result['label_str'] = f'{names[c]} {conf:.2f}'


    detection_result['bbox_warp'] = [x1, y1, x2, y2]

    # ----------------------------------
    # If it's a person (cls=0), do pose and color check
    # ----------------------------------
    if c == 0:
        keypoints_dict, warped_points = handle_pose_estimation(
            im0s,
            x1, y1, x2, y2,
            pose_model
        )
        if keypoints_dict and warped_points is not None:
            detection_result['keypoints'] = warped_points
            detection_result['keypoints_conf'] = keypoints_dict['confidence']

            # If clothes_colors is not None, do color matching
            if clothes_colors_histogram is not None:
                # Extract torso keypoints
                keypoints = {
                    'left_shoulder': keypoints_dict['points'][5],
                    'right_shoulder': keypoints_dict['points'][6],
                    'left_hip': keypoints_dict['points'][11],
                    'right_hip': keypoints_dict['points'][12]
                }
                # if bbox is too small, skip
                if (x2 - x1) * (y2 - y1) < 40000:
                    return None
                
                else:
                    # Extract colors from the person's torso
                    skelton_colors_histogram = extract_color_histogram_from_rotated_skelton(
                        im0s, keypoints
                    )

                    # Compute match score
                    score_val = compare_histograms(
                        clothes_colors_histogram,
                        skelton_colors_histogram
                    )
                    # print(f"Color match score: {score_val}")
                    detection_result['score'] = score_val

    return detection_result


def save_all_detections_txt(
    all_detections,
    txt_path,
    save_conf
):
    """
    Save all detections in YOLO format, in the *warped* coordinate domain.
    """
    txt_out = f'{txt_path}.txt'
    with open(txt_out, 'a') as f:
        for det in all_detections:
            c = det['cls']
            conf = det['conf']
            wx1, wy1, wx2, wy2 = det['bbox_warp']
            keypoints_global = []

            # If we have keypoints, flatten them
            if det['keypoints'] is not None and det['keypoints_conf'] is not None:
                points = det['keypoints']
                confs  = det['keypoints_conf']
                for (kx, ky), cpt in zip(points, confs):
                    keypoints_global.extend([kx, ky, float(cpt)])

            # YOLO format: class, x_center, y_center, w, h
            bw = wx2 - wx1
            bh = wy2 - wy1
            cx = wx1 + bw / 2
            cy = wy1 + bh / 2

            # normalize
            norm_cx = cx
            norm_cy = cy
            norm_w  = bw
            norm_h  = bh

            if save_conf:
                line = (c, norm_cx, norm_cy, norm_w, norm_h, conf, *keypoints_global)
            else:
                line = (c, norm_cx, norm_cy, norm_w, norm_h, *keypoints_global)

            f.write(('%g ' * len(line)).rstrip() % line + '\n' + '\n')


def draw_all_detections(im0_warped, all_detections, pose_model, line_thickness=1):
    """
    Draw bounding boxes and skeletons for all detections in the frame.
    This function is called once after we've processed all detections.
    """
    annotator = Annotator(im0_warped, line_width=line_thickness)

    for det in all_detections:
        # BBox data
        x1, y1, x2, y2 = det['bbox_warp']
        label_str = det['label_str']
        cls_id = det['cls']
        conf_val = det['conf']

        # Draw bounding box
        annotator.box_label([x1, y1, x2, y2], label_str, color=colors(cls_id, True))

        # Draw skeleton if any
        if det['keypoints'] is not None and det['keypoints_conf'] is not None:
            skel_info = {
                'points': det['keypoints'],
                'confidence': det['keypoints_conf']
            }
            onepose.visualize_keypoints(
                annotator.im,
                skel_info,
                pose_model.keypoint_info,
                pose_model.skeleton_info
            )

        # Optionally save crop if needed
        if det['save_crop']:
            save_cropped(
                annotator.im,
                x1, y1, x2, y2,
                det['class_name'],
                det['txt_path']
            )

    # Replace the original image with the annotated one
    final_img = annotator.result()
    im0_warped[:, :, :] = final_img  # copy back if needed so caller sees changes


def handle_pose_estimation(
    im0s,
    x1, y1, x2, y2,
    pose_model
):
    """
    Crops the region for a person and runs pose estimation (OnePose).
    Returns:
      keypoints_dict: the original dictionary from the model (with 'points' and 'confidence')
      clamped_keypoints: final coords (warped if perspective) for visualization.
    """
    x1_safe, y1_safe = max(0, x1), max(0, y1)
    x2_safe = min(im0s.shape[1], x2)
    y2_safe = min(im0s.shape[0], y2)

    # If bounding box too small, skip
    if (x2_safe - x1_safe) < 10 or (y2_safe - y1_safe) < 10:
        return None, []

    cropped_img = im0s[y1_safe:y2_safe, x1_safe:x2_safe]
    keypoints_dict = pose_model(cropped_img)  # must return { 'points': Nx2, 'confidence': Nx1 }
    points = keypoints_dict['points']         # shape: Nx2
    confidences = keypoints_dict['confidence']  # shape: Nx1

    # Shift from local crop to original image coords
    for idx in range(len(points)):
        points[idx][0] += x1_safe
        points[idx][1] += y1_safe


    # Return the original coords
    return keypoints_dict, points


def save_cropped(
    im_warped,
    wx1, wy1, wx2, wy2,
    class_name,
    txt_path
):
    """
    Save cropped region from warped image under `crops/<class_name>`.
    """
    h_warp, w_warp = im_warped.shape[:2]
    iw1 = max(int(wx1), 0)
    iw2 = min(int(wx2), w_warp)
    ih1 = max(int(wy1), 0)
    ih2 = min(int(wy2), h_warp)

    crop_img = im_warped[ih1:ih2, iw1:iw2]
    outdir = Path(txt_path).parent.parent / 'crops' / class_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Use the txt_path stem as a reference
    parent_stem = Path(txt_path).stem
    crop_path = outdir / f'{parent_stem}.jpg'
    cv2.imwrite(str(crop_path), crop_img)


def handle_view_img(p, windows, im0_warped_final):
    """
    Show results in a window, if user sets `view_img=True`.
    """
    if platform.system() == 'Linux' and p not in windows:
        windows.append(p)
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(str(p), im0_warped_final.shape[1], im0_warped_final.shape[0])
    cv2.imshow(str(p), im0_warped_final)
    cv2.waitKey(1)

def handle_save_results(
    mode,               # <--- a string or boolean indicating "image" vs. "video"
    i,
    im0_warped_final,
    save_path,
    vid_path,
    vid_writer
):
    """
    Save either a single image or frames of a video/stream.
    """
    if mode == 'image':
        # Save a single image
        cv2.imwrite(save_path, im0_warped_final)
    else:
        # Save video/stream
        if i < len(vid_path):
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()

                fps, w_vid, h_vid = 30, im0_warped_final.shape[1], im0_warped_final.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))
                print(f"Initializing video writer: {save_path}, FPS: {fps}, Size: ({w_vid}, {h_vid})")
                vid_writer[i] = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps, 
                    (w_vid, h_vid)
                )

            if not vid_writer[i].isOpened():
                print(f"Error: Failed to initialize video writer for {save_path}")

            vid_writer[i].write(im0_warped_final)


def summarize_and_cleanup(
    dt,
    seen,
    imgsz,
    save_dir,
    save_txt,
    save_img,
    update,
    weights
):
    """
    Print speed summary, final results, and handle any final cleanups/updates.
    """
    t = tuple(x.t / seen * 1E3 for x in dt) if seen else (0, 0, 0)
    LOGGER.info(
        f'Speed: {t[0]:.1f}ms pre-process, {t[1]:.1f}ms inference, '
        f'{t[2]:.1f}ms NMS per image at shape {(1, 3, *imgsz)}'
    )

    # Save summary
    if save_txt or save_img:
        label_files = list(save_dir.glob('labels/*.txt'))
        s = f"\n{len(label_files)} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    # Update model if needed
    if update and weights and len(weights) > 0:
        strip_optimizer(weights[0])

@smart_inference_mode()
def run(
    weights,         # model path or triton URL
    source,      # file/dir/URL/glob/screen/0(webcam)
    data,     # dataset.yaml path
    imgsz,                 # inference size (height, width)
    conf_thres,                  # confidence threshold
    iou_thres,                   # NMS IOU threshold
    max_det,                     # maximum detections per image
    device,                        # cuda device, i.e. 0, 0,1,2,3 or cpu
    view_img,                   # show results
    save_txt,                   # save results to *.txt
    save_conf,                  # save confidences in --save-txt labels
    save_crop,                  # save cropped prediction boxes
    nosave,                     # do not save images/videos
    classes,                     # filter by class: --class 0, or --class 0 2 3
    agnostic_nms,               # class-agnostic NMS
    augment,                    # augmented inference
    visualize,                  # visualize features
    update,                     # update all models
    project,     # save results to project/name
    name,                       # save results to project/name
    exist_ok,                   # existing project/name ok, do not increment
    line_thickness,                 # bounding box thickness (pixels)
    hide_labels,                # hide labels
    hide_conf,                  # hide confidences
    half,                       # use FP16 half-precision inference
    dnn,                        # use OpenCV DNN for ONNX inference
    vid_stride,                     # video frame-rate stride
    goalkeeper_clothes_colors_histogram_path,    # list of LAB colors for color matching
    ball_speed,                     # ball speed
    distance_threshold,
    movement_threshold,
    speed_threshold,
    jump_threshold,
    elbow_angle_threshold,
    use_tqdm
):
    """
    Main detection + pose estimation pipeline.

    If perspective_matrix is provided (via goal_image_coordinate),
    we will warp the entire image and bounding-box coordinates,
    then display/save the warped image + warped coords.
    Otherwise, fallback to the original image as usual.
    """

    # --- 1) Setup output directory ---
    save_dir = setup_output_dir(project, name, exist_ok, save_txt)
    save_img = not nosave and not str(source).endswith('.txt')

    # --- 2) Load YOLO model ---
    model, stride, names, pt = load_yolo_model(weights, device, dnn, data, half)
    imgsz = check_img_size(imgsz, s=stride)

    # --- 3) Load Pose model ---
    pose_model = load_pose_model()

    # --- 4) Create dataloader ---
    dataset, bs, webcam_mode = create_dataloader(
        source, imgsz, stride, pt, vid_stride
    )

    # --- 5) Warm up YOLO model ---
    warmup_yolo_model(model, pt, bs, imgsz)

    goalkeeper_clothes_colors_histogram = load_histogram(goalkeeper_clothes_colors_histogram_path)

    # --- 6) Inference over dataset (main loop) ---
    seen, windows, dt, behavior_dict = infer_on_dataset(
        dataset, 
        model,
        pose_model,
        names,
        pt,
        imgsz,
        conf_thres,
        iou_thres,
        max_det,
        classes,
        agnostic_nms,
        augment,
        visualize,
        goalkeeper_clothes_colors_histogram,
        ball_speed,
        save_img,
        save_txt,
        save_conf,
        save_crop,
        line_thickness,
        hide_labels,
        hide_conf,
        view_img,
        save_dir,
        distance_threshold,
        movement_threshold,
        speed_threshold,
        jump_threshold,
        elbow_angle_threshold,
        use_tqdm,
    )

    # --- 7) Summaries & Cleanup ---
    summarize_and_cleanup(
        dt,
        seen,
        imgsz,
        save_dir,
        save_txt,
        save_img,
        update,
        weights
    )

    print (behavior_dict)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0 ,32], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--goalkeeper_clothes_colors_histogram_path', type=str, default = None, help = 'numpy array of HSV colors histogram')
    parser.add_argument('--ball-speed', type=float, default=0, help='ball speed')
    parser.add_argument('--distance_threshold', type=float, default=100, help='distance threshold')
    parser.add_argument('--movement_threshold', type=float, default=300, help='movement threshold')
    parser.add_argument('--speed_threshold', type=float, default=50, help='speed threshold')
    parser.add_argument('--jump_threshold', type=float, default=100, help='jump threshold')
    parser.add_argument('--elbow_angle_threshold', type=float, default=160, help='elbow angle threshold')
    parser.add_argument('--use-tqdm', action='store_true', help='use tqdm for progress bar')
    opt = parser.parse_args()

    # Convert flat list to nested list of coordinates
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# sample usage
# python3 detect_onepose_v5_re.py --weights "./weight/yolov9-c-converted.pt" --source "./data/video/param2/8-1.mp4" --name 'test' --ball-speed 60 --goalkeeper_clothes_colors_histogram_path ./data/histograms/8-1_goalkeeper_hist.npy

# python3 detect_onepose_v5_re.py --weights "./weight/yolov9-c-converted.pt" --source "./data/video/param1/10-1.mp4" --name 'test' --ball-speed 60 --goalkeeper_clothes_colors_histogram_path ./data/histograms/10-1_goalkeeper_hist.npy

# python3 detect_onepose_v5_re.py --weights "./weight/yolov9-c-converted.pt" --source "./runs/detect/real_goal2-2/clips/" --name 'real_goal2_anly' --ball-speed 60 --goalkeeper_clothes_colors_histogram_path ./data/histograms/real_goal2.npy

# python3 detect_onepose_v5_re.py --weights "./weight/yolov9-c-converted.pt" --source "./data/video/demo_video/clips/" --name 'demo_video_high_conf_anly' --ball-speed 60 --goalkeeper_clothes_colors_histogram_path ./data/histograms/gk01_clothes.npy 