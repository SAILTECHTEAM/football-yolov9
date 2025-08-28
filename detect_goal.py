import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import numpy as np
import onepose
from idenfity_goalkeeper import extract_color_histogram_with_specific_background_color, extract_color_histogram_from_rotated_skelton, compare_histograms, load_histogram
from goalkeeper_motion_classification import classify_goalkeeper_behavior
from collections import deque
from yolov9.tools.extract_datetime import get_video_start_time_and_fps, calculate_real_timestamp
from yolov9.tools.extract_speed_data import find_max_speed_in_range
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


def compute_perspective_transform(pts, width, height):
    """Compute the perspective transform matrix from 4 points to (width x height)."""
    src = np.float32(pts)
    dst = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return M


def perspective_transform_points(points, M):
    """
    Transform a list (or array) of 2D points using a 3x3 perspective matrix M.
    points: np.array of shape (N, 2).
    returns: np.array of shape (N, 2) of transformed points.
    """
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_pts = cv2.perspectiveTransform(pts, M)
    return transformed_pts.reshape(-1, 2)


def prepare_perspective(goal_image_coordinate, goal_realworld_size):
    """
    Compute perspective matrix if 4 corner points provided. 
    Returns (perspective_matrix, have_perspective).
    """
    if goal_image_coordinate and len(goal_image_coordinate) == 4:
        perspective_matrix = compute_perspective_transform(
            goal_image_coordinate,
            goal_realworld_size[0],
            goal_realworld_size[1]
        )
        print("Perspective matrix:", perspective_matrix)
        return perspective_matrix
    else:
        raise TypeError("Perspective matrix is not calculated. Please provide 4 points.")


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
    perspective_matrix,
    goal_realworld_size,
    save_img,
    save_txt,
    save_conf,
    save_crop,
    line_thickness,
    hide_labels,
    hide_conf,
    view_img,
    save_dir,
    draw_bbox,
    radar_data_path,
    use_tqdm
):
    vid_path, vid_writer = [None] * 1, [None] * 1
    seen, windows = 0, []
    dt = (Profile(), Profile(), Profile())

    pre_trigger_frames_to_save = 150              # Number of frames to buffer before trigger
    frames_buffer = deque(maxlen=pre_trigger_frames_to_save)  # store the last frames for pre-trigger
    post_trigger_frames = 150                       # Number of frames to record after trigger
    skip_counter = 0                 # if > 0, skip checking “big ball” triggers
    clip_index = 0                   # to name each saved clip uniquely
    fps = 30                         # (optional) frames per second for each clip
    collection_of_speed_dict = []                  # store speed data, video time, real time for each clip
    prev_video_id = None  # Track the previous video's identifier
    video_start_time = None
    video_fps = fps
    video_start_frame_idx = 0

    # New state variables for post-trigger recording
    active_clip_writer = None      # when not None, we are recording post-trigger frames
    post_trigger_counter = 0       # counter for frames written after trigger

    # If 'dataset' supports len(), use it for total frames. Otherwise set a fixed total or remove total=...
    total_frames = len(dataset) if hasattr(dataset, '__len__') else None

    if use_tqdm:
        pbar = tqdm(total=total_frames, desc="Infer on dataset") if total_frames else None
    else:
        pbar = None

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # --------------------------------------
        # 1) Preprocessing & YOLO Inference
        # --------------------------------------
        current_video_id = Path(path).stem

        # If we're starting a new video, reset all state variables
        if prev_video_id is not None and current_video_id != prev_video_id:
            frames_buffer.clear()  # Clear the frame buffer
            skip_counter = 0       # Reset the skip counter
            LOGGER.info(f"New video detected ({current_video_id}). State reset.")
            video_start_frame_idx = frame_idx

        # Update the tracker for the next iteration
        prev_video_id = current_video_id

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]  # batch dimension

        with dt[1]:
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_path)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # --------------------------------------
        # 2) Process Predictions
        # --------------------------------------
        for i, det in enumerate(pred):
            seen += 1

            # Multi-webcam logic
            if hasattr(dataset, 'count') and isinstance(path, list):
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}'
            )

            # Possibly warp entire frame
            im0_warped = cv2.warpPerspective(
                im0,
                perspective_matrix,
                (goal_realworld_size[0], goal_realworld_size[1])
            )

            # Rescale boxes
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print detection summary
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            s += '%gx%g ' % im.shape[2:]

            # --------------------------------------
            # 3) Process each detection
            # --------------------------------------
            all_detections_for_frame = []
            for *xyxy, conf, cls in det:
                det_result = process_single_detection(
                    xyxy,
                    conf,
                    cls,
                    perspective_matrix,
                    save_crop,
                    hide_labels,
                    hide_conf,
                    names
                )
                if det_result:
                    all_detections_for_frame.append(det_result)

            # -------------
            # (A) Save detections to YOLO TXT
            # -------------
            if save_txt and goal_realworld_size is not None and len(all_detections_for_frame) > 0:
                save_all_detections_txt(
                    all_detections_for_frame,
                    txt_path,
                    goal_realworld_size,
                    save_conf
                )

            # -------------
            # (B) Draw bounding boxes
            # -------------
            if draw_bbox:
                draw_all_detections(
                    im0_warped,
                    all_detections_for_frame,
                    line_thickness
                )

            # (1) Store the final annotated frame in our buffer (pre-trigger frames)
            frames_buffer.append(im0_warped.copy())  # copy() so we don’t overwrite by reference

            # (2) If we are skipping, just decrement skip_counter
            if skip_counter > 0:
                skip_counter -= 1
            else:
                # (3) We are not skipping => check for big ball triggers
                im0_h, im0_w = im0_warped.shape[:2]
                triggered = False

                for det_result in all_detections_for_frame:
                    if det_result['cls'] == 32:  # ball class
                        wx1, wy1, wx2, wy2 = det_result['bbox_warp']
                        w = wx2 - wx1
                        h = wy2 - wy1
                        # Calculate center of warped bbox
                        cx = (wx1 + wx2) / 2.0
                        cy = (wy1 + wy2) / 2.0

                        if w >= 70 and h >= 70:
                            triggered = True
                            break

                # (4) If triggered and we are not already recording a clip, start recording post-trigger frames
                if triggered and active_clip_writer is None:
                    clip_index += 1
                    clip_name = f"clip_{clip_index}.mp4"
                    clip_path = str(save_dir / 'clips' / clip_name)
                    (save_dir / 'clips').mkdir(exist_ok=True)  # ensure subdir

                    # Create a new video writer
                    height, width = im0_warped.shape[:2]
                    writer = cv2.VideoWriter(
                        clip_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (width, height)
                    )

                    # (a) Write out the pre-trigger frames stored in the buffer
                    for old_frame in frames_buffer:
                        writer.write(old_frame)

                    # Instead of releasing writer here, we keep it open to record post-trigger frames
                    active_clip_writer = writer
                    post_trigger_counter = 0

                    # Compute timestamps and other metadata as before
                    video_start_time, video_fps = get_video_start_time_and_fps(path)
                    trigger_time, video_time = calculate_real_timestamp(
                        video_start_time,
                        video_start_frame_idx,
                        frame_idx,
                        video_fps,
                    )
                    speed = find_max_speed_in_range(
                        radar_data_path,
                        trigger_time,
                        time_buffer=60,
                        csv_utc_offset=8
                    )
                    clip_path_dict = {
                        'clip_path': clip_path,
                        'speed': speed,
                        'video_time': video_time,
                        'real_time': trigger_time
                    }
                    collection_of_speed_dict.append(clip_path_dict)
                    # Set skip counter to avoid immediate re-triggering
                    skip_counter = 900
                    LOGGER.info(f"Triggered and started recording clip => {clip_path}")

            # (Extra) If we are in post-trigger recording mode, record the current frame
            if active_clip_writer is not None:
                # Write the current frame (post-trigger)
                active_clip_writer.write(im0_warped.copy())
                post_trigger_counter += 1
                # Once we have recorded enough post-trigger frames, finish the clip
                if post_trigger_counter >= post_trigger_frames:
                    active_clip_writer.release()
                    active_clip_writer = None
                    post_trigger_counter = 0
                    LOGGER.info("Finished recording post-trigger frames for current clip.")

            # Show in a window
            if view_img:
                handle_view_img(p, windows, im0_warped)

            # Save image / video
            if save_img:
                handle_save_results(
                    dataset,
                    i,
                    im0_warped,
                    save_path,
                    vid_path,
                    vid_writer
                )

        if pbar:
            pbar.set_postfix_str(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            pbar.update(1)
        else:
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    if pbar:
        pbar.close()

    return seen, windows, dt, collection_of_speed_dict



def process_single_detection(
    xyxy,
    conf,
    cls,
    perspective_matrix,
    save_crop,
    hide_labels,
    hide_conf,
    names,
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

    # Warp bounding box corners
    corners_src = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    corners_dst = perspective_transform_points(corners_src, perspective_matrix)
    wx1, wy1 = corners_dst[:, 0].min(), corners_dst[:, 1].min()
    wx2, wy2 = corners_dst[:, 0].max(), corners_dst[:, 1].max()
    detection_result['bbox_warp'] = [wx1, wy1, wx2, wy2]


    return detection_result


def save_all_detections_txt(
    all_detections,
    txt_path,
    goal_realworld_size,
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
            norm_cx = cx / goal_realworld_size[0]
            norm_cy = cy / goal_realworld_size[1]
            norm_w  = bw / goal_realworld_size[0]
            norm_h  = bh / goal_realworld_size[1]

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
    have_perspective,
    perspective_matrix,
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

    # If perspective, warp skeleton coords
    if have_perspective and perspective_matrix is not None:
        skel_src = np.array(points, dtype=np.float32)
        skel_dst = perspective_transform_points(skel_src, perspective_matrix)

        # Optionally clamp them to the warped image size
        # but we can just pass them along if you don't need strict clamping
        return keypoints_dict, skel_dst  
    else:
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
    dataset,
    i,
    im0_warped_final,
    save_path,
    vid_path,
    vid_writer
):
    """
    Save either a single image or frames of a video/stream.
    """
    # print(f"Dataset mode: {dataset.mode}")
    if dataset.mode == 'image':
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

            print(f"Writing frame {i} to video: {save_path}")
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
    goal_image_coordinate,       # list of 4 points [[x,y], [x,y], [x,y], [x,y]]
    goal_realworld_size,         # output width x height
    draw_bbox,
    radar_data_path,
    use_tqdm,
):

    # --- 1) Prepare perspective transform if needed ---
    perspective_matrix = prepare_perspective(
        goal_image_coordinate, goal_realworld_size
    )

    # --- 2) Setup output directory ---
    save_dir = setup_output_dir(project, name, exist_ok, save_txt)
    save_img = not nosave and not str(source).endswith('.txt')

    # --- 3) Load YOLO model ---
    model, stride, names, pt = load_yolo_model(weights, device, dnn, data, half)
    imgsz = check_img_size(imgsz, s=stride)

    # --- 4) Create dataloader ---
    dataset, bs, webcam_mode = create_dataloader(
        source, imgsz, stride, pt, vid_stride
    )

    # --- 5) Warm up YOLO model ---
    warmup_yolo_model(model, pt, bs, imgsz)

    # --- 6) Inference over dataset (main loop) ---
    seen, windows, dt, collection_of_speed_dict = infer_on_dataset(
        dataset, 
        model,
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
        perspective_matrix,
        goal_realworld_size,
        save_img,
        save_txt,
        save_conf,
        save_crop,
        line_thickness,
        hide_labels,
        hide_conf,
        view_img,
        save_dir,
        draw_bbox,
        radar_data_path,
        use_tqdm,
    )

    # --- 8) Summaries & Cleanup ---
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

    print(collection_of_speed_dict)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
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
    parser.add_argument('--goal_image_coordinate', nargs='*' ,type=int, default=None, help='four points(x1,y1,...,x4,y4) for perspective transform')
    parser.add_argument('--goal_realworld_size', nargs='*' ,type=int, default=[2100, 700], help='output width x height')
    parser.add_argument('--draw-bbox', action='store_true', help='draw bounding boxes')
    parser.add_argument('--radar_data_path', type=str, default=None, help='radar data path csv file')
    parser.add_argument('--use-tqdm', action='store_true', help='use tqdm for progress bar')

    opt = parser.parse_args()

    # Convert flat list to nested list of coordinates
    if opt.goal_image_coordinate:
        if len(opt.goal_image_coordinate) != 8:
            raise ValueError("Please provide 4 points x,y coordinate for perspective transform.")
        
        opt.goal_image_coordinate = [
            [opt.goal_image_coordinate[i], opt.goal_image_coordinate[i + 1]] 
            for i in range(0, len(opt.goal_image_coordinate), 2)
        ]

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

# python3 detect_goal.py --weights "./weight/yolov9-c-converted.pt" --source "./data/video/GX010025.MP4" --name 'demo_video' --goal_image_coordinate 77 233 1665 247 1655 758 79 765 --goal_realworld_size 2400 800 --nosave --radar_data_path data/excel/PR_20250208_1739_session.csv 