import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import numpy as np
import supervision as sv

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
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
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        use_slicer=False,  # use supervision InferenceSlicer for large images
        slice_size=(640, 640),  # slice width and height
        nms_threshold=0.1,  # NMS threshold for slicer
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Check for supervision library if using slicer
    if use_slicer:
        try:
            import supervision as sv
        except ImportError:
            LOGGER.error("Supervision package is required for slice inference. Install with 'pip install supervision'")
            return

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for path, im, im0s, vid_cap, s in dataset:
        # Handle slice-based detection
        if use_slicer:
            # Process each image in the batch
            for i, im0 in enumerate(im0s if isinstance(im0s, list) else [im0s]):
                # Pre-processing - dt[0]
                with dt[0]:
                    # No preprocessing needed for the slicer mode as it's done inside the callback
                    pass
                
                # Create slicer callback dynamically to access the model
                def slicer_callback(image_slice: np.ndarray):
                    with torch.no_grad():
                        h, w = image_slice.shape[:2]
                        # Check if dimensions need padding to be same as expected slice size (1280*1280)
                        need_padding = (h != 1280) or (w != 1280)
                        if need_padding:
                            # Calculate padding needed
                            pad_h = (1280 - h) if h < 1280 else 0
                            pad_w = (1280 - w) if w < 1280 else 0

                            # Apply padding (right and bottom)
                            padded_slice = cv2.copyMakeBorder(
                                image_slice, 
                                0, pad_h, 
                                0, pad_w, 
                                cv2.BORDER_CONSTANT,
                                value=(114, 114, 114)  # Using gray color common in YOLO
                            )
                            # Use the padded image for further processing
                            image_slice = padded_slice
                            
                        # Convert image to tensor (similar to the original preprocessing)
                        img = torch.from_numpy(image_slice.transpose(2, 0, 1)).to(model.device)
                        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
                        img /= 255  # 0 - 255 to 0.0 - 1.0
                        if len(img.shape) == 3:
                            img = img[None]  # expand for batch dim

                        # img = img.clone().detach()
                        # Run inference (similar to original inference)
                        pred = model(img, augment=augment)
                    
                    # Apply NMS (similar to original NMS)
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                    
                    # Convert to supervision Detections format
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

                    return sv.Detections(
                        xyxy=boxes,
                        confidence=confidences,
                        class_id=class_ids
                    )
                
                # Create slicer on-demand for each frame
                current_overlap = (0.2, 0.2)  # overlap ratio for the slicer
                overlap_wh = (slice_size[0] * current_overlap[0], slice_size[1] * current_overlap[1])
                current_slicer = sv.InferenceSlicer(callback=slicer_callback, slice_wh=slice_size, overlap_ratio_wh=None, overlap_wh=overlap_wh)

                # Inference - dt[1]
                with dt[1]:
                    detections = current_slicer(im0).with_nms(threshold=nms_threshold)
                
                # NMS - dt[2]
                with dt[2]:
                    detections = tracker.update_with_detections(detections)
                
                # Process results (similar to original)
                p = Path(path) if isinstance(path, str) else Path(path[i])
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{getattr(dataset, "frame", 0)}')  # im.txt
                s += '%gx%g ' % (im0.shape[1], im0.shape[0])  # print string
                
                # Create annotated frame
                annotated_frame = im0.copy()
                if len(detections):
                    # Add bounding boxes
                    annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
                    
                    # Add labels and prepare text output
                    labels = []
                    # Count unique classes correctly
                    class_counts = {}
                    for class_id in detections.class_id:
                        class_name = names[int(class_id)]
                        if class_name not in class_counts:
                            class_counts[class_name] = 0
                        class_counts[class_name] += 1
                    
                    # Add counts to string in a single pass
                    for class_name, count in class_counts.items():
                        s += f"{count} {class_name}{'s' * (count > 1)}, "
                    
                    # Build labels for individual annotations
                    for class_id, confidence in zip(detections.class_id, detections.confidence):
                        labels.append(f"{names[class_id]} {confidence:.2f}")

                    annotated_frame = BOX_LABEL_ANNOTATOR.annotate(scene=annotated_frame, detections=detections, labels=labels)

                    # Write results to txt file if needed
                    if save_txt:
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        for xyxy, conf, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                            # Convert to PyTorch tensor
                            xyxy_tensor = torch.tensor(xyxy).view(1, 4)
                            # Convert to xywh format (normalized)
                            xywh = (xyxy2xywh(xyxy_tensor) / gn).view(-1).tolist()
                            line = (cls_id, *xywh, conf) if save_conf else (cls_id, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                
                # Save results
                if save_crop:
                    # Handle crop saving
                    for xyxy, cls_id in zip(detections.xyxy, detections.class_id):
                        xyxy_list = xyxy.tolist()
                        save_one_box(xyxy_list, im0.copy(), file=save_dir / 'crops' / names[cls_id] / f'{p.stem}.jpg', BGR=True)
                
                # View results
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(str(p), annotated_frame.shape[1], annotated_frame.shape[0])
                    cv2.imshow(str(p), annotated_frame)
                    cv2.waitKey(1)
                
                # Save results
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, annotated_frame)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, annotated_frame.shape[1], annotated_frame.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(annotated_frame)
                
                # Increment seen count
                seen += 1
                
            # Print time info
            LOGGER.info(f"{s}{'' if len(detections) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
            
        else:
            # Regular processing (non-slicer mode)
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--use-slicer', action='store_true', help='use supervision InferenceSlicer for large images')
    parser.add_argument('--slice-size', nargs='+', type=int, default=[640, 640], help='slice width and height')
    parser.add_argument('--nms-threshold', type=float, default=0.1, help='NMS threshold for slicer')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.slice_size = tuple(opt.slice_size)  # convert to tuple
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
