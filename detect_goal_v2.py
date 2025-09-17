import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import numpy as np
import onepose
import subprocess
import re
import json


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
from tools.goalkeeper_motion_classification import classify_goalkeeper_behavior
from collections import deque, defaultdict
from tools.extract_datetime import get_video_start_time_and_fps, calculate_real_timestamp
from tools.extract_speed_data import find_max_speed_in_range
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable

detections_per_file = defaultdict(lambda: defaultdict(list))

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


def parse_gopro_name(path_str: str) -> Tuple[str, int]:
    """
    GX010036 -> (group_key='0036', chapter=01).
    Fallback: (stem, 0)
    """
    stem = Path(path_str).stem
    m = re.match(r'(?i)^GX(\d{2})(\d{4})$', stem)
    if m:
        chap = int(m.group(1))
        base = m.group(2)
        return base, chap
    return stem, 0

def apply_cooldown(trigs: List[int], fps: float, cooldown_s: float) -> List[int]:
    if not trigs: return []
    cool = int(round(cooldown_s * fps))
    kept, last = [], -10**12
    for f in sorted(set(trigs)):
        if f - last >= cool:
            kept.append(f); last = f
    return kept

def build_windows(triggers: List[int], fps: float, pre_s: float, post_s: float) -> List[Tuple[int,int,int]]:
    """
    Returns merged [(start_gf, end_gf, repr_gf), ...]; end is exclusive.
    """
    if not triggers: return []
    pre = int(round(pre_s * fps)); post = int(round(post_s * fps))
    spans = [(max(0, f - pre), f + post, f) for f in triggers]
    spans.sort(key=lambda x: x[0])
    merged = []
    s, e, rf = spans[0]
    for s2, e2, rf2 in spans[1:]:
        if s2 <= e:
            e = max(e, e2)
        else:
            merged.append((s, e, rf)); s, e, rf = s2, e2, rf2
    merged.append((s, e, rf))
    return merged

def run_ffmpeg(cmd: List[str]):
    print("FFmpeg:", " ".join(cmd))
    out = subprocess.run(cmd, text=True, capture_output=True)
    if out.returncode != 0:
        print("---- FFmpeg STDERR ----")
        print(out.stderr)
        print("-----------------------")
        raise subprocess.CalledProcessError(out.returncode, cmd, out.stdout, out.stderr)

def ffmpeg_cut_piece(video_path, fps, start_frame, end_frame, out_path, reencode=False):
    start_sec = start_frame / fps
    dur_sec   = max(0.1, (end_frame - start_frame) / fps)
    if not reencode:
        cmd = ["ffmpeg","-y","-ss",f"{start_sec:.3f}","-i",str(video_path),
               "-t",f"{dur_sec:.3f}","-c","copy",str(out_path)]
    else:
        cmd = ["ffmpeg","-y","-ss",f"{start_sec:.3f}","-i",str(video_path),
               "-t",f"{dur_sec:.3f}","-c:v","libx264","-preset","veryfast","-crf","18",
               "-c:a","copy",str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _probe_fps(path: str):
    out = subprocess.run(
        ["ffprobe","-v","error","-select_streams","v:0",
         "-show_entries","stream=avg_frame_rate,r_frame_rate",
         "-of","default=nokey=1:noprint_wrappers=1", path],
        text=True, capture_output=True, check=True
    ).stdout.strip().splitlines()
    rat = next((l for l in out if l and l != "0/0"), "30")
    if "/" in rat:
        a,b = rat.split("/")
        fps = float(a) / float(b) if float(b) else 30.0
    else:
        fps = float(rat or 30.0)
    return fps, rat

def _dense_gop_encoder_args(fps_int: int, no_bframes: bool = True):
    try:
        encs = subprocess.run(["ffmpeg","-hide_banner","-encoders"],
                              text=True, capture_output=True, check=True).stdout
    except Exception:
        encs = ""
    force_each_sec = ["-force_key_frames","expr:gte(t,n_forced*1)"]
    if "libx264" in encs:
        x264_params = f"keyint={fps_int}:min-keyint={fps_int}:scenecut=0:open_gop=0"
        if no_bframes:
            x264_params += ":bframes=0:ref=1"
        return ["-c:v","libx264","-preset","veryfast","-crf","18",
                "-pix_fmt","yuv420p","-profile:v","high","-level","4.1",
                "-x264-params", x264_params, *force_each_sec]
    if "h264_nvenc" in encs:
        base = ["-c:v","h264_nvenc","-preset","p4","-cq","19",
                "-pix_fmt","yuv420p","-profile:v","high","-level","4.1",
                "-g", str(fps_int), "-forced-idr","1", *force_each_sec]
        if no_bframes:
            base += ["-bf","0"]
        return base
    return ["-c:v","mpeg4","-qscale:v","4","-pix_fmt","yuv420p"]

def concat_cross_files_dense_video(parts: List[str], out_path: str, src_fps_rat: str, fps_int: int):
    enc = _dense_gop_encoder_args(fps_int, no_bframes=True)
    cmd = ["ffmpeg","-y","-hide_banner","-loglevel","error"]
    for p in parts:
        cmd += ["-i", str(p)]
    n = len(parts)

    flines, vlabels = [], []
    for i in range(n):
        v_in = f"[{i}:v:0]"; v_out = f"[v{i}]"
        flines.append(
            f"{v_in}setpts=PTS-STARTPTS,scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,format=yuv420p{v_out}"
        )
        vlabels.append(v_out)
    flines.append("".join(vlabels) + f"concat=n={n}:v=1:a=0[v]")

    cmd += [
        "-filter_complex",";".join(flines),
        "-map","[v]",
        "-vsync","cfr","-r", src_fps_rat,
        *enc,
        "-an","-movflags","+faststart",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)

# =========================
# RAM-safe detection sharder
# =========================

class DetShardWriter:
    """
    Writes detections to shards per file_key to keep memory flat.
    One JSON line per detection:
      {"frame_local": int, "cls_idx": int, "cls_name": str, "conf": float, "bbox_xyxy": [..], "bbox_warp":[..]?}
    """
    def __init__(self, root: Path, shard_size: int = 5000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        # cache of open file handles { (fkey, shard_id) : file_obj }
        self._open: Dict[Tuple[str,int], "io.TextIOWrapper"] = {}

    def _path_for(self, fkey: str, shard_id: int) -> Path:
        d = self.root / fkey
        d.mkdir(parents=True, exist_ok=True)
        start = shard_id * self.shard_size
        end   = start + self.shard_size - 1
        return d / f"block_{start:06d}-{end:06d}.jsonl"

    def write_det(self, fkey: str, frame_local: int, record: dict):
        shard_id = frame_local // self.shard_size
        key = (fkey, shard_id)
        if key not in self._open:
            p = self._path_for(fkey, shard_id)
            self._open[key] = open(p, "a", encoding="utf-8")
        fh = self._open[key]
        record = dict(record)  # ensure copy
        record["frame_local"] = int(frame_local)
        fh.write(json.dumps(record) + "\n")

    def close(self):
        for fh in self._open.values():
            try: fh.close()
            except Exception: pass
        self._open.clear()

class DetShardReader:
    def __init__(self, root: Path, shard_size: int = 5000):
        self.root = Path(root)
        self.shard_size = int(shard_size)

    def _paths_for_range(self, fkey: str, start_local: int, end_local: int) -> List[Path]:
        """end_local is exclusive"""
        if end_local <= start_local: return []
        first = start_local // self.shard_size
        last  = (max(start_local, end_local - 1)) // self.shard_size
        out = []
        for sid in range(first, last + 1):
            start = sid * self.shard_size
            end   = start + self.shard_size - 1
            p = self.root / fkey / f"block_{start:06d}-{end:06d}.jsonl"
            if p.exists():
                out.append(p)
        return out

    def iter_range(self, fkey: str, start_local: int, end_local: int) -> Iterable[dict]:
        """Yield per-detection records whose frame_local ∈ [start_local, end_local)."""
        for p in self._paths_for_range(fkey, start_local, end_local):
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    fl = rec.get("frame_local")
                    if fl is None: continue
                    if start_local <= int(fl) < end_local:
                        yield rec

# =========================
# Core phases
# =========================

def _scan_pass(
    dataset, model, names, conf_thres, iou_thres, max_det, classes, agnostic_nms,
    augment, perspective_matrix, save_crop, hide_labels, hide_conf,
    use_tqdm: bool, batch_size: int, tmp_det_root: Path
):
    """
    Returns:
      seen, dt_profiles, cache dicts:
        - file_path_map, fps_per_file, start_time_per_file, first_frame_in_file, last_frame_in_file,
        - group_key_of_file, chapter_of_file, triggers_per_file
    Side-effect:
      writes per-detection JSONL shards under tmp_det_root/<fkey>/block_*.jsonl
    """
    from utils.general import LOGGER  # if you have it elsewhere, adjust
    dt = (Profile(), Profile(), Profile())
    shard_writer = DetShardWriter(tmp_det_root, shard_size=5000)

    triggers_per_file = defaultdict(list)
    fps_per_file, start_time_per_file = {}, {}
    first_frame_in_file, last_frame_in_file = {}, {}
    file_path_map, group_key_of_file, chapter_of_file = {}, {}, {}
    frame_counter_map = defaultdict(int)

    total_frames = len(dataset) if hasattr(dataset, '__len__') else None
    pbar = tqdm(total=total_frames, desc="Scan (two-pass)") if (use_tqdm and total_frames) else None

    # batch buffers
    batch_tensors, batch_in_shapes, batch_im0s = [], [], []
    batch_paths, batch_s_bases, batch_frames_local, batch_file_keys = [], [], [], []
    seen = 0

    # class id
    ball_cls_id = names.index('ball') if isinstance(names, (list, tuple)) and 'ball' in names else 32

    def _flush_batch():
        nonlocal seen
        if not batch_tensors: return

        with dt[0]:
            im = torch.stack(batch_tensors, 0).to(model.device)
            im = im.half() if getattr(model, "fp16", False) else im
            im /= 255.0
        with dt[1]:
            pred = model(im, augment=augment)
        with dt[2]:
            preds = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for k, det in enumerate(preds):
            seen += 1
            path = batch_paths[k]
            im0  = batch_im0s[k]
            in_h, in_w = batch_in_shapes[k]
            s = batch_s_bases[k]
            frame_local = batch_frames_local[k]
            fkey = batch_file_keys[k]

            # cache per-file once
            if fkey not in file_path_map:
                apath = str(Path(path).resolve())
                file_path_map[fkey] = apath
                try:
                    v_start_time, v_fps = get_video_start_time_and_fps(apath)
                except Exception:
                    v_start_time, v_fps = None, 30.0
                fps_per_file[fkey] = float(v_fps or 30.0)
                start_time_per_file[fkey] = v_start_time
                first_frame_in_file[fkey] = frame_local
                last_frame_in_file[fkey]  = frame_local

            last_frame_in_file[fkey] = max(last_frame_in_file[fkey], frame_local)

            # rescale & collect detections
            if det is not None and len(det):
                det[:, :4] = scale_boxes((in_h, in_w), det[:, :4], im0.shape).round()

            triggered = False
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(float, xyxy)
                    det_result = process_single_detection(
                        xyxy, conf, cls, perspective_matrix,
                        save_crop, hide_labels, hide_conf, names
                    )
                    rec = {
                        "file_key": fkey,
                        "cls_idx": int(cls.item() if hasattr(cls, "item") else cls),
                        "cls_name": names[int(cls)] if isinstance(names, (list, tuple)) and int(cls) < len(names) else str(int(cls)),
                        "conf": float(conf.item() if hasattr(conf, "item") else conf),
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                    if det_result and "bbox_warp" in det_result:
                        rec["bbox_warp"] = [float(v) for v in det_result["bbox_warp"]]
                    shard_writer.write_det(fkey, int(frame_local), rec)

                    # trigger logic
                    if det_result and int(det_result['cls']) == int(ball_cls_id):
                        wx1, wy1, wx2, wy2 = det_result['bbox_warp']
                        if (wx2 - wx1) >= 70 and (wy2 - wy1) >= 70:
                            triggered = True

            if triggered:
                triggers_per_file[fkey].append(int(frame_local))

            # progress
            if pbar:
                pbar.set_postfix_str(f"{s}{'' if (det is not None and len(det)) else '(no det), '}{dt[1].dt * 1E3:.1f}ms")
                pbar.update(1)
            else:
                LOGGER.info(f"{s}{'' if (det is not None and len(det)) else '(no det), '}{dt[1].dt * 1E3:.1f}ms")

        # clear batch
        batch_tensors.clear(); batch_in_shapes.clear(); batch_im0s.clear()
        batch_paths.clear(); batch_s_bases.clear(); batch_frames_local.clear(); batch_file_keys.clear()

    # iterate dataset -> build batches
    for _idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        if hasattr(dataset, 'count') and isinstance(path, list):
            p, im0, frame_local = path[0], im0s[0].copy(), dataset.count
            s_local = f'0: ' + s
        else:
            p, im0, frame_local = path, im0s.copy(), getattr(dataset, 'frame', None)
            s_local = s

        fkey = Path(p).stem
        gkey, chap = parse_gopro_name(p)
        group_key_of_file[fkey] = gkey
        chapter_of_file[fkey] = chap

        if frame_local is None:
            frame_local = frame_counter_map[fkey]
            frame_counter_map[fkey] += 1

        x = torch.from_numpy(im).to(model.device)
        x = x.half() if getattr(model, "fp16", False) else x.float()
        if len(x.shape) != 3:
            x = x.permute(2, 0, 1).contiguous()

        batch_tensors.append(x)
        H_in, W_in = x.shape[-2:]
        batch_in_shapes.append((H_in, W_in))
        batch_im0s.append(im0)
        batch_paths.append(p)
        batch_s_bases.append(s_local)
        batch_frames_local.append(int(frame_local))
        batch_file_keys.append(fkey)

        if len(batch_tensors) >= batch_size:
            _flush_batch()

    _flush_batch()
    if pbar: pbar.close()
    shard_writer.close()

    return (seen, dt,
            file_path_map, fps_per_file, start_time_per_file, first_frame_in_file, last_frame_in_file,
            group_key_of_file, chapter_of_file, triggers_per_file)

def _build_segments(
    file_path_map, fps_per_file, start_time_per_file,
    first_frame_in_file, last_frame_in_file,
    group_key_of_file, chapter_of_file
):
    groups_present = {group_key_of_file[fk] for fk in file_path_map.keys()}
    if len(groups_present) == 0:
        raise ValueError("No videos found in the dataset.")
    if len(groups_present) > 1:
        raise ValueError(f"Multiple GoPro groups detected in folder: {sorted(groups_present)}. "
                         f"Please pass a folder with a single chaptered sequence.")
    gkey = list(groups_present)[0]

    segs = []
    for fkey, path_abs in file_path_map.items():
        if group_key_of_file[fkey] != gkey: continue
        segs.append({
            'fkey': fkey,
            'path': path_abs,
            'chapter': chapter_of_file[fkey],
            'fps': fps_per_file.get(fkey, 30.0),
            'start_time': start_time_per_file.get(fkey),
            'first_frame': int(first_frame_in_file.get(fkey, 0)),
            'last_frame':  int(last_frame_in_file.get(fkey, 0)),
        })
    if not segs:
        return gkey, [], None, None

    segs.sort(key=lambda d: d['chapter'])
    group_fps = float(segs[0]['fps'])
    group_start_time = segs[0]['start_time']

    # global offsets across chapters
    offset = 0
    for seg in segs:
        seg_len = max(0, (int(seg['last_frame']) - int(seg['first_frame']) + 1))
        seg['g_start'] = offset
        seg['g_end']   = offset + seg_len
        offset = seg['g_end']

    return gkey, segs, group_fps, group_start_time

def _compute_global_triggers(segs, triggers_per_file) -> List[int]:
    seg_by_fkey = {s["fkey"]: s for s in segs}
    gtrigs = []
    for fkey, locs in triggers_per_file.items():
        seg = seg_by_fkey.get(fkey)
        if not seg: continue
        f0 = int(seg['first_frame'])
        g0 = int(seg['g_start'])
        for f in locs:
            gtrigs.append(g0 + (int(f) - f0))
    return gtrigs

def _local_to_global_builder(segs):
    seg_by_fkey = {s["fkey"]: s for s in segs}
    def local_to_global(fkey: str, frame_local: int) -> int:
        seg = seg_by_fkey[fkey]
        return int(seg['g_start'] + (int(frame_local) - int(seg['first_frame'])))
    return local_to_global

def _write_clip_jsonl_for_window(
    clip_path: Path, meta_info: dict, segs: List[dict],
    gs_global: int, ge_global: int, reader: DetShardReader, local_to_global
):
    out_path = clip_path.with_suffix(".jsonl")
    with open(out_path, "w", encoding="utf-8") as jf:
        jf.write(json.dumps({"type":"meta", **meta_info}) + "\n")

        for seg in segs:
            s_int = max(gs_global, int(seg['g_start']))
            e_int = min(ge_global, int(seg['g_end']))
            if s_int >= e_int: continue

            local_start = int(seg['first_frame'] + (s_int - seg['g_start']))
            local_end   = int(local_start + (e_int - s_int))  # exclusive
            fkey = seg["fkey"]

            for rec in reader.iter_range(fkey, local_start, local_end):
                f_global = local_to_global(fkey, int(rec["frame_local"]))
                f_clip   = int(f_global - gs_global)
                out = {
                    "type": "det",
                    "clip_frame": f_clip,
                    "global_frame": int(f_global),
                    "file_key": fkey,
                    "frame_local": int(rec["frame_local"]),
                    "cls_idx": int(rec["cls_idx"]),
                    "cls_name": rec.get("cls_name"),
                    "conf": float(rec["conf"]),
                    "bbox_xyxy": [float(v) for v in rec["bbox_xyxy"]],
                }
                if "bbox_warp" in rec:
                    out["bbox_warp"] = [float(v) for v in rec["bbox_warp"]]
                jf.write(json.dumps(out) + "\n")
    return str(out_path)

# =========================
# Public API
# =========================

def infer_on_dataset(
    dataset,
    model,
    names,
    conf_thres,
    iou_thres,
    max_det,
    classes,
    agnostic_nms,
    augment,
    visualize,              # unused in two-pass
    perspective_matrix,
    save_img,               # ignored in scan mode
    save_txt,               # unused
    save_conf,              # unused
    save_crop,
    hide_labels,
    hide_conf,
    view_img,               # ignored in scan mode
    save_dir,
    radar_data_path,
    use_tqdm,
    batch_size=32,
    pre_s=5.0,
    post_s=5.0,
    cooldown_s=30.0,
    ffmpeg_reencode=False,
):
    """
    Two-pass pipeline:
      Pass-1: scan + write detections to shards
      Pass-2: build segments/windows, cut clips, and write per-clip JSONL
    Returns:
      seen, windows, dt, collection_of_speed_dict
    """
    save_dir = Path(save_dir)
    clips_root = save_dir / "clips"
    tmp_det_root = save_dir / "tmp_dets"
    clips_root.mkdir(parents=True, exist_ok=True)
    tmp_det_root.mkdir(parents=True, exist_ok=True)

    # ---- Pass-1
    (seen, dt,
     file_path_map, fps_per_file, start_time_per_file, first_frame_in_file, last_frame_in_file,
     group_key_of_file, chapter_of_file, triggers_per_file) = _scan_pass(
        dataset, model, names, conf_thres, iou_thres, max_det, classes, agnostic_nms,
        augment, perspective_matrix, save_crop, hide_labels, hide_conf,
        use_tqdm, batch_size, tmp_det_root
    )

    # Quick out if empty
    if not file_path_map:
        return seen, [], dt, []

    # ---- Segments
    gkey, segs, group_fps, group_start_time = _build_segments(
        file_path_map, fps_per_file, start_time_per_file,
        first_frame_in_file, last_frame_in_file,
        group_key_of_file, chapter_of_file
    )
    if not segs:
        return seen, [], dt, []

    # fps + src rational
    src_fps_float, src_fps_rat = _probe_fps(segs[0]['path'])
    fps_int = max(1, int(round(src_fps_float)))

    # Global triggers -> windows
    global_triggers = _compute_global_triggers(segs, triggers_per_file)
    if not global_triggers:
        return seen, [], dt, []

    cooled = apply_cooldown(global_triggers, group_fps, cooldown_s)
    merged = build_windows(cooled, group_fps, pre_s, post_s)  # (gs, ge, repr_gf)

    # ---- Cut & JSONL
    out_group_dir = clips_root / f"{gkey}"
    out_group_dir.mkdir(parents=True, exist_ok=True)

    reader = DetShardReader(tmp_det_root, shard_size=5000)
    local_to_global = _local_to_global_builder(segs)

    collection_of_speed_dict = []
    result_windows = []

    for i, (gs, ge, repr_gf) in enumerate(merged, 1):
        parts = []
        try:
            # collect per-file parts
            for seg in segs:
                s_int = max(gs, int(seg['g_start']))
                e_int = min(ge, int(seg['g_end']))
                if s_int >= e_int: continue
                local_start = int(seg['first_frame'] + (s_int - seg['g_start']))
                local_end   = int(local_start + (e_int - s_int))
                part_path = out_group_dir / f"__part_{i:03d}_{int(seg['chapter']):02d}.mp4"
                ffmpeg_cut_piece(seg['path'], float(group_fps), int(local_start), int(local_end),
                                 part_path, reencode=ffmpeg_reencode)
                parts.append(str(part_path))

            final_name = f"{Path(segs[0]['path']).stem}_clip_{i:03d}.mp4"
            final_path = out_group_dir / final_name

            if len(parts) == 1:
                os.replace(parts[0], final_path)
            else:
                concat_cross_files_dense_video(parts, final_path, src_fps_rat, fps_int)
                for p in parts:
                    try: os.remove(p)
                    except Exception: pass

            # timestamps for representative frame (global)
            try:
                trigger_time, video_time = calculate_real_timestamp(group_start_time, 0, int(repr_gf), float(group_fps))
            except Exception:
                trigger_time, video_time = None, None

            # radar speed near trigger
            try:
                speed = find_max_speed_in_range(radar_data_path, trigger_time, time_buffer=60, csv_utc_offset=8)
            except Exception:
                speed = None

            collection_of_speed_dict.append({
                'clip_path': str(final_path),
                'speed': speed,
                'video_time': video_time,
                'real_time': trigger_time
            })
            result_windows.append({
                'start_frame_global': int(gs),
                'end_frame_global': int(ge),
                'fps': float(group_fps),
            })

            # Per-clip detections JSONL
            meta = {
                "clip_path": str(final_path),
                "group_key": gkey,
                "window_global": [int(gs), int(ge)],
                "fps": float(group_fps),
                "video_time_repr": video_time,
                "real_time_repr": str(trigger_time) if trigger_time else None,
            }
            _write_clip_jsonl_for_window(final_path, meta, segs, int(gs), int(ge), reader, local_to_global)

        except Exception as e:
            LOGGER.warning(f"Failed to build clip window {i}: {e}")

    return seen, result_windows, dt, collection_of_speed_dict

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


def draw_all_detections(im0, all_detections, pose_model, line_thickness=1):
    """
    Draw bounding boxes and skeletons for all detections in the frame.
    This function is called once after we've processed all detections.
    """
    annotator = Annotator(im0, line_width=line_thickness)

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
    im0[:, :, :] = final_img  # copy back if needed so caller sees changes


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


def handle_view_img(p, windows, im0_final):
    """
    Show results in a window, if user sets `view_img=True`.
    """
    if platform.system() == 'Linux' and p not in windows:
        windows.append(p)
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(str(p), im0_final.shape[1], im0_final.shape[0])
    cv2.imshow(str(p), im0_final)
    cv2.waitKey(1)


def handle_save_results(
    dataset,
    i,
    im0_final,
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
        cv2.imwrite(save_path, im0_final)
    else:
        # Save video/stream
        if i < len(vid_path):
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()

                fps, w_vid, h_vid = 30, im0_final.shape[1], im0_final.shape[0]
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
            vid_writer[i].write(im0_final)


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
    homography_path,
    # goal_image_coordinate,       # list of 4 points [[x,y], [x,y], [x,y], [x,y]]
    # goal_realworld_size,         # output width x height
    draw_bbox,
    radar_data_path,
    use_tqdm,
):

    # --- 1) Prepare perspective transform if needed ---
    # perspective_matrix = prepare_perspective(
    #     goal_image_coordinate, goal_realworld_size
    # )
    perspective_matrix = np.load(homography_path)

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
        conf_thres,
        iou_thres,
        max_det,
        classes,
        agnostic_nms,
        augment,
        visualize,
        perspective_matrix,
        save_img,
        save_txt,
        save_conf,
        save_crop,
        hide_labels,
        hide_conf,
        view_img,
        save_dir,
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
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='NMS IoU threshold')
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
    parser.add_argument("--homography_path", type=str, default=None, help="Single 3x3 homography .npy applied to all")
    # parser.add_argument('--goal_image_coordinate', nargs='*' ,type=int, default=None, help='four points(x1,y1,...,x4,y4) for perspective transform')
    # parser.add_argument('--goal_realworld_size', nargs='*' ,type=int, default=[2100, 700], help='output width x height')
    parser.add_argument('--draw-bbox', action='store_true', help='draw bounding boxes')
    parser.add_argument('--radar_data_path', type=str, default=None, help='radar data path csv file')
    parser.add_argument('--use-tqdm', action='store_true', help='use tqdm for progress bar')

    opt = parser.parse_args()

    # # Convert flat list to nested list of coordinates
    # if opt.goal_image_coordinate:
    #     if len(opt.goal_image_coordinate) != 8:
    #         raise ValueError("Please provide 4 points x,y coordinate for perspective transform.")
        
    #     opt.goal_image_coordinate = [
    #         [opt.goal_image_coordinate[i], opt.goal_image_coordinate[i + 1]] 
    #         for i in range(0, len(opt.goal_image_coordinate), 2)
    #     ]

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
# --goal_image_coordinate 77 233 1665 247 1655 758 79 765 --goal_realworld_size 640 213 
# python3 detect_goal_v2.py --weights "./weight/yolov9-s-converted.pt" --source "./data/video/GX010025_clips/" --name 'demo_video' --nosave --radar_data_path data/excel/PR_20250208_1739_session.csv --homography_path ./runs/detect/demo_video/homography_matrix.npy
# standed goal size should be 24ft wide x 8ft tall = 732.6 cm wide x 244 cm tall, this is smaller only 21ft wide x 7ft tall = 640 cm wide x 213 cm tall