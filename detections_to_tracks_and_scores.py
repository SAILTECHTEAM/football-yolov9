#!/usr/bin/env python3
import os, json, glob, argparse, math, subprocess, gc
import numpy as np
import cv2
import joblib

from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, OrderedDict
from multiprocessing import Pool, cpu_count
from tools.identify_goalkeeper import extract_color_histogram_with_specific_background_color, extract_color_histogram_from_rotated_skelton, compare_histograms, load_histogram
# ByteTrack
import supervision as sv

from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo

# limit threaded libs to keep CPU sane
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
cv2.setNumThreads(1)


def load_pose_model():
    """
    Load the ViTPose model instead of OnePose.
    """
    # Configuration paths for ViTPose
    pose_config = '/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_simple_coco_256x192.py'
    pose_checkpoint = '/ViTPose/checkpoints/vitpose-h-simple.pth'
    
    # Initialize the pose model
    pose_model = init_pose_model(
        pose_config, 
        pose_checkpoint, 
        device='cuda'
    )
    
    return pose_model


def handle_pose_estimation(
    im0s,
    x1, y1, x2, y2,
    pose_model
):
    """
    Crops the region for a person and runs pose estimation using ViTPose.
    Returns:
      keypoints_dict: the original dictionary with 'points' and 'confidence'
      clamped_keypoints: final coords for visualization.
    """
    x1_safe, y1_safe = max(0, int(x1)), max(0, int(y1))
    x2_safe = min(im0s.shape[1], int(x2))
    y2_safe = min(im0s.shape[0], int(y2))

    # If bounding box too small, skip
    if (x2_safe - x1_safe) < 10 or (y2_safe - y1_safe) < 10:
        return None, []

    # Prepare person detection result for ViTPose format
    person_result = [{'bbox': [x1_safe, y1_safe, x2_safe, y2_safe, 1.0]}]
    
    # Get dataset info
    dataset_info = pose_model.cfg.data.test.dataset_info
    dataset_info = DatasetInfo(dataset_info)
    
    # Run inference
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        im0s,
        person_result,
        bbox_thr=0.0,  # Already filtered
        format='xyxy',
        dataset_info=dataset_info
    )
    
    if not pose_results:
        return None, []
    
    # Extract keypoints and scores from ViTPose format
    keypoints = pose_results[0]['keypoints']  # shape: Nx3 (x, y, score)
    points = keypoints[:, :2]  # shape: Nx2
    confidences = keypoints[:, 2]  # shape: Nx1
    
    # Create return structure to match expected format
    keypoints_dict = {
        'points': points,
        'confidence': confidences
    }
    
    return keypoints_dict, points

# ------------ ffprobe helpers ------------
def _ffprobe_fps(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=avg_frame_rate",
             "-of","default=nokey=1:noprint_wrappers=1", path],
            text=True, capture_output=True, check=True
        ).stdout.strip()
        if "/" in out:
            a,b = out.split("/")
            a = float(a); b = float(b) if float(b) else 0.0
            return a / b if b > 0 else 0.0
        return float(out or 0.0)
    except Exception:
        return 0.0

def _ffprobe_size(path: str):
    try:
        out = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=width,height",
             "-of","csv=p=0:s=x", path],
            text=True, capture_output=True, check=True
        ).stdout.strip()
        if "x" in out:
            w_s,h_s = out.split("x")
            return int(h_s), int(w_s)
    except Exception:
        pass
    return 0, 0

# ------------ streaming JSONL reader ------------
def stream_frames_from_jsonl(path: Path):
    """
    Yields (frame_idx, list_of_dets) in ascending frame order.
    Assumes input jsonl is already sorted by clip_frame (your pipeline writes it that way).
    Also returns the discovered class list.
    """
    classes = OrderedDict()
    meta = {}
    cur_f = None
    bucket = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("type")
            if t == "meta":
                meta = rec
                continue
            if t != "det":
                continue

            cf = int(rec["clip_frame"])
            k = (int(rec.get("cls_idx", 0)), rec.get("cls_name"))
            if k not in classes:
                classes[k] = None

            if cur_f is None:
                cur_f = cf
            if cf != cur_f:
                # emit previous frame
                yield cur_f, bucket
                bucket = []
                cur_f = cf
            bucket.append(rec)

    if cur_f is not None and bucket:
        yield cur_f, bucket

    return meta, list(classes.keys())

def projected_point_from_warp(rec):
    bw = rec.get("bbox_warp")
    if not bw: return None
    x1, y1, x2, y2 = map(float, bw)
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]

def select_class_dets(dets, cls_idx: int, cls_name: str, conf_thres: float):
    picked = [d for d in dets
              if int(d.get("cls_idx", -9999)) == cls_idx
              and d.get("cls_name") == cls_name
              and float(d.get("conf", 0.0)) >= conf_thres]
    if not picked:
        return np.zeros((0,4), np.float32), np.zeros((0,), np.float32), []
    boxes  = np.array([p["bbox_xyxy"] for p in picked], dtype=np.float32)
    scores = np.array([p.get("conf", 1.0) for p in picked], dtype=np.float32)
    return boxes, scores, picked

def iou_matrix_np(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x11, y11, x12, y12 = np.split(a, 4, axis=1)
    x21, y21, x22, y22 = np.split(b, 4, axis=1)
    xi1 = np.maximum(x11, x21.T)
    yi1 = np.maximum(y11, y21.T)
    xi2 = np.minimum(x12, x22.T)
    yi2 = np.minimum(y12, y22.T)
    inter = np.clip(xi2 - xi1, 0, None) * np.clip(yi2 - yi1, 0, None)
    area_a = (x12 - x11) * (y12 - y11)
    area_b = (x22 - x21) * (y22 - y21)
    union = area_a + area_b.T - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0).astype(np.float32)

# ------------ stream writer ------------
class TrackJsonlStreamer:
    def __init__(self, out_path: str, meta_head: dict, flush_interval: int = 200, lost_thresh: int = 50, is_person=False):
        self.out_path = Path(out_path)
        self.flush_interval = max(50, int(flush_interval))
        self.lost_thresh = max(25, int(lost_thresh))
        self.is_person = is_person

        self.records = defaultdict(lambda: {
            "track_id": None,
            "frame_id": [],
            "conf": [],
            "bbox": [],
            "projected": [],
            **({"team_score": [], "skel": [], "skel_conf": []} if is_person else {})
        })
        self.last_seen = {}

        self.fh = self.out_path.open("w", encoding="utf-8")
        if meta_head:
            json.dump(meta_head, self.fh, ensure_ascii=False); self.fh.write("\n")

    def update(self, tid, frame_idx, bbox, conf, proj_pt, *, team_score=None, skel=None, skel_conf=None):
        rec = self.records[tid]
        if rec["track_id"] is None:
            rec["track_id"] = tid
        rec["frame_id"].append(int(frame_idx))
        rec["conf"].append(float(conf) if conf is not None else None)
        rec["bbox"].append([float(v) for v in bbox])
        rec["projected"].append(None if proj_pt is None else [float(v) for v in proj_pt])

        if self.is_person:
            rec["team_score"].append(None if team_score is None else float(team_score))
            rec["skel"].append(skel if skel is None else (skel.tolist() if isinstance(skel, np.ndarray) else skel))
            rec["skel_conf"].append(skel_conf if skel_conf is None else (skel_conf.tolist() if isinstance(skel_conf, np.ndarray) else skel_conf))

        self.last_seen[tid] = int(frame_idx)

    def maybe_flush(self, frame_idx):
        if frame_idx % self.flush_interval != 0:
            return
        stale = [tid for tid, last in self.last_seen.items()
                 if frame_idx - last > self.lost_thresh]
        for tid in stale:
            self._write_record(self.records.pop(tid))
            self.last_seen.pop(tid, None)
        # proactive GC
        gc.collect()

    def _write_record(self, rec):
        json.dump(rec, self.fh, ensure_ascii=False)
        self.fh.write("\n")

    def close(self):
        for rec in self.records.values():
            self._write_record(rec)
        self.fh.close()
        self.records.clear()
        self.last_seen.clear()

# ------------ ball helpers ------------
def _is_ball_class(ci, cn, args):
    name_ok = (args.ball_name is None) or (cn == args.ball_name)
    return (ci == args.ball_idx) and name_ok

def _valid_number(x):
    return (x is not None) and not (isinstance(x, float) and np.isnan(x))

def _bbox_center_xyxy(b):
    x1, y1, x2, y2 = map(float, b)
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _projected_point_from_warp(rec):
    bw = rec.get("bbox_warp")
    if not bw:
        return None
    x1, y1, x2, y2 = map(float, bw)
    return [(x1 + x2) * 0.5, (y1 + y2) * 0.5]

class BallFuser:
    """
    Stream-time fuser for the ball (no tracker).
    Writes a single fused 'track' line at close().

    Strategy per frame:
      - filter by conf >= conf_thres
      - if we already have a previous center, choose candidate with min center distance
      - else choose highest conf (fallback to first)
    """
    def __init__(self, out_path: Path, meta_head: dict, conf_thres: float = 0.10):
        self.out_path = Path(out_path)
        self.conf_thres = float(conf_thres)
        self.meta_head = dict(meta_head or {})
        # streaming buffers for the single fused track
        self.frame_ids = []
        self.boxes = []
        self.proj = []
        self.confs = []
        self._prev_center = None
        # open file and write meta
        self.fh = self.out_path.open("w", encoding="utf-8")
        if self.meta_head:
            json.dump(self.meta_head, self.fh, ensure_ascii=False); self.fh.write("\n")

    @staticmethod
    def _center_xyxy(b):
        x1, y1, x2, y2 = map(float, b)
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def update_from_dets(self, dets_for_this_frame: list, frame_idx: int):
        # keep only confident candidates
        cands = []
        for d in dets_for_this_frame or []:
            c = float(d.get("conf", 0.0))
            if c >= self.conf_thres:
                cands.append(d)
        if not cands:
            return

        # pick best candidate
        pick = None
        if self._prev_center is not None:
            best_d2 = float("inf")
            for d in cands:
                cx, cy = self._center_xyxy(d["bbox_xyxy"])
                d2 = (cx - self._prev_center[0])**2 + (cy - self._prev_center[1])**2
                if d2 < best_d2:
                    best_d2 = d2; pick = d
        else:
            # first: highest conf
            best_conf = -1.0
            for d in cands:
                cc = float(d.get("conf", 0.0))
                if cc > best_conf:
                    best_conf = cc; pick = d

        if pick is None:
            return

        b = [float(v) for v in pick["bbox_xyxy"]]
        self.frame_ids.append(int(frame_idx))
        self.boxes.append(b)
        self.confs.append(float(pick.get("conf", 0.0)))
        self.proj.append(projected_point_from_warp(pick))
        self._prev_center = self._center_xyxy(b)

    def close(self):
        # write a single per-track row if we have anything
        if self.frame_ids:
            rec = {
                "track_id": -1,
                "frame_id": self.frame_ids,
                "conf": self.confs,
                "bbox": self.boxes,
                "projected": self.proj,
            }
            json.dump(rec, self.fh, ensure_ascii=False); self.fh.write("\n")
        self.fh.close()

# ------------ utils ------------
def _is_person_class(ci, cn):
    return (ci == 0) or (isinstance(cn, str) and cn.lower() == "person")

def _is_ball_class(ci, cn):
    """
    Treat 'ball' robustly:
    - class index 32
    - or class name literally 'ball' (case-insensitive)
    - or class name equals '32' (some pipelines carry numeric name strings)
    """
    try:
        if int(ci) == 32:
            return True
    except Exception:
        pass
    if isinstance(cn, str):
        if cn.lower() == "ball":
            return True
        # tolerate numeric string
        try:
            if int(cn) == 32:
                return True
        except Exception:
            pass
    return False

def _resolve_hw(meta: dict, probe_video: bool = True):
    # try meta first
    H = int(meta.get("H", 0) or 0)
    W = int(meta.get("W", 0) or 0)
    if H > 0 and W > 0:
        return H, W
    if probe_video:
        h2, w2 = _ffprobe_size(meta.get("clip_path", ""))
        if h2 > 0 and w2 > 0:
            return h2, w2
    # fallback
    return 1080, 1920

def load_histogram_any(path: str):
    if not path: return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".npy", ".npz"):
            arr = np.load(path, allow_pickle=False)
            if isinstance(arr, np.lib.npyio.NpzFile):
                key = list(arr.keys())[0]; arr = arr[key]
        elif ext in (".pkl", ".pickle", ".joblib"):
            arr = joblib.load(path)
        else:
            arr = np.load(path, allow_pickle=False)
    except Exception:
        arr = joblib.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    s = float(arr.sum())
    if s > 0:
        arr /= s
    return arr

# ------------ core per-file ------------

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
    clothes_colors_histogram,
    skip_small_area,
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
        if keypoints_dict and len(warped_points) > 0:
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
                if (x2 - x1) * (y2 - y1) < skip_small_area:
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

def process_one_file(jsonl_path: Path, args) -> str:
    # read meta + classes quickly (first pass only over meta line)
    meta = {}
    classes = OrderedDict()
    with open(jsonl_path, "r", encoding="utf-8") as f_head:
        for ln in f_head:
            ln = ln.strip()
            if not ln: continue
            rec = json.loads(ln)
            if rec.get("type") == "meta":
                meta = rec
                break
    # second quick pass just to discover classes (without holding dets)
    with open(jsonl_path, "r", encoding="utf-8") as f_cls:
        for ln in f_cls:
            ln = ln.strip()
            if not ln: continue
            rec = json.loads(ln)
            if rec.get("type") != "det":
                continue
            k = (int(rec.get("cls_idx", 0)), rec.get("cls_name"))
            if k not in classes:
                classes[k] = None
    class_list = list(classes.keys())

    # fps
    fps = float(args.fps or meta.get("fps", 0.0) or 0.0)
    if fps <= 0:
        fps = _ffprobe_fps(meta.get("clip_path", "")) or 30.0

    # size
    H, W = _resolve_hw(meta, probe_video=True)
    imsz = [max(1, H), max(1, W)]

    # open video (only if person+pose is needed)
    need_pose = any(_is_person_class(ci, cn) for (ci,cn) in class_list) and (not args.no_pose)
    cap = None
    if need_pose:
        vp = meta.get("clip_path", "")
        if vp and os.path.exists(vp):
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                cap = None

    pose_model = load_pose_model() if (need_pose and not args.no_pose) else None

    # trackers & writers (but: ball class uses a fuser instead)
    trackers = {}
    writers  = {}
    fusers   = {}
    outputs  = []

    for (ci, cn) in class_list:
        tag = f"{cn}" if cn else f"cls{ci}"
        out_path = jsonl_path.with_suffix(f".cls_{tag}.tracks.jsonl")
        meta_head = {"type":"meta", **{k:v for k,v in meta.items() if k!="type"},
                     "class_idx": ci, "class_name": cn}

        if _is_ball_class(ci, cn):
            # no tracker for ball; create a fuser that writes meta now
            fusers[(ci,cn)] = BallFuser(out_path, meta_head, conf_thres=args.det_conf)
            outputs.append(str(out_path))
            continue

        # normal tracker + writer for non-ball classes
        tr_args = SimpleNamespace(
            track_activation_threshold=args.track_thresh,
            lost_track_buffer=args.track_buffer,
            minimum_matching_threshold=args.match_thresh,
            frame_rate=fps
        )
        trackers[(ci,cn)] = sv.ByteTrack(**vars(tr_args))
        writers[(ci,cn)]  = TrackJsonlStreamer(
            out_path, meta_head,
            flush_interval=args.flush_interval,
            lost_thresh=args.lost_thresh,
            is_person=_is_person_class(ci, cn)
        )
        outputs.append(str(out_path))

    # streaming through file per frame
    current_video_frame = 0  # where we are in the decoder
    pose_every = max(1, int(args.pose_every_k))
    frame_counter = 0

    for cf, dets in stream_frames_from_jsonl(jsonl_path):
        # advance video sequentially to cf (no caching)
        img = None
        if cap is not None:  # only if we actually need pose
            if cf < current_video_frame:
                img = None
            else:
                while current_video_frame < cf:
                    ok = cap.grab()
                    if not ok:
                        img = None; break
                    current_video_frame += 1
                if current_video_frame == cf:
                    ok, fr = cap.read()
                    if ok:
                        img = fr
                        current_video_frame += 1

        # 1) handle ball class without running a tracker
        for (ci, cn), fuser in list(fusers.items()):
            # keep only *this* class' dets (cheap filter)
            frame_ball_dets = [d for d in dets if int(d.get("cls_idx", -9999)) == ci]
            fuser.update_from_dets(frame_ball_dets, cf)

        # 2) track all other classes normally
        for (ci, cn), tracker in trackers.items():
            boxes, scores, det_list = select_class_dets(dets, ci, cn, args.det_conf)

            if len(det_list) == 0:
                tracker.update_with_tensors(np.zeros((0,5), np.float32))
                writers[(ci,cn)].maybe_flush(cf)
                continue

            det_np = np.concatenate([boxes, scores[:,None]], axis=1)
            online = tracker.update_with_tensors(det_np)

            if not online:
                writers[(ci,cn)].maybe_flush(cf)
                continue

            tids, tlbrs = [], []
            for t in online:
                if hasattr(t, "tlbr"):
                    box = np.asarray(t.tlbr, np.float32)
                elif hasattr(t, "tlwh"):
                    x,y,w,h = map(float, t.tlwh)
                    box = np.array([x,y,x+w,y+h], dtype=np.float32)
                else:
                    arr = np.asarray(t, np.float32).ravel()
                    box = arr[:4]
                tlbrs.append(box)
                tids.append(int(t.external_track_id))
            tlbrs = np.stack(tlbrs, axis=0) if len(tlbrs) else np.zeros((0,4), np.float32)

            # associate to pick conf + proj
            if tlbrs.size and boxes.size:
                M = iou_matrix_np(tlbrs, boxes)
                used_t, used_d = set(), set()
                pairs = []
                flat = [(M[i,j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1])]
                flat.sort(key=lambda x: x[0], reverse=True)
                for val, i_t, j_d in flat:
                    if val < 1e-6: break
                    if i_t in used_t or j_d in used_d: continue
                    used_t.add(i_t); used_d.add(j_d)
                    pairs.append((i_t, j_d))
            else:
                pairs = []

            WRT = writers[(ci,cn)]
            is_person = WRT.is_person
            do_pose_this_frame = (img is not None) and (frame_counter % pose_every == 0)

            for i_t, j_d in pairs:
                tid = tids[i_t]
                box = tlbrs[i_t].tolist()
                det_rec = det_list[j_d]
                conf = det_rec.get("conf", None)
                proj = projected_point_from_warp(det_rec)

                # defaults
                team_score = None; skel = None; skel_conf = None

                if is_person and do_pose_this_frame and (not args.no_pose):
                    x1,y1,x2,y2 = map(int, box)
                    if (x2 - x1) * (y2 - y1) >= args.skip_small_area:
                        det_pose = process_single_detection(
                            img, (x1,y1,x2,y2), 1.0, 0,
                            pose_model, False, True, True, ('person',),
                            clothes_colors_histogram=_HIST_,
                            skip_small_area=args.skip_small_area
                        )
                        if det_pose is not None:
                            team_score = float(det_pose.get("score", 0.0))
                            sk = det_pose.get("keypoints")
                            sc = det_pose.get("keypoints_conf")
                            if sk is not None:
                                skel = sk.tolist() if isinstance(sk, np.ndarray) else sk
                            if sc is not None:
                                skel_conf = sc.tolist() if isinstance(sc, np.ndarray) else sc

                WRT.update(tid, cf, box, conf, proj,
                           team_score=team_score, skel=skel, skel_conf=skel_conf)

            WRT.maybe_flush(cf)

        # per-frame cleanup
        del dets
        frame_counter += 1
        if (frame_counter % 500) == 0:
            gc.collect()

    # close: writers and ball fusers
    for w in writers.values():
        w.close()
    for f in fusers.values():
        f.close()
    if cap is not None:
        cap.release()

    return ";".join(outputs)

# ------------ discovery / CLI ------------
def find_all_jsonls(root: Path):
    out = []
    for p in glob.glob(str(root / "**/*.jsonl"), recursive=True):
        if p.endswith(".tracks.jsonl"):
            continue
        out.append(Path(p))
    return sorted(out)

_HIST_ = None

def _worker(args_tuple):
    return process_one_file(*args_tuple)

def main():
    ap = argparse.ArgumentParser("detections_to_tracks_and_scores_streaming")
    ap.add_argument("--clips-root", type=str, required=True)
    ap.add_argument("--det-conf", type=float, default=0.10)
    ap.add_argument("--fps", type=float, default=0.0)
    # ByteTrack
    ap.add_argument("--track-thresh", type=float, default=0.4)
    ap.add_argument("--track-buffer", type=int, default=60)
    ap.add_argument("--match-thresh", type=float, default=0.8)
    # Streaming/flush
    ap.add_argument("--flush-interval", type=int, default=200)
    ap.add_argument("--lost-thresh", type=int, default=50)
    # Pose + team
    ap.add_argument("--hist", type=str, default=None)
    ap.add_argument("--skip-small-area", type=int, default=5000)
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--pose-every-k", type=int, default=1, help="compute pose/team score every k-th frame per track")
    # Exec
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    args = ap.parse_args()

    global _HIST_
    _HIST_ = load_histogram_any(args.hist) if args.hist else None

    # If pose enabled, force single worker to avoid N× model + decoder in RAM
    if not args.no_pose and _HIST_ is not None and args.workers > 1:
        print("[info] pose enabled -> forcing workers=1 to avoid OOM")
        args.workers = 1

    root = Path(args.clips_root)
    files = find_all_jsonls(root)
    if not files:
        print("No detection JSONLs found under", root)
        return

    work = [(p, args) for p in files]

    if args.workers <= 1:
        for w in work:
            print("processing:", w[0])
            out = process_one_file(*w)
            print(" ->", out)
    else:
        with Pool(processes=args.workers) as pool:
            for out in pool.imap_unordered(_worker, work):
                print(" ->", out)

if __name__ == "__main__":
    main()


# python detections_to_tracks_and_scores.py --clips-root ./runs/detect/demo_video/clips/0025/ --hist ./data/histograms/gk01.npy