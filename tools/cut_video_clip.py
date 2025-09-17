import math
import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

def _ffprobe_json(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)

def _probe_fps(path: str) -> Tuple[float, str]:
    """Return (fps_float, fps_rational_str). Falls back to (30.0, '30')."""
    try:
        out = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=avg_frame_rate,r_frame_rate",
             "-of","default=nokey=1:noprint_wrappers=1", path],
            text=True, capture_output=True, check=True
        ).stdout.strip().splitlines()
        rat = next((l for l in out if l and l != "0/0"), "30")
        if "/" in rat:
            a,b = rat.split("/")
            fps = float(a) / float(b) if float(b) != 0 else 30.0
        else:
            fps = float(rat)
        return fps, rat
    except Exception:
        return 30.0, "30"

def _has_audio(path: str) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            text=True, capture_output=True, check=True
        ).stdout.strip()
        return out != ""
    except Exception:
        return False

def _dense_gop_encoder_args(fps_int: int) -> list:
    """
    Enforce 1 keyframe per second (closed GOP).
    fps_int should be ~30 for 29.97 inputs.
    """
    try:
        encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              text=True, capture_output=True, check=True).stdout
    except Exception:
        encs = ""
    force_each_sec = ["-force_key_frames", "expr:gte(t,n_forced*1)"]  # IDR at t=0,1,2,...
    if "libx264" in encs:
        return [
            "-c:v","libx264","-preset","veryfast","-crf","18",
            "-pix_fmt","yuv420p",
            "-x264-params", f"keyint={fps_int}:min-keyint={fps_int}:scenecut=0:open_gop=0",
            *force_each_sec
        ]
    if "h264_nvenc" in encs:
        return [
            "-c:v","h264_nvenc","-preset","p4","-cq","19",
            "-pix_fmt","yuv420p",
            "-g", str(fps_int), "-forced-idr","1",
            *force_each_sec
        ]
    # Fallback encoder (won’t strictly give IDR/1s, but keeps working)
    return ["-c:v","mpeg4","-qscale:v","4","-pix_fmt","yuv420p"]

def split_video_fixed_length(
    input_path: str,
    clip_len: float,
    unit: str = "min",
    output_dir: Optional[str] = None,
    base_name: Optional[str] = None,
    precise: bool = False,
    overwrite: bool = True,
) -> List[str]:
    """
    Split a video into multiple clips of equal target length.

    precise=True:
      - re-encode each clip
      - CFR at source FPS (e.g. 30000/1001)
      - dense GOP: 1 keyframe/sec (closed GOP)
      - perfect for later fast (-c copy) re-cuts

    precise=False:
      - fast segmenter with -c copy (keeps source GOP; no new keyframes)
    """
    if unit not in {"min", "sec"}:
        raise ValueError("unit must be 'min' or 'sec'")
    clip_seconds = float(clip_len) * (60.0 if unit == "min" else 1.0)
    if clip_seconds <= 0:
        raise ValueError("clip_len must be > 0")

    for bin_name in ("ffmpeg", "ffprobe"):
        if not shutil.which(bin_name):
            raise RuntimeError(f"Required binary '{bin_name}' not found in PATH")

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(input_path)

    if output_dir is None:
        output_dir = str(in_path.parent / f"{in_path.stem}_clips")
    os.makedirs(output_dir, exist_ok=True)
    base_name = in_path.stem if base_name is None else base_name.rstrip("_")

    meta = _ffprobe_json(str(in_path))
    duration = None
    if "format" in meta and "duration" in meta["format"]:
        duration = float(meta["format"]["duration"])
    if not duration and "streams" in meta:
        durs = [float(s.get("duration", 0)) for s in meta["streams"] if s.get("duration")]
        if durs:
            duration = max(durs)
    if not duration:
        raise RuntimeError("Unable to determine video duration via ffprobe")

    # Probe source fps → we expect 30000/1001 for 29.97
    src_fps_float, src_fps_rat = _probe_fps(str(in_path))
    fps_int = max(1, int(round(src_fps_float)))  # 29.97 → 30

    # ---- PATH 1: precise (re-encode with dense GOP) ----
    if precise:
        keep_audio = _has_audio(str(in_path))   # set False if you always want silent outputs
        enc_args = _dense_gop_encoder_args(fps_int)
        outputs: List[str] = []
        n_parts = math.ceil(duration / clip_seconds)
        for idx in range(n_parts):
            start = idx * clip_seconds
            end = min(duration, start + clip_seconds)  # EXCLUSIVE
            if end <= start:
                continue
            out_path = os.path.join(output_dir, f"{base_name}_{idx:03d}.mp4")

            # Trim by time, reset PTS, normalize dimensions/SAR
            v_chain = (f"trim=start={start:.6f}:end={end:.6f},"
                       f"setpts=PTS-STARTPTS,"
                       f"scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,format=yuv420p[v]")
            if keep_audio:
                a_chain = (f"atrim=start={start:.6f}:end={end:.6f},"
                           f"asetpts=PTS-STARTPTS,"
                           f"aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo[a]")
                filtergraph = f"[0:v:0]{v_chain};[0:a:0]{a_chain}"
                map_opts = ["-map","[v]","-map","[a]","-c:a","aac","-b:a","128k"]
            else:
                filtergraph = f"[0:v:0]{v_chain}"
                map_opts = ["-map","[v]","-an"]

            cmd = [
                "ffmpeg",
                "-y" if overwrite else "-n",
                "-hide_banner","-loglevel","error",
                "-i", str(in_path),
                "-filter_complex", filtergraph,
                *map_opts,
                # Lock CFR so every clip has identical timebase/fps
                "-vsync","cfr","-r", src_fps_rat,          # e.g. 30000/1001
                *enc_args,                                 # dense, closed GOP + forced 1s IDRs
                "-movflags","+faststart",
                out_path,
            ]
            subprocess.run(cmd, check=True)
            outputs.append(out_path)
        return outputs

    # ---- PATH 2: fast (segment muxer, copy) ----
    pattern = os.path.join(output_dir, f"{base_name}_%03d.mp4")
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner","-loglevel","error",
        "-i", str(in_path),
        "-c","copy",
        "-map","0",
        "-f","segment",
        "-segment_time", f"{clip_seconds:.6f}",
        "-reset_timestamps","1",
        "-movflags","+faststart",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    outs = sorted(str(p) for p in Path(output_dir).glob(f"{base_name}_*.mp4"))
    return outs

if __name__ == "__main__":
    # 29.97 fps → outputs with CFR=30000/1001 and 1 IDR/sec (GoPro-like)
    out_files = split_video_fixed_length(
        "../data/video/GX010025.MP4",
        clip_len=1,
        unit="min",
        precise=True,   # re-encode with dense GOP so later -c copy cuts are clean
    )
    print("\n".join(out_files))
