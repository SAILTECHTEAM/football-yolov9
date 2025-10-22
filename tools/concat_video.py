import os
import subprocess
from pathlib import Path

def run_ffmpeg(cmd):
    # show stderr so we can actually see the error
    print("FFmpeg:", " ".join(cmd))
    out = subprocess.run(cmd, text=True, capture_output=True)
    if out.returncode != 0:
        print("---- FFmpeg STDERR ----")
        print(out.stderr)
        print("-----------------------")
        raise subprocess.CalledProcessError(out.returncode, cmd, out.stdout, out.stderr)

def ensure_files(parts):
    for p in parts:
        pp = Path(p)
        if not pp.exists():
            raise FileNotFoundError(f"Missing part: {pp}")
        if pp.stat().st_size == 0:
            raise ValueError(f"Empty part file: {pp}")

def _write_concat_list(parts, list_file: Path):
    list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{Path(p).resolve().as_posix()}'\n")

def ffmpeg_concat_copy(list_file: Path, out_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    run_ffmpeg(cmd)

def pick_video_encoder() -> list:
    try:
        encs = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              text=True, capture_output=True, check=True).stdout
    except Exception:
        encs = ""
    if "libx264" in encs:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    if "h264_nvenc" in encs:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19"]
    return ["-c:v", "mpeg4", "-qscale:v", "4"]

def has_audio(p: str) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", p],
            text=True, capture_output=True, check=True
        ).stdout.strip()
        return out != ""
    except Exception:
        return False  # if ffprobe missing, assume no audio to be safe

def ffmpeg_concat_filter(parts, out_path, keep_audio=True):
    """
    Concat via filter_complex (re-encode). Normalizes SAR/size/PTS.
    Avoids fps=... to bypass 0/0 issues.
    """
    out_path = Path(out_path)
    enc_args = pick_video_encoder()

    cmd = ["ffmpeg", "-y"]
    parts = [str(Path(p)) for p in parts]
    for p in parts:
        cmd += ["-i", p]
    n = len(parts)

    # normalize each input:
    # - scale to even dims
    # - setsar=1
    # - settb to AVTB and reset PTS
    v_norm_labels, a_norm_labels, filter_lines = [], [], []
    for i in range(n):
        v_in = f"[{i}:v:0]"; v_out = f"[v{i}]"
        filter_lines.append(f"{v_in}scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1,settb=AVTB,setpts=PTS-STARTPTS{v_out}")
        v_norm_labels.append(v_out)
        if keep_audio:
            a_in = f"[{i}:a:0]"; a_out = f"[a{i}]"
            # resample & reset PTS; 48k stereo is safest
            filter_lines.append(f"{a_in}aformat=sample_fmts=s16:sample_rates=48000:channel_layouts=stereo,asetpts=PTS-STARTPTS{a_out}")
            a_norm_labels.append(a_out)

    if keep_audio:
        filter_lines.append("".join(v_norm_labels + a_norm_labels) + f"concat=n={n}:v=1:a=1[v][a]")
        map_opts = ["-map", "[v]", "-map", "[a]"]
        audio_opts = ["-c:a", "aac", "-b:a", "192k"]
    else:
        filter_lines.append("".join(v_norm_labels) + f"concat=n={n}:v=1:a=0[v]")
        map_opts = ["-map", "[v]"]
        audio_opts = ["-an"]

    filtergraph = ";".join(filter_lines)
    cmd += [
        "-filter_complex", filtergraph,
        *map_opts,
        *enc_args,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        *audio_opts,
        str(out_path),
    ]
    run_ffmpeg(cmd)

def concat_or_reencode(parts: list, out_path: str) -> str:
    ensure_files(parts)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) try demuxer copy-concat (fast, no re-encode)
    list_file = Path(str(out_path) + ".list.txt")
    try:
        _write_concat_list(parts, list_file)
        ffmpeg_concat_copy(list_file, out_path)
        return "copy"
    except subprocess.CalledProcessError:
        pass
    finally:
        try: list_file.unlink()
        except Exception: pass

    # 2) filter concat; keep audio only if ALL parts have audio
    all_have_audio = all(has_audio(p) for p in parts)
    try:
        ffmpeg_concat_filter(parts, out_path, keep_audio=all_have_audio)
        return "filter_av" if all_have_audio else "filter_v"
    except subprocess.CalledProcessError:
        # last resort: force video-only
        ffmpeg_concat_filter(parts, out_path, keep_audio=False)
        return "filter_v_forced"

if __name__ == "__main__":
    a = [
        '../runs/detect/demo_video2/clips/0025/__part_004_02.mp4',
        '../runs/detect/demo_video2/clips/0025/__part_004_03.mp4'
    ]
    b = '../runs/detect/demo_video2/clips/0025/GX010025_clip_004.mp4'
    mode = concat_or_reencode(a, b)
    print("concat mode:", mode)
