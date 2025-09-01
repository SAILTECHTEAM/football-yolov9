import json
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from numpy.linalg import svd

# --------------- helpers ---------------

def is_ball_track(item) -> bool:
    """
    Heuristic: if team_conf is a list and *every* entry is an empty dict,
    treat it as ball (copy through).
    """
    tc = item.get("team_conf")
    if isinstance(tc, list) and len(tc) > 0:
        return all(isinstance(d, dict) and len(d) == 0 for d in tc)
    return False

def fill_small_gaps(mask, max_gap):
    if max_gap <= 0:
        return mask
    m = mask.copy(); n = len(m); i = 0
    while i < n:
        while i < n and m[i]: i += 1
        g0 = i
        while i < n and not m[i]: i += 1
        g1 = i
        if g1 - g0 > 0 and g0 > 0 and g1 < n and (g1 - g0) <= max_gap and m[g0-1] and m[g1]:
            m[g0:g1] = True
    return m

def segments_from_mask(mask):
    segs = []
    i, n = 0, len(mask)
    while i < n:
        while i < n and not mask[i]:
            i += 1
        if i >= n: break
        s = i
        while i < n and mask[i]:
            i += 1
        e = i - 1
        segs.append([int(s), int(e)])
    return segs

def _monotonicity_ratio(y):
    if len(y) < 5: return 0.0
    dy = np.diff(y)
    peak = int(np.argmax(y)); valley = int(np.argmin(y))
    pr = 0.0
    if 0 < peak < len(y)-1:
        left, right = dy[:peak], dy[peak:]
        rise = np.sum(left > 0)/(len(left)+1e-12) if len(left) else 0.0
        fall = np.sum(right < 0)/(len(right)+1e-12) if len(right) else 0.0
        pr = 0.5*(rise+fall)
    vr = 0.0
    if 0 < valley < len(y)-1:
        left, right = dy[:valley], dy[valley:]
        fall2 = np.sum(left < 0)/(len(left)+1e-12) if len(left) else 0.0
        rise2 = np.sum(right > 0)/(len(right)+1e-12) if len(right) else 0.0
        vr = 0.5*(fall2+rise2)
    return max(pr, vr)

def _quad_curvature_abs(y):
    n = len(y)
    if n < 5: return 0.0
    t = np.linspace(-1, 1, n)
    A = np.column_stack([t**2, t, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return abs(coef[0])

# --------------- detector (windowed PCA) ---------------

def detect_long_waves_2d_sharp_windowed(
    xs, ys,
    window_size=301,   # choose per FPS; 21–301 is typical
    step=60,          # overlap ~ window_size - step
    prominence=2.0,   # peak strength on local wobble
    min_wave_len=6,   # min frames of a wave
    max_wave_len=60,  # max frames of a wave
    speed_std_factor=None,        # None to disable; or 0.3–0.8
    smooth_window=7, savgol_poly=2,
    min_steepness=0.2,            # amp/duration
    min_quad_curv=0.08,           # quadratic curvature
    min_monotonic_ratio=0.6,      # clean rise+fall
    max_gap_size=10               # fill tiny holes between detections
):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    n = len(xs)
    if n < 7:
        return np.zeros(n, dtype=bool)

    # global speed gate (optional)
    sp = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    sp_thresh = (np.mean(sp) + (speed_std_factor*np.std(sp))) if (speed_std_factor is not None and len(sp)>0) else 0.0

    out_mask = np.zeros(n, dtype=bool)
    w = int(max(21, window_size))
    s = int(max(1, step))

    for start in range(0, max(1, n - w + 1), s):
        end = min(start + w, n)
        if end - start < 7:
            continue

        X = xs[start:end]; Y = ys[start:end]
        P = np.column_stack([X, Y]); P0 = P - P.mean(axis=0, keepdims=True)

        try:
            _, _, Vt = svd(P0, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        pc2 = Vt[1] if Vt.shape[0] >= 2 else np.array([0.0, 1.0])

        z = (P0 @ pc2)

        sw = max(3, int(smooth_window) | 1)
        if sw > len(z): sw = max(3, (len(z)//2)*2 + 1)
        z_s = savgol_filter(z, window_length=sw, polyorder=min(savgol_poly, sw-1), mode='interp')

        peaks, _   = find_peaks(z_s,  prominence=prominence)
        valleys, _ = find_peaks(-z_s, prominence=prominence)

        def _accept(l, r):
            dur = r - l + 1
            if dur < min_wave_len or dur > max_wave_len: return False
            seg = z_s[l:r+1]
            amp = float(np.max(seg) - np.min(seg))
            steep = amp / max(dur, 1)
            curv = _quad_curvature_abs(seg)
            mono = _monotonicity_ratio(seg)
            seg_sp = np.sqrt(np.diff(X[l:r+1])**2 + np.diff(Y[l:r+1])**2)
            avg_sp = float(np.mean(seg_sp)) if len(seg_sp) else 0.0
            if steep < min_steepness:       return False
            if curv  < min_quad_curv:       return False
            if mono  < min_monotonic_ratio: return False
            if avg_sp < sp_thresh:          return False
            return True

        # expand from peaks (rise→fall)
        for p in peaks:
            l = p
            while l > 0 and z_s[l-1] < z_s[l]: l -= 1
            r = p
            while r < len(z_s)-1 and z_s[r+1] < z_s[r]: r += 1
            if _accept(l, r):
                out_mask[start+l:start+r+1] = True

        # expand from valleys (fall→rise)
        for v in valleys:
            l = v
            while l > 0 and z_s[l-1] > z_s[l]: l -= 1
            r = v
            while r < len(z_s)-1 and z_s[r+1] > z_s[r]: r += 1
            if _accept(l, r):
                out_mask[start+l:start+r+1] = True

    return fill_small_gaps(out_mask, max_gap_size)

# --------------- correction (replace with straight line) ---------------

def apply_linear_bridge(xs, ys, segments):
    """
    For each [s, e], replace xs[s:e+1], ys[s:e+1] with a straight line
    between endpoints (same number of points).
    """
    xs = np.array(xs, float).copy()
    ys = np.array(ys, float).copy()
    n = len(xs)
    for s, e in segments:
        s = max(0, min(n-1, int(s)))
        e = max(0, min(n-1, int(e)))
        if e <= s:
            continue
        xs[s:e+1] = np.linspace(xs[s], xs[e], e - s + 1)
        ys[s:e+1] = np.linspace(ys[s], ys[e], e - s + 1)
    return xs, ys

# --------------- streaming: detect & replace & write ---------------

def process_jsonl_detect_replace(
    input_path,
    output_path,
    detector_kwargs=None,
    overwrite_projected=True  # set False to keep original and add "projected_corrected"
):
    if detector_kwargs is None:
        detector_kwargs = {}

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ball → copy through
            if is_ball_track(item):
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            proj = item.get("projected", [])
            if not proj or len(proj) < 2:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            xs = [float(p[0]) for p in proj]
            ys = [float(p[1]) for p in proj]

            # detect sharp spans
            mask = detect_long_waves_2d_sharp_windowed(xs, ys, **detector_kwargs)
            segments = segments_from_mask(mask)

            # replace points inside each span with a straight line
            if segments:
                xs_corr, ys_corr = apply_linear_bridge(xs, ys, segments)
                corrected = [[float(x), float(y)] for x, y in zip(xs_corr, ys_corr)]
                if overwrite_projected:
                    item["projected"] = corrected
                else:
                    item["projected_corrected"] = corrected
            # else: no change

            # write out (no extra debug fields)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

# --------------- CLI ---------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--keep_original", action="store_true",
                        help="If set, keeps original 'projected' and writes 'projected_corrected' instead of overwriting.")
    args = parser.parse_args()

    # start with permissive gates; tighten later
    DETECTOR = dict(
        window_size=2201,
        step=450,
        prominence=2,
        min_wave_len=0,
        max_wave_len=60,
        speed_std_factor=None,       # enable later (e.g., 0.5) if needed
        smooth_window=0, savgol_poly=2,
        min_steepness=0.1,
        min_quad_curv=0.9,
        min_monotonic_ratio=0.5,
        max_gap_size=30
    )

    process_jsonl_detect_replace(
        input_path=args.input,
        output_path=args.output,
        detector_kwargs=DETECTOR,
        overwrite_projected=(not args.keep_original)
    )
