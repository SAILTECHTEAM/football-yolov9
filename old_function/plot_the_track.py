import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.cm as cm
from scipy.signal import find_peaks, peak_widths, savgol_filter, savgol_coeffs
from scipy.linalg import svd


def fill_small_gaps(mask, max_gap):
    if max_gap <= 0:
        return mask
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        while i < n and m[i]:
            i += 1
        gap_start = i
        while i < n and not m[i]:
            i += 1
        gap_end = i
        if gap_end - gap_start > 0 and gap_end < n and gap_start > 0:
            if (gap_end - gap_start) <= max_gap and m[gap_start-1] and m[gap_end]:
                m[gap_start:gap_end] = True
    return m

def _monotonicity_ratio(y):
    if len(y) < 5:
        return 0.0
    dy = np.diff(y)
    peak = int(np.argmax(y)); valley = int(np.argmin(y))
    peak_ratio = 0.0
    if 0 < peak < len(y)-1:
        left, right = dy[:peak], dy[peak:]
        rise = np.sum(left > 0)/(len(left)+1e-12) if len(left) else 0.0
        fall = np.sum(right < 0)/(len(right)+1e-12) if len(right) else 0.0
        peak_ratio = 0.5*(rise+fall)
    valley_ratio = 0.0
    if 0 < valley < len(y)-1:
        left, right = dy[:valley], dy[valley:]
        fall2 = np.sum(left < 0)/(len(left)+1e-12) if len(left) else 0.0
        rise2 = np.sum(right > 0)/(len(right)+1e-12) if len(right) else 0.0
        valley_ratio = 0.5*(fall2+rise2)
    return max(peak_ratio, valley_ratio)

def _quad_curvature_abs(y):
    """
    Fit y ~ a*(t')^2 + b*(t') + c on t' normalized to [-1,1] to make |a| scale-comparable.
    Return |a|.
    """
    n = len(y)
    if n < 5:
        return 0.0
    t = np.linspace(-1, 1, n)
    A = np.column_stack([t**2, t, np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = coef[0]
    return abs(a)

def detect_long_waves_2d_sharp_windowed(
    xs, ys,
    # sliding window
    window_size=301,     # 視窗長度（odd較好，便於平滑）
    step=8,             # 步長
    # detection on local wobble (PC2)
    prominence=3.0,
    min_wave_len=8,
    max_wave_len=60,
    # 2D speed gating (mean + k*std) — 可設 None 關閉
    speed_std_factor=None,
    # smoothing of local wobble
    smooth_kind='savgol', smooth_window=7, savgol_poly=2,
    # SHAPE constraints
    min_steepness=0.25,     # amplitude / duration (units per frame)
    min_quad_curv=0.08,     # |a| on normalized time
    min_monotonic_ratio=0.7,
    # post-process
    max_gap_size=10
):
    """
    局部 PCA + 滑窗：在每個視窗上做 PCA→取 PC2 擺動→找峰谷成段→套形狀/速度規則→合併到全域 mask。
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n < 200:
        return np.zeros(n, dtype=bool)

    # 全域速度統計（用於動態門檻）
    dx_all = np.diff(xs); dy_all = np.diff(ys)
    sp_all = np.sqrt(dx_all**2 + dy_all**2)
    if speed_std_factor is not None and len(sp_all) > 0:
        sp_thresh = np.mean(sp_all) + speed_std_factor*np.std(sp_all)
    else:
        sp_thresh = 0.0

    out_mask = np.zeros(n, dtype=bool)

    # 滑窗遍歷
    w = int(window_size)
    s = int(step)
    if w < 200: w = 200  # 太短不穩
    for start in range(0, max(1, n - w + 1), max(1, s)):
        end = start + w
        if end > n:
            end = n
        if end - start < 7:
            continue

        # 取局部片段
        X = xs[start:end]
        Y = ys[start:end]

        # 局部 PCA
        P = np.column_stack([X, Y])
        P0 = P - P.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = svd(P0, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        pc2 = Vt[1] if Vt.shape[0] >= 2 else np.array([0.0, 1.0])

        # 擺動訊號
        z = (P0 @ pc2)

        # 平滑
        if smooth_kind == 'savgol':
            sw = max(3, int(smooth_window) | 1)
            if sw > len(z): sw = max(3, (len(z)//2)*2+1)  # 保證 odd 且 <= len
            z_s = savgol_filter(z, window_length=sw, polyorder=min(savgol_poly, sw-1), mode='interp')
        else:
            if smooth_window and smooth_window > 1:
                k = np.ones(int(smooth_window))/float(smooth_window)
                z_s = np.convolve(z, k, mode='same')
            else:
                z_s = z.copy()

        # 峰/谷
        peaks, _   = find_peaks(z_s,  prominence=prominence)
        valleys, _ = find_peaks(-z_s, prominence=prominence)

        # 規則檢查函式
        def _accept(seg_l, seg_r):
            dur = seg_r - seg_l + 1
            if dur < min_wave_len or dur > max_wave_len:
                return False

            seg = z_s[seg_l:seg_r+1]
            amp = float(np.max(seg) - np.min(seg))
            steep = amp / max(dur, 1)

            # 局部 2D 平均速度（回到原座標）
            seg_sp = np.sqrt(np.diff(X[seg_l:seg_r+1])**2 + np.diff(Y[seg_l:seg_r+1])**2)
            avg_sp = float(np.mean(seg_sp)) if len(seg_sp) else 0.0

            curv = _quad_curvature_abs(seg)
            mono = _monotonicity_ratio(seg)

            if steep < min_steepness:           return False
            if curv  < min_quad_curv:           return False
            if mono  < min_monotonic_ratio:     return False
            if avg_sp < sp_thresh:              return False

            return True

        # 從峰擴張（升→降）
        for p in peaks:
            l = p
            while l > 0 and z_s[l-1] < z_s[l]:
                l -= 1
            r = p
            while r < len(z_s)-1 and z_s[r+1] < z_s[r]:
                r += 1
            if _accept(l, r):
                out_mask[start+l:start+r+1] = True

        # 從谷擴張（降→升）
        for v in valleys:
            l = v
            while l > 0 and z_s[l-1] > z_s[l]:
                l -= 1
            r = v
            while r < len(z_s)-1 and z_s[r+1] > z_s[r]:
                r += 1
            if _accept(l, r):
                out_mask[start+l:start+r+1] = True

    # 合併後補小縫
    return fill_small_gaps(out_mask, max_gap_size)

# Find wave segment boundaries
def find_wave_segments(wave_mask):
    """Find start and end indices of continuous wave segments"""
    segments = []
    if not np.any(wave_mask):
        return segments
        
    # Find transitions
    padded = np.concatenate([[False], wave_mask, [False]])
    transitions = np.diff(padded.astype(int))
    
    starts = np.where(transitions == 1)[0]  # False to True
    ends = np.where(transitions == -1)[0]   # True to False
    
    for start, end in zip(starts, ends):
        segments.append((start, end - 1))  # end-1 because we want inclusive
        
    return segments

def plot_track_on_field(
    xs,
    ys,
    track_id,
    field_w=1060,
    field_h=660,
    cmap_name='viridis',
    point_size=3,
    line_width=0,
    fig_width_inches=12,   # make it bigger without scaling data
    save=True,
    bg_img_path=None
):

    # Detect wave segments
    wave_segments = detect_long_waves_2d_sharp_windowed(xs, ys, min_wave_len=0, prominence=2, max_gap_size=30 ,min_steepness=0.1, min_quad_curv=0.9, min_monotonic_ratio=0.5, window_size=2201, step=450,)
    wave_segment_boundaries = find_wave_segments(wave_segments)

    # 3) Figure size with aspect ratio
    fig_h_inches = fig_width_inches * (field_h / field_w)
    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_h_inches))
    if bg_img_path is not None:
        img = np.array(Image.open(bg_img_path))  # or PNG etc.
        ax.imshow(img, extent=[0, field_w, 0, field_h])
    else:
        ax.set_facecolor('white')  # white background

    # 4) Time-based color for non-wave segments
    t = np.linspace(0.0, 1.0, len(xs))
    cmap = plt.get_cmap(cmap_name)
    time_colors = cmap(t)

    # 5) Draw each segment, overriding with red for wave
    for i in range(len(xs) - 1):
        if wave_segments[i]:
            seg_color = 'red'
        else:
            seg_color = time_colors[i]
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=seg_color, linewidth=line_width, solid_capstyle='round')

    # 6) Add blue connecting lines for wave segments
    for start_idx, end_idx in wave_segment_boundaries:
        if start_idx < len(xs) and end_idx < len(xs):
            # Get start and end points of the wave segment
            start_x, start_y = xs[start_idx], ys[start_idx]
            end_x, end_y = xs[end_idx], ys[end_idx]
            
            # Calculate number of points in the wave segment
            num_wave_points = end_idx - start_idx + 1
            
            # Create interpolated points for the connecting line
            connect_x = np.linspace(start_x, end_x, num_wave_points)
            connect_y = np.linspace(start_y, end_y, num_wave_points)
            
            # Plot the blue connecting line
            ax.plot(connect_x, connect_y, color='blue', linewidth=0, 
                   alpha=0.7, linestyle='-', zorder=3, label='Wave connector' if start_idx == wave_segment_boundaries[0][0] else "")
            
            # Plot blue points along the connecting line
            ax.scatter(connect_x, connect_y, c='blue', s=point_size, 
                      alpha=0.7, zorder=4, edgecolors='none')

    # 6) Draw scatter points with same logic (wave points in red)
    point_colors = []
    for i in range(len(xs)):
        if i < len(wave_segments) and wave_segments[i]:
            point_colors.append('red')
        else:
            point_colors.append(time_colors[i])
    sc = ax.scatter(xs, ys, c=point_colors, s=point_size, edgecolors='none')

    # 7) Start/End markers
    ax.scatter([xs[0]], [ys[0]], s=120, facecolors='none', edgecolors='green', linewidths=2, zorder=5, label='start')
    ax.scatter([xs[-1]], [ys[-1]], s=120, facecolors='none', edgecolors='red', linewidths=2, zorder=5, label='end')
    ax.legend(loc='upper right')

    # 8) Axes settings
    ax.set_xlim(0, field_w)
    ax.set_ylim(0, field_h)       # origin='lower' -> no need to invert
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Track {track_id} trajectory")

    # 9) Colorbar for time progression
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Time progression (start → end)')

    plt.tight_layout()

    if save:
        out = f"track_{track_id}_on_field.png"
        fig.savefig(out, dpi=220)
        print(f"Saved: {out}")

    plt.show()


if __name__ == "__main__":

    jsonl_path="./runs/detect/test_4k_converted/test_correct_peak.jsonl"
    track_id="10753a"
    # bg_img_path="./data/images/mongkok_football_field.png",
    
    xs, ys, frames = None, None, None
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if str(item.get("track_id")) == str(track_id):
                proj = item.get("projected", [])
                if not proj:
                    print(f"Track {track_id} has no projected points.")
                    break
                xs = np.array([p[0] for p in proj], dtype=float)
                ys = np.array([p[1] for p in proj], dtype=float)
                frames = item.get("frames") or item.get("frame_id")
                break

    plot_track_on_field(
        xs, ys, track_id,
        field_w=1060, field_h=660,
        cmap_name='viridis', point_size=3, line_width=0,
        fig_width_inches=12, save=True,
        bg_img_path=None
    )