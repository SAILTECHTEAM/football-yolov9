import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from scipy.interpolate import RBFInterpolator
import joblib

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def is_person_track(track):
    """A track is considered a person if any team_conf entry is non-empty."""
    return any(len(conf) > 0 for conf in track.get("team_conf", []))

def extract_height_and_position(data, conf_thresh=0.3, outlier_thresh=2000):
    heights = []
    positions = []

    for item in data:
        bboxes = item.get("bbox", [])
        projections = item.get("projected", [])

        if not is_person_track(item):
            continue  # skip ball or unknown tracks
        
        if len(bboxes) != len(projections):
            continue  # skip malformed data

        for bbox, proj in zip(bboxes, projections):
            if not bbox or not proj:
                continue
            x1, y1, x2, y2, conf = bbox
            if conf < conf_thresh:
                continue
            h = y2 - y1
            if h <= 0:
                continue
            X, Y = proj
            heights.append(h)
            positions.append((X, Y))

    # Convert to numpy arrays
    positions = np.array(positions)
    heights = np.array(heights)

    # Remove outliers based on projected coordinates
    valid_mask = (np.abs(positions[:, 0]) < outlier_thresh) & (np.abs(positions[:, 1]) < outlier_thresh)
    positions = positions[valid_mask]
    heights = heights[valid_mask]

    print("Height stats:")
    print("Min:", np.min(heights))
    print("Max:", np.max(heights))
    print("Mean:", np.mean(heights))

    return positions, heights

def build_height_map(positions, heights, grid_size=50):
    x = positions[:, 0]
    y = positions[:, 1]

    print("X range:", np.min(x), "to", np.max(x))
    print("Y range:", np.min(y), "to", np.max(y))
    print("Unusual X values:", np.sum(np.abs(x) > 1e4))
    print("Unusual Y values:", np.sum(np.abs(y) > 1e4))

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)

    height_map = np.zeros((grid_size - 1, grid_size - 1))
    count_map = np.zeros_like(height_map)

    for i in range(len(heights)):
        xi = np.searchsorted(x_bins, x[i]) - 1
        yi = np.searchsorted(y_bins, y[i]) - 1
        if 0 <= xi < grid_size - 1 and 0 <= yi < grid_size - 1:
            height_map[yi, xi] += heights[i]
            count_map[yi, xi] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_height_map = np.true_divide(height_map, count_map)
        avg_height_map[count_map == 0] = np.nan

    return avg_height_map, x_bins, y_bins

def plot_height_map(avg_height_map, x_bins, y_bins):
    plt.figure(figsize=(10, 6))
    plt.imshow(avg_height_map, origin='lower',
               extent=(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]),
               aspect='auto', cmap='viridis')
    # # set limits to 1060x660 to match field size
    # plt.xlim(0, 1060)
    # plt.ylim(0, 660)
    plt.colorbar(label='Expected BBox Height (px)')
    plt.title("Expected BBox Height Over Field")
    plt.xlabel("Projected X")
    plt.ylabel("Projected Y")
    plt.grid(True)
    plt.show()

def get_expected_height(x, y, height_map, x_bins, y_bins):
    """
    Given a projected (x, y) coordinate, return the expected height from the grid.
    """
    # Find the closest bin index
    x_idx = np.digitize([x], x_bins)[0] - 1
    y_idx = np.digitize([y], y_bins)[0] - 1

    # Clip to valid range
    x_idx = np.clip(x_idx, 0, len(x_bins) - 2)
    y_idx = np.clip(y_idx, 0, len(y_bins) - 2)

    # row = y-axis, col = x-axis
    return height_map[y_idx, x_idx]

def build_height_model_from_jsonl(path, sample_size=50000, conf_thresh=0.5, outlier_thresh=2000):
    data = load_jsonl(path)
    np.random.seed(42)
    positions, heights = extract_height_and_position(data, conf_thresh, outlier_thresh)

    print(f"Total points: {len(positions)}")

    # 隨機抽樣，避免 OOM
    if len(positions) > sample_size:
        indices = np.random.choice(len(positions), sample_size, replace=False)
        positions = positions[indices]
        heights = heights[indices]
        print(f"Downsampled to {sample_size} points for model fitting.")

    print("Fitting RBF model...")
    rbf_model = RBFInterpolator(positions, heights, smoothing=5)
    print("Model fitting complete.")
    joblib.dump(rbf_model, 'rbf_model.pkl')
    # return rbf_model

    # create a heatmap for visualization
    height_map, x_bins, y_bins = build_height_map(positions, heights)
    plot_height_map(height_map, x_bins, y_bins)

def predict_height(rbf_model, x, y):
    rbf_model = joblib.load(rbf_model)
    return rbf_model(np.array([[x, y]]))[0]



if __name__ == "__main__":
    # path = "./runs/detect/test_4k_converted/team_tracking.jsonl"  # 替換為你的jsonl檔案路徑
    # data = load_jsonl(path)
    # positions, heights = extract_height_and_position(data)
    # height_map, x_bins, y_bins = build_height_map(positions, heights)
    # # Save the grid and bins to file
    # np.savez("height_map_grid.npz", height_map=height_map, x_bins=x_bins, y_bins=y_bins)
    # print("Saved height map to height_map_grid.npz")
    # plot_height_map(height_map, x_bins, y_bins)

    # Load the saved height map and bin edges
    data = np.load("height_map_grid.npz")
    height_map = data["height_map"]
    x_bins = data["x_bins"]
    y_bins = data["y_bins"]
    expected= get_expected_height(530, 330, height_map, x_bins, y_bins)  # Example usage
    print(f"Expected height at : {expected}")
    # plot_height_map(height_map, x_bins, y_bins)


    # # Load the model
    # build_height_model_from_jsonl(path)
    # # Predict expected height at (x, y)
    # x, y = 0, 0  # 替換為你想要預測的座標
    # rbf_model_path = 'rbf_model.pkl'
    # predicted_height = predict_height(rbf_model_path, x, y)
    # print(f"Predicted height at ({x}, {y}): {predicted_height}")
