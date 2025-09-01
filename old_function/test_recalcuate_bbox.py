import json
import numpy as np
from pathlib import Path
from gen_high_map import get_expected_height, load_jsonl, is_person_track
from yolov9.tools.homography_matrix import compute_homography, apply_homography_to_point

def process_jsonl_with_height_map_stream(input_path, output_path, height_map, x_bins, y_bins, threshold_ratio, H):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            item = json.loads(line)

            # If it's not a person, just copy directly
            if not is_person_track(item):
                outfile.write(json.dumps(item) + "\n")
                continue

            new_bboxes = []
            new_projs = []

            for bbox, proj in zip(item["bbox"], item["projected"]):
                # bbox: [x1, y1, x2, y2, conf]
                x1, y1, x2, y2, conf = bbox
                height = y2 - y1
                x_center = (x1 + x2) / 2

                expected_height = get_expected_height(proj[0], proj[1], height_map, x_bins, y_bins)

                # Avoid division by zero
                if expected_height <= 0:
                    new_bboxes.append(bbox)
                    new_projs.append(proj)
                    continue

                diff_ratio = abs(height - expected_height) / expected_height

                if diff_ratio <= threshold_ratio:
                    new_bboxes.append(bbox)
                    new_projs.append(proj)
                else:
                    # calculate new bbox based on expected height
                    new_y2 = y1 + expected_height
                    new_bbox = [x1, y1, x2, new_y2, conf]
                    # Recalculate projection using bottom middle point
                    new_proj = apply_homography_to_point((x_center, new_y2), H)

                    new_bboxes.append(new_bbox)
                    new_projs.append(new_proj)

            item["bbox"] = new_bboxes
            item["projected"] = new_projs

            outfile.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    input_path = "./runs/detect/test_4k_converted/team_tracking.jsonl"
    output_path = "./runs/detect/test_4k_converted/team_tracking_corrected.jsonl"

    homography_src_points = [(172, 1104), (2101, 895), (3800, 1021), (3458, 2057)]
    homography_dst_points = [(530, 0), (530, 660), (1060, 660), (1060, 0)]

    H = compute_homography(
        np.array(homography_src_points, dtype=np.float32),
        np.array(homography_dst_points, dtype=np.float32)
    )[0]

    # Load the height map and bin edges
    data = np.load("height_map_grid.npz")
    height_map = data["height_map"]
    x_bins = data["x_bins"]
    y_bins = data["y_bins"]

    # Process the JSONL file with the height map
    process_jsonl_with_height_map_stream(input_path, output_path, height_map, x_bins, y_bins, threshold_ratio=0.2, H=H)

    print("✅ Height correction complete!")


