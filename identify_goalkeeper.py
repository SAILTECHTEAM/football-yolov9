# TODO: ADD score base one skeleton size
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def extract_color_histogram_with_specific_background_color_old(image, mask=None, bins=(50, 60, 60), ignore_color_range=None):
    """
    Extract the color histogram from an image in HSV format, with the option to ignore specific color ranges.

    Args:
        image (np.ndarray): Input image (BGR format).
        mask (np.ndarray): Optional mask to focus on specific regions.
        bins (tuple): Number of bins for Hue, Saturation, and Value.
        ignore_color_range (tuple): Range of HSV values to ignore ((low_H, low_S, low_V), (high_H, high_S, high_V)).

    Returns:
        np.ndarray: Normalized color histogram.
    """
    # multi input format, if image iis a path, read the image, if it is a numpy array, use it directly
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not read image from {image}")
    elif not isinstance(image, np.ndarray):
        raise TypeError("Image must be a file path or a numpy array.")
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # If ignore_color_range is provided, mask out the specified range
    if ignore_color_range:
        lower_bound = np.array(ignore_color_range[0], dtype=np.uint8)
        upper_bound = np.array(ignore_color_range[1], dtype=np.uint8)
        ignore_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Invert the ignore mask to keep only valid colors
        if mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(ignore_mask))
        else:
            mask = cv2.bitwise_not(ignore_mask)

    # Calculate the histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])

    # Normalize the histogram
    hist = hist / np.sum(hist)  # Normalize to make it independent of image size
    return hist


def extract_color_histogram_with_specific_background_color(
    image, 
    mask=None, 
    bins=(50, 60, 60), 
    ignore_color_range=None
):
    """
    Extract a color histogram from an image in HSV format, with the option to ignore specific color ranges.

    Args:
        image (str | np.ndarray): Input image path or image array (BGR format).
        mask (np.ndarray): Optional binary mask to restrict histogram region.
        bins (tuple): Number of bins for Hue, Saturation, and Value.
        ignore_color_range (tuple): ((low_H, low_S, low_V), (high_H, high_S, high_V)) to ignore.

    Returns:
        np.ndarray: Normalized HSV color histogram.
    """
    # Read from file path if necessary
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not read image from path: {image}")
    elif not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array or file path.")

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply color exclusion mask if provided
    if ignore_color_range is not None:
        lower_bound = np.array(ignore_color_range[0], dtype=np.uint8)
        upper_bound = np.array(ignore_color_range[1], dtype=np.uint8)
        ignore_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        valid_mask = cv2.bitwise_not(ignore_mask)

        if mask is not None:
            mask = cv2.bitwise_and(mask, valid_mask)
        else:
            mask = valid_mask

    # Calculate the histogram over masked region
    hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])

    # Normalize to sum = 1
    hist_sum = np.sum(hist)
    if hist_sum == 0:
        return np.zeros_like(hist)
    return hist / hist_sum


def extract_color_histogram_from_rotated_skelton(image, skeleton_keypoints, bins=(50, 60, 60), expand_ratio=0.2, ignore_skin=False):
    """
    Extract the clothes histogram from an image using a rotated bounding box based on skeleton keypoints.

    Args:
        image (np.ndarray): Input image (BGR format).
        skeleton_keypoints (dict): Detected skeleton keypoints with body part names as keys and (x, y) tuples as values.
        bins (tuple): Number of bins for Hue, Saturation, and Value.
        expand_ratio (float): Ratio to expand the bounding box dimensions.
        ignore_skin (bool): Whether to exclude skin tones from the histogram.

    Returns:
        np.ndarray: Normalized clothes histogram.
    """
    # Define torso keypoints
    left_shoulder = skeleton_keypoints["left_shoulder"]
    right_shoulder = skeleton_keypoints["right_shoulder"]
    left_hip = skeleton_keypoints["left_hip"]
    right_hip = skeleton_keypoints["right_hip"]

    # Calculate the center of the torso
    center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
    center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4

    # Calculate the torso orientation (angle)
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Calculate the width and height of the torso bounding box
    torso_width = np.sqrt(dx**2 + dy**2) * (1 + expand_ratio)
    torso_height = np.sqrt((right_hip[0] - left_shoulder[0])**2 + (right_hip[1] - left_shoulder[1])**2) * (1 + expand_ratio)

    # Create a rotated rectangle
    rect = ((center_x, center_y), (torso_width, torso_height), angle)

    # Generate a mask for the rotated rectangle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    box = cv2.boxPoints(rect).astype(int)  # Convert to corner points
    cv2.fillPoly(mask, [box], 255)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Exclude skin tones if needed
    if ignore_skin:
        skin_lower = np.array([0, 40, 60], dtype=np.uint8)   # Adjust based on dataset
        skin_upper = np.array([20, 150, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv_image, skin_lower, skin_upper)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(skin_mask))

    # Extract the histogram for the masked region
    hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])

    # Normalize the histogram
    hist = hist / np.sum(hist)
    return hist


def compare_histograms(hist1, hist2, method=cv2.HISTCMP_BHATTACHARYYA):
    """
    Compare two histograms and compute their similarity.

    Args:
        hist1 (np.ndarray): First histogram.
        hist2 (np.ndarray): Second histogram.
        method (int): OpenCV method for histogram comparison (e.g., cv2.HISTCMP_BHATTACHARYYA).
        threshold (float): Minimum similarity score to accept as a match (0-1 for Bhattacharyya).

    Returns:
        float: Similarity score (higher is better).
    """
    # Compare the histograms
    similarity = 1 - cv2.compareHist(hist1, hist2, method)  # Invert for similarity (1 = identical)

    return similarity


def plot_hsv_histogram(hist, bins=(50, 60, 60)):
    """
    Plot the HSV histogram as a 3D bar chart.

    Args:
        hist (np.ndarray): The HSV histogram (normalized).
        bins (tuple): Number of bins for Hue, Saturation, and Value.
    """
    # Extract bin counts
    h_bins, s_bins, v_bins = bins
    h_range = np.linspace(0, 180, h_bins)
    s_range = np.linspace(0, 256, s_bins)
    v_range = np.linspace(0, 256, v_bins)

    # Reshape the histogram for plotting
    hist = hist.reshape((h_bins, s_bins, v_bins))

    # Create a 3D bar chart
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare bar positions
    xpos, ypos, zpos = np.meshgrid(
        np.arange(h_bins),
        np.arange(s_bins),
        np.arange(v_bins),
        indexing="ij"
    )
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.8  # Bar width
    dz = hist.flatten()

    # Normalize bar colors based on Hue and Saturation
    colors = plt.cm.hsv(xpos / h_bins)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
    ax.set_xlabel('Hue')
    ax.set_ylabel('Saturation')
    ax.set_zlabel('Value')

    plt.title('HSV Histogram')
    plt.show()


def save_histogram(hist, file_path):
    """
    Save the histogram to a file for later use.

    Args:
        hist (np.ndarray): The histogram to save.
        file_path (str): Path to save the histogram file.
    """
    np.save(file_path, hist)
    print(f"Histogram saved to {file_path}")


def load_histogram(file_path):
    """
    Load the histogram from a file.

    Args:
        file_path (str): Path to the saved histogram file.

    Returns:
        np.ndarray: Loaded histogram.
    """
    hist = np.load(file_path)
    print(f"Histogram loaded from {file_path}")
    return hist


def load_team_histograms_from_folder(folder_path, file_ext=".npy"):
    """
    Load all histogram .npy files from a folder and organize them by team name.

    Args:
        folder_path (str): Path to the folder containing histogram .npy files.
        file_ext (str): Extension of the histogram files (default: '.npy').

    Returns:
        dict: {team_name: [hist1, hist2, ...]} grouped histograms per team.
    """
    team_histograms = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_ext):
            file_path = os.path.join(folder_path, file_name)
            hist = np.load(file_path)

            # Extract team name from the filename (assumes format like teamA_01.npy)
            team_name = file_name.split('_')[0]

            if team_name not in team_histograms:
                team_histograms[team_name] = []
            team_histograms[team_name].append(hist)

    return team_histograms


def match_histograms_to_teams(crop_hists, team_histograms, method=cv2.HISTCMP_BHATTACHARYYA):
    """
    Compare each player's crop histogram to each team's reference histograms.

    Args:
        crop_hists (List[np.ndarray]): histograms of current detections
        team_histograms (Dict[str, List[np.ndarray]]): team_id -> list of histograms

    Returns:
        List[Dict[str, float]]: list of team score dictionaries for each detection
    """
    all_scores = []
    for hist in crop_hists:
        scores = {}
        for team_id, team_hist_list in team_histograms.items():
            similarities = [
                compare_histograms(hist, team_hist, method)
                for team_hist in team_hist_list
            ]
            # You can average or take max depending on your preference
            scores[team_id] = float(np.mean(similarities))  # or np.max(similarities)
        all_scores.append(scores)
    return all_scores


def parse_opt():
    parser = argparse.ArgumentParser(description="Extract and save color histogram from an image.")
    
    parser.add_argument('--image', type=str,
                        help="Path to the goalkeeper's clothes image.")
    parser.add_argument('--histogram_save_path', type=str,
                        help="Path to save the histogram file.")
    opt = parser.parse_args()
    return opt


def run(image, histogram_save_path):
    # Load the image
    goalkeeper_clothe_image = cv2.imread(image)
    if goalkeeper_clothe_image is None:
        print(f"Error: Could not load image from {image}")
        exit(1)

    # Extract histograms, ignoring white
    goalkeeper_hist_from_image = extract_color_histogram_with_specific_background_color(goalkeeper_clothe_image)
    
    # Plot the histogram
    # plot_hsv_histogram(goalkeeper_hist_from_image, bins=tuple(args.bins))
    
    # Save the histogram
    save_histogram(goalkeeper_hist_from_image, histogram_save_path)


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
# python3 idenfity_goalkeeper.py --image ./data/images/goalkeeper_clothes.jpg --histogram_save_path ./data/histograms/goalkeeper_clothes.npy