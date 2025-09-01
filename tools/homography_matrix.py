import numpy as np
import cv2

def compute_homography(src_points, dst_points, method=cv2.RANSAC, ransac_thresh=5.0):
    """
    Compute homography matrix from two sets of corresponding points.

    Args:
        src_points (np.ndarray): Source points, shape (N, 2)
        dst_points (np.ndarray): Destination points, shape (N, 2)
        method (int): Method used to compute the homography (default: cv2.RANSAC)
        ransac_thresh (float): RANSAC reprojection threshold (default: 5.0)

    Returns:
        H (np.ndarray): Homography matrix, shape (3, 3)
        mask (np.ndarray): Mask of inliers used by RANSAC
    """
    src_pts = np.asarray(src_points, dtype=np.float32)
    dst_pts = np.asarray(dst_points, dtype=np.float32)

    if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        raise ValueError("At least 4 point correspondences are required to compute homography.")

    H, mask = cv2.findHomography(src_pts, dst_pts, method=method, ransacReprojThreshold=ransac_thresh)
    return H, mask


def apply_homography_to_point(point, matrix):
    """
    Applies a 3x3 homography matrix to a single (x, y) point.

    Args:
        point (tuple or list): (x, y)
        matrix (np.ndarray): 3x3 homography matrix

    Returns:
        tuple: (x', y') — the transformed point
    """
    x, y = point
    src = np.array([x, y, 1.0], dtype=np.float32).reshape(3, 1)
    dst = matrix @ src
    dst /= dst[2, 0]  # normalize by z
    return (dst[0, 0], dst[1, 0])


if __name__ == "__main__":
    src = [(172, 1104), (2101, 895), (3800, 1021), (3458, 2057)]
    dst = [(530, 0), (530, 660), (1060, 660), (1060, 0)]

    H, mask = compute_homography(src, dst)
    print("Homography Matrix:\n", H)

    # point = (150, 150)
    # transformed_point = apply_homography_to_point(point, H)
    # print("Transformed Point:", transformed_point)

