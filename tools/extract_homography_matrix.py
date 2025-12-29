#!/usr/bin/env python3
import argparse
import numpy as np
import cv2
import ast
import os


def compute_homography(src_points, dst_points, method=cv2.RANSAC, ransac_thresh=5.0):
    """
    Compute homography matrix from two sets of corresponding points.

    Args:
        src_points (list of tuples): Source points [(x,y), ...]
        dst_points (list of tuples): Destination points [(x,y), ...]
        method (int): Method used to compute the homography (default: cv2.RANSAC)
        ransac_thresh (float): RANSAC reprojection threshold

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
    """
    x, y = point
    src = np.array([x, y, 1.0], dtype=np.float32).reshape(3, 1)
    dst = matrix @ src
    dst /= dst[2, 0]  # normalize
    return (dst[0, 0], dst[1, 0])


def parse_points(points_str):
    """
    Parse string like "86,242 1658,258 1644,766 100,771"
    into [(86,242), (1658,258), (1644,766), (100,771)]
    """
    pts = []
    for token in points_str.split():
        x, y = token.split(",")
        pts.append((float(x), float(y)))
    return pts


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Compute homography matrix from point correspondences")
    ap.add_argument("--src", required=True, type=str,
                    help='Source points: e.g. "86,242 1658,258 1644,766 100,771"')
    ap.add_argument("--dst", required=True, type=str,
                    help='Destination points: e.g. "0,0 2400,0 2400,800 0,800"')
    ap.add_argument("--out", required=True, type=str,
                    help="Output .npy file to save the homography matrix")
    ap.add_argument("--ransac-thresh", type=float, default=5.0,
                    help="RANSAC reprojection threshold")
    args = ap.parse_args()

    src_pts = parse_points(args.src)
    dst_pts = parse_points(args.dst)

    H, mask = compute_homography(src_pts, dst_pts, method=cv2.RANSAC, ransac_thresh=args.ransac_thresh)

    print("Homography Matrix:\n", H)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, H)
    print(f"Saved matrix to {args.out}")
