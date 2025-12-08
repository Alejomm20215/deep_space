"""
Lightweight multi-view pose estimation using OpenCV features.
Designed to be GPU-light: uses SIFT if available, else ORB.
Returns camera-to-world poses and intrinsics for use in unprojection.
"""
import cv2
import numpy as np
from typing import List, Tuple


class PoseEstimator:
    def __init__(self, max_matches: int = 800):
        self.max_matches = max_matches

        # Prefer SIFT; fallback to ORB if not available
        if hasattr(cv2, "SIFT_create"):
            self.detector = cv2.SIFT_create()
            self.norm = cv2.NORM_L2
        else:
            self.detector = cv2.ORB_create(2000)
            self.norm = cv2.NORM_HAMMING

    def estimate(self, image_paths: List[str]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Estimate camera-to-world poses for a list of images.
        Returns:
          poses: list of 4x4 camera-to-world matrices
          K: intrinsics matrix (fx, fy, cx, cy)
        """
        if len(image_paths) == 0:
            raise ValueError("No images provided for pose estimation")

        # Load first image for intrinsics
        ref_img = cv2.imread(image_paths[0])
        if ref_img is None:
            raise ValueError(f"Cannot read image: {image_paths[0]}")
        h, w = ref_img.shape[:2]
        fx = fy = max(h, w) * 0.8
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)

        # Detect features
        keypoints, descriptors = [], []
        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                keypoints.append(None)
                descriptors.append(None)
                continue
            kps, desc = self.detector.detectAndCompute(img, None)
            keypoints.append(kps)
            descriptors.append(desc)

        # First pose is identity
        poses = [np.eye(4, dtype=np.float32)]

        # Matcher
        matcher = cv2.BFMatcher(self.norm, crossCheck=True)

        # Match each view to the reference (view 0)
        ref_desc = descriptors[0]
        ref_kps = keypoints[0]
        if ref_desc is None or ref_kps is None:
            # Fallback: all identity
            return [np.eye(4, dtype=np.float32) for _ in image_paths], K

        for i in range(1, len(image_paths)):
            if descriptors[i] is None or keypoints[i] is None:
                poses.append(np.eye(4, dtype=np.float32))
                continue

            matches = matcher.match(ref_desc, descriptors[i])
            # Sort by distance and keep top-N
            matches = sorted(matches, key=lambda m: m.distance)[: self.max_matches]

            if len(matches) < 8:
                poses.append(np.eye(4, dtype=np.float32))
                continue

            pts_ref = np.float32([ref_kps[m.queryIdx].pt for m in matches])
            pts_i = np.float32([keypoints[i][m.trainIdx].pt for m in matches])

            # Essential matrix
            E, mask = cv2.findEssentialMat(pts_ref, pts_i, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
            if E is None:
                poses.append(np.eye(4, dtype=np.float32))
                continue

            _, R, t, mask_pose = cv2.recoverPose(E, pts_ref, pts_i, K)

            # Camera-to-world: invert [R|t] (which maps ref->i)
            R_wc = R.T
            t_wc = -R.T @ t
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_wc
            pose[:3, 3] = t_wc[:, 0]
            poses.append(pose)

        return poses, K

