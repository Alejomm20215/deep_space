"""
3D Reconstruction from multi-view images.
Uses Depth Anything V2 for depth estimation.
Assumes turntable-style capture (camera orbits around object).
"""
import asyncio
import os
import math
from typing import List, Dict, Any

import cv2
import numpy as np
from backend.config import PipelineConfig
from backend.core.depth_estimator import get_depth_estimator


class Fast3RReconstructor:
    """
    Multi-view reconstruction:
    1. Estimate depth for each image using AI
    2. Assume turntable camera positions (evenly spaced around object)
    3. Unproject to 3D and merge point clouds
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.depth_estimator = None

    def _get_depth_estimator(self):
        if self.depth_estimator is None:
            self.depth_estimator = get_depth_estimator()
        return self.depth_estimator

    async def reconstruct(self, image_paths: List[str], output_dir: str, progress_callback=None) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        num_images = len(image_paths)
        if num_images == 0:
            raise RuntimeError("No images provided.")

        all_points = []
        all_colors = []

        for idx, img_path in enumerate(image_paths):
            if progress_callback:
                await progress_callback(
                    int((idx / num_images) * 70),
                    f"Processing image {idx + 1}/{num_images}"
                )

            pts, clr = await asyncio.to_thread(
                self._process_single_image,
                img_path,
                idx,
                num_images
            )

            if pts is not None and len(pts) > 0:
                all_points.append(pts)
                all_colors.append(clr)

        if not all_points:
            raise RuntimeError("No valid images for reconstruction.")

        # Merge all point clouds
        merged_pts = np.concatenate(all_points, axis=0)
        merged_clr = np.concatenate(all_colors, axis=0)

        # Subsample if too many points
        max_points = 200000
        if len(merged_pts) > max_points:
            indices = np.random.choice(len(merged_pts), max_points, replace=False)
            merged_pts = merged_pts[indices]
            merged_clr = merged_clr[indices]

        if progress_callback:
            await progress_callback(80, "Writing point cloud...")

        # Center the point cloud
        centroid = merged_pts.mean(axis=0)
        merged_pts = merged_pts - centroid

        # Scale to unit box
        scale = np.abs(merged_pts).max()
        if scale > 1e-6:
            merged_pts = merged_pts / scale * 0.5

        # Write PLY
        ply_path = os.path.join(output_dir, "sparse_pc.ply")
        await asyncio.to_thread(self._write_ply, ply_path, merged_pts, merged_clr)

        # Write dummy poses
        poses_path = os.path.join(output_dir, "poses.npy")
        np.save(poses_path, np.eye(4, dtype=np.float32))

        if progress_callback:
            await progress_callback(90, f"Point cloud: {len(merged_pts)} points")

        return {
            "point_cloud": ply_path,
            "poses": poses_path,
            "point_count": len(merged_pts)
        }

    def _process_single_image(self, img_path: str, img_idx: int, total_images: int):
        """Process one image: depth estimate + unproject to 3D."""
        img = cv2.imread(img_path)
        if img is None:
            return None, None

        # Resize for speed
        h, w = img.shape[:2]
        max_dim = 512
        scale = max_dim / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        # Estimate depth
        estimator = self._get_depth_estimator()
        depth = estimator.estimate(img)

        # Camera intrinsics (approximate)
        fx = fy = max(h, w) * 0.8
        cx, cy = w / 2.0, h / 2.0

        # Turntable camera pose: rotate around Y axis
        # Assume images are taken in a circle looking at center
        angle = (img_idx / total_images) * 2 * math.pi
        radius = 1.5  # Distance from object center

        # Camera position on circle
        cam_x = radius * math.sin(angle)
        cam_z = radius * math.cos(angle)
        cam_y = 0.0

        # Camera looks toward origin
        # Rotation matrix: camera Z points toward origin
        forward = np.array([-cam_x, -cam_y, -cam_z])
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=1)  # 3x3 rotation
        t = np.array([cam_x, cam_y, cam_z])  # translation

        # Unproject pixels to 3D
        step = 2  # Sample every 2 pixels for speed
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        zs = depth[ys, xs]

        # Filter out very close/far depths
        mask = (zs > 0.05) & (zs < 0.95)
        ys, xs, zs = ys[mask], xs[mask], zs[mask]

        # Depth to actual distance (invert and scale)
        # Depth map: 0 = close, 1 = far -> convert to actual depth
        actual_depth = 0.2 + zs * 1.0  # Range 0.2 to 1.2 meters

        # Unproject to camera space
        x_cam = (xs - cx) * actual_depth / fx
        y_cam = (ys - cy) * actual_depth / fy
        z_cam = actual_depth

        pts_cam = np.stack([x_cam, -y_cam, -z_cam], axis=-1)  # Flip Y and Z for OpenGL convention

        # Transform to world space
        pts_world = pts_cam @ R.T + t

        # Get colors
        colors = img[ys, xs][:, ::-1] / 255.0  # BGR to RGB, normalize

        return pts_world.reshape(-1, 3).astype(np.float32), colors.reshape(-1, 3).astype(np.float32)

    def _write_ply(self, path: str, pts: np.ndarray, colors: np.ndarray):
        """Write colored point cloud as PLY."""
        colors_uint8 = (colors * 255).clip(0, 255).astype(np.uint8)

        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for p, c in zip(pts, colors_uint8):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
