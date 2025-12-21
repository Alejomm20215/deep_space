"""
3D Reconstruction from images.
Uses Depth Anything V2 for high-quality depth estimation.
Creates dense 2.5D relief mesh from best image.
"""
import asyncio
import os
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from backend.config import PipelineConfig
from backend.core.depth_estimator import get_depth_estimator
from backend.core.poses.factory import create_best_available_pose_backend


class Fast3RReconstructor:
    """
    High-quality 3D reconstruction using AI depth estimation.
    Now supports multi-view with lightweight pose estimation (OpenCV E-matrix).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.depth_estimator = None
        self._pose_backend = None

    def _get_depth_estimator(self):
        if self.depth_estimator is None:
            self.depth_estimator = get_depth_estimator(use_refinement=True)
        return self.depth_estimator

    def _get_pose_backend(self):
        if self._pose_backend is None:
            self._pose_backend = create_best_available_pose_backend(self.config.pose_backend)
        return self._pose_backend

    def _postprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Depth Anything (and similar) often returns *relative* depth with outliers.
        For our lightweight mesh, we aggressively smooth + normalize to reduce spikes.
        """
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim != 2:
            d = d.squeeze()
        # Replace NaNs/Infs
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to [0, 1]
        dmin = float(d.min())
        dmax = float(d.max())
        if dmax - dmin > 1e-6:
            d = (d - dmin) / (dmax - dmin)

        # Clip extreme outliers (prevents "exploding" triangles)
        lo = float(np.quantile(d, 0.02))
        hi = float(np.quantile(d, 0.98))
        if hi - lo > 1e-6:
            d = np.clip(d, lo, hi)
            d = (d - lo) / (hi - lo)

        # Edge-preserving smoothing (keeps object boundaries sharper than a blur)
        try:
            d = cv2.bilateralFilter(d, d=5, sigmaColor=0.08, sigmaSpace=5)
        except Exception:
            pass

        return d.astype(np.float32)

    async def reconstruct(self, image_paths: List[str], output_dir: str, progress_callback=None) -> Dict[str, Any]:
        """
        Reconstruct 3D model from images.
        - Multi-view: estimate poses (COLMAP if available, else OpenCV), depth per view, fuse meshes.
        - Single-image fallback: dense 2.5D relief.
        """
        os.makedirs(output_dir, exist_ok=True)

        if len(image_paths) == 0:
            raise RuntimeError("No images provided.")

        if len(image_paths) == 1:
            return await self._single_image_relief(image_paths[0], output_dir, progress_callback)

        pose_backend = self._get_pose_backend()
        sfm_dir = os.path.join(output_dir, "sfm")
        print(
            f"[reconstruction] multi-view: {len(image_paths)} images | pose_backend={getattr(pose_backend, 'name', type(pose_backend).__name__)} | sfm_dir={sfm_dir}",
            flush=True,
        )
        if progress_callback:
            await progress_callback(6, f"Estimating poses ({pose_backend.name})...")
        try:
            pose_result = await asyncio.to_thread(pose_backend.estimate, image_paths, sfm_dir)
        except Exception as e:
            # If COLMAP fails to bootstrap (common with low overlap / low texture),
            # fall back to OpenCV so the pipeline can still proceed end-to-end.
            if (getattr(pose_backend, "name", "") or "") not in ("opencv", "cv"):
                from backend.core.poses.opencv_backend import OpenCVPoseBackend

                # COLMAP can fail on low-texture/low-overlap frames (no initial pair). We fall back.
                print(
                    f"[reconstruction] pose backend '{getattr(pose_backend, 'name', type(pose_backend).__name__)}' failed (often: no good initial image pair / weak matches); falling back to OpenCV. Error={type(e).__name__}: {e}",
                    flush=True,
                )
                if progress_callback:
                    await progress_callback(7, f"Pose backend '{pose_backend.name}' failed; falling back to OpenCV...")
                pose_backend = OpenCVPoseBackend()
                pose_result = await asyncio.to_thread(pose_backend.estimate, image_paths, sfm_dir)
            else:
                raise
        poses = pose_result.poses_c2w
        K = pose_result.K
        print(
            f"[reconstruction] poses ready: backend={pose_result.backend} | valid={sum(1 for v in pose_result.valid if v)}/{len(pose_result.valid)} | K_shape={getattr(K, 'shape', None)}",
            flush=True,
        )

        # IMPORTANT QUALITY FIX:
        # The previous implementation built a full dense relief mesh for *every* view and
        # concatenated them. That explodes triangle count (e.g. 8 frames -> ~8x triangles)
        # and makes the mesh look messy/duplicated.
        #
        # For low-resource + good-looking output, build the mesh from ONE best frame only,
        # while still saving poses/intrinsics for downstream tasks.
        estimator = self._get_depth_estimator()

        best_path = self._select_best_image(image_paths)
        best_idx = image_paths.index(best_path) if best_path in image_paths else 0

        if progress_callback:
            await progress_callback(12, "Depth (best view)...")

        img = cv2.imread(best_path)
        if img is None:
            return await self._single_image_relief(image_paths[0], output_dir, progress_callback)

        h, w = img.shape[:2]
        # Smaller max_dim keeps triangles reasonable and makes export/viewing fast.
        max_dim = 640
        scale = max_dim / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

        depth = await asyncio.to_thread(estimator.estimate, img, True)
        depth = self._postprocess_depth(depth)

        # Use the pose of the selected frame to place mesh in world coords
        verts, cols, faces = await asyncio.to_thread(
            self._create_dense_relief_multiview,
            img, depth, K, poses[best_idx]
        )

        vertices, colors = verts, cols

        if progress_callback:
            await progress_callback(80, "Writing output...")

        ply_path = os.path.join(output_dir, "sparse_pc.ply")
        await asyncio.to_thread(self._write_mesh_ply, ply_path, vertices, colors, faces)

        poses_path = os.path.join(output_dir, "poses.npy")
        np.save(poses_path, np.stack(poses, axis=0))

        K_path = os.path.join(output_dir, "intrinsics.npy")
        np.save(K_path, K)

        if progress_callback:
            await progress_callback(95, f"Mesh: {len(vertices)} vertices, {len(faces)} faces")

        return {
            "point_cloud": ply_path,
            "poses": poses_path,
            "intrinsics": K_path,
            "sfm_dir": sfm_dir,
            "point_count": len(vertices),
            "face_count": len(faces)
        }

    async def _single_image_relief(self, image_path: str, output_dir: str, progress_callback=None):
        if progress_callback:
            await progress_callback(15, "Loading image...")

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        h, w = img.shape[:2]
        max_dim = 1024
        scale = max_dim / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

        estimator = self._get_depth_estimator()
        depth = await asyncio.to_thread(estimator.estimate, img, True)
        depth = self._postprocess_depth(depth)

        if progress_callback:
            await progress_callback(60, "Creating 3D mesh...")

        vertices, colors, faces = await asyncio.to_thread(
            self._create_dense_relief_single,
            img, depth
        )

        if progress_callback:
            await progress_callback(80, "Writing output...")

        ply_path = os.path.join(output_dir, "sparse_pc.ply")
        await asyncio.to_thread(self._write_mesh_ply, ply_path, vertices, colors, faces)

        poses_path = os.path.join(output_dir, "poses.npy")
        np.save(poses_path, np.eye(4, dtype=np.float32))

        K_path = os.path.join(output_dir, "intrinsics.npy")
        # Approx intrinsics for single image (matches PoseEstimator heuristic)
        fx = fy = max(img.shape[0], img.shape[1]) * 0.8
        cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        np.save(K_path, K)

        if progress_callback:
            await progress_callback(95, f"Mesh: {len(vertices)} vertices, {len(faces)} faces")

        return {
            "point_cloud": ply_path,
            "poses": poses_path,
            "intrinsics": K_path,
            "point_count": len(vertices),
            "face_count": len(faces)
        }

    def _select_best_image(self, image_paths: List[str]) -> str:
        """
        Select the best image based on sharpness and size.
        """
        best_path = image_paths[0]
        best_score = -1

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue

            # Score = resolution * sharpness
            h, w = img.shape[:2]
            resolution_score = h * w

            # Laplacian variance = sharpness measure
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            score = resolution_score * (1 + sharpness / 1000)

            if score > best_score:
                best_score = score
                best_path = path

        return best_path

    def _create_dense_relief_single(self, img: np.ndarray, depth: np.ndarray):
        """Create dense 2.5D relief mesh from one image/depth."""
        h, w = img.shape[:2]

        # Use every pixel for maximum quality
        # For very large images, subsample slightly
        step = 1 if h * w < 500000 else 2

        # Create grid of vertices
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        grid_h, grid_w = ys.shape

        # Get depth values (already post-processed to [0,1])
        zs = depth[ys, xs].astype(np.float32)

        # Normalize coordinates to [-0.5, 0.5] range
        x_norm = (xs.astype(np.float32) / w - 0.5)
        y_norm = -(ys.astype(np.float32) / h - 0.5)  # Flip Y for 3D

        # Depth scaling: keep relief subtle to avoid warped "spikes"
        depth_scale = 0.12
        z_norm = (0.5 - zs) * depth_scale

        # Stack into vertices (N, 3)
        vertices = np.stack([
            x_norm.flatten(),
            y_norm.flatten(),
            z_norm.flatten()
        ], axis=-1).astype(np.float32)

        # Get colors (RGB normalized)
        colors = img[ys, xs][:, :, ::-1].reshape(-1, 3) / 255.0
        colors = colors.astype(np.float32)

        # Create triangle faces (2 triangles per quad)
        faces = []
        for i in range(grid_h - 1):
            for j in range(grid_w - 1):
                # Vertex indices
                v00 = i * grid_w + j
                v01 = i * grid_w + (j + 1)
                v10 = (i + 1) * grid_w + j
                v11 = (i + 1) * grid_w + (j + 1)

                # Two triangles per quad
                faces.append([v00, v10, v01])  # Lower-left triangle
                faces.append([v01, v10, v11])  # Upper-right triangle

        faces = np.array(faces, dtype=np.int32)

        return vertices, colors, faces

    def _create_dense_relief_multiview(self, img: np.ndarray, depth: np.ndarray,
                                       K: np.ndarray, pose_c2w: np.ndarray):
        """Create dense mesh for one view and place it in world coordinates using pose."""
        h, w = img.shape[:2]

        step = 1 if h * w < 500000 else 2
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        grid_h, grid_w = ys.shape

        # Depth is *relative*; keep it normalized + bounded to avoid huge spikes.
        zs = depth[ys, xs].astype(np.float32)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # Map relative depth to a stable pseudo-metric range.
        # Keep z positive (COLMAP/3DGS expect forward-facing camera Z).
        z_cam = 1.0 + (0.5 - zs) * 0.8
        x_cam = (xs - cx) * z_cam / fx
        y_cam = (ys - cy) * z_cam / fy

        pts_cam = np.stack([x_cam, -y_cam, z_cam], axis=-1).reshape(-1, 3).astype(np.float32)

        R = pose_c2w[:3, :3]
        t = pose_c2w[:3, 3]
        pts_world = (pts_cam @ R.T) + t

        colors = img[ys, xs][:, :, ::-1].reshape(-1, 3) / 255.0

        faces = []
        for i in range(grid_h - 1):
            for j in range(grid_w - 1):
                v00 = i * grid_w + j
                v01 = i * grid_w + (j + 1)
                v10 = (i + 1) * grid_w + j
                v11 = (i + 1) * grid_w + (j + 1)
                faces.append([v00, v10, v01])
                faces.append([v01, v10, v11])
        faces = np.array(faces, dtype=np.int32)

        return pts_world, colors.astype(np.float32), faces

    def _write_mesh_ply(self, path: str, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray):
        """Write mesh as PLY with vertices, colors, and faces."""
        colors_uint8 = (colors * 255).clip(0, 255).astype(np.uint8)

        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices
            for v, c in zip(vertices, colors_uint8):
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def _write_ply(self, path: str, pts: np.ndarray, colors: np.ndarray):
        """Write colored point cloud as PLY (legacy compatibility)."""
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
