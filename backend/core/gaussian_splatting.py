import asyncio
import os
from typing import Tuple, List

import numpy as np
from backend.config import PipelineConfig

from backend.core.gaussians.colmap_scene_builder import build_colmap_scene
from backend.core.gaussians.docker_3dgs_trainer import (
    train_3dgs_docker,
    train_3dgs_adaptive,
)
from backend.core.gaussians.lm_refine import maybe_refine_with_lm


class GaussianSplattingOptimizer:
    """
    Lightweight "splat" generator.

    This project previously used a placeholder that copied the input PLY to `.splat`.
    Until a full 3DGS training backend is integrated, we generate a Gaussian-compatible
    PLY (position, scale, rotation, SH DC color) that can be rendered by common web
    splat viewers (e.g. GaussianSplats3D).

    Note: This is NOT full 3DGS optimization. It is a fast, local fallback that
    produces a real splat-like artifact for preview + downstream tooling.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def optimize(self, initial_points: str, poses: str, output_dir: str, progress_callback=None) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_splat = os.path.join(output_dir, "model_gaussians.ply")

        # If configured, run real 3DGS training (Docker). Otherwise use the fast local fallback.
        if self.config.gaussian_backend == "docker_3dgs" and self.config.use_gaussians:
            recon_dir = os.path.dirname(os.path.abspath(poses))
            frames_dir = os.path.abspath(os.path.join(os.path.dirname(recon_dir), "frames"))
            sfm_dir = os.path.join(recon_dir, "sfm")

            if progress_callback:
                await progress_callback(5, "Preparing COLMAP scene for 3DGS...")

            def train():
                # Build scene folder expected by 3DGS repo
                scene_dir = os.path.join(output_dir, "colmap_scene")
                build_colmap_scene(frames_dir=frames_dir, colmap_work_dir=sfm_dir, scene_dir=scene_dir)

                model_dir = os.path.join(output_dir, "trained_model")
                image = os.environ.get("GS_3DGS_DOCKER_IMAGE", "deep_space_3dgs_trainer:cu117")

                # Phase 8: Adaptive Training with interleaved pruning
                if self.config.adaptive_prune_enabled:
                    # Use adaptive training loop that prunes during training
                    adaptive_result = train_3dgs_adaptive(
                        image=image,
                        scene_dir=scene_dir,
                        model_dir=model_dir,
                        total_iterations=self.config.gaussian_iterations,
                        prune_interval=self.config.adaptive_prune_interval,
                        prune_opacity=self.config.adaptive_prune_opacity,
                        resolution=self.config.gaussian_resolution,
                        densify_from_iter=self.config.gaussian_densify_from,
                        densify_until_iter=self.config.gaussian_densify_until or self.config.gaussian_iterations,
                        densify_interval=self.config.gaussian_densify_interval or 100,
                        progress_callback=None,  # Could pipe to progress_callback if needed
                    )
                    ply_path = adaptive_result.gaussian_ply
                    print(f"[Adaptive Training] {adaptive_result.message}")

                # DashGaussian-style schedule: stage at low-res then resume at higher res.
                elif self.config.dashgaussian_schedule:
                    stage1_iters = max(500, int(self.config.gaussian_iterations * 0.35))
                    stage2_iters = int(self.config.gaussian_iterations)

                    # stage 1: cheaper resolution
                    r1 = max(self.config.gaussian_resolution, 4)
                    res1 = train_3dgs_docker(
                        image=image,
                        scene_dir=scene_dir,
                        model_dir=model_dir,
                        iterations=stage1_iters,
                        resolution=r1,
                        densify_from_iter=self.config.gaussian_densify_from,
                        densify_until_iter=min(self.config.gaussian_densify_until, stage1_iters),
                        densify_interval=self.config.gaussian_densify_interval or 100,
                        start_checkpoint=None,
                    )

                    # stage 2: resume with target resolution
                    # Prefer latest checkpoint inside model_dir if present.
                    ckpt = res1.checkpoint_path
                    res2 = train_3dgs_docker(
                        image=image,
                        scene_dir=scene_dir,
                        model_dir=model_dir,
                        iterations=stage2_iters,
                        resolution=self.config.gaussian_resolution,
                        densify_from_iter=self.config.gaussian_densify_from,
                        densify_until_iter=self.config.gaussian_densify_until,
                        densify_interval=self.config.gaussian_densify_interval or 100,
                        start_checkpoint=ckpt,
                    )
                    ply_path = res2.gaussian_ply
                else:
                    res = train_3dgs_docker(
                        image=image,
                        scene_dir=scene_dir,
                        model_dir=model_dir,
                        iterations=self.config.gaussian_iterations,
                        resolution=self.config.gaussian_resolution,
                        densify_from_iter=self.config.gaussian_densify_from,
                        densify_until_iter=self.config.gaussian_densify_until or self.config.gaussian_iterations,
                        densify_interval=self.config.gaussian_densify_interval or 100,
                        start_checkpoint=None,
                    )
                    ply_path = res.gaussian_ply

                # Optional LM refine hook (no-op unless enabled)
                lm_res = maybe_refine_with_lm(model_dir=model_dir, gaussian_ply=ply_path)
                ply_path = lm_res.refined_ply

                # Copy into stable output filename
                import shutil

                shutil.copy2(ply_path, output_splat)
                return output_splat

            try:
                trained = await asyncio.to_thread(train)
                if progress_callback:
                    await progress_callback(100, "3DGS training complete")
                return trained
            except Exception as e:
                # Fall back to local fast splat generation
                if progress_callback:
                    await progress_callback(10, f"3DGS training unavailable ({e}); using lightweight splat...")

        def build_gaussian_ply():
            xyz, rgb01 = self._read_vertex_rgb_ply(initial_points)
            # Cap gaussians to avoid enormous files (relief meshes can be huge)
            max_n = 200_000
            if xyz.shape[0] > max_n:
                idx = np.random.RandomState(0).choice(xyz.shape[0], size=max_n, replace=False)
                xyz = xyz[idx]
                rgb01 = rgb01[idx]

            self._write_gaussian_ply(output_splat, xyz, rgb01)

        if progress_callback:
            await progress_callback(5, "Reading point cloud...")

        await asyncio.to_thread(build_gaussian_ply)

        if progress_callback:
            await progress_callback(100, "Splat ready")

        return output_splat

    def _read_vertex_rgb_ply(self, ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read vertices (x,y,z) and colors (r,g,b) from an ASCII PLY.
        Accepts the mesh PLY produced by this repo (has vertex colors, may have faces).
        """
        with open(ply_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        header_end = 0
        num_vertices = 0
        props: List[str] = []
        in_vertex_element = False

        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("element vertex"):
                num_vertices = int(s.split()[-1])
                in_vertex_element = True
            elif s.startswith("element ") and not s.startswith("element vertex"):
                in_vertex_element = False
            elif in_vertex_element and s.startswith("property"):
                # property <type> <name>
                parts = s.split()
                if len(parts) >= 3:
                    props.append(parts[-1])
            elif s == "end_header":
                header_end = i + 1
                break

        if num_vertices <= 0:
            raise RuntimeError(f"No vertices found in PLY: {ply_path}")

        # Locate indices (default to standard order)
        def idx_of(name: str, default: int) -> int:
            try:
                return props.index(name)
            except ValueError:
                return default

        ix = idx_of("x", 0)
        iy = idx_of("y", 1)
        iz = idx_of("z", 2)
        ir = idx_of("red", 3)
        ig = idx_of("green", 4)
        ib = idx_of("blue", 5)

        xyz = np.zeros((num_vertices, 3), dtype=np.float32)
        rgb = np.zeros((num_vertices, 3), dtype=np.float32)

        for vi in range(num_vertices):
            parts = lines[header_end + vi].split()
            xyz[vi, 0] = float(parts[ix])
            xyz[vi, 1] = float(parts[iy])
            xyz[vi, 2] = float(parts[iz])

            # Handle either 0-255 uint8 or 0-1 floats
            r = float(parts[ir])
            g = float(parts[ig])
            b = float(parts[ib])
            if r > 1.0 or g > 1.0 or b > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            rgb[vi] = [r, g, b]

        rgb = np.clip(rgb, 0.0, 1.0)
        return xyz, rgb

    def _write_gaussian_ply(self, out_path: str, xyz: np.ndarray, rgb01: np.ndarray) -> None:
        """
        Write a 3DGS-style Gaussian PLY (ASCII) with DC SH color only.

        Many viewers use:
          rgb = 0.5 + C0 * f_dc, where C0 = 0.28209479177387814
        So we store:
          f_dc = (rgb - 0.5) / C0
        """
        n = int(xyz.shape[0])
        C0 = 0.28209479177387814
        f_dc = (rgb01 - 0.5) / C0

        # Heuristic scale: based on scene extent
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        extent = float(np.linalg.norm(maxs - mins) + 1e-6)
        base_scale = max(extent * 0.002, 1e-4)

        # Identity rotation quaternion (w,x,y,z)
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        opacity = 1.0

        # 45 rest coeffs for SH degree 3 (zeros)
        rest = [0.0] * 45

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float f_dc_0\n")
            f.write("property float f_dc_1\n")
            f.write("property float f_dc_2\n")
            for i in range(45):
                f.write(f"property float f_rest_{i}\n")
            f.write("property float opacity\n")
            f.write("property float scale_0\n")
            f.write("property float scale_1\n")
            f.write("property float scale_2\n")
            f.write("property float rot_0\n")
            f.write("property float rot_1\n")
            f.write("property float rot_2\n")
            f.write("property float rot_3\n")
            f.write("end_header\n")

            for i in range(n):
                x, y, z = xyz[i].tolist()
                r0, r1, r2 = f_dc[i].tolist()
                # Small random jitter on scale to reduce perfect-grid artifacts in some viewers
                s = base_scale * (0.9 + 0.2 * ((i * 2654435761) % 1000) / 1000.0)
                row = [x, y, z, r0, r1, r2, *rest, opacity, s, s, s, qw, qx, qy, qz]
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
