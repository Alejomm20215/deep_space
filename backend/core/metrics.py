"""
Metrics and lightweight introspection for produced artifacts.

We keep this dependency-free and fast: just parse PLY headers and filesystem sizes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ArtifactMetrics:
    elapsed_seconds: Optional[float]
    frames: int
    point_vertices: Optional[int]
    point_faces: Optional[int]
    gaussian_count: Optional[int]
    mesh_faces: Optional[int]
    sizes_bytes: Dict[str, int]
    # Quality metrics (Phase 6)
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "frames": self.frames,
            "point_vertices": self.point_vertices,
            "point_faces": self.point_faces,
            "gaussian_count": self.gaussian_count,
            "mesh_faces": self.mesh_faces,
            "sizes_bytes": self.sizes_bytes,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips": self.lpips,
        }


def _file_size(path: Optional[str]) -> int:
    if not path:
        return 0
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def _ply_counts(ply_path: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """
    Returns (vertex_count, face_count) for a PLY if header is readable.
    Supports both ASCII and binary PLY by parsing only the header.
    """
    if not ply_path or not os.path.exists(ply_path):
        return None, None
    v = None
    f = None
    try:
        # Read as bytes to avoid UnicodeDecodeError for binary PLY bodies.
        with open(ply_path, "rb") as fh:
            while True:
                line_b = fh.readline()
                if not line_b:
                    break
                line = line_b.decode("ascii", errors="ignore")
                if line.startswith("element vertex"):
                    try:
                        v = int(line.split()[-1])
                    except Exception:
                        pass
                elif line.startswith("element face"):
                    try:
                        f = int(line.split()[-1])
                    except Exception:
                        pass
                elif line.strip() == "end_header":
                    break
    except OSError:
        return None, None
    return v, f


def compute_metrics(
    *,
    frames: int,
    elapsed_seconds: Optional[float],
    point_cloud_ply: Optional[str],
    gaussian_ply: Optional[str],
    mesh_ply: Optional[str],
    glb_path: Optional[str],
    psnr: Optional[float] = None,
    ssim: Optional[float] = None,
    lpips: Optional[float] = None,
) -> ArtifactMetrics:
    pc_v, pc_f = _ply_counts(point_cloud_ply)
    gs_v, _ = _ply_counts(gaussian_ply)
    mesh_v, mesh_f = _ply_counts(mesh_ply)

    sizes = {
        "point_cloud_ply": _file_size(point_cloud_ply),
        "gaussian_ply": _file_size(gaussian_ply),
        "mesh_ply": _file_size(mesh_ply),
        "glb": _file_size(glb_path),
    }

    return ArtifactMetrics(
        elapsed_seconds=elapsed_seconds,
        frames=frames,
        point_vertices=pc_v,
        point_faces=pc_f,
        gaussian_count=gs_v,
        mesh_faces=mesh_f,
        sizes_bytes=sizes,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
    )


def write_metrics_json(path: str, metrics: ArtifactMetrics) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    return path

