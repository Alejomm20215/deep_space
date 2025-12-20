from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from backend.core.poses.base import PoseBackend, PoseBackendUnavailable, PoseResult


@dataclass(frozen=True)
class _ColmapModelPaths:
    database_path: str
    sparse_dir: str
    txt_dir: str


def _run(cmd: List[str], cwd: str) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """
    COLMAP quaternion is (qw, qx, qy, qz) and represents world->cam rotation.
    """
    qw, qx, qy, qz = qvec.tolist()
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _parse_cameras_txt(cameras_txt: str) -> np.ndarray:
    """
    Parse intrinsics from COLMAP cameras.txt.
    We pick the first camera and assume shared intrinsics.
    Supports SIMPLE_PINHOLE / PINHOLE.
    """
    with open(cameras_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            if len(parts) < 5:
                continue
            model = parts[1]
            width = float(parts[2])
            height = float(parts[3])
            params = list(map(float, parts[4:]))

            if model == "SIMPLE_PINHOLE":
                # f, cx, cy
                f0, cx, cy = params[:3]
                fx = fy = f0
            elif model == "PINHOLE":
                # fx, fy, cx, cy
                fx, fy, cx, cy = params[:4]
            else:
                # Fallback: approximate focal length if unsupported
                fx = fy = max(width, height) * 0.8
                cx, cy = width / 2.0, height / 2.0

            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            return K

    raise RuntimeError("COLMAP cameras.txt had no camera lines")


def _parse_images_txt(images_txt: str) -> Dict[str, np.ndarray]:
    """
    Parse images.txt into a mapping: image_name -> c2w 4x4 pose.
    COLMAP stores qvec,tvec for world->cam. We invert to get cam->world.
    """
    poses: Dict[str, np.ndarray] = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            # Image lines have:
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            # followed by a 2D points line (which we skip).
            if len(parts) >= 10 and parts[0].isdigit():
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                name = parts[9]

                qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
                tvec = np.array([tx, ty, tz], dtype=np.float64).reshape(3, 1)

                R_w2c = _qvec2rotmat(qvec)
                # world->cam: Xc = R*Xw + t
                # cam->world: Xw = R^T * (Xc - t)
                R_c2w = R_w2c.T
                t_c2w = (-R_w2c.T @ tvec).reshape(3)

                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = R_c2w.astype(np.float32)
                pose[:3, 3] = t_c2w.astype(np.float32)
                poses[name] = pose

                # Skip the next line (2D points)
                next(f, None)

    return poses


class ColmapPoseBackend(PoseBackend):
    name = "colmap"

    def __init__(self, colmap_exe: str | None = None):
        self._colmap = colmap_exe or shutil.which("colmap")
        if not self._colmap:
            raise PoseBackendUnavailable("COLMAP executable not found in PATH")

    def _paths(self, work_dir: str) -> _ColmapModelPaths:
        return _ColmapModelPaths(
            database_path=os.path.join(work_dir, "database.db"),
            sparse_dir=os.path.join(work_dir, "sparse"),
            txt_dir=os.path.join(work_dir, "txt"),
        )

    def estimate(self, image_paths: List[str], work_dir: str) -> PoseResult:
        os.makedirs(work_dir, exist_ok=True)
        paths = self._paths(work_dir)
        os.makedirs(paths.sparse_dir, exist_ok=True)
        os.makedirs(paths.txt_dir, exist_ok=True)

        # COLMAP expects an images directory; we already have frame paths.
        images_dir = os.path.dirname(os.path.abspath(image_paths[0]))

        # 1) Feature extraction
        _run(
            [
                self._colmap,
                "feature_extractor",
                "--database_path",
                paths.database_path,
                "--image_path",
                images_dir,
                "--ImageReader.single_camera",
                "1",
            ],
            cwd=work_dir,
        )

        # 2) Matching
        _run(
            [
                self._colmap,
                "exhaustive_matcher",
                "--database_path",
                paths.database_path,
            ],
            cwd=work_dir,
        )

        # 3) Mapping (sparse reconstruction)
        _run(
            [
                self._colmap,
                "mapper",
                "--database_path",
                paths.database_path,
                "--image_path",
                images_dir,
                "--output_path",
                paths.sparse_dir,
            ],
            cwd=work_dir,
        )

        # Pick the first model (usually sparse/0)
        model0 = os.path.join(paths.sparse_dir, "0")
        if not os.path.isdir(model0):
            raise RuntimeError("COLMAP mapper did not produce sparse/0")

        # 4) Convert to TXT so we can parse it without extra deps
        _run(
            [
                self._colmap,
                "model_converter",
                "--input_path",
                model0,
                "--output_path",
                paths.txt_dir,
                "--output_type",
                "TXT",
            ],
            cwd=work_dir,
        )

        cameras_txt = os.path.join(paths.txt_dir, "cameras.txt")
        images_txt = os.path.join(paths.txt_dir, "images.txt")
        if not os.path.exists(cameras_txt) or not os.path.exists(images_txt):
            raise RuntimeError("COLMAP model_converter did not produce cameras.txt/images.txt")

        K = _parse_cameras_txt(cameras_txt)
        name_to_pose = _parse_images_txt(images_txt)

        # Align to input ordering
        poses_c2w: List[np.ndarray] = []
        valid: List[bool] = []
        for p in image_paths:
            name = os.path.basename(p)
            pose = name_to_pose.get(name)
            if pose is None:
                poses_c2w.append(np.eye(4, dtype=np.float32))
                valid.append(False)
            else:
                poses_c2w.append(pose.astype(np.float32))
                valid.append(True)

        return PoseResult(poses_c2w=poses_c2w, K=K, valid=valid, backend=self.name, sfm_dir=work_dir)

