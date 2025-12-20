from __future__ import annotations

import os
import subprocess
from typing import List

import numpy as np

from backend.core.poses.base import PoseBackend, PoseBackendUnavailable, PoseResult
from backend.core.poses.colmap_backend import _parse_cameras_txt, _parse_images_txt


def _to_host_path(container_path: str) -> str:
    host_pwd = os.environ.get("HOST_PWD")
    if not host_pwd:
        return os.path.abspath(container_path)
    
    # In container, everything is under /app. 
    # If path starts with /app, replace it with HOST_PWD
    abs_container = os.path.abspath(container_path).replace("\\", "/")
    if abs_container.startswith("/app"):
        return abs_container.replace("/app", host_pwd, 1)
    return abs_container

class ColmapDockerPoseBackend(PoseBackend):
    """
    COLMAP via Docker image (no host install required).

    Uses the official `colmap/colmap` image by default.
    """

    name = "colmap_docker"

    def __init__(self, image: str | None = None):
        self._image = image or os.environ.get("COLMAP_DOCKER_IMAGE", "colmap/colmap")

        # Light check: docker available?
        try:
            subprocess.run(["docker", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            raise PoseBackendUnavailable("Docker not available for colmap_docker backend") from e

    def estimate(self, image_paths: List[str], work_dir: str) -> PoseResult:
        os.makedirs(work_dir, exist_ok=True)
        images_dir = os.path.dirname(os.path.abspath(image_paths[0]))

        # Use a work dir inside the container
        # We'll mount work_dir and images_dir separately.
        db_path = os.path.join(work_dir, "database.db")
        sparse_dir = os.path.join(work_dir, "sparse")
        txt_dir = os.path.join(work_dir, "txt")
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        w_mount = _to_host_path(work_dir)
        i_mount = _to_host_path(images_dir)

        def run_colmap(args: List[str]) -> None:
            cmd = [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{w_mount}:/work",
                "-v",
                f"{i_mount}:/images",
                self._image,
                "colmap",
                *args,
            ]
            subprocess.run(cmd, check=True)

        # Feature extraction / matching / mapping
        run_colmap(
            [
                "feature_extractor",
                "--database_path",
                "/work/database.db",
                "--image_path",
                "/images",
                "--ImageReader.single_camera",
                "1",
            ]
        )
        run_colmap(["exhaustive_matcher", "--database_path", "/work/database.db"])
        run_colmap(
            [
                "mapper",
                "--database_path",
                "/work/database.db",
                "--image_path",
                "/images",
                "--output_path",
                "/work/sparse",
            ]
        )

        model0 = os.path.join(work_dir, "sparse", "0")
        if not os.path.isdir(model0):
            raise RuntimeError("COLMAP (docker) mapper did not produce sparse/0")

        # Convert to TXT for parsing
        run_colmap(
            [
                "model_converter",
                "--input_path",
                "/work/sparse/0",
                "--output_path",
                "/work/txt",
                "--output_type",
                "TXT",
            ]
        )

        cameras_txt = os.path.join(work_dir, "txt", "cameras.txt")
        images_txt = os.path.join(work_dir, "txt", "images.txt")
        K = _parse_cameras_txt(cameras_txt)
        name_to_pose = _parse_images_txt(images_txt)

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

        return PoseResult(poses_c2w=poses_c2w, K=K.astype(np.float32), valid=valid, backend=self.name, sfm_dir=work_dir)


