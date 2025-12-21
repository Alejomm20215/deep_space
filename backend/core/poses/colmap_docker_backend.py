from __future__ import annotations

import os
import shlex
import subprocess
import re
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
    host_pwd_norm = host_pwd.replace("\\", "/").rstrip("/")
    abs_container = os.path.abspath(container_path).replace("\\", "/")
    if abs_container.startswith("/app"):
        abs_container = abs_container.replace("/app", host_pwd_norm, 1)

    abs_container = abs_container.replace("\\", "/")

    # If we're a Linux container talking to Docker Desktop daemon, Windows drive paths
    # like C:/... must be converted to the daemon's internal host mount path.
    # Otherwise Docker errors out with exit status 125.
    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):/(.*)$", abs_container)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2)
            return f"/run/desktop/mnt/host/{drive}/{rest}"

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
        print(f"[colmap_docker] start: images={len(image_paths)} | images_dir={images_dir} | work_dir={work_dir}", flush=True)

        # Use a work dir inside the container
        # We'll mount work_dir and images_dir separately.
        db_path = os.path.join(work_dir, "database.db")
        sparse_dir = os.path.join(work_dir, "sparse")
        txt_dir = os.path.join(work_dir, "txt")
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(txt_dir, exist_ok=True)

        w_mount = _to_host_path(work_dir)
        i_mount = _to_host_path(images_dir)
        print(f"[colmap_docker] mounts: -v {w_mount}:/work  -v {i_mount}:/images", flush=True)

        def run_colmap(args: List[str]) -> None:
            # Hard force CPU-only for stability on Windows / older drivers.
            # (GPU COLMAP in docker frequently fails with nvidia-container-cli / CUDA mismatch.)
            use_gpu = False

            # Even without --gpus all, COLMAP may still default to GPU SIFT and fail.
            # Force CPU SIFT/matching unless explicitly enabled.
            if not use_gpu and args:
                sub = args[0]
                if sub == "feature_extractor":
                    # COLMAP 3.13 uses FeatureExtraction.* flags (not SiftExtraction.use_gpu)
                    args = [*args, "--FeatureExtraction.use_gpu", "0"]
                elif sub in ("exhaustive_matcher", "sequential_matcher", "spatial_matcher"):
                    # COLMAP 3.13 uses FeatureMatching.* flags (not SiftMatching.use_gpu)
                    args = [*args, "--FeatureMatching.use_gpu", "0"]
                elif sub == "mapper":
                    # Bundle adjustment GPU flag
                    args = [*args, "--Mapper.ba_use_gpu", "0"]

            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{w_mount}:/work",
                "-v",
                f"{i_mount}:/images",
                self._image,
                "colmap",
                *args,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                # Bubble up the actual COLMAP error; very useful for debugging.
                tail = (proc.stderr or proc.stdout or "").strip()
                tail = "\n".join(tail.splitlines()[-60:])  # keep it readable
                raise RuntimeError(
                    "COLMAP docker command failed.\n"
                    f"Command: {shlex.join(cmd)}\n"
                    f"Exit code: {proc.returncode}\n"
                    f"Output (tail):\n{tail}\n"
                )
            else:
                # Keep it short; COLMAP is chatty.
                if args:
                    print(f"[colmap_docker] ok: {args[0]}", flush=True)

        def _run_mapper(*, relaxed: bool) -> None:
            base = [
                "mapper",
                "--database_path",
                "/work/database.db",
                "--image_path",
                "/images",
                "--output_path",
                "/work/sparse",
            ]
            if not relaxed:
                run_colmap(base)
                return

            # More permissive init for weak scenes / low overlap.
            run_colmap(
                [
                    *base,
                    "--Mapper.min_num_matches",
                    "6",
                    "--Mapper.init_min_num_inliers",
                    "20",
                    "--Mapper.init_max_error",
                    "10",
                    "--Mapper.abs_pose_min_num_inliers",
                    "15",
                    "--Mapper.filter_max_reproj_error",
                    "10",
                    "--Mapper.max_reg_trials",
                    "8",
                    "--Mapper.multiple_models",
                    "1",
                ]
            )

        # Feature extraction / matching / mapping
        print("[colmap_docker] step: feature_extractor", flush=True)
        run_colmap(
            [
                "feature_extractor",
                "--database_path",
                "/work/database.db",
                "--image_path",
                "/images",
                "--ImageReader.single_camera",
                "1",
                # Slightly more robust defaults for small, difficult sequences
                "--SiftExtraction.max_image_size",
                "2000",
                "--SiftExtraction.max_num_features",
                "16384",
            ]
        )

        matcher = (os.environ.get("COLMAP_MATCHER", "exhaustive") or "exhaustive").strip().lower()
        if matcher not in ("exhaustive", "sequential", "both"):
            matcher = "exhaustive"

        # Matching: exhaustive is best for unordered; sequential is best for video frames.
        if matcher in ("exhaustive", "both"):
            print("[colmap_docker] step: exhaustive_matcher", flush=True)
            run_colmap(["exhaustive_matcher", "--database_path", "/work/database.db"])
        if matcher in ("sequential", "both"):
            print("[colmap_docker] step: sequential_matcher", flush=True)
            run_colmap(
                [
                    "sequential_matcher",
                    "--database_path",
                    "/work/database.db",
                    "--SequentialMatching.overlap",
                    "10",
                ]
            )

        # Mapping (retry strategy)
        try:
            print("[colmap_docker] step: mapper", flush=True)
            _run_mapper(relaxed=False)
        except RuntimeError as e1:
            print(f"[colmap_docker] mapper failed; retrying relaxed... ({type(e1).__name__})", flush=True)
            try:
                _run_mapper(relaxed=True)
            except RuntimeError as e2:
                # Last resort: if we didn't already, try sequential matcher then relaxed mapper.
                if matcher == "exhaustive":
                    print("[colmap_docker] mapper still failing; trying sequential_matcher + relaxed mapper...", flush=True)
                    run_colmap(
                        [
                            "sequential_matcher",
                            "--database_path",
                            "/work/database.db",
                            "--SequentialMatching.overlap",
                            "10",
                        ]
                    )
                    _run_mapper(relaxed=True)
                else:
                    raise e2

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


