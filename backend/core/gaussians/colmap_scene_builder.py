from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ColmapScenePaths:
    scene_dir: str
    images_dir: str
    sparse0_dir: str


def _to_host_path(path: str) -> str:
    """
    Translate container paths to host paths for Docker-in-Docker on Windows.
    Produces /run/desktop/mnt/host/<drive>/... when running inside a Linux container.
    """
    host_pwd = os.environ.get("HOST_PWD")
    if not host_pwd:
        return os.path.abspath(path)

    host_pwd_norm = host_pwd.replace("\\", "/").rstrip("/")
    abs_p = os.path.abspath(path).replace("\\", "/")
    if abs_p.startswith("/app"):
        abs_p = abs_p.replace("/app", host_pwd_norm, 1)

    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):/(.*)$", abs_p)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2)
            return f"/run/desktop/mnt/host/{drive}/{rest}"
    return abs_p


def _docker_colmap(scene_dir: str, args: list[str]) -> None:
    """
    Run COLMAP in a sibling container (CPU-only by default for stability).
    """
    image = os.environ.get("COLMAP_DOCKER_IMAGE", "colmap/colmap")
    use_gpu = os.environ.get("COLMAP_USE_GPU", "0") == "1"
    mount = _to_host_path(scene_dir)
    cmd = [
        "docker",
        "run",
        "--rm",
        *(["--gpus", "all"] if use_gpu else []),
        "-v",
        f"{mount}:/scene",
        image,
        "colmap",
        *args,
    ]
    subprocess.run(cmd, check=True)


def build_colmap_scene(*, frames_dir: str, colmap_work_dir: str, scene_dir: str) -> ColmapScenePaths:
    """
    Build a COLMAP scene folder compatible with graphdeco-inria/gaussian-splatting.

    Expected output:
      scene_dir/
        images/      (jpg/png)
        sparse/0/    (cameras.bin, images.bin, points3D.bin)
    """
    images_dir = os.path.join(scene_dir, "images")
    sparse0_dir = os.path.join(scene_dir, "sparse", "0")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse0_dir, exist_ok=True)

    # Copy images (avoid symlinks on Windows by default)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    for name in sorted(os.listdir(frames_dir)):
        if not name.lower().endswith(valid_ext):
            continue
        src = os.path.join(frames_dir, name)
        dst = os.path.join(images_dir, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Copy sparse model from COLMAP work dir
    src_sparse0 = os.path.join(colmap_work_dir, "sparse", "0")
    if not os.path.isdir(src_sparse0):
        raise RuntimeError(f"COLMAP sparse model not found at: {src_sparse0}")

    for name in os.listdir(src_sparse0):
        src = os.path.join(src_sparse0, name)
        dst = os.path.join(sparse0_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # 3DGS requires undistorted camera models (PINHOLE / SIMPLE_PINHOLE).
    # COLMAP's mapper often produces SIMPLE_RADIAL / OPENCV models by default.
    # Run image_undistorter to create an undistorted dataset in-place.
    try:
        undist_dir = os.path.join(scene_dir, "_undistorted")
        os.makedirs(undist_dir, exist_ok=True)

        _docker_colmap(
            scene_dir,
            [
                "image_undistorter",
                "--image_path",
                "/scene/images",
                "--input_path",
                "/scene/sparse/0",
                "--output_path",
                "/scene/_undistorted",
                "--output_type",
                "COLMAP",
                "--max_image_size",
                "2000",
            ],
        )

        # Replace images/ and sparse/0 with undistorted outputs
        und_images = os.path.join(undist_dir, "images")
        und_sparse = os.path.join(undist_dir, "sparse")

        if os.path.isdir(und_images) and os.path.isdir(und_sparse):
            # Move original aside (keep for debugging)
            orig_images = os.path.join(scene_dir, "_orig_images")
            orig_sparse = os.path.join(scene_dir, "_orig_sparse0")
            if os.path.isdir(images_dir) and not os.path.exists(orig_images):
                shutil.move(images_dir, orig_images)
            if os.path.isdir(sparse0_dir) and not os.path.exists(orig_sparse):
                shutil.move(sparse0_dir, orig_sparse)

            # Install undistorted
            shutil.move(und_images, images_dir)
            # und_sparse contains cameras/images/points, but not necessarily /0
            os.makedirs(sparse0_dir, exist_ok=True)
            for name in os.listdir(und_sparse):
                shutil.copy2(os.path.join(und_sparse, name), os.path.join(sparse0_dir, name))
    except Exception:
        # If undistortion fails, keep original. Trainer will error with a clear message.
        pass

    return ColmapScenePaths(scene_dir=scene_dir, images_dir=images_dir, sparse0_dir=sparse0_dir)

