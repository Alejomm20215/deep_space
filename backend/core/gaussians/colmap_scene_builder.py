from __future__ import annotations

import os
import shutil
from dataclasses import dataclass


@dataclass(frozen=True)
class ColmapScenePaths:
    scene_dir: str
    images_dir: str
    sparse0_dir: str


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

    return ColmapScenePaths(scene_dir=scene_dir, images_dir=images_dir, sparse0_dir=sparse0_dir)

