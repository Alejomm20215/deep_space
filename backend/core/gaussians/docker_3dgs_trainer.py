from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Callable


@dataclass(frozen=True)
class TrainResult:
    model_dir: str
    gaussian_ply: str
    checkpoint_path: Optional[str]


# Re-export AdaptiveTrainResult for convenience
from backend.core.gaussians.adaptive_trainer import (
    AdaptiveTrainResult,
    train_adaptive,
)


def _which_docker() -> str:
    # Let subprocess find it via PATH; return name for nicer errors.
    return "docker"


def _latest_pointcloud_ply(model_dir: str) -> str:
    """
    graphdeco 3DGS writes:
      model_dir/point_cloud/iteration_<N>/point_cloud.ply
    """
    pc_root = os.path.join(model_dir, "point_cloud")
    if not os.path.isdir(pc_root):
        raise RuntimeError(f"3DGS output missing point_cloud dir: {pc_root}")

    best_n = -1
    best_path = None
    for name in os.listdir(pc_root):
        m = re.match(r"iteration_(\d+)$", name)
        if not m:
            continue
        n = int(m.group(1))
        cand = os.path.join(pc_root, name, "point_cloud.ply")
        if os.path.exists(cand) and n > best_n:
            best_n = n
            best_path = cand

    if not best_path:
        raise RuntimeError(f"Could not find iteration_*/point_cloud.ply under: {pc_root}")
    return best_path


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
    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):/(.*)$", abs_container)
        if m:
            drive = m.group(1).lower()
            rest = m.group(2)
            return f"/run/desktop/mnt/host/{drive}/{rest}"

    return abs_container

def train_3dgs_docker(
    *,
    image: str,
    scene_dir: str,
    model_dir: str,
    iterations: int,
    resolution: int,
    densify_from_iter: int,
    densify_until_iter: int,
    densify_interval: int,
    start_checkpoint: Optional[str] = None,
    save_iterations: Optional[List[int]] = None,
    checkpoint_iterations: Optional[List[int]] = None,
    eval_split: bool = False,
) -> TrainResult:
    """
    Run 3DGS training via Docker container.
    
    Args:
        image: Docker image name
        scene_dir: COLMAP scene directory
        model_dir: Output model directory
        iterations: Total training iterations
        resolution: Resolution downscale factor
        densify_from_iter: Start densification at this iteration
        densify_until_iter: Stop densification at this iteration
        densify_interval: Densify every N iterations
        start_checkpoint: Optional checkpoint to resume from
        save_iterations: Optional list of iterations to save point_cloud.ply
        checkpoint_iterations: Optional list of iterations to save checkpoints
        eval_split: Whether to hold out images for evaluation
    
    Returns:
        TrainResult with model directory, PLY path, and checkpoint path
    """
    os.makedirs(model_dir, exist_ok=True)

    # Windows path -> docker mount path works with Docker Desktop; we keep it absolute.
    docker = _which_docker()

    print(
        f"[3dgs] docker train: image={image} | iters={iterations} | res={resolution} | densify_from={densify_from_iter} | densify_until={densify_until_iter} | densify_interval={densify_interval} | eval={eval_split}",
        flush=True,
    )

    cmd = [
        docker,
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{_to_host_path(scene_dir)}:/scene",
        "-v",
        f"{_to_host_path(model_dir)}:/model",
        image,
        "--scene",
        "/scene",
        "--model",
        "/model",
        "--iterations",
        str(iterations),
        "--resolution",
        str(resolution),
        "--densify_from_iter",
        str(densify_from_iter),
        "--densify_until_iter",
        str(densify_until_iter),
        "--densify_interval",
        str(densify_interval),
    ]

    if eval_split:
        cmd += ["--eval"]

    if start_checkpoint:
        # start_checkpoint must be inside /model for the container
        ck = os.path.abspath(start_checkpoint)
        if not ck.startswith(os.path.abspath(model_dir)):
            raise ValueError("start_checkpoint must be inside model_dir for docker mounting")
        rel = os.path.relpath(ck, os.path.abspath(model_dir)).replace("\\", "/")
        cmd += ["--start_checkpoint", f"/model/{rel}"]

    if save_iterations:
        cmd += ["--save_iterations", ",".join(str(i) for i in save_iterations)]

    if checkpoint_iterations:
        cmd += ["--checkpoint_iterations", ",".join(str(i) for i in checkpoint_iterations)]

    try:
        print(f"[3dgs] docker cmd: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("Docker not found. Install Docker Desktop or run without docker training.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError("3DGS docker training failed (see logs above).") from e

    ply = _latest_pointcloud_ply(model_dir)

    # Optional checkpoint (graphdeco writes 'chkpnt<iter>.pth' into model dir)
    ckpt = None
    for name in os.listdir(model_dir):
        if name.startswith("chkpnt") and name.endswith(".pth"):
            ckpt = os.path.join(model_dir, name)
    print(f"[3dgs] done: ply={ply} | ckpt={ckpt}", flush=True)
    return TrainResult(model_dir=model_dir, gaussian_ply=ply, checkpoint_path=ckpt)


def train_3dgs_adaptive(
    *,
    image: str,
    scene_dir: str,
    model_dir: str,
    total_iterations: int,
    prune_interval: int,
    prune_opacity: float,
    resolution: int,
    densify_from_iter: int,
    densify_until_iter: int,
    densify_interval: int,
    eval_split: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> AdaptiveTrainResult:
    """
    Run adaptive 3DGS training with interleaved pruning.
    
    This is the Phase 8 implementation: instead of pruning only after training,
    we interleave pruning into training for better fidelity-per-parameter.
    
    Args:
        image: Docker image for 3DGS trainer
        scene_dir: COLMAP scene directory
        model_dir: Output model directory
        total_iterations: Total training iterations
        prune_interval: Prune every N iterations
        prune_opacity: Minimum opacity threshold for pruning
        resolution: Resolution downscale factor
        densify_from_iter: Start densification at this iteration
        densify_until_iter: Stop densification at this iteration
        densify_interval: Densify every N iterations
        eval_split: Whether to hold out images for evaluation
        progress_callback: Optional callback(phase, message)
    
    Returns:
        AdaptiveTrainResult with final model paths and stats
    """
    return train_adaptive(
        image=image,
        scene_dir=scene_dir,
        model_dir=model_dir,
        total_iterations=total_iterations,
        prune_interval=prune_interval,
        prune_opacity=prune_opacity,
        resolution=resolution,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_interval=densify_interval,
        eval_split=eval_split,
        progress_callback=progress_callback,
    )

