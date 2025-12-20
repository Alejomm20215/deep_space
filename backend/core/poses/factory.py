from __future__ import annotations

from backend.core.poses.base import PoseBackend, PoseBackendUnavailable
from backend.core.poses.colmap_backend import ColmapPoseBackend
from backend.core.poses.colmap_docker_backend import ColmapDockerPoseBackend
from backend.core.poses.opencv_backend import OpenCVPoseBackend


def create_pose_backend(name: str) -> PoseBackend:
    """
    Create requested backend, raising PoseBackendUnavailable if impossible.
    """
    name = (name or "").strip().lower()
    if name in ("opencv", "cv"):
        return OpenCVPoseBackend()
    if name in ("colmap",):
        return ColmapPoseBackend()
    if name in ("colmap_docker", "colmap-docker"):
        return ColmapDockerPoseBackend()
    raise PoseBackendUnavailable(f"Unknown pose backend: {name}")


def create_best_available_pose_backend(preferred: str) -> PoseBackend:
    """
    Try preferred backend first; fall back to OpenCV if unavailable.
    """
    try:
        return create_pose_backend(preferred)
    except PoseBackendUnavailable:
        # If user asked for colmap but it's not installed, try docker colmap as the next best.
        if (preferred or "").strip().lower() in ("colmap",):
            try:
                return ColmapDockerPoseBackend()
            except PoseBackendUnavailable:
                pass
        return OpenCVPoseBackend()

