"""
Pose backend abstraction.

We keep this intentionally lightweight: different machines can use different
pose estimation stacks (COLMAP / Fast3R / OpenCV), but downstream code only
needs consistent outputs (poses + intrinsics, aligned to input images).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class PoseResult:
    """
    Outputs are aligned to `image_paths` passed into `estimate()`.

    - poses_c2w: list of 4x4 camera-to-world matrices
    - K: 3x3 intrinsics matrix (shared intrinsics assumption for simplicity)
    - valid: whether the pose for that frame is estimated (vs identity fallback)
    """

    poses_c2w: List[np.ndarray]
    K: np.ndarray
    valid: List[bool]
    backend: str
    sfm_dir: Optional[str] = None


class PoseBackend(Protocol):
    name: str

    def estimate(self, image_paths: List[str], work_dir: str) -> PoseResult:  # pragma: no cover
        ...


class PoseBackendUnavailable(RuntimeError):
    """Raised when a backend is requested but not available on this machine."""

