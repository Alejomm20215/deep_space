from __future__ import annotations

import os
from typing import List

import numpy as np

from backend.core.pose_estimator import PoseEstimator
from backend.core.poses.base import PoseBackend, PoseResult


class OpenCVPoseBackend(PoseBackend):
    name = "opencv"

    def __init__(self):
        self._estimator = PoseEstimator()

    def estimate(self, image_paths: List[str], work_dir: str) -> PoseResult:
        os.makedirs(work_dir, exist_ok=True)
        poses, K = self._estimator.estimate(image_paths)
        valid = [True] * len(poses)
        return PoseResult(
            poses_c2w=[p.astype(np.float32) for p in poses],
            K=K.astype(np.float32),
            valid=valid,
            backend=self.name,
            sfm_dir=work_dir,
        )

