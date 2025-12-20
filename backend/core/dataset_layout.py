"""
Job folder layout utilities.

Purpose
-------
Keep all pipelines writing to a consistent, inspectable directory structure so that:
- every stage is independently debuggable
- later modules (poses / gaussians / compression) can rely on stable paths
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class JobLayout:
    """
    Canonical on-disk layout for a processing job.

    root/
      frames/
      sfm/
      recon/
      gaussians/
      mesh/
      texture/
      manifest.json
    """

    root: str

    @property
    def frames_dir(self) -> str:
        return os.path.join(self.root, "frames")

    @property
    def sfm_dir(self) -> str:
        return os.path.join(self.root, "sfm")

    @property
    def recon_dir(self) -> str:
        return os.path.join(self.root, "recon")

    @property
    def gaussians_dir(self) -> str:
        return os.path.join(self.root, "gaussians")

    @property
    def mesh_dir(self) -> str:
        return os.path.join(self.root, "mesh")

    @property
    def texture_dir(self) -> str:
        return os.path.join(self.root, "texture")

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, "manifest.json")

    def ensure_dirs(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.sfm_dir, exist_ok=True)
        os.makedirs(self.recon_dir, exist_ok=True)
        os.makedirs(self.gaussians_dir, exist_ok=True)
        os.makedirs(self.mesh_dir, exist_ok=True)
        os.makedirs(self.texture_dir, exist_ok=True)


def write_manifest(
    *,
    layout: JobLayout,
    job_id: str,
    mode: str,
    input_path: str,
    frame_paths: List[str],
    config_dict: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write a minimal manifest describing this job for reproducibility/debugging.
    """
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "mode": mode,
        "input_path": input_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frames": [os.path.relpath(p, layout.root).replace("\\", "/") for p in frame_paths],
        "config": config_dict,
    }
    if extra:
        payload["extra"] = extra

    os.makedirs(layout.root, exist_ok=True)
    with open(layout.manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return layout.manifest_path

