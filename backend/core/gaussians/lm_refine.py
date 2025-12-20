"""
3DGS-LM optional refinement hook.

We don't bundle 3DGS-LM directly into the main trainer image to keep builds stable.
Instead, this provides a structured "hook" you can enable once you have an image
or local checkout available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LMRefineResult:
    refined_ply: str
    used: bool
    message: str


def maybe_refine_with_lm(*, model_dir: str, gaussian_ply: str) -> LMRefineResult:
    """
    Placeholder hook.

    Enable by setting:
      - GS_LM_ENABLE=1
      - GS_LM_COMMAND="<command template>"

    Where the command template can reference:
      {model_dir} and {gaussian_ply}

    Example:
      set GS_LM_ENABLE=1
      set GS_LM_COMMAND=docker run --rm --gpus all -v "{model_dir}:/model" my-3dgs-lm:latest --model_dir /model
    """
    if os.environ.get("GS_LM_ENABLE", "0") != "1":
        return LMRefineResult(refined_ply=gaussian_ply, used=False, message="LM refine disabled")

    cmd_tmpl = os.environ.get("GS_LM_COMMAND")
    if not cmd_tmpl:
        return LMRefineResult(refined_ply=gaussian_ply, used=False, message="GS_LM_COMMAND not set")

    # We keep this as a hook only. The actual integration depends on the chosen LM implementation.
    return LMRefineResult(
        refined_ply=gaussian_ply,
        used=False,
        message="LM refine hook configured but not executed (integration requires your LM runner command).",
    )

