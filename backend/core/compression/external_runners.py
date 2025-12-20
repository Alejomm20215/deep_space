"""
Hooks for research compression methods (FCGS, HAC++).

These methods require large research repos and (often) pretrained weights.
We provide a robust "runner" abstraction that can invoke:
- a local checkout (path specified by env var), or
- a docker image (name specified by env var).

This keeps the main application local-first and avoids forcing heavy deps.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExternalCompressionResult:
    method: str
    input_path: str
    output_path: str
    used: bool
    message: str


def _run_shell(command: str) -> None:
    subprocess.run(command, check=True, shell=True)


def maybe_run_fcgs(*, gaussian_ply: str, out_path: str) -> ExternalCompressionResult:
    """
    Hook for FCGS (ICLR 2025).

    Enable by setting:
      - FCGS_ENABLE=1
      - FCGS_COMMAND="... {input} ... {output} ..."
    """
    if os.environ.get("FCGS_ENABLE", "0") != "1":
        return ExternalCompressionResult("fcgs", gaussian_ply, out_path, False, "FCGS disabled")
    cmd = os.environ.get("FCGS_COMMAND")
    if not cmd:
        return ExternalCompressionResult("fcgs", gaussian_ply, out_path, False, "FCGS_COMMAND not set")

    cmd = cmd.format(input=gaussian_ply, output=out_path)
    try:
        _run_shell(cmd)
        return ExternalCompressionResult("fcgs", gaussian_ply, out_path, True, "FCGS completed")
    except Exception as e:
        return ExternalCompressionResult("fcgs", gaussian_ply, out_path, False, f"FCGS failed: {e}")


def maybe_run_hacpp(*, gaussian_ply: str, out_path: str) -> ExternalCompressionResult:
    """
    Hook for HAC++ (arXiv 2025).

    Enable by setting:
      - HACPP_ENABLE=1
      - HACPP_COMMAND="... {input} ... {output} ..."
    """
    if os.environ.get("HACPP_ENABLE", "0") != "1":
        return ExternalCompressionResult("hacpp", gaussian_ply, out_path, False, "HAC++ disabled")
    cmd = os.environ.get("HACPP_COMMAND")
    if not cmd:
        return ExternalCompressionResult("hacpp", gaussian_ply, out_path, False, "HACPP_COMMAND not set")

    cmd = cmd.format(input=gaussian_ply, output=out_path)
    try:
        _run_shell(cmd)
        return ExternalCompressionResult("hacpp", gaussian_ply, out_path, True, "HAC++ completed")
    except Exception as e:
        return ExternalCompressionResult("hacpp", gaussian_ply, out_path, False, f"HAC++ failed: {e}")

