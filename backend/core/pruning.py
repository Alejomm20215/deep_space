"""
Safe, local pruning utilities for Gaussian PLYs.

This is not Speedy-Splat or PUP (those are research methods with specific scoring),
but it provides a practical baseline: prune by opacity and/or overly tiny scales.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PruneResult:
    input_path: str
    output_path: str
    kept: int
    removed: int
    message: str


def prune_gaussian_ply_ascii(
    *,
    input_path: str,
    output_path: str,
    min_opacity: float = 0.05,
    min_scale: Optional[float] = None,
) -> PruneResult:
    """
    Prune an ASCII Gaussian PLY by:
    - removing rows with opacity < min_opacity
    - optionally removing rows with any scale component < min_scale

    Assumes headers include 'opacity' and 'scale_0/1/2' (as produced by 3DGS).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse header, find vertex count + property order
    header_end = None
    vertex_count = None
    props = []
    in_vertex = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("element vertex"):
            vertex_count = int(s.split()[-1])
            in_vertex = True
        elif s.startswith("element ") and not s.startswith("element vertex"):
            in_vertex = False
        elif in_vertex and s.startswith("property"):
            parts = s.split()
            if len(parts) >= 3:
                props.append(parts[-1])
        elif s == "end_header":
            header_end = i + 1
            break

    if header_end is None or vertex_count is None:
        raise RuntimeError("Invalid PLY header (missing end_header or element vertex)")

    def idx(name: str) -> int:
        try:
            return props.index(name)
        except ValueError as e:
            raise RuntimeError(f"PLY missing required property: {name}") from e

    i_op = idx("opacity")
    i_s0 = idx("scale_0")
    i_s1 = idx("scale_1")
    i_s2 = idx("scale_2")

    kept_rows = []
    removed = 0

    for vi in range(vertex_count):
        row = lines[header_end + vi].strip().split()
        if not row:
            removed += 1
            continue
        op = float(row[i_op])
        if op < min_opacity:
            removed += 1
            continue
        if min_scale is not None:
            s0 = float(row[i_s0])
            s1 = float(row[i_s1])
            s2 = float(row[i_s2])
            if min(s0, s1, s2) < float(min_scale):
                removed += 1
                continue
        kept_rows.append(" ".join(row))

    kept = len(kept_rows)

    # Rewrite header with updated vertex count
    out_lines = []
    for i, line in enumerate(lines[:header_end]):
        if line.strip().startswith("element vertex"):
            out_lines.append(f"element vertex {kept}\n")
        else:
            out_lines.append(line)
    out_lines.extend([r + "\n" for r in kept_rows])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return PruneResult(
        input_path=input_path,
        output_path=output_path,
        kept=kept,
        removed=removed,
        message=f"Pruned {removed} / {vertex_count} gaussians (kept {kept})",
    )


