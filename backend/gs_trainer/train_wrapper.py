"""
Container entrypoint for 3DGS training.

This wrapper exists so the backend can call a stable interface regardless of
upstream repo changes.

Supports:
- --checkpoint_iterations: Iterations at which to save checkpoints (comma-separated)
- --save_iterations: Iterations at which to save point_cloud.ply (comma-separated)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Mounted COLMAP scene dir (contains images/ and sparse/0)")
    ap.add_argument("--model", required=True, help="Mounted output model dir")
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--densify_from_iter", type=int, default=500)
    ap.add_argument("--densify_until_iter", type=int, default=1500)
    ap.add_argument("--densify_interval", type=int, default=100)
    ap.add_argument("--start_checkpoint", default=None)
    ap.add_argument(
        "--checkpoint_iterations",
        default=None,
        help="Comma-separated list of iterations to save checkpoints, e.g. '1000,2000,3000'"
    )
    ap.add_argument(
        "--save_iterations",
        default=None,
        help="Comma-separated list of iterations to save point_cloud.ply, e.g. '1000,2000,3000'"
    )
    args = ap.parse_args()

    os.makedirs(args.model, exist_ok=True)

    cmd = [
        sys.executable,
        "train.py",
        "-s",
        args.scene,
        "-m",
        args.model,
        "--iterations",
        str(args.iterations),
        "-r",
        str(args.resolution),
        "--densify_from_iter",
        str(args.densify_from_iter),
        "--densify_until_iter",
        str(args.densify_until_iter),
        "--densify_interval",
        str(args.densify_interval),
    ]

    if args.start_checkpoint:
        cmd += ["--start_checkpoint", args.start_checkpoint]

    # Pass checkpoint save iterations to upstream train.py
    if args.checkpoint_iterations:
        # Upstream expects space-separated list via --checkpoint_iterations
        iters_list = [s.strip() for s in args.checkpoint_iterations.split(",")]
        cmd += ["--checkpoint_iterations"] + iters_list

    # Pass save iterations for point_cloud.ply to upstream train.py
    if args.save_iterations:
        # Upstream expects space-separated list via --save_iterations
        iters_list = [s.strip() for s in args.save_iterations.split(",")]
        cmd += ["--save_iterations"] + iters_list

    print("[train_wrapper] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

