# Current State Report

Backend runs (FastAPI + WebSocket) with CPU/GPU depth estimation using Depth Anything v2 small ONNX, multi-stage depth refinement, 2.5D relief mesh via Delaunay, and texture baking (original colors, double-sided). Outputs served under /outputs/....

Pose estimation: Fast3R integration code is present but not usable on this machine because Fast3R requires PyTorch ≥2.2/CUDA ≥11.8 while your GPU driver (516.40) is CUDA 11.7. Fallback pose estimation uses OpenCV E-matrix; multi-view works but lower quality.

GPU usage cap: Depth estimator respects GPU_MAX_DEPTH_RUNS env; Fast3R runner limits images and uses half precision if ever enabled.

Video parsing: Not implemented yet (images only).

3D quality: Front-facing relief looks decent; back side invisible because pipeline is 2.5D (single-sided depth → no true backside geometry). Depth is inverted for pop-out effect, but no full volumetric reconstruction.

Build/deps: Dockerfile on CUDA 11.7.1 + cuDNN8; Fast3R repo cloned and installed editable; requirements pin numpy<2.0, torch 2.0.1, torchvision 0.15.2, onnxruntime-gpu 1.18.0, plus Fast3R deps. libjpeg-dev/libpng-dev added for torchvision. Docker Compose requests all GPUs.

-----

Known issues:

- Fast3R unusable until GPU driver is upgraded to a CUDA 11.8+/12.x compatible version.

- 3D remains relief-like; no full backside reconstruction.
Video-to-frames ingestion missing.

- Build times are long due to heavy deps; model downloads happen on first run.
