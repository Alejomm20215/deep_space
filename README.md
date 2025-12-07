## Deep Space 3D – CPU-First Reconstruction Pipeline

### What this solves
- **No GPU? Still usable.** Replaces heavy Fast3R/3DGS with a CPU-friendly pipeline.
- **Responsive API.** Long steps offloaded to threads; FastAPI/WebSockets keep updating progress.
- **Downloadable artifacts.** Backend now returns browser-ready URLs under `/outputs/{job_id}/...`.
- **Automatic depth model fetch.** Downloads a small ONNX depth model if missing; falls back to pseudo depth if offline.

### Pipeline (current CPU-first flow)
1. **Frame selection**: Keyframes extracted/resized (~640px) to stay quick.
2. **Depth estimation**: Pseudo-depth (lightweight); ONNX path removed to keep deps small.
3. **Fusion**: Depth -> sparse point cloud (lightweight, no Open3D).
4. **Meshing**: Stub mesh generation (Open3D-free).
5. **Texture (stub)**: Placeholder GLB to keep flow; ready for a real bake step.
6. **Export**: Copies GLB/SPLAT/PLY to `backend/outputs/{job_id}` and exposes URLs.

### Key files
- `backend/core/reconstruction.py` – pseudo-depth + point cloud (minimal deps).
- `backend/core/mesh_extraction.py` – stub mesh (no Open3D).
- `backend/core/gaussian_splatting.py` – lightweight passthrough splat.
- `backend/core/texture_baker.py` – stub GLB writer (placeholder).
- `backend/core/exporter.py` – emits browser URLs under `/outputs/{job_id}/...`.

### Environment & setup
1) Install deps:
```
pip install -r backend/requirements.txt
```
2) (Optional) Set depth model path/URL:
```
export DEPTH_ANYTHING_MODEL=models/depth_anything_v2_small.onnx
export DEPTH_ANYTHING_URL=https://huggingface.co/LiheYoung/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_small.onnx
```
3) Run backend:
```
python -m backend.app
```
The app serves `/api` and static outputs at `/outputs`.

### Frontend notes
- Use Vite proxy in dev; set `VITE_API_TARGET` / `VITE_WS_TARGET` in prod to point to the backend host.
- Download buttons expect `/outputs/{job_id}/...` URLs returned by the API.

### Next improvements (research-friendly)
- Swap stub texture bake with a real UV unwrap + multi-view projection.
- Optional GPU path: replace depth with a larger model or Fast3R when a CUDA box is available.
- Add retention/cleanup policy for outputs and temps.

