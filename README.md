# Deep Space 3D – Reconstruction Service

FastAPI + React/Vite service that turns a short video or a small set of images into downloadable 3D artifacts (GLB, SPLAT, PLY). The pipeline is CPU-first but can opportunistically use GPU acceleration when available.

## Highlights
- CPU-friendly pipeline with automatic fallbacks (no GPU required; CUDA used if available by ONNX).
- Three quality presets (`fastest`, `balanced`, `quality`) tuned for speed vs. fidelity.
- Background processing with WebSocket progress events and REST polling.
- Browser-ready artifacts exposed at `/outputs/{job_id}/...` (served by FastAPI).
- Depth model auto-download (Depth Anything V2 ONNX) with an edge-based fallback when offline.
- Optional **true 3D Gaussian Splatting (3DGS)** training via a separate CUDA11.7 Docker trainer image (local-only).
- Optional progressive training schedule (DashGaussian-style) with checkpoint resume to reduce wall time.

## Architecture & Pipeline
- **Backend:** FastAPI (`backend/app.py`) orchestrates pipelines and exposes REST + WebSocket endpoints.
- **Frontend:** React/Vite app (`frontend/`) with proxying for `/api`, `/ws`, `/outputs`.
- **Pipeline stages:**
  1. Keyframe extraction from video or image folder (resized to preset resolutions).
  2. Depth estimation via Depth Anything V2 ONNX (GPU if available); falls back to edge-based depth.
  3. Sparse reconstruction with a turntable camera assumption → colored point cloud (`reconstruction.py`).
  4. Gaussian splatting:
     - **Fast fallback:** generate a renderable Gaussian PLY from point colors (no training)
     - **True 3DGS (optional):** run graphdeco-inria 3DGS in a dedicated Docker container (CUDA 11.7)
  5. Mesh extraction via SciPy Delaunay + filtering (`mesh_extraction.py`).
  6. Texture/GLB generation as a holographic-style mesh (`texture_baker.py`).
  7. Exporter copies artifacts to `backend/outputs/{job_id}` and returns HTTP URLs.
  8. Metrics (`metrics.json`) + optional gzip splat + optional research compression hooks.

## Quality Presets (from `backend/config.py`)
- **fastest** — 4 keyframes, skips 3DGS, quick Poisson-like mesh + simple texture. ~15–30s.
- **balanced** — 8 keyframes, light 3DGS passthrough, filtered mesh, blended texture. ~45–90s.
- **quality** — 16 keyframes, highest res/iterations, refined mesh, neural-blend texture. ~2–4m.

## API Quick Reference
- `POST /api/upload` — multipart `files` (one video _or_ multiple images), form `mode` = `fastest|balanced|quality` (default `balanced`). Returns `{ job_id }`.
- `GET /api/status/{job_id}` — current stage/progress.
- `GET /api/result/{job_id}` — artifact URLs after completion.
- `WS /ws/{job_id}` — push events: `init`, `progress` (`stage`, `progress`, `detail`), `complete`, `error`.
- Static files: `/outputs/{job_id}/model.glb`, `/model.splat`, `/pointcloud.ply`.

Example upload:
```
curl -X POST http://localhost:8000/api/upload ^
  -F "files=@sample.mp4" ^
  -F "mode=balanced"
```
Multiple images:
```
curl -X POST http://localhost:8000/api/upload ^
  -F "files=@img1.jpg" -F "files=@img2.jpg" -F "files=@img3.jpg" ^
  -F "mode=fastest"
```

## Running Locally (non-Docker)
### Backend
1) Python 3.10+ recommended.  
2) Install deps:
```
pip install -r backend/requirements.txt
```
   Optional GPU extras (matching Dockerfile):
```
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install onnxruntime-gpu==1.16.3
```
3) (Optional) override depth model path:
```
set DEPTH_MODEL_PATH=backend/models/depth_anything_v2_small.onnx
```
   If missing, the ONNX model is downloaded automatically; offline mode falls back to edge-based depth.  
4) Run:
```
python -m backend.app
```
   Serves REST/WebSocket at `http://localhost:8000` and static outputs at `/outputs`.

### Frontend
1) Node 18+ recommended.  
2) Install and run:
```
cd frontend
npm install
npm run dev
```
   - Dev proxy targets `http://localhost:8000` by default.  
   - For remote/prod, set `VITE_API_TARGET` and `VITE_WS_TARGET` to the backend host.

## Docker / Compose
- Build and run both services:
```
docker-compose up --build
```
- Ports: backend `8000`, frontend `5173`. Outputs persist in the `backend_outputs` volume.

## Outputs
- Files are written to `backend/outputs/{job_id}`:
  - `model.glb` — holographic GLB
  - `model.splat.ply` — Gaussian splat PLY (renderable)
  - `model.splat.ply.gz` — gzip-compressed splat (for storage/transfer)
  - `pointcloud.ply` — colored sparse cloud
  - `metrics.json` — counts/sizes/timing
- Returned URLs map to the same paths under `/outputs/{job_id}/...`.

## True 3DGS Training (local, Docker)
This project can run **real 3D Gaussian Splatting** training using a separate Docker image (so the API container stays small).

### 1) Install prerequisites
- **Docker Desktop** (with NVIDIA GPU support enabled)
- **COLMAP** available on PATH (for best results; otherwise pose estimation falls back to OpenCV)

### 2) Build the trainer image (once)
From repo root:
```
docker build -t deep_space_3dgs_trainer:cu117 -f backend/gs_trainer/Dockerfile .
```

### 3) Run a job (Balanced/Quality)
Balanced/Quality presets default to:
- `pose_backend="colmap"` (auto-fallback to OpenCV if COLMAP missing)
- `gaussian_backend="docker_3dgs"`
- `dashgaussian_schedule=True`

If Docker training fails (missing Docker / image / GPU), the pipeline automatically falls back to the lightweight Gaussian PLY generator.

### Optional knobs (env vars)
- `GS_3DGS_DOCKER_IMAGE`: override image name (default `deep_space_3dgs_trainer:cu117`)
- `GS_LM_ENABLE=1` and `GS_LM_COMMAND=...`: enable the 3DGS-LM hook (requires your own LM runner)
- `FCGS_ENABLE=1` + `FCGS_COMMAND="..."`: enable FCGS compression hook
- `HACPP_ENABLE=1` + `HACPP_COMMAND="..."`: enable HAC++ compression hook

## Repository Layout
- `backend/app.py` — FastAPI entrypoint, job orchestration, WebSockets.
- `backend/pipelines/` — quality-specific pipelines using shared core components.
- `backend/core/` — keyframes, depth, reconstruction, Gaussian splatting placeholder, meshing, texture/GLB baker, exporter.
- `frontend/` — React/Vite client with upload, progress, and viewer components.
- `docker-compose.yml` — dev orchestration; `backend/Dockerfile`, `frontend/Dockerfile`.

## Notes & Limitations
- Reconstruction assumes a turntable-style capture (camera orbit around subject); arbitrary trajectories may degrade quality.
- Gaussian splatting and texturing are simplified placeholders; GLB is holographic, not photorealistic.
- Depth model download requires internet on first run; offline runs will use the edge-based fallback.
- No automated test suite is included yet.
