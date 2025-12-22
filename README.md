# Deep Space

Local 3D reconstruction from **video** or **a set of images**, producing a **mesh preview (GLB)** or a **Gaussian Splat (PLY)**.

---

## Abstract

This project implements a practical, end-to-end 3D reconstruction service designed for local execution. Given a casual capture (video or images), it estimates camera motion (COLMAP when viable, otherwise OpenCV), infers relative depth, and produces lightweight artifacts for inspection and downstream use. The system emphasizes **robust completion** over brittle “perfect capture” assumptions by combining conservative defaults, fallbacks, and size-control steps to keep output viewable and exportable.

---

## System overview

- **Inputs**: one video (`.mp4/.mov/.avi/.mkv`) or multiple images (`.jpg/.jpeg/.png/.bmp`)
- **Interfaces**: REST (`/api/*`), WebSocket progress (`/ws/{job_id}`), static outputs (`/outputs/{job_id}/*`)
- **Design goal**: finish reliably on local machines; degrade gracefully when COLMAP/GPU fails

Outputs are written per job under `backend/outputs/{job_id}/` and served at:
- `http://localhost:8000/outputs/<job_id>/...`

---

## Artifacts (what you get)

- **`model.glb`**: a **browser-friendly preview mesh** generated from **relative depth**. It can look relief-like (front-facing geometry) on difficult scenes.
- **`model.splat.ply`**: the **Gaussian splat** artifact. Typically more faithful 3D, but **web preview is intentionally disabled** because splat renderers can spike browser RAM.
- **`pointcloud.ply`**: point cloud (debug/inspection).
- **`metrics.json`**: run metadata (sizes + counts + elapsed).

When the pipeline produces a Gaussian Splat, the rendered color is formed by front-to-back compositing of ordered contributions \(k\):

$$
C=\sum_k\left(\alpha_k\prod_{m=1}^{k-1}(1-\alpha_m)\right)c_k
$$

---

## Method (pipeline)

### 1) Keyframe selection
Extract a contiguous, overlapping set of frames from video (or resize the input image set). Overlap matters because downstream geometry estimation relies on consistent feature correspondences.

### 2) Camera model (projection)
All geometric estimation assumes a pinhole-style projection:

$$
x \sim K [R \mid t] X
$$

$$
(u,v)=\left(f_x\frac{X_c}{Z_c}+c_x,\; f_y\frac{Y_c}{Z_c}+c_y\right),\quad X_c = RX+t
$$

### 3) Pose estimation (COLMAP SfM → OpenCV fallback)

**Primary (COLMAP / SfM + BA)**  
Structure-from-Motion recovers poses and sparse structure; bundle adjustment refines them by minimizing reprojection error:

$$
\min_{\{R_i,t_i,X_j\}} \sum_{(i,j)\in\mathcal{O}} \left\lVert \pi\!\left(K_i(R_i X_j + t_i)\right) - x_{ij} \right\rVert_2^2
$$

**Fallback (OpenCV / Essential matrix + RANSAC)**  
If SfM fails, a robust pairwise estimate is used:

- Epipolar constraint: \(x'^T E x = 0\), with \(E=[t]_\times R\)
- RANSAC selection (conceptually):

$$
\arg\max_\theta \sum_k \mathbf{1}\!\left(e_k(\theta) < \tau\right)
$$

### 4) Depth estimation + stabilization (preview path)
The depth model outputs **relative** depth. For a stable preview surface we normalize, clip extremes, and apply edge-preserving smoothing:

- Normalize:
  \(d\leftarrow\frac{d-\min(d)}{\max(d)-\min(d)}\)
- Clip extremes:
  \(d\leftarrow\mathrm{clip}(d,q_{0.02},q_{0.98})\)
- Bilateral smoothing:

$$
\hat{I}(p)=\frac{1}{W_p}\sum_{q\in\Omega}
e^{-\frac{\lVert p-q\rVert^2}{2\sigma_s^2}}
e^{-\frac{(I(p)-I(q))^2}{2\sigma_r^2}}
I(q)
$$

### 5) Mesh preview (triangulation + decimation)
The preview mesh is built by lifting pixels into a surface and connecting them (Delaunay-style triangulation via SciPy), then reducing triangle count to keep GLB size and viewer cost bounded.

Quadric Error Metrics (QEM) decimation uses a quadratic form to score collapses:

$$
E(v)=v^TQv
$$

### 6) Gaussian Splatting (optional)
If enabled, the trainer path runs graphdeco-style 3DGS. Each element is an anisotropic 3D Gaussian with density:

$$
G(X)=\exp\!\left(-\tfrac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)
$$

Rendering uses projected splats and alpha compositing (see artifact section for the compositing equation).

### 7) Export
Artifacts are copied to `backend/outputs/{job_id}` and URLs are returned by the API.

---

## Quickstart (Docker Compose — recommended)

### Prerequisites
- **Docker Desktop**
- Optional: **NVIDIA GPU + NVIDIA Container Toolkit** (for GPU acceleration where available)

### Run
From repo root:

```bash
docker compose up -d --build
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

### Windows note: HOST_PWD
`docker-compose.yml` passes `HOST_PWD=${PWD}` so the backend (container) can mount Windows host paths when launching sibling containers (COLMAP / 3DGS trainer).

If `${PWD}` doesn’t expand in your shell:

```powershell
$env:PWD = (Get-Location).Path
docker compose up -d --build
```

---

## Quickstart (local dev without Docker)

### Backend
- Python **3.10+**

```bash
pip install -r backend/requirements.txt
python -m backend.app
```

### Frontend
- Node **18+**

```bash
cd frontend
npm install
npm run dev
```

---

## API (PowerShell-friendly)

### Upload
`POST /api/upload` (multipart):
- `files`: one video OR multiple images
- `mode`: `fastest | balanced | quality` (default `balanced`)

Video:

```powershell
curl.exe -X POST http://localhost:8000/api/upload `
  -F "files=@sample.mp4" `
  -F "mode=balanced"
```

Images:

```powershell
curl.exe -X POST http://localhost:8000/api/upload `
  -F "files=@img1.jpg" -F "files=@img2.jpg" -F "files=@img3.jpg" -F "files=@img4.jpg" `
  -F "mode=fastest"
```

### Track progress + fetch results
- `WS /ws/{job_id}`: `init`, `progress`, `complete`, `error`
- `GET /api/status/{job_id}`
- `GET /api/result/{job_id}`

---

## Presets (modes)

Defined in `backend/config.py`:

- **`fastest`**: minimal frames, minimal compute, fast preview.
- **`balanced`**: best default; attempts COLMAP + (optional) 3DGS, with fallbacks.
- **`quality`**: more frames/iterations; higher compute/time.

Note: weak overlap/texture, motion blur, or reflections can break SfM. The system will fall back to OpenCV and still finish.

---

## Outputs (per job)

Artifacts are written to `backend/outputs/{job_id}/`:

| File | What it is | Typical use |
|---|---|---|
| `model.glb` | mesh preview | quick inspection / download |
| `model.splat.ply` | Gaussian splat | higher-fidelity 3D representation |
| `model.splat.ply.gz` | compressed splat | storage/transfer |
| `pointcloud.ply` | point cloud | debugging / inspection |
| `metrics.json` | run metadata | sizes, counts, elapsed |

---

## 3DGS trainer (Gaussian Splatting)

The backend can run graphdeco-style 3DGS inside a separate Docker image.

Build (once):

```bash
docker build -t deep_space_3dgs_trainer:cu117 -f backend/gs_trainer/Dockerfile .
```

Fast rebuild when only the wrapper changed:

```bash
docker build -t deep_space_3dgs_trainer:cu117 -f backend/gs_trainer/Dockerfile.wrapper_only .
```

---

## Troubleshooting

### GitHub math rendering looks wrong
This README uses `$$ ... $$` for block math. If your GitHub view still shows raw LaTeX, ensure you’re viewing on GitHub (some renderers strip math).

### CUDA / driver mismatch (Windows)
Symptoms:
- `nvidia-container-cli: requirement error: unsatisfied condition: cuda>=...`
- `CUDA driver version is insufficient for CUDA runtime version`

Fix: host driver/toolkit must match container requirements. Otherwise run CPU-only paths and/or avoid GPU-enabled docker containers.

### COLMAP: “No good initial image pair found”
Fixes:
- move slower (more overlap)
- avoid blur
- use diffuse lighting; avoid reflections
- increase frames (balanced/quality)

### Backend rebuild takes long
If you only changed Python code under `backend/`, you often just need:

```bash
docker compose restart backend
```

Rebuild only when `backend/Dockerfile` or `backend/requirements.txt` changes.

---

## Repository layout

- `backend/app.py`: FastAPI server + job orchestration
- `backend/pipelines/`: fastest/balanced/quality pipelines
- `backend/core/`: keyframes, poses, depth, reconstruction, meshing, gaussian, exporter
- `frontend/`: React/Vite UI
- `docker-compose.yml`: dev orchestration

---

## Acknowledgements

This project builds on widely-used components and research directions including:
- COLMAP (SfM + BA)
- OpenCV (epipolar geometry + RANSAC)
- 3D Gaussian Splatting (graphdeco-style training/rendering)
- SciPy + trimesh (triangulation + decimation)
