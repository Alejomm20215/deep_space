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

## Method (pipeline, with the math where it belongs)

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

## 3DGS trainer (true Gaussian Splatting)

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

# Deep Space 3D (FastAPI + React)

Local 3D reconstruction from **a short video** or **a small set of images**, producing a **mesh preview (GLB)** and an optional **Gaussian Splat (PLY)** with robust fallbacks for constrained hardware.

---

## Abstract

This project implements a practical, end-to-end 3D reconstruction service designed for local execution. Given a casual capture (video or images), it estimates camera motion (COLMAP when viable, otherwise OpenCV), infers relative depth, and produces lightweight artifacts for inspection and downstream use. The system emphasizes **robust completion** over brittle “perfect capture” assumptions by combining conservative defaults, fallbacks, and size-control steps to keep output viewable and exportable.

---

## At a glance

- **Inputs**: one video (`.mp4/.mov/.avi/.mkv`) or multiple images (`.jpg/.jpeg/.png/.bmp`)
- **Outputs** (per job):
  - `model.glb` (mesh preview)
  - `model.splat.ply` (Gaussian splat, download-only)
  - `pointcloud.ply` (debug/inspection)
  - `metrics.json` (sizes + counts + timing)
- **Interfaces**: REST (`/api/*`), WebSocket progress (`/ws/{job_id}`), static outputs (`/outputs/{job_id}/*`)
- **Design goal**: finish reliably on local machines; degrade gracefully when COLMAP/GPU fails

---

## Key algorithms (term → role → math)

### Camera projection (pinhole model)
Role: maps a 3D point to a 2D measurement under intrinsics and pose.

$$
x \sim K [R \mid t] X
$$

$$
(u,v)=\left(f_x\frac{X_c}{Z_c}+c_x,\; f_y\frac{Y_c}{Z_c}+c_y\right),\quad X_c = RX+t
$$

### SfM + Bundle Adjustment (COLMAP)
Role: estimates camera poses and sparse structure by minimizing reprojection error over many views.

$$
\min_{\{R_i,t_i,X_j\}} \sum_{(i,j)\in\mathcal{O}} \left\lVert \pi\!\left(K_i(R_i X_j + t_i)\right) - x_{ij} \right\rVert_2^2
$$

### Essential matrix + RANSAC (OpenCV fallback)
Role: robust relative pose when full SfM fails or data is too weak.

- Epipolar constraint: $x'^T E x = 0$, with $E=[t]_\times R$
- RANSAC selection (conceptually):

$$
\arg\max_\theta \sum_k \mathbf{1}\!\left(e_k(\theta) < \tau\right)
$$

### Depth stabilization (preview path)
Role: convert relative monocular depth into a stable surface signal for preview meshing.

- Normalize: $d\leftarrow\frac{d-\min(d)}{\max(d)-\min(d)}$
- Clip extremes: $d\leftarrow\mathrm{clip}(d,q_{0.02},q_{0.98})$
- Bilateral smoothing:

$$
\hat{I}(p)=\frac{1}{W_p}\sum_{q\in\Omega}
e^{-\frac{\lVert p-q\rVert^2}{2\sigma_s^2}}
e^{-\frac{(I(p)-I(q))^2}{2\sigma_r^2}}
I(q)
$$

### Triangulation + decimation (mesh size control)
Role: connect the surface and reduce triangles to keep GLB size and viewer cost bounded.

- Connectivity: Delaunay-style triangulation (SciPy)
- QEM core score:

$$
E(v)=v^TQv
$$

### GS / 3DGS (Gaussian Splatting)
Role: represent the scene with anisotropic Gaussians and optimize them so renders match images.

$$
G(X)=\exp\!\left(-\tfrac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)
$$

$$
C=\sum_k\left(\alpha_k\prod_{m=1}^{k-1}(1-\alpha_m)\right)c_k
$$

---

## Artifacts and what they mean

- **`model.glb`**: a **browser-friendly preview mesh**. It is generated from **relative depth** and can look relief-like (front-facing geometry) on difficult scenes.
- **`model.splat.ply`**: the **Gaussian splat** artifact. This is typically the more faithful 3D representation, but **web preview is intentionally disabled** because splat renderers can spike browser RAM.

Outputs are served at:
- `http://localhost:8000/outputs/<job_id>/...`

---

## Method (pipeline)

The pipeline is staged to keep intermediate outputs usable:

1. **Keyframe selection**: extract a contiguous, overlapping set of frames from video (or resize the input image set).
2. **Pose estimation**:
   - primary: COLMAP (Docker) Structure-from-Motion
   - fallback: OpenCV essential-matrix poses
3. **Depth estimation**: monocular relative depth (ONNX) + stabilization (normalization, quantile clipping, bilateral smoothing).
4. **Mesh preview**:
   - lift pixels to a depth surface
   - triangulate
   - decimate to a target face budget
5. **Gaussian (optional)**:
   - lightweight fallback splat, or
   - true 3DGS training via a separate trainer container
6. **Export**: copy artifacts to `backend/outputs/{job_id}` and emit URLs.

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

## 3DGS trainer (true Gaussian Splatting)

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

# Deep Space 3D (FastAPI + React)

Local 3D reconstruction from **a short video** or **a small set of images**, producing a **mesh preview (GLB)** and an optional **Gaussian Splat (PLY)** with robust fallbacks for constrained hardware.

---

## Abstract

This project implements a practical, end-to-end 3D reconstruction service designed for local execution. Given a casual capture (video or images), it estimates camera motion (COLMAP when viable, otherwise OpenCV), infers relative depth, and produces lightweight artifacts for inspection and downstream use. The system emphasizes **robust completion** over brittle “perfect capture” assumptions by combining conservative defaults, fallbacks, and size-control steps to keep output viewable and exportable.

---

## At a glance

- **Inputs**: one video (`.mp4/.mov/.avi/.mkv`) or multiple images (`.jpg/.jpeg/.png/.bmp`)
- **Outputs** (per job):
  - `model.glb` (mesh preview)
  - `model.splat.ply` (Gaussian splat, download-only)
  - `pointcloud.ply` (debug/inspection)
  - `metrics.json` (sizes + counts + timing)
- **Interfaces**: REST (`/api/*`), WebSocket progress (`/ws/{job_id}`), static outputs (`/outputs/{job_id}/*`)
- **Design goal**: “finish reliably” on local machines; degrade gracefully when COLMAP/GPU fails

---

## Artifacts and what they mean (read this first)

- **`model.glb`**: a **browser-friendly preview mesh**. It is generated from **relative depth** and can look “relief-like” (front-facing geometry) on difficult scenes.
- **`model.splat.ply`**: the **Gaussian splat** artifact. This is typically the more faithful 3D representation, but **web preview is intentionally disabled** because splat renderers can spike browser RAM.

Outputs are served at:
- `http://localhost:8000/outputs/<job_id>/...`

---

## Method (pipeline)

The pipeline is staged to keep intermediate outputs usable:

1. **Keyframe selection**: extract a contiguous, overlapping set of frames from video (or resize the input image set).
2. **Pose estimation**:
   - primary: COLMAP (Docker) Structure-from-Motion
   - fallback: OpenCV essential-matrix poses
3. **Depth estimation**: monocular relative depth (ONNX) + stabilization (normalization, quantile clipping, bilateral smoothing).
4. **Mesh preview**:
   - lift pixels to a depth surface
   - triangulate
   - decimate to a target face budget
5. **Gaussian (optional)**:
   - lightweight fallback splat, or
   - true 3DGS training via a separate trainer container
6. **Export**: copy artifacts to `backend/outputs/{job_id}` and emit URLs.

---

## Key algorithms (term → role → math)

### Camera projection (pinhole model)
Role: maps a 3D point to a 2D measurement under intrinsics and pose.

$$
x \sim K [R \mid t] X
$$

$$
(u,v)=\left(f_x\frac{X_c}{Z_c}+c_x,\; f_y\frac{Y_c}{Z_c}+c_y\right),\quad X_c = RX+t
$$

### SfM + Bundle Adjustment (COLMAP)
Role: estimates camera poses and sparse structure by minimizing reprojection error over many views.

$$
\min_{\{R_i,t_i,X_j\}} \sum_{(i,j)\in\mathcal{O}} \left\lVert \pi\!\left(K_i(R_i X_j + t_i)\right) - x_{ij} \right\rVert_2^2
$$

### Essential matrix + RANSAC (OpenCV fallback)
Role: robust relative pose when full SfM fails or data is too weak.

- Epipolar constraint: $x'^T E x = 0$, with $E=[t]_\times R$
- RANSAC selection (conceptually):

$$
\arg\max_\theta \sum_k \mathbf{1}\!\left(e_k(\theta) < \tau\right)
$$

### Depth stabilization (preview path)
Role: convert relative monocular depth into a stable surface signal for preview meshing.

- Normalize: $d\leftarrow\frac{d-\min(d)}{\max(d)-\min(d)}$
- Clip extremes: $d\leftarrow\mathrm{clip}(d,q_{0.02},q_{0.98})$
- Bilateral smoothing:

$$
\hat{I}(p)=\frac{1}{W_p}\sum_{q\in\Omega}
e^{-\frac{\lVert p-q\rVert^2}{2\sigma_s^2}}
e^{-\frac{(I(p)-I(q))^2}{2\sigma_r^2}}
I(q)
$$

### Triangulation + decimation (mesh size control)
Role: connect the surface and reduce triangles to keep GLB size and viewer cost bounded.

- Connectivity: Delaunay-style triangulation (SciPy)
- QEM core score:

$$
E(v)=v^TQv
$$

### GS / 3DGS (Gaussian Splatting)
Role: represent the scene with anisotropic Gaussians and optimize them so renders match images.

$$
G(X)=\exp\!\left(-\tfrac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)
$$

$$
C=\sum_k\left(\alpha_k\prod_{m=1}^{k-1}(1-\alpha_m)\right)c_k
$$

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

## 3DGS trainer (true Gaussian Splatting)

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
Use `$$ ... $$` for block math. (This README does.) If your GitHub view still shows raw LaTeX, ensure you’re viewing the repo on GitHub (not a renderer that strips math).

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

## Acknowledgements (upstream ideas/tools)

This project builds on widely-used components and research directions including:
- COLMAP (SfM + BA)
- OpenCV (epipolar geometry + RANSAC)
- 3D Gaussian Splatting (graphdeco-style training/rendering)
- SciPy + trimesh (triangulation + decimation)

# Deep Space 3D (FastAPI + React) — Local 3D Reconstruction

Convert **a short video** or **a set of images** into downloadable 3D artifacts (mesh + Gaussian splat), entirely **on your machine**.

This repo is built for:
- **Local-only** workflows (no cloud)
- **Reasonable defaults** that keep running even when parts fail (COLMAP / GPU / 3DGS trainer)
- **Windows + Docker Desktop** friendliness

---

## What this app outputs (important)

- **`model.glb`**: a **mesh preview** intended to be browser-friendly.
  - This is currently derived from **depth** and can look like a “depth relief” (front-facing geometry) on hard scenes.
- **`model.splat.ply`**: the **Gaussian splat** output (download-only in the UI).
  - **Preview is intentionally removed** from the UI because splat renderers can explode browser RAM.
- **`pointcloud.ply`**: colored point cloud artifact (mostly for debugging / inspection)
- **`metrics.json`**: sizes + counts + timing

Outputs are served at:
- `http://localhost:8000/outputs/<job_id>/...`

### Algorithms behind these outputs (term → meaning → math)

**Pinhole camera model (projection)**  
Meaning: maps a 3D point \(X\) to a 2D pixel \(x\) given intrinsics \(K\) and pose \((R,t)\).

Math:

$$
x \sim K [R \mid t] X
$$

$$
(u,v)=\left(f_x\frac{X_c}{Z_c}+c_x,\; f_y\frac{Y_c}{Z_c}+c_y\right),\quad X_c = RX+t
$$

**SfM (Structure-from-Motion) / BA (Bundle Adjustment) — COLMAP backend**  
Meaning: estimates camera poses + sparse 3D points from multiple images; BA refines them by minimizing reprojection error.

Math:

$$
\min_{\{R_i,t_i,X_j\}} \sum_{(i,j)\in\mathcal{O}} \left\lVert \pi\!\left(K_i(R_i X_j + t_i)\right) - x_{ij} \right\rVert_2^2
$$

**Essential matrix \(E\) + RANSAC — OpenCV fallback poses**  
Meaning: robustly estimates relative motion from noisy matches when SfM fails.

Math: epipolar constraint \(x'^T E x = 0\), with \(E=[t]_\times R\). RANSAC picks \(\theta\) maximizing inliers:

$$
\arg\max_\theta \sum_k \mathbf{1}\!\left(e_k(\theta) < \tau\right)
$$

**Monocular depth + smoothing — mesh preview path (`model.glb`)**  
Meaning: depth is relative; we normalize + clip outliers + smooth to avoid spiky/warped meshes.

Math: normalize \(d\leftarrow\frac{d-\min(d)}{\max(d)-\min(d)}\), clip \(d\leftarrow\mathrm{clip}(d,q_{0.02},q_{0.98})\). One bilateral filter form:

$$
\hat{I}(p)=\frac{1}{W_p}\sum_{q\in\Omega}
e^{-\frac{\lVert p-q\rVert^2}{2\sigma_s^2}}
e^{-\frac{(I(p)-I(q))^2}{2\sigma_r^2}}
I(q)
$$

**Delaunay triangulation + QEM decimation — mesh construction**  
Meaning: triangulate the lifted surface; reduce triangles to keep the GLB small and the viewer fast.

Math (QEM core):

$$
E(v)=v^TQv
$$

**GS / 3DGS (3D Gaussian Splatting) — splat output (`model.splat.ply`)**  
Meaning: represents the scene as many anisotropic 3D Gaussians; optimizes them so rendered images match the input images.

Math (core pieces):

$$
G(X)=\exp\!\left(-\tfrac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)
$$

$$
C=\sum_k\left(\alpha_k\prod_{m=1}^{k-1}(1-\alpha_m)\right)c_k
$$

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
`docker-compose.yml` passes `HOST_PWD=${PWD}` so the backend (container) can correctly mount Windows paths when it launches sibling containers (COLMAP / 3DGS trainer).

If `${PWD}` doesn’t expand in your shell:

```powershell
$env:PWD = (Get-Location).Path
docker compose up -d --build
```

---

## Quickstart (Local dev without Docker)

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

## Using the API (PowerShell-friendly)

### Upload
`POST /api/upload` with multipart form fields:
- `files`: **one video** (`.mp4/.mov/.avi/.mkv`) OR **multiple images** (`.jpg/.jpeg/.png/.bmp`)
- `mode`: `fastest | balanced | quality` (default `balanced`)

Example (video):

```powershell
curl.exe -X POST http://localhost:8000/api/upload `
  -F "files=@sample.mp4" `
  -F "mode=balanced"
```

Example (images):

```powershell
curl.exe -X POST http://localhost:8000/api/upload `
  -F "files=@img1.jpg" -F "files=@img2.jpg" -F "files=@img3.jpg" -F "files=@img4.jpg" `
  -F "mode=fastest"
```

### Progress + results
- `WS /ws/{job_id}`: progress events (`init`, `progress`, `complete`, `error`)
- `GET /api/status/{job_id}`: current stage/progress
- `GET /api/result/{job_id}`: artifact URLs once done

---

## Quality modes (what they actually do)

Configured in `backend/config.py`:

- **`fastest`**
  - minimal frames
  - no 3DGS training
  - quickest mesh + texture
- **`balanced`**
  - more frames
  - attempts COLMAP (Docker) poses; falls back to OpenCV if COLMAP fails
  - attempts 3DGS training via trainer container (if enabled/available); otherwise falls back to lightweight splat
- **`quality`**
  - more frames + iterations
  - higher compute/time

Reality check: if your capture has low overlap, motion blur, glossy reflections, or low texture, COLMAP may fail and you’ll get OpenCV poses (lower quality). The pipeline still completes.

---

## Outputs (file list)

Outputs are written to:
- `backend/outputs/{job_id}/`

Common files:
- `model.glb`
- `model.splat.ply`
- `model.splat.ply.gz`
- `pointcloud.ply`
- `metrics.json`

---

## Pipeline overview (high level)

1. **Keyframe extraction** (`backend/core/keyframe_extractor.py`)
   - video → frames, or image directory → resized frames
2. **Pose estimation**
   - COLMAP (Docker) when possible
   - OpenCV fallback when COLMAP fails
3. **Depth estimation** (ONNX) + smoothing
4. **Mesh preview** generation + decimation
5. **Gaussian output**
   - lightweight fallback, or
   - trainer-container-based 3DGS
6. **Export** + `metrics.json` + optional gzip

---

## 3DGS trainer (true Gaussian Splatting)

The backend can run graphdeco-style 3D Gaussian Splatting inside a separate Docker image.

### Build trainer image (once)

```bash
docker build -t deep_space_3dgs_trainer:cu117 -f backend/gs_trainer/Dockerfile .
```

### Fast rebuild when only wrapper changed

```bash
docker build -t deep_space_3dgs_trainer:cu117 -f backend/gs_trainer/Dockerfile.wrapper_only .
```

---

## Troubleshooting (common failures)

### CUDA / driver mismatch errors (Windows)
Examples:
- `nvidia-container-cli: requirement error: unsatisfied condition: cuda>=...`
- `CUDA driver version is insufficient for CUDA runtime version`

These are **host driver ↔ container runtime** mismatches. Fix by updating the host driver/toolkit, or run CPU-only paths.

### COLMAP: “No good initial image pair found”
Means frames don’t have enough overlap/texture.

Fixes:
- move slower (more overlap)
- avoid motion blur
- use diffuse lighting; avoid reflections
- use more frames (balanced/quality)

### Backend rebuild takes long
The backend image includes heavyweight deps (PyTorch, ONNXRuntime GPU).

Tips:
- if you only changed Python code under `backend/`, you often only need:

```bash
docker compose restart backend
```

Rebuild only when changing `backend/Dockerfile` or `backend/requirements.txt`.

---

## Repo layout
- `backend/app.py`: FastAPI server + job orchestration
- `backend/pipelines/`: fastest/balanced/quality pipelines
- `backend/core/`: keyframes, poses, depth, reconstruction, meshing, gaussian, exporter
- `frontend/`: React/Vite UI
- `docker-compose.yml`: dev orchestration

w
