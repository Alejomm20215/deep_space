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

## Math / algorithms used (practical overview)

This is a “what’s under the hood” summary of the main mathematical pieces used in the pipeline.

### Camera + geometry basics
- **Pinhole camera model**:
  - Pixel projection (ignoring distortion): \(x \sim K [R \mid t] X\)
  - \(K\) is intrinsics, \(R,t\) are pose, \(X\) is 3D point.
  - In inhomogeneous image coordinates, with \(X_c = R X + t\): \((u, v) = \left(f_x \frac{X_c}{Z_c} + c_x,\; f_y \frac{Y_c}{Z_c} + c_y\right)\)

### Pose estimation (two backends)
- **COLMAP (SfM)** (when it succeeds):
  - Feature detection + matching (SIFT-style), geometric verification, and **incremental Structure-from-Motion**
  - Uses **bundle adjustment** (nonlinear least squares) to refine camera poses + 3D points by minimizing reprojection error:
    \[
      \min_{\{R_i,t_i,X_j\}} \sum_{(i,j)\in\mathcal{O}} \left\lVert \pi\!\left(K_i(R_i X_j + t_i)\right) - x_{ij} \right\rVert_2^2
    \]
- **OpenCV fallback** (when COLMAP fails):
  - Pairwise epipolar geometry via the **Essential matrix** \(E\) with **RANSAC** outlier rejection
  - Decomposes \(E\) to recover relative \(R,t\) up to scale (sufficient for our lightweight pipeline).
  - Epipolar constraint (normalized image points): \(x'^T E x = 0\), with \(E = [t]_\times R\)

### Depth processing (mesh preview path)
- **Monocular depth estimation** (ONNX model): produces **relative depth** (not metric).
- **Robust normalization + outlier clipping**:
  - Normalize depth to \([0,1]\), clip extremes using quantiles (e.g., 2%–98%) to prevent “spikes”.
- **Edge-preserving smoothing**:
  - Uses a **bilateral filter** (spatial + range weighting) to reduce noise while keeping edges sharper than a blur.
  - One common bilateral form:
    \[
      \hat{I}(p)=\frac{1}{W_p}\sum_{q\in\Omega} \exp\!\left(-\frac{\lVert p-q\rVert^2}{2\sigma_s^2}\right)\exp\!\left(-\frac{(I(p)-I(q))^2}{2\sigma_r^2}\right)I(q)
    \]

### Mesh construction + simplification
- **Depth-to-mesh lifting**:
  - Create a grid of vertices from pixels; convert depth \(z\) into a small relief displacement.
- **Triangulation**:
  - Uses **Delaunay triangulation** (via SciPy) for triangle connectivity in the generated surface.
- **Decimation (size control)**:
  - Uses **Quadric Error Metrics (QEM)** “quadratic decimation” (via `trimesh`) to reduce faces while preserving shape as much as possible.
  - QEM idea: each vertex has a quadric \(Q\) (sum of plane quadrics), and collapsing to \(v\) minimizes:
    \[
      E(v) = v^T Q v
    \]

### Gaussian splatting (true 3DGS path)
When enabled, the trainer runs a standard 3D Gaussian Splatting optimizer (graphdeco-style):
- Scene represented as many **anisotropic 3D Gaussians** (mean \(\mu\), covariance \(\Sigma\), opacity \(\alpha\), and color).
- Rendering is done by projecting Gaussians to screen space and **alpha-compositing** contributions along the ray.
- Optimization is gradient-based:
  - Minimizes an image reconstruction loss (e.g., photometric L2/SSIM-like components in the upstream repo).
- **Densification / pruning**:
  - Periodically splits/creates Gaussians in high-error regions and prunes low-contribution ones (opacity/size thresholds).

Useful core equations:
- **3D Gaussian density**:
  \[
    G(X)=\exp\!\left(-\tfrac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)
  \]
- **Alpha compositing** (front-to-back) for ordered contributions \(k\):
  \[
    C = \sum_k \left(\alpha_k \prod_{m<k}(1-\alpha_m)\right)c_k
  \]
- **Typical training objective** (simplified):
  \[
    \min_\theta \sum_{i\in \text{pixels}} \lVert \hat{C}_i(\theta) - C_i \rVert_2^2
  \]

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

