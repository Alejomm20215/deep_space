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

This section is intentionally organized as **term → meaning → math**.

### Pinhole camera model (projection)
**Meaning:** maps a 3D point \(X\) to a 2D pixel \(x\) given intrinsics \(K\) and pose \((R,t)\).

**Math:**
- Homogeneous form: \(x \sim K [R \mid t] X\)
- Inhomogeneous pixel coordinates (with \(X_c = R X + t\)):
  \[
  (u, v) = \left(f_x \frac{X_c}{Z_c} + c_x,\; f_y \frac{Y_c}{Z_c} + c_y\right)
  \]

### SfM (Structure-from-Motion) — COLMAP backend
**Meaning:** estimates camera poses + sparse 3D points from many overlapping images by matching features and enforcing geometry.

**Math (core objective via bundle adjustment):**
\[
\min_{\{R_i,t_i,X_j\}} \sum_{(i,j)\in\mathcal{O}} \left\lVert \pi\!\left(K_i(R_i X_j + t_i)\right) - x_{ij} \right\rVert_2^2
\]
Where \(\mathcal{O}\) is the set of observed 2D measurements \(x_{ij}\), and \(\pi(\cdot)\) projects to image coordinates.

### BA (Bundle Adjustment)
**Meaning:** the nonlinear least-squares refinement step inside SfM that “tightens” poses/points by reducing reprojection error.

**Math:** BA is typically solved with Gauss–Newton / Levenberg–Marquardt on the same objective above (often with robust losses in practice).

### Essential matrix \(E\) — OpenCV fallback pose backend
**Meaning:** estimates relative motion between two calibrated views using epipolar geometry (works even when full SfM fails).

**Math:**
- Epipolar constraint: \(x'^T E x = 0\)
- Relationship to motion: \(E = [t]_\times R\) where \([t]_\times\) is the skew-symmetric cross-product matrix.

### RANSAC
**Meaning:** robustly fits a model (e.g., \(E\)) when many matches are outliers by repeatedly sampling minimal sets and scoring consensus.

**Math (idea):** choose model parameters \(\theta\) maximizing inliers under an error threshold \(\tau\):
\[
\arg\max_\theta \sum_k \mathbf{1}\big(e_k(\theta) < \tau\big)
\]

### Monocular depth (relative depth) + post-processing
**Meaning:** the depth model outputs **relative** depth; we normalize and smooth it so the mesh preview doesn’t explode with spikes.

**Math (common steps):**
- Normalize: \(d \leftarrow \frac{d-\min(d)}{\max(d)-\min(d)}\)
- Quantile clipping: \(d \leftarrow \mathrm{clip}(d, q_{0.02}, q_{0.98})\) then renormalize

### Bilateral filter (edge-preserving smoothing)
**Meaning:** smooths noise while preserving edges by weighting neighbors by distance and intensity similarity.

**Math:**
\[
\hat{I}(p)=\frac{1}{W_p}\sum_{q\in\Omega} \exp\!\left(-\frac{\lVert p-q\rVert^2}{2\sigma_s^2}\right)\exp\!\left(-\frac{(I(p)-I(q))^2}{2\sigma_r^2}\right)I(q)
\]

### Delaunay triangulation (mesh connectivity)
**Meaning:** builds triangles from points such that triangles avoid skinny shapes (circumcircle property); used to connect the depth-lifted surface.

### QEM (Quadric Error Metrics) decimation
**Meaning:** reduces triangle count by collapsing edges while minimizing geometric error (keeps GLB size manageable).

**Math (core scoring idea):**
\[
E(v) = v^T Q v
\]
Where \(Q\) accumulates plane quadrics from incident faces; collapsing chooses \(v\) that minimizes \(E(v)\).

### GS / 3DGS (3D Gaussian Splatting) — trainer backend
**Meaning:** represents the scene as many **anisotropic 3D Gaussians** and optimizes them so renders match training images.

**Math (key pieces):**
- 3D Gaussian density:
  \[
  G(X)=\exp\!\left(-\tfrac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)\right)
  \]
- Front-to-back alpha compositing (ordered contributions \(k\)):
  \[
  C = \sum_k \left(\alpha_k \prod_{m<k}(1-\alpha_m)\right)c_k
  \]
- Simplified training objective:
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

