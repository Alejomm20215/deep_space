# syntax=docker/dockerfile:1.4
#
# 3D Gaussian Splatting trainer image (CUDA 11.7).
#
# This image is intentionally isolated from the backend container. The backend
# calls it via `docker run ...` to avoid forcing C++/CUDA build toolchains into
# the API image.
#
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# During `docker build` there is no GPU access, so torch can't auto-detect
# GPU compute capability for compiling CUDA extensions. Provide a safe default
# that works for most RTX 20/30 class GPUs and includes PTX for forward-compat.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /opt/gs

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential \
    git cmake ninja-build \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Torch + friends (cu117)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip && \
    pip3 install "numpy<2.0" && \
    pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Clone the original 3DGS reference implementation.
RUN git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git /opt/gs/gaussian-splatting


WORKDIR /opt/gs/gaussian-splatting

# Init submodules (diff-gaussian-rasterization, simple-knn, etc.)
RUN git submodule update --init --recursive

# Install python deps + build CUDA extensions.
# Upstream repo layout changes occasionally; avoid hard-depending on a root requirements.txt.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
      "numpy<2.0" \
      plyfile tqdm Pillow scipy imageio opencv-python-headless matplotlib && \
    ( test -f requirements.txt && pip3 install -r requirements.txt || true ) && \
    pip3 install --no-build-isolation ./submodules/diff-gaussian-rasterization ./submodules/simple-knn

COPY backend/gs_trainer/train_wrapper.py /opt/gs/train_wrapper.py

ENTRYPOINT ["python3", "/opt/gs/train_wrapper.py"]

