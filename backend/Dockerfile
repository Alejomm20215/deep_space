# syntax=docker/dockerfile:1.4
# CUDA 11.7 + cuDNN 8 (required for ONNX GPU, matches driver 516.40)
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# System deps (including libjpeg/libpng for torchvision)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv build-essential git \
    libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt .

# Install GPU-enabled torch + torchvision (cu117) + runtime deps with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip && \
    pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install onnxruntime-gpu==1.16.3 && \
    pip3 install -r requirements.txt

# ALL Fast3R dependencies (from their requirements.txt)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
    lightning torchmetrics torchinfo \
    hydra-core hydra-colorlog omegaconf \
    rootutils rich \
    einops roma \
    matplotlib seaborn plotly \
    scikit-image scikit-learn \
    "huggingface-hub[torch]>=0.22"

# Clone Fast3R and install in editable mode (keeps source for subpackages)
RUN git clone --depth 1 https://github.com/facebookresearch/fast3r.git /opt/fast3r && \
    cd /opt/fast3r && \
    pip3 install -e . --no-deps

# Copy the backend code
COPY backend/ ./backend/

# Expose port (FastAPI default)
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "backend.app"]
