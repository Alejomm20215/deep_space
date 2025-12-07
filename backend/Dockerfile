# syntax=docker/dockerfile:1.4
# CUDA 11.7 base (matches driver 516.40)
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt .

# Install GPU-enabled torch (cu117) + runtime deps with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip && \
    pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install onnxruntime-gpu==1.16.3 && \
    pip3 install -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/

# Expose port (FastAPI default)
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "backend.app"]
