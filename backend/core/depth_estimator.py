"""
Depth Anything V2 estimator using ONNX runtime.
Downloads the model automatically if missing.
Includes GPU status checking and depth refinement.
"""
import os
import urllib.request
import numpy as np
import cv2

from backend.core.depth_refiner import refine_depth

# Model config - Depth Anything V2 Small (official ONNX export from HuggingFace)
# Using FP16 version: 49.6 MB, good balance of speed and quality
MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx"
MODEL_PATH = os.environ.get("DEPTH_MODEL_PATH", "backend/models/depth_anything_v2_small_fp16.onnx")

# GPU usage guard: limit number of GPU inferences before forcing CPU
GPU_MAX_RUNS = int(os.environ.get("GPU_MAX_DEPTH_RUNS", "0"))  # 0 means no limit
_gpu_runs = 0

def check_gpu_status() -> dict:
    """
    Check GPU availability for ONNX Runtime and PyTorch.
    Returns status dict with details.
    """
    status = {
        "cuda_available": False,
        "onnx_gpu": False,
        "torch_gpu": False,
        "gpu_name": None,
        "gpu_memory": None,
        "details": []
    }

    # Check ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        status["onnx_providers"] = providers
        if "CUDAExecutionProvider" in providers:
            status["onnx_gpu"] = True
            status["details"].append("✅ ONNX Runtime CUDA available")
        else:
            status["details"].append("❌ ONNX Runtime CUDA NOT available")
    except Exception as e:
        status["details"].append(f"❌ ONNX Runtime error: {e}")

    # Check PyTorch
    try:
        import torch
        status["torch_gpu"] = torch.cuda.is_available()
        if status["torch_gpu"]:
            status["cuda_available"] = True
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            status["details"].append(f"✅ PyTorch CUDA: {status['gpu_name']} ({status['gpu_memory']})")
        else:
            status["details"].append("❌ PyTorch CUDA NOT available")
    except Exception as e:
        status["details"].append(f"❌ PyTorch error: {e}")

    # Overall status
    status["cuda_available"] = status["onnx_gpu"] or status["torch_gpu"]

    return status


class DepthEstimator:
    """
    Real depth estimation using Depth Anything V2 Small.
    Includes multi-stage refinement for sharp, accurate depth.
    """

    def __init__(self, use_refinement: bool = True):
        self.session = None
        self.input_size = 518  # Depth Anything V2 expects 518x518
        self.use_refinement = use_refinement
        self.gpu_status = None
        self._load_model()

    def _load_model(self):
        """Load ONNX model, downloading if needed."""
        try:
            import onnxruntime as ort

            if not os.path.exists(MODEL_PATH):
                print(f"📥 Downloading depth model to {MODEL_PATH}...")
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                print("✅ Download complete.")

            # Check GPU and prefer it if available
            self.gpu_status = check_gpu_status()
            
            # Enforce GPU run limit
            global _gpu_runs
            gpu_allowed = self.gpu_status["onnx_gpu"]
            if GPU_MAX_RUNS > 0 and _gpu_runs >= GPU_MAX_RUNS:
                gpu_allowed = False
                print(f"⚠️ GPU usage limit reached ({GPU_MAX_RUNS}); forcing CPU for depth.")

            if gpu_allowed:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("🚀 Using GPU for depth estimation")
                _gpu_runs += 1
            else:
                providers = ['CPUExecutionProvider']
                print("💻 Using CPU for depth estimation (GPU not available)")

            self.session = ort.InferenceSession(MODEL_PATH, providers=providers)
            active = self.session.get_providers()
            print(f"📊 Active providers: {active}")

        except Exception as e:
            print(f"⚠️ Could not load depth model: {e}")
            print("Falling back to edge-based depth estimation.")
            self.session = None

    def get_gpu_status(self) -> dict:
        """Return GPU status info."""
        if self.gpu_status is None:
            self.gpu_status = check_gpu_status()
        return self.gpu_status

    def estimate(self, image: np.ndarray, apply_refinement: bool = None) -> np.ndarray:
        """
        Estimate depth from BGR image.
        
        Args:
            image: BGR image (H, W, 3), uint8
            apply_refinement: Override instance setting for refinement
            
        Returns:
            Depth map (H, W), float32, 0-1 range (higher = farther)
        """
        # Get raw depth
        if self.session is not None:
            raw_depth = self._model_depth(image)
        else:
            raw_depth = self._fallback_depth(image)

        # Apply refinement if enabled
        do_refine = apply_refinement if apply_refinement is not None else self.use_refinement
        if do_refine:
            refined_depth = refine_depth(raw_depth, image)
            return refined_depth
        else:
            return raw_depth

    def _model_depth(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Anything V2 model for depth estimation."""
        h, w = image.shape[:2]

        # Preprocess: resize, normalize, transpose to NCHW
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        img_norm = img_resized.astype(np.float32) / 255.0

        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std

        # NCHW format
        img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: img_input})[0]

        # Output shape is (1, H, W) or (1, 1, H, W)
        depth = output.squeeze()

        # Depth Anything outputs relative depth (higher = farther)
        # Resize back to original size
        depth = cv2.resize(depth, (w, h))

        # Normalize to 0-1
        depth = depth - depth.min()
        if depth.max() > 1e-6:
            depth = depth / depth.max()

        return depth.astype(np.float32)

    def _fallback_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Fallback depth from edges + blur.
        Better than pure luminance but not great.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection gives object boundaries
        edges = cv2.Canny(gray, 50, 150)

        # Distance transform from edges gives rough depth
        dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

        # Combine with blurred luminance
        blur = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)

        # Blend: edges provide structure, blur provides smoothness
        depth = 0.7 * dist + 0.3 * blur

        # Normalize
        depth = depth - depth.min()
        if depth.max() > 1e-6:
            depth = depth / depth.max()

        return depth.astype(np.float32)


# Singleton instance
_estimator = None


def get_depth_estimator(use_refinement: bool = True) -> DepthEstimator:
    global _estimator
    if _estimator is None:
        _estimator = DepthEstimator(use_refinement=use_refinement)
    return _estimator
