"""
Depth Anything V2 estimator using ONNX runtime.
Downloads the model automatically if missing.
"""
import os
import urllib.request
import numpy as np
import cv2

# Model config
MODEL_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.onnx"
MODEL_PATH = os.environ.get("DEPTH_MODEL_PATH", "backend/models/depth_anything_v2_small.onnx")


class DepthEstimator:
    """
    Real depth estimation using Depth Anything V2.
    Falls back to edge-based depth if model unavailable.
    """

    def __init__(self):
        self.session = None
        self.input_size = 518  # Model expects 518x518
        self._load_model()

    def _load_model(self):
        """Load ONNX model, downloading if needed."""
        try:
            import onnxruntime as ort

            if not os.path.exists(MODEL_PATH):
                print(f"Downloading depth model to {MODEL_PATH}...")
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                print("Download complete.")

            # Prefer GPU if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(MODEL_PATH, providers=providers)
            active = self.session.get_providers()
            print(f"Depth model loaded. Providers: {active}")

        except Exception as e:
            print(f"Could not load depth model: {e}")
            print("Falling back to edge-based depth estimation.")
            self.session = None

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from BGR image.
        Returns normalized depth map (0-1, higher = farther).
        """
        if self.session is not None:
            return self._model_depth(image)
        else:
            return self._fallback_depth(image)

    def _model_depth(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Anything V2 model."""
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
        img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: img_input})[0]

        # Output shape is (1, H, W) or (1, 1, H, W)
        depth = output.squeeze()

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


def get_depth_estimator() -> DepthEstimator:
    global _estimator
    if _estimator is None:
        _estimator = DepthEstimator()
    return _estimator

