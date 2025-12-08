"""
Lightweight wrapper to run Fast3R for camera pose estimation.
Uses half-precision on GPU to reduce memory on 4GB cards.
If Fast3R is unavailable or fails, callers should fallback.
"""
import numpy as np
from typing import List, Tuple, Optional


class Fast3RRunner:
    def __init__(self, device: Optional[str] = None, max_images: int = 6, size: int = 512):
        self.device = device
        self.max_images = max_images
        self.size = size
        self._model = None
        self._lit = None
        self._torch = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from fast3r.models.fast3r import Fast3R
        from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
        model = model.to(device)
        model = model.half()  # save memory
        model.eval()

        self._model = model
        self._lit = MultiViewDUSt3RLitModule.load_for_inference(model)
        self._device = device

    def estimate_poses(self, image_paths: List[str]) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Returns (poses_c2w list, K intrinsics).
        Poses: list of 4x4 float32 camera-to-world matrices.
        """
        if len(image_paths) == 0:
            raise ValueError("No images provided to Fast3R.")

        self._lazy_load()
        torch = self._torch
        device = self._device

        # Limit number of images to fit memory
        if len(image_paths) > self.max_images:
            # even spread
            idxs = np.linspace(0, len(image_paths) - 1, self.max_images, dtype=int).tolist()
            image_paths = [image_paths[i] for i in idxs]

        from fast3r.dust3r.utils.image import load_images
        from fast3r.dust3r.inference_multiview import inference

        images = load_images(image_paths, size=self.size, verbose=False)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            output_dict, _ = inference(
                images,
                self._model,
                device,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                verbose=False,
                profiling=False,
            )

            poses_batch, focals = self._lit.estimate_camera_poses(
                output_dict["preds"],
                niter_PnP=50,
                focal_length_estimation_method="first_view_from_global_head",
            )

        poses_c2w = poses_batch[0]  # list/array of shape (N,4,4)
        poses = [p.astype(np.float32) for p in poses_c2w]

        # Build intrinsics from focal; assume square resize to self.size
        f = float(focals[0][0]) if isinstance(focals, (list, tuple)) else float(focals)
        cx = cy = self.size / 2.0
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float32)

        # Free memory
        if device == "cuda":
            torch.cuda.empty_cache()

        return poses, K

