"""
Multi-Stage Depth Refinement Pipeline.

Implements mathematically rigorous depth enhancement:
1. Edge-Aware Bilateral Filter (color-guided)
2. Gradient-Domain Sharpening
3. Normal-Based Consistency

All operations are vectorized numpy for CPU efficiency.
"""
import numpy as np
import cv2
from scipy.ndimage import convolve, uniform_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg


class DepthRefiner:
    """
    Refines raw depth maps using multi-stage mathematical processing.
    """

    def __init__(self, config=None):
        self.config = config or {}
        # Stage 1 params
        self.sigma_spatial = self.config.get('sigma_spatial', 9)
        self.sigma_color = self.config.get('sigma_color', 0.1)
        # Stage 2 params
        self.sharpen_lambda = self.config.get('sharpen_lambda', 0.3)
        self.sharpen_k = self.config.get('sharpen_k', 5.0)
        # Stage 3 params
        self.normal_sigma = self.config.get('normal_sigma', 3)

    def refine(self, depth: np.ndarray, color_image: np.ndarray) -> np.ndarray:
        """
        Full refinement pipeline.
        
        Args:
            depth: Raw depth map (H, W), float32, 0-1 range
            color_image: BGR image (H, W, 3), uint8
            
        Returns:
            Refined depth map (H, W), float32
        """
        # Ensure correct formats
        depth = depth.astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / depth.max()

        # Convert color to LAB for better perceptual distance
        color_lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

        # Stage 1: Edge-aware bilateral filter
        d1 = self._bilateral_filter(depth, color_lab)

        # Stage 2: Gradient-domain sharpening
        d2 = self._gradient_sharpen(d1, color_lab)

        # Stage 3: Normal-based refinement
        d3 = self._normal_refine(d2)

        return d3

    def _bilateral_filter(self, depth: np.ndarray, color: np.ndarray) -> np.ndarray:
        """
        Stage 1: Joint/Cross Bilateral Filter.
        
        D₁(p) = Σ Gₛ(||p-q||) · Gᵣ(||I(p)-I(q)||) · D(q) / W
        
        Uses separable approximation for O(n) complexity.
        """
        h, w = depth.shape
        sigma_s = self.sigma_spatial
        sigma_r = self.sigma_color

        # Kernel size (3 sigma rule)
        ksize = int(sigma_s * 3) | 1  # Ensure odd

        # Use OpenCV's joint bilateral filter if available
        # Otherwise fall back to our implementation
        try:
            # OpenCV bilateral with color guidance
            guide = (color * 255).astype(np.uint8)
            depth_uint16 = (depth * 65535).astype(np.uint16)
            
            # Joint bilateral filter
            filtered = cv2.ximgproc.jointBilateralFilter(
                guide, depth_uint16, ksize, sigma_r * 65535, sigma_s
            )
            return filtered.astype(np.float32) / 65535.0
        except:
            # Fallback: standard bilateral (less accurate but faster)
            depth_uint8 = (depth * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(depth_uint8, ksize, sigma_r * 255, sigma_s)
            return filtered.astype(np.float32) / 255.0

    def _gradient_sharpen(self, depth: np.ndarray, color: np.ndarray) -> np.ndarray:
        """
        Stage 2: Gradient-Domain Sharpening.
        
        Amplifies depth gradients at edges:
        ∇D_enhanced = ∇D + λ · tanh(k·∇D)
        
        Then reconstructs via Poisson solve.
        """
        # Compute gradients
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

        # Also compute color gradients for edge guidance
        color_gray = cv2.cvtColor((color * 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        color_gray = cv2.cvtColor(color_gray, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        edge_weight = cv2.Sobel(color_gray, cv2.CV_32F, 1, 0, ksize=3)**2 + \
                      cv2.Sobel(color_gray, cv2.CV_32F, 0, 1, ksize=3)**2
        edge_weight = np.sqrt(edge_weight)
        edge_weight = edge_weight / (edge_weight.max() + 1e-8)

        # Amplify gradients using tanh (soft clamp)
        lam = self.sharpen_lambda
        k = self.sharpen_k

        # Enhanced gradients (more amplification at color edges)
        boost = 1.0 + edge_weight * 2.0  # More sharpening at edges
        grad_x_enhanced = grad_x + lam * np.tanh(k * grad_x) * boost
        grad_y_enhanced = grad_y + lam * np.tanh(k * grad_y) * boost

        # Poisson reconstruction from gradients
        # ∇²D = ∂/∂x(grad_x) + ∂/∂y(grad_y)
        laplacian = cv2.Sobel(grad_x_enhanced, cv2.CV_32F, 1, 0, ksize=1) + \
                    cv2.Sobel(grad_y_enhanced, cv2.CV_32F, 0, 1, ksize=1)

        # Simple iterative Poisson solve (Gauss-Seidel)
        result = self._poisson_solve_fast(laplacian, depth)

        return result

    def _normal_refine(self, depth: np.ndarray) -> np.ndarray:
        """
        Stage 3: Normal-Based Refinement.
        
        1. Compute surface normals from depth gradient
        2. Filter normals (bilateral smoothing)
        3. Re-integrate normals via Poisson equation
        
        n = normalize([-∂D/∂x, -∂D/∂y, 1])
        ∇²D = ∇·n
        """
        # Compute gradients
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

        # Compute normals: n = normalize([-dD/dx, -dD/dy, 1])
        nx = -grad_x
        ny = -grad_y
        nz = np.ones_like(depth)

        # Normalize
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        # Filter normals (Gaussian smooth - could use bilateral for better edges)
        sigma = self.normal_sigma
        nx = cv2.GaussianBlur(nx, (0, 0), sigma)
        ny = cv2.GaussianBlur(ny, (0, 0), sigma)
        nz = cv2.GaussianBlur(nz, (0, 0), sigma)

        # Re-normalize after filtering
        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx, ny, nz = nx / norm, ny / norm, nz / norm

        # Compute divergence of normal field
        # ∇·n = ∂nx/∂x + ∂ny/∂y
        div_x = cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=1)
        div_y = cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=1)
        divergence = div_x + div_y

        # Poisson solve: ∇²D = ∇·n
        result = self._poisson_solve_fast(divergence, depth)

        return result

    def _poisson_solve_fast(self, laplacian: np.ndarray, initial: np.ndarray, 
                            iterations: int = 50) -> np.ndarray:
        """
        Fast Poisson solve using Gauss-Seidel iteration.
        
        Solves ∇²D = f with Dirichlet boundary (edges fixed to initial).
        """
        h, w = laplacian.shape
        result = initial.copy()

        # Gauss-Seidel iteration
        for _ in range(iterations):
            # Interior update: D[i,j] = (D[i-1,j] + D[i+1,j] + D[i,j-1] + D[i,j+1] - f[i,j]) / 4
            result[1:-1, 1:-1] = (
                result[0:-2, 1:-1] +  # up
                result[2:, 1:-1] +    # down
                result[1:-1, 0:-2] +  # left
                result[1:-1, 2:] -    # right
                laplacian[1:-1, 1:-1]
            ) / 4.0

        # Normalize to original range
        result = result - result.min()
        if result.max() > 1e-6:
            result = result / result.max()

        # Blend with original to preserve overall structure
        alpha = 0.7
        result = alpha * result + (1 - alpha) * initial

        return result


def refine_depth(depth: np.ndarray, color_image: np.ndarray, config: dict = None) -> np.ndarray:
    """
    Convenience function to refine a depth map.
    
    Args:
        depth: Raw depth (H, W), float, any range
        color_image: BGR image (H, W, 3), uint8
        config: Optional parameters dict
        
    Returns:
        Refined depth (H, W), float32, 0-1 range
    """
    refiner = DepthRefiner(config)
    return refiner.refine(depth, color_image)

