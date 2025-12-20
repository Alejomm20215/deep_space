from enum import Enum
from typing import Tuple, Dict, Any, Optional
from pydantic import BaseModel

class QualityMode(str, Enum):
    FASTEST = "fastest"
    BALANCED = "balanced"
    QUALITY = "quality"

class PipelineConfig(BaseModel):
    name: str
    description: str
    estimated_time: str
    
    # Keyframe extraction
    max_keyframes: int
    keyframe_resolution: Tuple[int, int]
    
    # Reconstruction
    # "fast3r" is kept for backwards-compat, but current implementation uses
    # depth estimation + pose backend (COLMAP/OpenCV).
    reconstruction_model: str = "depth_pose"
    pose_backend: str = "opencv"  # "colmap" (preferred if installed) or "opencv"
    confidence_threshold: float
    
    # 3D Gaussian Splatting
    use_gaussians: bool
    gaussian_iterations: int
    gaussian_densify_interval: Optional[int] = None
    gaussian_densify_until: Optional[int] = None
    gaussian_backend: str = "lightweight"  # "lightweight" or "docker_3dgs"
    gaussian_resolution: int = 2  # graphdeco resolution downscale factor (1,2,4,8)
    gaussian_densify_from: int = 500
    dashgaussian_schedule: bool = False
    
    # Mesh extraction
    mesh_method: str  # "poisson", "sugar", "sugar_refined"
    mesh_smoothing: bool = False
    mesh_depth: int = 8
    target_triangles: int
    
    # Texture
    texture_resolution: int
    texture_method: str # "simple_projection", "weighted_blend", "neural_blend"
    generate_normal_map: bool = False
    
    # Export
    export_splat: bool = True
    export_glb: bool = True
    draco_compression: bool = True
    export_metrics: bool = True
    export_splat_gzip: bool = True
    export_fcgs: bool = False
    export_hacpp: bool = False

    # Pruning (safe baseline)
    prune_enabled: bool = True
    prune_min_opacity: float = 0.02

    # Symmetry (SymGS inspired)
    symmetry_enabled: bool = False
    symmetry_axis: str = "x"
    symmetry_tolerance: float = 0.05

    # Adaptive Training & Pruning Loop (Phase 8)
    # Instead of post-training pruning only, interleave pruning into training
    # for better fidelity-per-parameter.
    adaptive_prune_enabled: bool = False
    adaptive_prune_interval: int = 1000  # Prune every N iterations
    adaptive_prune_opacity: float = 0.01  # More aggressive during training


QUALITY_PRESETS: Dict[QualityMode, PipelineConfig] = {
    QualityMode.FASTEST: PipelineConfig(
        name="⚡ Fastest",
        description="Quick preview, some artifacts okay",
        estimated_time="15-30 seconds",
        max_keyframes=4,
        keyframe_resolution=(384, 288),
        pose_backend="opencv",
        confidence_threshold=0.3,
        use_gaussians=False,
        gaussian_iterations=0,
        gaussian_backend="lightweight",
        dashgaussian_schedule=False,
        mesh_method="poisson",
        mesh_depth=8,
        target_triangles=30_000,
        texture_resolution=1024,
        texture_method="simple_projection",
    ),
    QualityMode.BALANCED: PipelineConfig(
        name="⚖️ Balanced",
        description="Good quality for most use cases",
        estimated_time="45-90 seconds",
        max_keyframes=8,
        keyframe_resolution=(512, 384),
        pose_backend="colmap",
        confidence_threshold=0.5,
        use_gaussians=True,
        gaussian_iterations=3000,
        gaussian_densify_interval=100,
        gaussian_densify_until=1500,
        gaussian_densify_from=500,
        gaussian_resolution=2,
        gaussian_backend="docker_3dgs",
        dashgaussian_schedule=True,
        mesh_method="sugar",
        target_triangles=80_000,
        texture_resolution=2048,
        texture_method="weighted_blend",
        # Adaptive pruning for smaller, higher quality models
        adaptive_prune_enabled=True,
        adaptive_prune_interval=1000,
        adaptive_prune_opacity=0.01,
    ),
    QualityMode.QUALITY: PipelineConfig(
        name="💎 Max Quality",
        description="Best possible output, takes longer",
        estimated_time="2-4 minutes",
        max_keyframes=16,
        keyframe_resolution=(768, 576),
        pose_backend="colmap",
        confidence_threshold=0.7,
        use_gaussians=True,
        gaussian_iterations=7000,
        gaussian_densify_interval=50,
        gaussian_densify_until=3500,
        gaussian_densify_from=500,
        gaussian_resolution=1,
        gaussian_backend="docker_3dgs",
        dashgaussian_schedule=True,
        mesh_method="sugar_refined",
        mesh_smoothing=True,
        target_triangles=150_000,
        texture_resolution=4096,
        texture_method="neural_blend",
        generate_normal_map=True,
        # More prune phases for max quality (every 1500 iters)
        adaptive_prune_enabled=True,
        adaptive_prune_interval=1500,
        adaptive_prune_opacity=0.005,
    )
}
