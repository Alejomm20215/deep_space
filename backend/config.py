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
    
    # Reconstruction (Fast3R)
    reconstruction_model: str = "fast3r"
    confidence_threshold: float
    
    # 3D Gaussian Splatting
    use_gaussians: bool
    gaussian_iterations: int
    gaussian_densify_interval: Optional[int] = None
    gaussian_densify_until: Optional[int] = None
    
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

QUALITY_PRESETS: Dict[QualityMode, PipelineConfig] = {
    QualityMode.FASTEST: PipelineConfig(
        name="⚡ Fastest",
        description="Quick preview, some artifacts okay",
        estimated_time="15-30 seconds",
        max_keyframes=4,
        keyframe_resolution=(384, 288),
        confidence_threshold=0.3,
        use_gaussians=False,
        gaussian_iterations=0,
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
        confidence_threshold=0.5,
        use_gaussians=True,
        gaussian_iterations=300,
        gaussian_densify_interval=100,
        gaussian_densify_until=300,
        mesh_method="sugar",
        target_triangles=80_000,
        texture_resolution=2048,
        texture_method="weighted_blend",
    ),
    QualityMode.QUALITY: PipelineConfig(
        name="💎 Max Quality",
        description="Best possible output, takes longer",
        estimated_time="2-4 minutes",
        max_keyframes=16,
        keyframe_resolution=(768, 576),
        confidence_threshold=0.7,
        use_gaussians=True,
        gaussian_iterations=1000,
        gaussian_densify_interval=50,
        gaussian_densify_until=500,
        mesh_method="sugar_refined",
        mesh_smoothing=True,
        target_triangles=150_000,
        texture_resolution=4096,
        texture_method="neural_blend",
        generate_normal_map=True,
    )
}
