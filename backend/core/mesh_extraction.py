import time
import os
from backend.config import PipelineConfig

class MeshExtractor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    async def extract_mesh(self, input_model: str, output_dir: str, progress_callback=None) -> str:
        """
        Extracts a mesh from the point cloud or gaussian splats.
        Simulates SuGaR or Poisson reconstruction.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_mesh = os.path.join(output_dir, "mesh_raw.ply")
        
        method = self.config.mesh_method
        print(f"Extracting mesh using {method}...")
        
        # Simulate processing time
        if progress_callback:
            await progress_callback(0, "Initializing mesh extraction...")
        
        time.sleep(1.0)
        
        if progress_callback:
            await progress_callback(50, "Running surface reconstruction...")
            
        time.sleep(1.0)
        
        # Create dummy mesh file
        self._create_dummy_mesh(output_mesh)
        
        return output_mesh

    def _create_dummy_mesh(self, path: str):
        # Create a simple PLY triangle
        header = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
3 0 1 2
"""
        with open(path, "w") as f:
            f.write(header)
