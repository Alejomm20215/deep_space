import os
import shutil
from typing import Dict
from backend.config import PipelineConfig

class Exporter:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    async def export_files(self, artifacts: Dict[str, str], output_dir: str) -> Dict[str, str]:
        """
        Prepares final artifacts for download.
        """
        final_files = {}
        os.makedirs(output_dir, exist_ok=True)
        
        # Export GLB
        if self.config.export_glb and "textured_mesh" in artifacts:
            src = artifacts["textured_mesh"]
            dst = os.path.join(output_dir, "model.glb")
            shutil.copy2(src, dst)
            final_files["glb"] = dst
            
        # Export Splat
        if self.config.export_splat and "splat" in artifacts:
            src = artifacts["splat"]
            dst = os.path.join(output_dir, "model.splat")
            shutil.copy2(src, dst)
            final_files["splat"] = dst
            
        # Export Point Cloud
        if "point_cloud" in artifacts:
            src = artifacts["point_cloud"]
            dst = os.path.join(output_dir, "pointcloud.ply")
            shutil.copy2(src, dst)
            final_files["ply"] = dst
            
        return final_files
