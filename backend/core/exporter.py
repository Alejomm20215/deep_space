import os
import shutil
from typing import Dict
from backend.config import PipelineConfig


class Exporter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    async def export_files(self, artifacts: Dict[str, str], output_dir: str) -> Dict[str, str]:
        """
        Copies artifacts to the public outputs directory and returns
        browser-accessible URLs (not filesystem paths).
        """
        final_files = {}
        os.makedirs(output_dir, exist_ok=True)

        job_id = os.path.basename(output_dir.rstrip("/\\"))

        def to_url(name: str) -> str:
            return f"/outputs/{job_id}/{name}"

        # Export GLB
        if self.config.export_glb and "textured_mesh" in artifacts:
            src = artifacts["textured_mesh"]
            dst = os.path.join(output_dir, "model.glb")
            shutil.copy2(src, dst)
            final_files["glb"] = to_url("model.glb")

        # Export Splat
        if self.config.export_splat and "splat" in artifacts:
            src = artifacts["splat"]
            dst = os.path.join(output_dir, "model.splat")
            shutil.copy2(src, dst)
            final_files["splat"] = to_url("model.splat")

        # Export Point Cloud
        if "point_cloud" in artifacts:
            src = artifacts["point_cloud"]
            dst = os.path.join(output_dir, "pointcloud.ply")
            shutil.copy2(src, dst)
            final_files["ply"] = to_url("pointcloud.ply")

        return final_files
