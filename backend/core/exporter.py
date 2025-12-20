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
            if str(src).lower().endswith(".ply"):
                dst_name = "model.splat.ply"
            else:
                dst_name = "model.splat"
            dst = os.path.join(output_dir, dst_name)
            shutil.copy2(src, dst)
            final_files["splat"] = to_url(dst_name)

        # Export compressed splat (.gz)
        if self.config.export_splat_gzip and "splat_gz" in artifacts:
            src = artifacts["splat_gz"]
            dst = os.path.join(output_dir, "model.splat.ply.gz")
            shutil.copy2(src, dst)
            final_files["splat_gz"] = to_url("model.splat.ply.gz")

        # Export research compression outputs (optional)
        if self.config.export_fcgs and "fcgs" in artifacts:
            src = artifacts["fcgs"]
            dst = os.path.join(output_dir, "model.fcgs.bin")
            shutil.copy2(src, dst)
            final_files["fcgs"] = to_url("model.fcgs.bin")

        if self.config.export_hacpp and "hacpp" in artifacts:
            src = artifacts["hacpp"]
            dst = os.path.join(output_dir, "model.hacpp.bin")
            shutil.copy2(src, dst)
            final_files["hacpp"] = to_url("model.hacpp.bin")

        # Export Point Cloud
        if "point_cloud" in artifacts:
            src = artifacts["point_cloud"]
            dst = os.path.join(output_dir, "pointcloud.ply")
            shutil.copy2(src, dst)
            final_files["ply"] = to_url("pointcloud.ply")

        # Export metrics
        if self.config.export_metrics and "metrics" in artifacts:
            src = artifacts["metrics"]
            dst = os.path.join(output_dir, "metrics.json")
            shutil.copy2(src, dst)
            final_files["metrics"] = to_url("metrics.json")

        return final_files
