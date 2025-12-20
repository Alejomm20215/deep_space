import os
import time
from backend.pipelines import BasePipeline
from backend.core.dataset_layout import write_manifest
from backend.core.metrics import compute_metrics, write_metrics_json
from backend.core.compression.gzip_compressor import gzip_file
from backend.core.compression.external_runners import maybe_run_fcgs, maybe_run_hacpp
from backend.core.pruning import prune_gaussian_ply_ascii

class FastPipeline(BasePipeline):
    async def run(self, video_path: str, progress_callback):
        t0 = time.perf_counter()
        # 1. Keyframes
        await progress_callback("keyframes", 0, "Extracting 4 keyframes...")
        image_paths = self.extractor.extract_frames(video_path, self.layout.frames_dir)
        write_manifest(
            layout=self.layout,
            job_id=self.job_id,
            mode="fastest",
            input_path=video_path,
            frame_paths=image_paths,
            config_dict=self.config.model_dump(),
        )
        
        # 2. Reconstruction (depth + poses)
        await progress_callback("reconstruction", 20, "Reconstructing (depth + poses)...")
        recon_data = await self.reconstructor.reconstruct(
            image_paths, 
            self.layout.recon_dir,
            lambda p, msg: progress_callback("reconstruction", 20 + int(p * 0.2), msg)
        )
        
        # 3. Generate lightweight Gaussian splat (fast preview)
        await progress_callback("gaussians", 40, "Generating lightweight splat...")
        splat_path = await self.optimizer.optimize(
            recon_data["point_cloud"],
            recon_data["poses"],
            self.layout.gaussians_dir,
            lambda p, msg: progress_callback("gaussians", 40 + int(p * 0.1), msg)
        )
        
        # 4. Mesh Extraction (Poisson from Points)
        await progress_callback("mesh", 50, "Generating mesh from points...")
        mesh_path = os.path.join(self.layout.mesh_dir, "mesh.ply")
        mesh_path = await self.mesher.extract_mesh(
            recon_data["point_cloud"],
            mesh_path,
            lambda p, msg: progress_callback("mesh", 50 + int(p * 0.2), msg)
        )
        
        # 5. Texture (Simple)
        await progress_callback("texture", 70, "Projecting texture...")
        textured_path = os.path.join(self.layout.texture_dir, "model.glb")
        textured_path = await self.baker.bake_texture(
            mesh_path,
            image_paths,
            textured_path,
            lambda p, msg: progress_callback("texture", 70 + int(p * 0.2), msg)
        )

        elapsed = time.perf_counter() - t0
        metrics = compute_metrics(
            frames=len(image_paths),
            elapsed_seconds=elapsed,
            point_cloud_ply=recon_data.get("point_cloud"),
            gaussian_ply=splat_path,
            mesh_ply=mesh_path,
            glb_path=textured_path,
        )
        if self.config.prune_enabled:
            pruned_path = os.path.join(self.output_dir, "model_gaussians_pruned.ply")
            try:
                splat_path = prune_gaussian_ply_ascii(
                    input_path=splat_path,
                    output_path=pruned_path,
                    min_opacity=self.config.prune_min_opacity,
                ).output_path
            except Exception:
                pass

        metrics_path = write_metrics_json(os.path.join(self.output_dir, "metrics.json"), metrics)
        splat_gz_path = gzip_file(splat_path).output_path

        fcgs_path = os.path.join(self.output_dir, "model.fcgs.bin")
        hacpp_path = os.path.join(self.output_dir, "model.hacpp.bin")
        fcgs_res = maybe_run_fcgs(gaussian_ply=splat_path, out_path=fcgs_path)
        hacpp_res = maybe_run_hacpp(gaussian_ply=splat_path, out_path=hacpp_path)
        
        # 6. Export
        await progress_callback("export", 90, "Finalizing files...")
        artifacts = {
            "point_cloud": recon_data["point_cloud"],
            "textured_mesh": textured_path,
            "splat": splat_path,
            "splat_gz": splat_gz_path,
            "metrics": metrics_path,
            **({"fcgs": fcgs_path} if fcgs_res.used and os.path.exists(fcgs_path) else {}),
            **({"hacpp": hacpp_path} if hacpp_res.used and os.path.exists(hacpp_path) else {}),
        }
        final_files = await self.exporter.export_files(artifacts, f"backend/outputs/{self.job_id}")
        final_files["stats"] = {
            "time": f"{elapsed:.1f}s",
            "triangles": metrics.mesh_faces,
            "gaussians": metrics.gaussian_count,
        }
        
        await progress_callback("complete", 100, "Done!")
        return final_files
