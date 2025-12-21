import os
import time
from backend.pipelines import BasePipeline
from backend.core.dataset_layout import write_manifest
from backend.core.metrics import compute_metrics, write_metrics_json
from backend.core.compression.gzip_compressor import gzip_file
from backend.core.compression.external_runners import maybe_run_fcgs, maybe_run_hacpp
from backend.core.pruning import prune_gaussian_ply_ascii, prune_speedy_splat
from backend.core.gaussians.symmetry import apply_symmetry_pruning

class BalancedPipeline(BasePipeline):
    async def run(self, video_path: str, progress_callback):
        t0 = time.perf_counter()
        # 1. Keyframes
        await progress_callback("keyframes", 0, "Extracting 8 keyframes...")
        image_paths = self.extractor.extract_frames(video_path, self.layout.frames_dir)
        write_manifest(
            layout=self.layout,
            job_id=self.job_id,
            mode="balanced",
            input_path=video_path,
            frame_paths=image_paths,
            config_dict=self.config.model_dump(),
        )
        
        # 2. Reconstruction (depth + poses)
        await progress_callback("reconstruction", 10, "Reconstructing (depth + poses)...")
        recon_data = await self.reconstructor.reconstruct(
            image_paths, 
            self.layout.recon_dir,
            lambda p, msg: progress_callback("reconstruction", 10 + int(p * 0.15), msg)
        )
        
        # 3. Lightweight Gaussian splat (preview + downstream tooling)
        await progress_callback("gaussians", 25, "Generating lightweight splat...")
        splat_path = await self.optimizer.optimize(
            recon_data["point_cloud"],
            recon_data["poses"],
            self.layout.gaussians_dir,
            lambda p, msg: progress_callback("gaussians", 25 + int(p * 0.25), msg)
        )
        
        # 4. Mesh Extraction
        # Use the reconstruction mesh directly (it already has faces). The current lightweight
        # splat is a Gaussian point set and isn't suitable input for our mesh extractor.
        await progress_callback("mesh", 50, "Preparing mesh...")
        mesh_path = os.path.join(self.layout.mesh_dir, "mesh.ply")
        mesh_path = await self.mesher.extract_mesh(
            recon_data["point_cloud"],
            mesh_path,
            lambda p, msg: progress_callback("mesh", 50 + int(p * 0.2), msg)
        )
        
        # 5. Texture (Weighted Blend)
        await progress_callback("texture", 70, "Baking Weighted Texture...")
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
        # Optional post-training pruning (safe baseline)
        # Skip if adaptive pruning was used (already pruned during training)
        if self.config.prune_enabled and not self.config.adaptive_prune_enabled:
            pruned_path = os.path.join(self.output_dir, "model_gaussians_pruned.ply")
            try:
                splat_path = prune_gaussian_ply_ascii(
                    input_path=splat_path,
                    output_path=pruned_path,
                    min_opacity=self.config.prune_min_opacity,
                ).output_path
            except Exception:
                # Keep original splat if pruning fails
                pass

        # Optional Speedy-Splat rendering acceleration (Phase 4)
        if self.config.speedy_prune_enabled:
            speedy_path = os.path.join(self.output_dir, "model_gaussians_speedy.ply")
            try:
                splat_path = prune_speedy_splat(
                    input_path=splat_path,
                    output_path=speedy_path,
                    importance_threshold=self.config.speedy_prune_threshold,
                ).output_path
            except Exception:
                pass

        # Optional symmetry pruning (SymGS baseline)
        if self.config.symmetry_enabled:
            sym_path = os.path.join(self.output_dir, "model_gaussians_sym.ply")
            try:
                splat_path = apply_symmetry_pruning(
                    input_path=splat_path,
                    output_path=sym_path,
                    axis=self.config.symmetry_axis,
                    tolerance=self.config.symmetry_tolerance,
                ).pruned_path
            except Exception:
                pass

        metrics_path = write_metrics_json(os.path.join(self.output_dir, "metrics.json"), metrics)
        splat_gz_path = gzip_file(splat_path).output_path

        # Optional research compression runners (no-op unless enabled by env + config)
        fcgs_path = os.path.join(self.output_dir, "model.fcgs.bin")
        hacpp_path = os.path.join(self.output_dir, "model.hacpp.bin")
        fcgs_res = maybe_run_fcgs(gaussian_ply=splat_path, out_path=fcgs_path)
        hacpp_res = maybe_run_hacpp(gaussian_ply=splat_path, out_path=hacpp_path)
        
        # 6. Export
        await progress_callback("export", 90, "Exporting GLB & SPLAT...")
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
