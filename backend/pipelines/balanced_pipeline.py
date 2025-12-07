from backend.pipelines import BasePipeline

class BalancedPipeline(BasePipeline):
    async def run(self, video_path: str, progress_callback):
        # 1. Keyframes
        await progress_callback("keyframes", 0, "Extracting 8 keyframes...")
        frames_dir = f"{self.output_dir}/frames"
        image_paths = self.extractor.extract_frames(video_path, frames_dir)
        
        # 2. Reconstruction
        await progress_callback("reconstruction", 10, "Running Fast3R...")
        recon_dir = f"{self.output_dir}/recon"
        recon_data = await self.reconstructor.reconstruct(
            image_paths, 
            recon_dir,
            lambda p, msg: progress_callback("reconstruction", 10 + int(p * 0.15), msg)
        )
        
        # 3. 3DGS Optimization (300 iters)
        await progress_callback("gaussians", 25, "Optimizing 3D Gaussians...")
        splat_dir = f"{self.output_dir}/splat"
        splat_path = await self.optimizer.optimize(
            recon_data["point_cloud"],
            recon_data["poses"],
            splat_dir,
            lambda p, msg: progress_callback("gaussians", 25 + int(p * 0.25), msg)
        )
        
        # 4. Mesh Extraction (SuGaR)
        await progress_callback("mesh", 50, "Extracting SuGaR Mesh...")
        mesh_dir = f"{self.output_dir}/mesh"
        mesh_path = await self.mesher.extract_mesh(
            splat_path, 
            mesh_dir,
            lambda p, msg: progress_callback("mesh", 50 + int(p * 0.2), msg)
        )
        
        # 5. Texture (Weighted Blend)
        await progress_callback("texture", 70, "Baking Weighted Texture...")
        tex_dir = f"{self.output_dir}/texture"
        textured_path = await self.baker.bake_texture(
            mesh_path, 
            image_paths, 
            tex_dir,
            lambda p, msg: progress_callback("texture", 70 + int(p * 0.2), msg)
        )
        
        # 6. Export
        await progress_callback("export", 90, "Exporting GLB & SPLAT...")
        artifacts = {
            "point_cloud": recon_data["point_cloud"],
            "textured_mesh": textured_path,
            "splat": splat_path
        }
        final_files = await self.exporter.export_files(artifacts, f"backend/outputs/{self.job_id}")
        
        await progress_callback("complete", 100, "Done!")
        return final_files
