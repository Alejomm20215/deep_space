import os
from backend.pipelines import BasePipeline

class FastPipeline(BasePipeline):
    async def run(self, video_path: str, progress_callback):
        # 1. Keyframes
        await progress_callback("keyframes", 0, "Extracting 4 keyframes...")
        frames_dir = f"{self.output_dir}/frames"
        image_paths = self.extractor.extract_frames(video_path, frames_dir)
        
        # 2. Fast3R Reconstruction
        await progress_callback("reconstruction", 20, "Running Fast3R (Single Pass)...")
        recon_dir = f"{self.output_dir}/recon"
        recon_data = await self.reconstructor.reconstruct(
            image_paths, 
            recon_dir,
            lambda p, msg: progress_callback("reconstruction", 20 + int(p * 0.2), msg)
        )
        
        # 3. Skip 3DGS Optimization (Direct Point Cloud Export)
        await progress_callback("gaussians", 40, "Skipping 3DGS (Fastest Mode)...")
        splat_path = recon_data["point_cloud"] 
        
        # 4. Mesh Extraction (Poisson from Points)
        await progress_callback("mesh", 50, "Generating mesh from points...")
        mesh_dir = f"{self.output_dir}/mesh"
        os.makedirs(mesh_dir, exist_ok=True)
        mesh_path = os.path.join(mesh_dir, "mesh.ply")
        mesh_path = await self.mesher.extract_mesh(
            splat_path,
            mesh_path,
            lambda p, msg: progress_callback("mesh", 50 + int(p * 0.2), msg)
        )
        
        # 5. Texture (Simple)
        await progress_callback("texture", 70, "Projecting texture...")
        tex_dir = f"{self.output_dir}/texture"
        os.makedirs(tex_dir, exist_ok=True)
        textured_path = os.path.join(tex_dir, "model.glb")
        textured_path = await self.baker.bake_texture(
            mesh_path,
            image_paths,
            textured_path,
            lambda p, msg: progress_callback("texture", 70 + int(p * 0.2), msg)
        )
        
        # 6. Export
        await progress_callback("export", 90, "Finalizing files...")
        artifacts = {
            "point_cloud": recon_data["point_cloud"],
            "textured_mesh": textured_path,
            # For fast mode, we might just use the point cloud ply as the 'splat' equivalent
            "splat": splat_path 
        }
        final_files = await self.exporter.export_files(artifacts, f"backend/outputs/{self.job_id}")
        
        await progress_callback("complete", 100, "Done!")
        return final_files
