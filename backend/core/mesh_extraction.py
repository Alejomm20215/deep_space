"""
Mesh extraction from point cloud.
Now primarily passes through the mesh created in reconstruction,
since we create a proper 2.5D relief mesh directly.
"""
import asyncio
import os
import numpy as np
from backend.config import PipelineConfig


class MeshExtractor:
    """
    Extracts/refines mesh from point cloud.
    
    If the input PLY already contains faces (from reconstruction),
    we pass it through. Otherwise, we create a simple mesh.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def extract(self, input_model: str, output_mesh: str, progress_callback=None) -> str:
        """
        Extract or pass-through mesh from PLY.
        """
        if progress_callback:
            await progress_callback(10, "Checking mesh format...")

        # Check if input already has faces
        has_faces = await asyncio.to_thread(self._ply_has_faces, input_model)

        if has_faces:
            # Already a mesh - just copy/pass through
            if progress_callback:
                await progress_callback(50, "Using pre-built mesh...")
            
            await asyncio.to_thread(self._copy_file, input_model, output_mesh)
        else:
            # Point cloud only - create simple mesh
            if progress_callback:
                await progress_callback(30, "Creating mesh from points...")
            
            await asyncio.to_thread(self._create_mesh_from_points, input_model, output_mesh)

        if progress_callback:
            await progress_callback(100, "Mesh ready")

        return output_mesh

    # Backward-compat alias (some pipelines call extract_mesh)
    async def extract_mesh(self, input_model: str, output_mesh: str, progress_callback=None) -> str:
        return await self.extract(input_model, output_mesh, progress_callback)

    def _ply_has_faces(self, ply_path: str) -> bool:
        """Check if PLY file contains face data."""
        try:
            with open(ply_path, 'r') as f:
                for line in f:
                    if line.startswith('element face'):
                        parts = line.split()
                        if len(parts) >= 3:
                            num_faces = int(parts[2])
                            return num_faces > 0
                    if line.strip() == 'end_header':
                        break
        except:
            pass
        return False

    def _copy_file(self, src: str, dst: str):
        """Copy file from src to dst."""
        import shutil
        shutil.copy2(src, dst)

    def _create_mesh_from_points(self, ply_path: str, output_path: str):
        """
        Create a simple mesh from point cloud using grid-based triangulation.
        Fallback for point clouds without faces.
        """
        # Read points and colors from PLY
        vertices = []
        colors = []
        
        with open(ply_path, 'r') as f:
            lines = f.readlines()

        # Parse header
        header_end = 0
        num_vertices = 0
        for i, line in enumerate(lines):
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            if line.strip() == "end_header":
                header_end = i + 1
                break

        if num_vertices == 0:
            # Create dummy mesh
            self._write_dummy_mesh(output_path)
            return

        # Read vertex data
        for i in range(num_vertices):
            parts = lines[header_end + i].split()
            if len(parts) >= 6:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

        vertices = np.array(vertices, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)

        if len(vertices) < 3:
            self._write_dummy_mesh(output_path)
            return

        # Try to create a grid-based mesh
        # Assume points are roughly in a grid pattern
        n = len(vertices)
        grid_size = int(np.sqrt(n))
        
        if grid_size * grid_size == n:
            # Perfect grid - create proper mesh
            faces = self._create_grid_faces(grid_size, grid_size)
        else:
            # Not a grid - create simple triangle fan from center
            faces = self._create_fan_faces(len(vertices))

        # Write output PLY with faces
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for v, c in zip(vertices, colors):
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def _create_grid_faces(self, rows: int, cols: int) -> list:
        """Create triangle faces for a grid of vertices."""
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                v00 = i * cols + j
                v01 = i * cols + (j + 1)
                v10 = (i + 1) * cols + j
                v11 = (i + 1) * cols + (j + 1)
                faces.append([v00, v10, v01])
                faces.append([v01, v10, v11])
        return faces

    def _create_fan_faces(self, num_vertices: int) -> list:
        """Create triangle fan from vertices (fallback)."""
        if num_vertices < 3:
            return []
        
        # Use first vertex as center
        faces = []
        for i in range(1, num_vertices - 1):
            faces.append([0, i, i + 1])
        return faces

    def _write_dummy_mesh(self, output_path: str):
        """Write a minimal dummy mesh."""
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 3\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("element face 1\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            f.write("0 0 0 128 128 128\n")
            f.write("1 0 0 128 128 128\n")
            f.write("0 1 0 128 128 128\n")
            f.write("3 0 1 2\n")
