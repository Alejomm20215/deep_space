"""
Mesh extraction from point cloud.
Uses Poisson-like surface reconstruction via scipy.
"""
import asyncio
import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist
from backend.config import PipelineConfig


class MeshExtractor:
    """
    Converts a colored point cloud (PLY) into a triangle mesh.
    Uses 3D Delaunay tetrahedralization and extracts the surface.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def extract_mesh(self, input_model: str, output_dir: str, progress_callback=None) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_mesh = os.path.join(output_dir, "mesh_raw.ply")

        if progress_callback:
            await progress_callback(0, "Loading point cloud...")

        def build_mesh():
            pts, colors = self._load_ply(input_model)
            if len(pts) < 10:
                raise RuntimeError("Not enough points for mesh.")

            # Subsample for speed
            max_pts = 30000
            if len(pts) > max_pts:
                indices = np.random.choice(len(pts), max_pts, replace=False)
                pts = pts[indices]
                colors = colors[indices]

            # Try 3D Delaunay, extract surface triangles
            try:
                faces = self._extract_surface_triangles(pts)
            except Exception as e:
                print(f"Delaunay failed: {e}, using convex hull")
                hull = ConvexHull(pts)
                faces = hull.simplices

            # Filter bad triangles
            faces = self._filter_triangles(pts, faces, max_edge=0.1)

            if len(faces) == 0:
                # Fallback: create a simple convex hull
                hull = ConvexHull(pts)
                faces = hull.simplices

            self._write_mesh_ply(output_mesh, pts, colors, faces)
            return len(pts), len(faces)

        num_verts, num_faces = await asyncio.to_thread(build_mesh)

        if progress_callback:
            await progress_callback(100, f"Mesh: {num_verts} verts, {num_faces} faces")

        return output_mesh

    def _extract_surface_triangles(self, pts: np.ndarray) -> np.ndarray:
        """
        Extract surface triangles from 3D Delaunay tetrahedralization.
        Surface triangles are those that appear in only one tetrahedron.
        """
        # 3D Delaunay
        tri = Delaunay(pts)

        # Count triangle occurrences
        triangle_count = {}

        for tetra in tri.simplices:
            # Each tetrahedron has 4 triangular faces
            faces = [
                tuple(sorted([tetra[0], tetra[1], tetra[2]])),
                tuple(sorted([tetra[0], tetra[1], tetra[3]])),
                tuple(sorted([tetra[0], tetra[2], tetra[3]])),
                tuple(sorted([tetra[1], tetra[2], tetra[3]])),
            ]
            for f in faces:
                triangle_count[f] = triangle_count.get(f, 0) + 1

        # Surface triangles appear exactly once
        surface = [list(f) for f, count in triangle_count.items() if count == 1]

        return np.array(surface, dtype=np.int32) if surface else np.zeros((0, 3), dtype=np.int32)

    def _filter_triangles(self, pts: np.ndarray, faces: np.ndarray, max_edge: float) -> np.ndarray:
        """Remove triangles with edges longer than max_edge."""
        if len(faces) == 0:
            return faces

        valid = []
        for f in faces:
            v0, v1, v2 = pts[f[0]], pts[f[1]], pts[f[2]]
            e0 = np.linalg.norm(v1 - v0)
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v0 - v2)
            if max(e0, e1, e2) < max_edge:
                valid.append(f)

        return np.array(valid, dtype=np.int32) if valid else np.zeros((0, 3), dtype=np.int32)

    def _load_ply(self, path: str):
        """Load ASCII PLY with XYZ + RGB."""
        pts = []
        colors = []
        in_header = True
        vertex_count = 0

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if in_header:
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[-1])
                    if line == "end_header":
                        in_header = False
                    continue

                parts = line.split()
                if len(parts) >= 6:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                    pts.append([x, y, z])
                    colors.append([r, g, b])

                if len(pts) >= vertex_count:
                    break

        return np.array(pts, dtype=np.float32), np.array(colors, dtype=np.uint8)

    def _write_mesh_ply(self, path: str, pts: np.ndarray, colors: np.ndarray, faces: np.ndarray):
        """Write mesh as ASCII PLY with vertex colors."""
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            for p, c in zip(pts, colors):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
