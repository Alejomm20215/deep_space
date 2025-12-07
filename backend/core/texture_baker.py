import asyncio
import json
import os
import struct
from typing import List

import numpy as np
from backend.config import PipelineConfig


class TextureBaker:
    """
    Converts mesh PLY to GLB with holographic blue coloring (Tony Stark style).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def bake_texture(self, mesh_path: str, image_paths: List[str], output_dir: str, progress_callback=None) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_glb = os.path.join(output_dir, "mesh_textured.glb")

        if progress_callback:
            await progress_callback(20, "Loading mesh...")

        def build_glb():
            pts, _, faces = self._load_mesh_ply(mesh_path)
            if len(pts) == 0 or len(faces) == 0:
                pts, faces = self._fallback_cube()
            
            # Apply holographic blue coloring
            colors = self._holographic_blue(pts)
            return self._create_glb(pts, colors, faces)

        glb_bytes = await asyncio.to_thread(build_glb)

        if progress_callback:
            await progress_callback(80, "Writing GLB...")

        with open(output_glb, "wb") as f:
            f.write(glb_bytes)

        if progress_callback:
            await progress_callback(100, "Hologram ready")

        return output_glb

    def _holographic_blue(self, pts: np.ndarray) -> np.ndarray:
        """
        Generate Tony Stark-style holographic blue coloring.
        Uses position-based gradients for that sci-fi look.
        """
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Normalize positions to 0-1 range
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        range_pt = max_pt - min_pt
        range_pt[range_pt < 1e-6] = 1.0  # Avoid division by zero
        normalized = (pts - min_pt) / range_pt

        # Base hologram colors (RGB)
        # Electric cyan: (0, 1, 1)
        # Deep blue: (0, 0.3, 0.8)
        # Bright blue: (0.2, 0.6, 1)

        colors = np.zeros((len(pts), 3), dtype=np.float32)

        # Height-based gradient (Y axis usually up)
        height = normalized[:, 1]
        
        # Radial distance from center (XZ plane)
        center_xz = np.array([0.5, 0.5])
        dist = np.sqrt((normalized[:, 0] - center_xz[0])**2 + (normalized[:, 2] - center_xz[1])**2)
        dist = np.clip(dist / 0.7, 0, 1)  # Normalize

        # Add some noise for that flickering hologram feel
        noise = np.random.uniform(0.85, 1.0, len(pts))

        # Blue channel: strong everywhere
        colors[:, 2] = (0.7 + 0.3 * height) * noise

        # Green channel: cyan tint, stronger at edges
        colors[:, 1] = (0.4 + 0.4 * dist + 0.2 * height) * noise

        # Red channel: subtle, adds some white/glow at peaks
        colors[:, 0] = (0.05 + 0.15 * height * dist) * noise

        # Clamp to valid range
        colors = np.clip(colors, 0, 1)

        return colors

    def _load_mesh_ply(self, path: str):
        """Load mesh PLY with XYZ + optional RGB + faces."""
        pts = []
        colors = []
        faces = []
        in_header = True
        vertex_count = 0
        face_count = 0
        reading_vertices = True

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if in_header:
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith("element face"):
                        face_count = int(line.split()[-1])
                    elif line == "end_header":
                        in_header = False
                    continue

                parts = line.split()
                if reading_vertices and len(pts) < vertex_count:
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        pts.append([x, y, z])
                        if len(parts) >= 6:
                            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                            colors.append([r / 255.0, g / 255.0, b / 255.0])
                        else:
                            colors.append([0.0, 0.5, 1.0])  # Default blue
                    if len(pts) >= vertex_count:
                        reading_vertices = False
                elif len(faces) < face_count:
                    if len(parts) >= 4:
                        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

        return (
            np.array(pts, dtype=np.float32),
            np.array(colors, dtype=np.float32),
            np.array(faces, dtype=np.uint32)
        )

    def _fallback_cube(self):
        """Simple cube if mesh is empty."""
        pts = np.array([
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [3, 2, 6], [3, 6, 5], [0, 7, 1], [0, 4, 7],
            [0, 3, 5], [0, 5, 4], [1, 7, 6], [1, 6, 2],
        ], dtype=np.uint32)
        return pts, faces

    def _create_glb(self, pts: np.ndarray, colors: np.ndarray, faces: np.ndarray) -> bytes:
        """Create GLB from mesh vertices, colors, and faces."""
        indices = faces.flatten().astype(np.uint32)

        min_pt = pts.min(axis=0).tolist()
        max_pt = pts.max(axis=0).tolist()

        pos_bytes = pts.astype(np.float32).tobytes()
        col_bytes = colors.astype(np.float32).tobytes()
        idx_bytes = indices.astype(np.uint32).tobytes()

        def pad4(data: bytes) -> bytes:
            remainder = len(data) % 4
            return data + b'\x00' * (4 - remainder) if remainder else data

        pos_bytes = pad4(pos_bytes)
        col_bytes = pad4(col_bytes)
        idx_bytes = pad4(idx_bytes)

        bin_data = pos_bytes + col_bytes + idx_bytes
        pos_offset = 0
        col_offset = len(pos_bytes)
        idx_offset = col_offset + len(col_bytes)

        gltf = {
            "asset": {"version": "2.0", "generator": "DeepSpace-Hologram"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "HolographicMesh"}],
            "meshes": [{
                "primitives": [{
                    "attributes": {"POSITION": 0, "COLOR_0": 1},
                    "indices": 2,
                    "mode": 4
                }]
            }],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": len(pts),
                    "type": "VEC3",
                    "min": min_pt,
                    "max": max_pt
                },
                {
                    "bufferView": 1,
                    "componentType": 5126,
                    "count": len(colors),
                    "type": "VEC3"
                },
                {
                    "bufferView": 2,
                    "componentType": 5125,
                    "count": len(indices),
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(pos_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": col_offset, "byteLength": len(col_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": idx_offset, "byteLength": len(idx_bytes), "target": 34963}
            ],
            "buffers": [{"byteLength": len(bin_data)}]
        }

        json_str = json.dumps(gltf, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        json_pad = (4 - len(json_bytes) % 4) % 4
        json_bytes += b' ' * json_pad

        json_chunk = struct.pack('<I', len(json_bytes)) + b'JSON' + json_bytes
        bin_chunk = struct.pack('<I', len(bin_data)) + b'BIN\x00' + bin_data

        total_length = 12 + len(json_chunk) + len(bin_chunk)
        header = b'glTF' + struct.pack('<II', 2, total_length)

        return header + json_chunk + bin_chunk
