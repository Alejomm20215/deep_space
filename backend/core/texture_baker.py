"""
Texture baking and GLB export.
Applies holographic blue shader effect and exports final model.
"""
import asyncio
import os
import struct
import numpy as np
from typing import List
from backend.config import PipelineConfig


class TextureBaker:
    """
    Bakes vertex colors onto mesh and exports as GLB.
    Applies holographic blue-cyan shader effect.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def bake(self, mesh_path: str, image_paths: List[str], 
                   output_textured_mesh: str, progress_callback=None) -> str:
        """
        Bake colors and export as GLB.
        """
        if progress_callback:
            await progress_callback(10, "Loading mesh...")

        # Load mesh from PLY
        vertices, colors, faces = await asyncio.to_thread(self._load_ply_mesh, mesh_path)

        if progress_callback:
            await progress_callback(40, "Applying holographic effect...")

        # Keep original colors (no holographic tint)
        final_colors = colors

        if progress_callback:
            await progress_callback(70, "Exporting GLB...")

        # Export as binary GLB
        await asyncio.to_thread(self._write_glb, output_textured_mesh, vertices, final_colors, faces)

        if progress_callback:
            await progress_callback(100, "Export complete")

        return output_textured_mesh

    # Backward-compat alias (pipelines may call bake_texture)
    async def bake_texture(self, mesh_path: str, image_paths: List[str],
                           output_textured_mesh: str, progress_callback=None) -> str:
        return await self.bake(mesh_path, image_paths, output_textured_mesh, progress_callback)

    def _load_ply_mesh(self, ply_path: str):
        """Load mesh data from PLY file."""
        vertices = []
        colors = []
        faces = []

        with open(ply_path, 'r') as f:
            lines = f.readlines()

        # Parse header
        header_end = 0
        num_vertices = 0
        num_faces = 0
        
        for i, line in enumerate(lines):
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                num_faces = int(line.split()[-1])
            elif line.strip() == "end_header":
                header_end = i + 1
                break

        # Read vertices
        for i in range(num_vertices):
            parts = lines[header_end + i].split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(parts) >= 6:
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
            else:
                colors.append([128, 128, 128])

        # Read faces
        for i in range(num_faces):
            parts = lines[header_end + num_vertices + i].split()
            if len(parts) >= 4:
                # First number is vertex count (usually 3)
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

        return (
            np.array(vertices, dtype=np.float32),
            np.array(colors, dtype=np.uint8),
            np.array(faces, dtype=np.uint32)
        )

    # Legacy hook retained for compatibility; now returns original colors unchanged.
    def _apply_holographic_effect(self, vertices: np.ndarray, colors: np.ndarray) -> np.ndarray:
        return colors

    def _write_glb(self, path: str, vertices: np.ndarray, colors: np.ndarray, faces: np.ndarray):
        """
        Write mesh as binary GLB (glTF 2.0).
        """
        import json

        # Prepare binary buffer data
        # Positions (float32 x 3)
        positions_data = vertices.astype(np.float32).tobytes()
        
        # Colors (normalized to float for glTF)
        colors_float = (colors.astype(np.float32) / 255.0)
        colors_data = colors_float.astype(np.float32).tobytes()
        
        # Indices (uint32 for large meshes)
        indices_data = faces.flatten().astype(np.uint32).tobytes()

        # Calculate buffer offsets (must be 4-byte aligned)
        def align4(x):
            return (x + 3) & ~3

        positions_offset = 0
        positions_length = len(positions_data)
        
        colors_offset = align4(positions_offset + positions_length)
        colors_length = len(colors_data)
        colors_padding = colors_offset - (positions_offset + positions_length)
        
        indices_offset = align4(colors_offset + colors_length)
        indices_length = len(indices_data)
        indices_padding = indices_offset - (colors_offset + colors_length)

        total_buffer_length = indices_offset + indices_length

        # Build binary buffer
        buffer_data = bytearray()
        buffer_data.extend(positions_data)
        buffer_data.extend(b'\x00' * colors_padding)
        buffer_data.extend(colors_data)
        buffer_data.extend(b'\x00' * indices_padding)
        buffer_data.extend(indices_data)

        # Calculate bounding box
        pos_min = vertices.min(axis=0).tolist()
        pos_max = vertices.max(axis=0).tolist()

        # Build glTF JSON
        gltf = {
            "asset": {"version": "2.0", "generator": "DeepSpace Holographic"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{
                "primitives": [{
                    "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "indices": 2,
                        "mode": 4,  # TRIANGLES
                        "material": 0
                }]
            }],
                "materials": [{
                    # Enable rendering of back faces so the relief is visible from behind
                    "doubleSided": True
                }],
            "accessors": [
                {  # Positions
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": len(vertices),
                    "type": "VEC3",
                    "min": pos_min,
                    "max": pos_max
                },
                {  # Colors
                    "bufferView": 1,
                    "componentType": 5126,  # FLOAT
                    "count": len(colors),
                    "type": "VEC3"
                },
                {  # Indices
                    "bufferView": 2,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": len(faces) * 3,
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                {  # Positions
                    "buffer": 0,
                    "byteOffset": positions_offset,
                    "byteLength": positions_length,
                    "target": 34962  # ARRAY_BUFFER
                },
                {  # Colors
                    "buffer": 0,
                    "byteOffset": colors_offset,
                    "byteLength": colors_length,
                    "target": 34962
                },
                {  # Indices
                    "buffer": 0,
                    "byteOffset": indices_offset,
                    "byteLength": indices_length,
                    "target": 34963  # ELEMENT_ARRAY_BUFFER
                }
            ],
            "buffers": [{"byteLength": len(buffer_data)}]
        }

        # Serialize JSON
        json_str = json.dumps(gltf, separators=(',', ':'))
        json_data = json_str.encode('utf-8')
        
        # Pad JSON to 4-byte alignment
        json_padding = (4 - (len(json_data) % 4)) % 4
        json_data += b' ' * json_padding

        # Build GLB file
        # Header: magic + version + length
        glb_magic = 0x46546C67  # "glTF"
        glb_version = 2
        
        # Chunk 0: JSON
        json_chunk_type = 0x4E4F534A  # "JSON"
        json_chunk_length = len(json_data)
        
        # Chunk 1: BIN
        bin_chunk_type = 0x004E4942  # "BIN\0"
        bin_padding = (4 - (len(buffer_data) % 4)) % 4
        bin_chunk_length = len(buffer_data) + bin_padding
        
        # Total file length
        total_length = 12 + 8 + json_chunk_length + 8 + bin_chunk_length

        # Write GLB
        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', glb_magic))
            f.write(struct.pack('<I', glb_version))
            f.write(struct.pack('<I', total_length))
            
            # JSON chunk
            f.write(struct.pack('<I', json_chunk_length))
            f.write(struct.pack('<I', json_chunk_type))
            f.write(json_data)
            
            # BIN chunk
            f.write(struct.pack('<I', bin_chunk_length))
            f.write(struct.pack('<I', bin_chunk_type))
            f.write(buffer_data)
            f.write(b'\x00' * bin_padding)
