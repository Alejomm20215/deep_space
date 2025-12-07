import asyncio
import os
import shutil
from backend.config import PipelineConfig


class GaussianSplattingOptimizer:
    """
    Placeholder optimizer: simply forwards the point cloud as the splat.
    Keeps async-friendly progress updates without blocking.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def optimize(self, initial_points: str, poses: str, output_dir: str, progress_callback=None) -> str:
        os.makedirs(output_dir, exist_ok=True)
        output_splat = os.path.join(output_dir, "model.splat")

        if progress_callback:
            await progress_callback(10, "Preparing splat")

        # Lightweight copy, offloaded to thread to avoid blocking
        def copy_file():
            shutil.copy2(initial_points, output_splat)

        await asyncio.to_thread(copy_file)

        if progress_callback:
            await progress_callback(100, "Splat ready")

        return output_splat
