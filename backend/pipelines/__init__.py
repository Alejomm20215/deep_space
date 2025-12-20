from typing import Dict, Any, Callable
import os
from backend.config import PipelineConfig
from backend.core import (
    KeyframeExtractor, 
    Fast3RReconstructor, 
    GaussianSplattingOptimizer, 
    MeshExtractor, 
    TextureBaker, 
    Exporter
)
from backend.core.dataset_layout import JobLayout

class BasePipeline:
    def __init__(self, job_id: str, config: PipelineConfig, base_dir: str = "temp_processing"):
        self.job_id = job_id
        self.config = config
        self.output_dir = os.path.join(base_dir, job_id)
        self.layout = JobLayout(self.output_dir)
        self.layout.ensure_dirs()
        
        # Initialize core components
        self.extractor = KeyframeExtractor(config)
        self.reconstructor = Fast3RReconstructor(config)
        self.optimizer = GaussianSplattingOptimizer(config)
        self.mesher = MeshExtractor(config)
        self.baker = TextureBaker(config)
        self.exporter = Exporter(config)

    async def run(self, video_path: str, progress_callback: Callable[[str, int, str], None]) -> Dict[str, Any]:
        """
        Main execution method to be implemented by subclasses or used directly.
        """
        raise NotImplementedError
