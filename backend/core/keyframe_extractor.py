
import cv2
import numpy as np
import os
from typing import List, Tuple
from backend.config import PipelineConfig

class KeyframeExtractor:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def extract_frames(self, input_path: str, output_dir: str) -> List[str]:
        """
        Extracts high-quality keyframes from a video OR processes existing images from a directory.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input not found: {input_path}")
            
        os.makedirs(output_dir, exist_ok=True)
        extracted_paths = []
        target_res = self.config.keyframe_resolution

        # Case 1: Input is a directory of images
        if os.path.isdir(input_path):
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = sorted([
                f for f in os.listdir(input_path) 
                if f.lower().endswith(valid_exts)
            ])
            
            # Limit to max_keyframes if needed, or take all if close
            # For 4-image mode, we likely want all of them.
            if len(image_files) > self.config.max_keyframes:
                 indices = np.linspace(0, len(image_files) - 1, self.config.max_keyframes, dtype=int)
                 image_files = [image_files[i] for i in indices]
            
            for i, filename in enumerate(image_files):
                img_path = os.path.join(input_path, filename)
                frame = cv2.imread(img_path)
                
                if frame is None:
                    continue

                if frame.shape[1] != target_res[0] or frame.shape[0] != target_res[1]:
                    frame = cv2.resize(frame, target_res)
                
                out_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
                cv2.imwrite(out_path, frame)
                extracted_paths.append(out_path)
                
            return extracted_paths

        # Case 2: Input is a video file
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        indices = np.linspace(0, total_frames - 1, self.config.max_keyframes, dtype=int)
        
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            if frame.shape[1] != target_res[0] or frame.shape[0] != target_res[1]:
                frame = cv2.resize(frame, target_res)
            
            frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_paths.append(frame_path)
            
        cap.release()
        return extracted_paths
