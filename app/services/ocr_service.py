import os
import tempfile
from pathlib import Path
from typing import Union
from PIL import Image
from pipelines.ocr_pipeline import OCRPipeline


class OCRService:
    def __init__(self, device: str = None, tesseract_cmd: str = None):
        self.device = device or "cpu"
        self.pipeline = OCRPipeline(device=self.device, tesseract_cmd=tesseract_cmd)

    def extract_text(self, image_source: Union[str, Path, Image.Image, bytes], suffix: str = ".png") -> dict:
        """Extracts text from an image path, PIL Image, or bytes."""
        if isinstance(image_source, bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_source)
                tmp_path = tmp.name
            try:
                return self.pipeline.extract_text(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return self.pipeline.extract_text(image_source)

    def extract_layout(self, image_source: Union[str, Path, Image.Image, bytes], suffix: str = ".png") -> dict:
        """Extracts text and layout bounding box positions."""
        if isinstance(image_source, bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_source)
                tmp_path = tmp.name
            try:
                return self.pipeline.extract_layout_data(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return self.pipeline.extract_layout_data(image_source)