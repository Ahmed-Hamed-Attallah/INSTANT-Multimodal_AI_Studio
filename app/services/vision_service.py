from pathlib import Path
from PIL import Image
import io
from pipelines.vision_pipeline import VisionPipeline

class VisionService:
    def __init__(self, device: str | None = None):
        self.pipeline = VisionPipeline(device=device)

    def process_image_upload(self, file_bytes_or_path: bytes | str | Path | Image.Image) -> Image.Image:
        if isinstance(file_bytes_or_path, Image.Image):
            return file_bytes_or_path
        if isinstance(file_bytes_or_path, bytes):
            return Image.open(io.BytesIO(file_bytes_or_path)).convert("RGB")
        return Image.open(file_bytes_or_path).convert("RGB")

    def get_caption(self, image_input: bytes | str | Path | Image.Image) -> dict:
        img = self.process_image_upload(image_input)
        return self.pipeline.run_captioning(img)

    def ask_visual_question(self, image_input: bytes | str | Path | Image.Image, question: str) -> dict:
        img = self.process_image_upload(image_input)
        return self.pipeline.run_vqa(img, question)