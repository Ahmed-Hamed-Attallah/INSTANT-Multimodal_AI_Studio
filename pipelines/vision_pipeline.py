from pathlib import Path
from typing import Union, Optional
from PIL import Image
from inference.vision_inference import VisionInferenceEngine


class VisionPipeline:
    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.engine = VisionInferenceEngine(device=self.device)

    def generate_caption(self, image_source: Union[str, Path, Image.Image], max_tokens: int = 50) -> dict:
        res = self.engine.generate_caption(image_source, max_tokens=max_tokens)
        if res.get("status") == "success":
            return {"success": True, "caption": res.get("caption", "")}
        return {"success": False, "error": res.get("error", "Unknown error")}

    def answer_question(self, image_source: Union[str, Path, Image.Image], question: str) -> dict:
        res = self.engine.answer_question(image_source, question)
        if res.get("status") == "success":
            return {"success": True, "answer": res.get("answer", "")}
        return {"success": False, "error": res.get("error", "Unknown error")}