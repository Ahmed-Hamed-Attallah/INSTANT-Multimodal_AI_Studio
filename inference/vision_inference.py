from pathlib import Path
from typing import Union, Optional
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)


class VisionInferenceEngine:
    def __init__(self, device: Optional[str] = None):
        self.device = device or self._detect_device()
        self._caption_model = None
        self._caption_processor = None
        self._vqa_model = None
        self._vqa_processor = None

    @staticmethod
    def _detect_device() -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            _ = torch.zeros(1, device="cuda") + 1
            return "cuda"
        except RuntimeError:
            return "cpu"

    def _load_caption_engine(self):
        if self._caption_model is None:
            model_id = "Salesforce/blip-image-captioning-base"
            self._caption_processor = BlipProcessor.from_pretrained(model_id)
            try:
                self._caption_model = BlipForConditionalGeneration.from_pretrained(model_id).to(self.device)
            except RuntimeError:
                self.device = "cpu"
                self._caption_model = BlipForConditionalGeneration.from_pretrained(model_id).to("cpu")
            self._caption_model.eval()

    def _load_vqa_engine(self):
        if self._vqa_model is None:
            model_id = "Salesforce/blip-vqa-base"
            self._vqa_processor = BlipProcessor.from_pretrained(model_id)
            try:
                self._vqa_model = BlipForQuestionAnswering.from_pretrained(model_id).to(self.device)
            except RuntimeError:
                self.device = "cpu"
                self._vqa_model = BlipForQuestionAnswering.from_pretrained(model_id).to("cpu")
            self._vqa_model.eval()

    def generate_caption(self, image_source: Union[str, Path, Image.Image], max_tokens: int = 50) -> dict:
        try:
            self._load_caption_engine()
            image = (
                Image.open(image_source).convert("RGB")
                if isinstance(image_source, (str, Path))
                else image_source.convert("RGB")
            )
            inputs = self._caption_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self._caption_model.generate(**inputs, max_new_tokens=max_tokens)
            caption = self._caption_processor.decode(out[0], skip_special_tokens=True)
            return {"status": "success", "caption": caption.capitalize()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def answer_question(self, image_source: Union[str, Path, Image.Image], question: str) -> dict:
        try:
            self._load_vqa_engine()
            image = (
                Image.open(image_source).convert("RGB")
                if isinstance(image_source, (str, Path))
                else image_source.convert("RGB")
            )
            inputs = self._vqa_processor(images=image, text=question, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self._vqa_model.generate(**inputs)
            answer = self._vqa_processor.decode(out[0], skip_special_tokens=True)
            return {"status": "success", "answer": answer}
        except Exception as e:
            return {"status": "error", "error": str(e)}