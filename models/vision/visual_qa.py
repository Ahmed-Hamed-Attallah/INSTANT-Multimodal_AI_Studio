import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering

class VisualQAModel:
    def __init__(self, model_id: str = "dandelin/vilt-b32-finetuned-vqa", device: str = "cpu"):
        self.device = torch.device(device)
        self.processor = ViltProcessor.from_pretrained(model_id)
        self.model = ViltForQuestionAnswering.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def answer_question(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
        return self.model.config.id2label[idx]