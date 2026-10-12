from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TextInferenceEngine:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6", device: Optional[str] = None):
        self.device = device or self._detect_device()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception:
            # Fallback to CPU if GPU device allocation or kernel execution fails
            self.device = "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cpu")
            self.model.eval()

    @staticmethod
    def _detect_device() -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            _ = torch.zeros(1, device="cuda") + 1
            return "cuda"
        except RuntimeError:
            return "cpu"

    def summarize_text(self, text: str, max_length: int = 100, min_length: int = 20) -> dict:
        if not text.strip():
            return {"status": "error", "error": "Input text is empty."}
        try:
            word_count = len(text.split())
            if word_count < 25:
                return {"status": "success", "summary": text.strip()}

            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            max_len = min(max_length, max(word_count, min_length + 5))
            min_len = min(min_length, max_len - 5)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_len,
                    min_length=min_len,
                    num_beams=2,
                    early_stopping=True
                )

            summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return {"status": "success", "summary": summary_text.strip()}
        except Exception as e:
            return {"status": "error", "error": str(e)}