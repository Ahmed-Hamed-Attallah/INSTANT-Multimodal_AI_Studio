from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class QAInferenceEngine:
    def __init__(self, model_name: str = "google/flan-t5-base", device: Optional[str] = None):
        self.device = device or self._detect_device()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception:
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

    def answer_question(self, question: str, context: str) -> dict:
        if not context.strip():
            return {"status": "error", "error": "No context transcript provided."}
        if not question.strip():
            return {"status": "error", "error": "Question is empty."}

        try:
            # Prompt engineered for accurate Q&A from context
            prompt = (
                f"Answer the question based only on the following context.\n\n"
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.1,
                    do_sample=False
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            if not answer:
                answer = "I could not find an answer in the transcript."

            return {"status": "success", "answer": answer}
        except Exception as e:
            return {"status": "error", "error": str(e)}