from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
import whisper


class AudioInferenceEngine:
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        self.device = device or self._detect_device()
        self.model = self._load_model_safe(model_size)

    @staticmethod
    def _detect_device() -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            _ = torch.zeros(1, device="cuda") + 1
            return "cuda"
        except RuntimeError:
            return "cpu"

    def _load_model_safe(self, model_size: str):
        try:
            return whisper.load_model(model_size, device=self.device)
        except RuntimeError as e:
            if "cudaErrorNoKernelImageForDevice" in str(e) or "CUDA error" in str(e):
                self.device = "cpu"
                return whisper.load_model(model_size, device="cpu")
            raise e

    def transcribe(
        self,
        audio_source: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        fp16: bool = False
    ) -> dict:
        try:
            use_fp16 = fp16 if self.device == "cuda" else False
            kwargs = {"fp16": use_fp16}
            if language:
                kwargs["language"] = language

            result = self.model.transcribe(audio_source, **kwargs)
            return {
                "status": "success",
                "text": result.get("text", "").strip(),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "text": "",
                "segments": []
            }