import torch
import whisper

class WhisperEngine:
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.device = device
        self.model = self._load_whisper_safe(model_size, self.device)

    def _load_whisper_safe(self, model_size: str, target_device: str):
        try:
            # Try loading directly to chosen device
            return whisper.load_model(model_size, device=target_device)
        except RuntimeError as e:
            if "cudaErrorNoKernelImageForDevice" in str(e) or "CUDA error" in str(e):
                print(f"[!] CUDA kernel mismatch on device '{target_device}'. Switching Whisper model to CPU.")
                self.device = "cpu"
                return whisper.load_model(model_size, device="cpu")
            raise e

    def transcribe(self, audio_path: str, fp16: bool = False) -> dict:
        # fp16 must be False on CPU to avoid PyTorch half-precision CPU errors
        use_fp16 = fp16 if self.device == "cuda" else False
        return self.model.transcribe(audio_path, fp16=use_fp16)