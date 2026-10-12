import os
import tempfile
from pathlib import Path
from typing import Union
import torch
from pipelines.audio_pipeline import AudioPipeline

def get_safe_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        x = torch.zeros(1, device="cuda")
        _ = x + 1
        return "cuda"
    except RuntimeError:
        return "cpu"

class AudioService:
    def __init__(self, device: str = None):
        target_device = device or get_safe_device()
        try:
            self.pipeline = AudioPipeline(device=target_device)
        except RuntimeError:
            self.pipeline = AudioPipeline(device="cpu")

    def transcribe(self, audio_source: Union[str, Path, bytes], suffix: str = ".wav", language: str = None) -> dict:
        if isinstance(audio_source, bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_source)
                tmp_path = tmp.name
            try:
                return self.pipeline.transcribe(tmp_path, language=language)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return self.pipeline.transcribe(audio_source, language=language)

    def summarize_audio(
        self,
        audio_source: Union[str, Path, bytes],
        suffix: str = ".wav",
        max_summary_words: int = 100
    ) -> dict:
        if isinstance(audio_source, bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_source)
                tmp_path = tmp.name
            try:
                return self.pipeline.transcribe_and_summarize(
                    tmp_path, max_summary_words=max_summary_words
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return self.pipeline.transcribe_and_summarize(
            audio_source, max_summary_words=max_summary_words
        )

    def summarize(self, text: str, max_length: int = 100) -> dict:
        return self.pipeline.text_engine.summarize_text(text, max_length=max_length)

    def ask_transcript(self, question: str, transcript: str) -> dict:
        return self.pipeline.ask_transcript(question, transcript)