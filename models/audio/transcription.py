from pathlib import Path
import numpy as np
from .whisper_model import WhisperEngine

class AudioTranscriber:
    def __init__(self, engine: WhisperEngine):
        self.engine = engine

    def transcribe(self, audio: str | Path | np.ndarray, language: str | None = None) -> dict:
        """
        Transcribes audio. Accepts a file path or a 16kHz float32 numpy array.
        Returns a dict containing 'text' and 'segments'.
        """
        options = {}
        if language:
            options["language"] = language
            
        result = self.engine.model.transcribe(audio, **options)
        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", [])
        }