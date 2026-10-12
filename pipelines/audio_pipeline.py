from pathlib import Path
from typing import Union, Optional
import numpy as np

from inference.audio_inference import AudioInferenceEngine
from inference.text_inference import TextInferenceEngine
from inference.qa_inference import QAInferenceEngine


class AudioPipeline:
    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.audio_engine = AudioInferenceEngine(device=device) if device else AudioInferenceEngine()
        self.text_engine = TextInferenceEngine(device=device) if device else TextInferenceEngine()
        self.qa_engine = QAInferenceEngine(device=device) if device else QAInferenceEngine()

    def transcribe(
        self,
        audio_source: Union[str, Path, np.ndarray],
        language: Optional[str] = None
    ) -> dict:
        source = str(audio_source) if isinstance(audio_source, Path) else audio_source
        result = self.audio_engine.transcribe(source, language=language)
        if result.get("status") == "success":
            return {
                "success": True,
                "transcript": result.get("text", "").strip(),
                "segments": result.get("segments", [])
            }
        return {
            "success": False,
            "error": result.get("error", "Transcription error occurred.")
        }

    def transcribe_and_summarize(
        self,
        audio_source: Union[str, Path, np.ndarray],
        max_summary_words: int = 100
    ) -> dict:
        source = str(audio_source) if isinstance(audio_source, Path) else audio_source
        transcription_res = self.audio_engine.transcribe(source)

        if transcription_res.get("status") != "success":
            return {
                "success": False,
                "error": transcription_res.get("error", "Transcription failed.")
            }

        raw_text = transcription_res.get("text", "")
        if not raw_text.strip():
            return {
                "success": False,
                "error": "Transcription returned empty audio text."
            }

        summary_res = self.text_engine.summarize_text(raw_text, max_length=max_summary_words)
        if summary_res.get("status") != "success":
            return {
                "success": False,
                "error": summary_res.get("error", "Summarization failed."),
                "transcript": raw_text
            }

        return {
            "success": True,
            "transcript": raw_text,
            "summary": summary_res.get("summary", ""),
            "segments": transcription_res.get("segments", [])
        }

    def ask_transcript(self, question: str, transcript: str) -> dict:
        """Asks a question about a provided transcript."""
        res = self.qa_engine.answer_question(question, transcript)
        if res.get("status") == "success":
            return {"success": True, "answer": res.get("answer", "")}
        return {"success": False, "error": res.get("error", "Failed to retrieve answer.")}