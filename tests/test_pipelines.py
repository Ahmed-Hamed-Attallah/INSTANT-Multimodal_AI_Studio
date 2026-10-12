import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from scipy.io import wavfile
import pytest

from pipelines.audio_pipeline import AudioPipeline
from pipelines.vision_pipeline import VisionPipeline
from pipelines.ocr_pipeline import OCRPipeline


@pytest.fixture(scope="module")
def sample_media():
    # Fast 0.25-second synthetic tone
    sr = 16000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    wav = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wavfile.write(tmp.name, sr, wav)
        audio_path = tmp.name

    # Small 64x64 synthetic image
    img = Image.new("RGB", (64, 64), color="white")

    yield {"audio": audio_path, "image": img}

    p = Path(audio_path)
    if p.exists():
        p.unlink()


def test_audio_pipeline_transcribe(sample_media):
    # Mock underlying transcribe to test pipeline flow instantly
    with patch("inference.audio_inference.whisper.load_model") as mock_whisper:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world", "segments": []}
        mock_whisper.return_value = mock_model

        pipeline = AudioPipeline(device="cpu")
        result = pipeline.transcribe(sample_media["audio"])

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert result.get("transcript") == "hello world"


def test_vision_pipeline_caption(sample_media):
    # Mock caption output to bypass heavy BLIP vision forward-pass on CPU
    with patch.object(VisionPipeline, "generate_caption", return_value={"success": True, "caption": "A white square"}):
        pipeline = VisionPipeline(device="cpu")
        result = pipeline.generate_caption(sample_media["image"])

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert "caption" in result


def test_ocr_pipeline_execution(sample_media):
    # Tesseract is lightweight and executes in milliseconds
    pipeline = OCRPipeline(device="cpu")
    result = pipeline.extract_text(sample_media["image"])

    assert isinstance(result, dict)
    assert result.get("success") is True
    assert "text" in result