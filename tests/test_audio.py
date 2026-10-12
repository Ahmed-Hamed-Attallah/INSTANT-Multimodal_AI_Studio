import tempfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import pytest

from inference.audio_inference import AudioInferenceEngine


@pytest.fixture(scope="module")
def sample_wav_file():
    """Generates a 1-second 16kHz sine-wave audio WAV file."""
    samplerate = 16000
    duration = 1.0  # seconds
    frequency = 440.0  # A4 note
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wavfile.write(tmp.name, samplerate, waveform)
        tmp_path = tmp.name

    yield tmp_path

    path_obj = Path(tmp_path)
    if path_obj.exists():
        path_obj.unlink()


def test_audio_engine_initialization():
    engine = AudioInferenceEngine(model_size="tiny", device="cpu")
    assert engine is not None
    assert engine.device == "cpu"


def test_audio_transcription(sample_wav_file):
    engine = AudioInferenceEngine(model_size="tiny", device="cpu")
    result = engine.transcribe(sample_wav_file)

    assert isinstance(result, dict)
    assert result.get("status") == "success"
    assert "text" in result
    assert "segments" in result