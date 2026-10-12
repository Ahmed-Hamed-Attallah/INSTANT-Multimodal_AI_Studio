from pathlib import Path
import librosa
import numpy as np

def load_audio_file(file_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Loads an audio file and resamples it to target_sr (Whisper expects 16kHz mono)."""
    audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
    return audio, sr