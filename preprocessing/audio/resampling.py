import librosa
import numpy as np

def resample_waveform(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resamples audio array to target sample rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)