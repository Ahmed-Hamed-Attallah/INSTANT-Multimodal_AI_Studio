import numpy as np

def pad_or_trim(audio: np.ndarray, target_length: int = 16000 * 30) -> np.ndarray:
    """Pads or trims audio to fit fixed windows (default 30 seconds for Whisper)."""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    return audio

def get_duration(audio: np.ndarray, sr: int = 16000) -> float:
    return len(audio) / sr