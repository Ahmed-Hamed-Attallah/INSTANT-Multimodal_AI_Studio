import numpy as np

class AudioProcessor:
    @staticmethod
    def ensure_float32(audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            return audio.astype(np.float32)
        return audio