import easyocr
import numpy as np
from PIL import Image

class OCRDetector:
    def __init__(self, languages: list[str] = ["en"], gpu: bool = False):
        self.reader = easyocr.Reader(languages, gpu=gpu)

    def detect_and_recognize(self, image: Image.Image | np.ndarray) -> list[dict]:
        """
        Returns structured detection output:
        [{"box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], "text": str, "confidence": float}]
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        raw_results = self.reader.readtext(image)
        formatted = []
        for box, text, conf in raw_results:
            formatted.append({
                "box": box,
                "text": text,
                "confidence": float(conf)
            })
        return formatted