from PIL import Image
from .text_detection import OCRDetector

class TextRecognizer:
    def __init__(self, detector: OCRDetector):
        self.detector = detector

    def extract_full_text(self, image: Image.Image) -> str:
        results = self.detector.detect_and_recognize(image)
        return "\n".join([item["text"] for item in results])