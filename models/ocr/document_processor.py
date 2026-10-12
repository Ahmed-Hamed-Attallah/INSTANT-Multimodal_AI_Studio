from PIL import Image
from .text_detection import OCRDetector
from .layout_analysis import LayoutAnalyzer

class DocumentProcessor:
    def __init__(self, detector: OCRDetector):
        self.detector = detector
        self.analyzer = LayoutAnalyzer()

    def process_page(self, page_image: Image.Image) -> dict:
        raw_detections = self.detector.detect_and_recognize(page_image)
        sorted_detections = self.analyzer.sort_by_reading_order(raw_detections)
        full_text = " ".join([d["text"] for d in sorted_detections])
        return {
            "blocks": sorted_detections,
            "full_text": full_text
        }