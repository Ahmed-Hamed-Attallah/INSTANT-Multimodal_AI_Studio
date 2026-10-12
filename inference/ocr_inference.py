from pathlib import Path
from PIL import Image
from app.config import DEVICE
from preprocessing.documents.pdf_processor import pdf_to_images
from preprocessing.documents.page_processor import enhance_page_for_ocr
from preprocessing.image.image_utils import load_image
from models.ocr.text_detection import OCRDetector
from models.ocr.document_processor import DocumentProcessor

class OCRInferenceEngine:
    def __init__(self, device: str = DEVICE):
        use_gpu = (device == "cuda")
        self.detector = OCRDetector(languages=["en"], gpu=use_gpu)
        self.processor = DocumentProcessor(self.detector)

    def extract_from_image(self, image_input: str | Path | Image.Image, preprocess: bool = True) -> dict:
        try:
            image = load_image(image_input) if not isinstance(image_input, Image.Image) else image_input
            if preprocess:
                image = enhance_page_for_ocr(image)

            doc_result = self.processor.process_page(image)
            return {
                "status": "success",
                "full_text": doc_result["full_text"],
                "blocks": doc_result["blocks"]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def extract_from_pdf(self, pdf_path: str | Path) -> dict:
        try:
            pages = pdf_to_images(pdf_path)
            all_text = []
            page_results = []

            for idx, page in enumerate(pages):
                page_data = self.processor.process_page(page)
                page_results.append({"page_num": idx + 1, "blocks": page_data["blocks"]})
                all_text.append(page_data["full_text"])

            return {
                "status": "success",
                "total_pages": len(pages),
                "full_text": "\n\n".join(all_text),
                "pages": page_results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}