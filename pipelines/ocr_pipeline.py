from pathlib import Path
from typing import Union, Optional
from PIL import Image
import pytesseract


class OCRPipeline:
    def __init__(
        self,
        device: Optional[str] = None,
        tesseract_cmd: Optional[str] = None
    ):
        self.device = device or "cpu"
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image_source: Union[str, Path, Image.Image]) -> dict:
        """Extracts raw text from an image."""
        try:
            image = (
                Image.open(image_source).convert("RGB")
                if isinstance(image_source, (str, Path))
                else image_source.convert("RGB")
            )
            extracted_text = pytesseract.image_to_string(image)
            return {
                "success": True,
                "text": extracted_text.strip()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def extract_layout_data(self, image_source: Union[str, Path, Image.Image]) -> dict:
        """Extracts structured word bounding boxes and layout data."""
        try:
            image = (
                Image.open(image_source).convert("RGB")
                if isinstance(image_source, (str, Path))
                else image_source.convert("RGB")
            )
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            blocks = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                word = data["text"][i].strip()
                if word:
                    blocks.append({
                        "text": word,
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                        "conf": data["conf"][i]
                    })

            return {
                "success": True,
                "blocks": blocks,
                "raw_text": pytesseract.image_to_string(image).strip()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }