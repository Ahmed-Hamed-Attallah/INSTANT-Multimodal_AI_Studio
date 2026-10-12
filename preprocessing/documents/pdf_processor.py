from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium

def pdf_to_images(pdf_path: str | Path, dpi: int = 150) -> list[Image.Image]:
    """Converts pages of a PDF document to a list of PIL Images."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    images = []
    # 72 points per inch default, calculate scale factor
    scale = dpi / 72.0
    for page in pdf:
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        images.append(pil_image.convert("RGB"))
    return images