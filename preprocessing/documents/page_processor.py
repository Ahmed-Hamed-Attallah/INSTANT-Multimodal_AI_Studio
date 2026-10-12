from PIL import Image, ImageOps, ImageFilter
import numpy as np

def enhance_page_for_ocr(image: Image.Image, grayscale: bool = True, contrast_stretch: bool = True) -> Image.Image:
    """
    Enhances scanned document image quality to improve downstream OCR accuracy.
    """
    if grayscale and image.mode != "L":
        image = image.convert("L")
    
    if contrast_stretch:
        # Autocontrast maximizes image contrast
        image = ImageOps.autocontrast(image, cutoff=2)
        
    return image

def binarize_page(image: Image.Image, threshold: int = 180) -> Image.Image:
    """
    Converts a grayscale page image to pure binary (black & white).
    """
    if image.mode != "L":
        image = image.convert("L")
    return image.point(lambda p: 255 if p > threshold else 0, mode="1")

def split_page_vertically(image: Image.Image, num_splits: int = 2) -> list[Image.Image]:
    """
    Splits multi-column document pages into vertical strips (useful for column-wise reading layouts).
    """
    width, height = image.size
    split_width = width // num_splits
    slices = []
    
    for i in range(num_splits):
        box = (i * split_width, 0, (i + 1) * split_width if i < num_splits - 1 else width, height)
        slices.append(image.crop(box))
        
    return slices

def resize_page(image: Image.Image, max_dim: int = 2000) -> Image.Image:
    """
    Constrains huge scanned page resolutions while preserving aspect ratio
    to prevent out-of-memory errors in OCR and layout analysis.
    """
    width, height = image.size
    if max(width, height) <= max_dim:
        return image
        
    if width > height:
        new_w = max_dim
        new_h = int((height / width) * max_dim)
    else:
        new_h = max_dim
        new_w = int((width / height) * max_dim)
        
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)