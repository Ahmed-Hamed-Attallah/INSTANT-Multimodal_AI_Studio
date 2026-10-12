from pathlib import Path
from PIL import Image
import io

def load_image(source: str | Path | bytes | io.BytesIO) -> Image.Image:
    """Loads an image from a path or byte stream into RGB format."""
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    else:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        img = Image.open(source)
    return img.convert("RGB")