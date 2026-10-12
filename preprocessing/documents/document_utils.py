from PIL import Image

def crop_bounding_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """
    Crops a region from an image.
    box format: (xmin, ymin, xmax, ymax)
    """
    return image.crop(box)