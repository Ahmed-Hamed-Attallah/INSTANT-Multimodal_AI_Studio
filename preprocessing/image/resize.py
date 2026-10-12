from PIL import Image

def resize_with_pad(
        image: Image.Image,
        target_size: tuple = (384, 384)) -> Image.Image:
    """Resizes an image preserving aspect ratio and pads to target size."""
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    paste_position = (
        (target_size[0] - image.size[0]) // 2,
        (target_size[1] - image.size[1]) // 2
    )
    new_image.paste(image, paste_position)
    return new_image