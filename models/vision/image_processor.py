from PIL import Image

class VisionProcessor:
    @staticmethod
    def prepare_image(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            return image.convert("RGB")
        return image