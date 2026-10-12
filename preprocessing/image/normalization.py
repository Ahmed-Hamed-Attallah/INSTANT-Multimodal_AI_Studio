import torchvision.transforms as transforms
from PIL import Image
import torch

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def normalize_image(image: Image.Image) -> torch.Tensor:
    """Converts a PIL image to a normalized tensor."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return DEFAULT_TRANSFORM(image)