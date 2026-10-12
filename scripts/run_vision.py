import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Disable cuDNN auto-tuner to prevent engine search errors
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

def run_caption(image_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running on device: {device}")

    model_id = "Salesforce/blip-image-captioning-base"
    print("[*] Loading model...")
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"\n[+] Generated Caption: {caption}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_vision.py <path_to_image>")
        sys.exit(1)
    run_caption(sys.argv[1])