import sys
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

def run_vqa(image_path: str, question: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    print(f"\n[?] Question: {question}")
    print(f"[+] Answer: {answer}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_multimodal.py <path_to_image> \"<question>\"")
        sys.exit(1)
    run_vqa(sys.argv[1], sys.argv[2])