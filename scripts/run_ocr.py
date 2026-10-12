import sys
from PIL import Image
import pytesseract
# Or if you use easyocr:
# import easyocr

def run_ocr(image_path: str):
    print(f"[*] Extracting text from {image_path}...")
    try:
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)
        print(f"\n[+] Extracted OCR Text:\n{extracted_text.strip()}\n")
    except Exception as e:
        print(f"[-] OCR Error: {e}")
        print("Hint: Install tesseract-ocr system binary (`sudo apt install tesseract-ocr`) or use easyocr.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_ocr.py <path_to_image>")
        sys.exit(1)
    run_ocr(sys.argv[1])