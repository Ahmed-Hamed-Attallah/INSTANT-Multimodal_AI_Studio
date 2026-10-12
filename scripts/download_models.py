import os
import whisper
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

def prefetch_models():
    print("Downloading BLIP Captioning model...")
    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    print("Downloading Whisper model...")
    whisper.load_model("base")

    print("Downloading BART Summarizer model...")
    AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    print("All base models downloaded successfully.")

if __name__ == "__main__":
    prefetch_models()