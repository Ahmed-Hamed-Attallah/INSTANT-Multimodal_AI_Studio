import sys
import torch
from transformers import pipeline

def run_audio(audio_path: str):
    device = 0 if torch.cuda.is_available() else -1
    print(f"[*] Loading Whisper on {'GPU' if device == 0 else 'CPU'}...")

    # Speech to text
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=device
    )
    result = transcriber(audio_path)
    text = result["text"]
    print(f"\n[+] Transcription:\n{text}")

    # Audio Summarization
    if len(text.split()) > 25:
        print("\n[*] Summarizing transcript...")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
        summary = summarizer(text, max_length=60, min_length=15, do_sample=False)
        print(f"[+] Summary:\n{summary[0]['summary_text']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_audio.py <path_to_audio_file>")
        sys.exit(1)
    run_audio(sys.argv[1])