import os
import shutil
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SAMPLES_DIR = BASE_DIR / "data" / "samples"

def prepare_text_samples(n_samples: int = 15):
    parquet_path = RAW_DIR / "text" / "test-00000-of-00001.parquet"
    out_dir = SAMPLES_DIR / "text"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        sample_df = df.head(n_samples)
        sample_df.to_json(out_dir / "samples.json", orient="records", indent=2)
        print(f"[Text] Successfully sampled {n_samples} rows to {out_dir / 'samples.json'}")
    else:
        print(f"[Text] Warning: {parquet_path} not found.")

def prepare_image_samples(n_samples: int = 10):
    val_dir = RAW_DIR / "images" / "val2014"
    out_dir = SAMPLES_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if val_dir.exists():
        images = [f for f in val_dir.glob("*.jpg")][:n_samples]
        for img in images:
            shutil.copy(img, out_dir / img.name)
        print(f"[Images] Successfully copied {len(images)} sample images.")
    else:
        print(f"[Images] Warning: {val_dir} not found.")

def prepare_document_samples(n_samples: int = 5):
    doc_img_dir = RAW_DIR / "documents" / "testing_data" / "images"
    doc_ann_dir = RAW_DIR / "documents" / "testing_data" / "annotations"
    
    out_img_dir = SAMPLES_DIR / "documents" / "images"
    out_ann_dir = SAMPLES_DIR / "documents" / "annotations"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)
    
    if doc_img_dir.exists():
        images = list(doc_img_dir.glob("*.*"))[:n_samples]
        for img in images:
            shutil.copy(img, out_img_dir / img.name)
            # Find matching annotation with same base name
            ann_matches = list(doc_ann_dir.glob(f"{img.stem}.*"))
            for ann in ann_matches:
                shutil.copy(ann, out_ann_dir / ann.name)
        print(f"[Documents] Successfully copied {len(images)} document images and annotations.")
    else:
        print(f"[Documents] Warning: {doc_img_dir} not found.")

def prepare_audio_samples(n_samples: int = 5):
    audio_root = RAW_DIR / "audio" / "test-clean"
    out_dir = SAMPLES_DIR / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not audio_root.exists():
        print(f"[Audio] Warning: {audio_root} not found.")
        return

    flac_files = list(audio_root.rglob("*.flac"))[:n_samples]
    metadata = {}

    for flac in flac_files:
        shutil.copy(flac, out_dir / flac.name)
        # Locate corresponding trans.txt in the same directory
        trans_file = list(flac.parent.glob("*.trans.txt"))
        if trans_file:
            with open(trans_file[0], "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2 and parts[0] == flac.stem:
                        metadata[flac.name] = parts[1]
                        break

    with open(out_dir / "transcriptions.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Audio] Successfully copied {len(flac_files)} audio samples and extracted transcripts.")

if __name__ == "__main__":
    print("Preparing project samples from raw datasets...")
    prepare_text_samples()
    prepare_image_samples()
    prepare_document_samples()
    prepare_audio_samples()
    print("Done!")