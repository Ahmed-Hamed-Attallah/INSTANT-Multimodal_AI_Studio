import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image

# Resolves to project_root/outputs
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def save_output(
    task_name: str,
    results: Dict[str, Any],
    input_file_name: Optional[str] = None,
    image_to_save: Optional[Image.Image] = None
) -> Path:
    """
    Saves inference outputs to the outputs/ directory.
    - task_name: 'captioning', 'vqa', 'speech_to_text', 'audio_summarization', 'ocr'
    - results: Dict containing model outputs, metadata, latency, etc.
    """
    task_dir = OUTPUTS_DIR / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{Path(input_file_name).stem if input_file_name else 'run'}"

    # 1. Save results as JSON
    json_path = task_dir / f"{base_name}_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "input_source": input_file_name,
                **results
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    # 2. Also save clean text copy if text exists
    text_content = results.get("transcript") or results.get("caption") or results.get("summary") or results.get("text")
    if text_content and isinstance(text_content, str):
        txt_path = task_dir / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_content.strip())

    # 3. Optional: save copy of the processed image if provided
    if image_to_save is not None:
        img_path = task_dir / f"{base_name}_input.png"
        image_to_save.save(img_path)

    return json_path