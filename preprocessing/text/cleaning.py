import re

def clean_text(text: str) -> str:
    """Cleans raw text for downstream models."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove unwanted control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", text)
    return text