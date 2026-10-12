import string

def normalize_for_eval(text: str) -> str:
    """Prepares text for WER/BLEU/ROUGE evaluation."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text