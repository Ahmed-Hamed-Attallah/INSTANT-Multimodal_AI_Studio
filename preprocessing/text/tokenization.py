def truncate_words(text: str, max_words: int = 512) -> str:
    """Truncates text by approximate word count to avoid transformer context overflow."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text