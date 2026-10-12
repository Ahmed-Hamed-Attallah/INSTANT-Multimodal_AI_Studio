from pydantic import BaseModel
from typing import Optional

class AudioTranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = "en"
    duration_seconds: Optional[float] = None