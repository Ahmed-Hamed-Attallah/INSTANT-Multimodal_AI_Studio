from pydantic import BaseModel
from typing import List

class OCRBox(BaseModel):
    text: str
    confidence: float

class OCRResponse(BaseModel):
    full_text: str
    detections: List[OCRBox] = []