from pydantic import BaseModel, Field
from typing import List, Optional

class VisionRequest(BaseModel):
    image_url: Optional[str] = None
    prompt: Optional[str] = Field(default="Describe this image")

class VisionResponse(BaseModel):
    description: str
    tags: List[str] = []