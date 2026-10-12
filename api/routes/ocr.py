from fastapi import APIRouter, UploadFile, File
from api.schemas.ocr import OCRResponse, OCRBox

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/extract-text", response_model=OCRResponse)
async def extract_text(file: UploadFile = File(...)):
    # Add OCR engine processing logic here
    return OCRResponse(
        full_text="Sample extracted text",
        detections=[OCRBox(text="Sample", confidence=0.98)]
    )