from fastapi import APIRouter, UploadFile, File
from api.schemas.audio import AudioTranscriptionResponse

router = APIRouter(prefix="/audio", tags=["Audio"])

@router.post("/transcribe", response_model=AudioTranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    # Add speech-to-text model logic here
    return AudioTranscriptionResponse(
        text="Sample transcribed speech.",
        language="en",
        duration_seconds=3.2
    )