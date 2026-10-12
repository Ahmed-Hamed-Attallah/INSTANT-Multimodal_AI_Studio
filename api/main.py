from fastapi import FastAPI
from api.routes import audio, vision, ocr, multimodal

app = FastAPI(
    title="Multimodal AI Studio API",
    version="1.0.0"
)

# Register routes
app.include_router(audio.router)
app.include_router(vision.router)
app.include_router(ocr.router)
app.include_router(multimodal.router)

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Multimodal AI Studio"}