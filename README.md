# 🚀 INSTANT-Multimodal_AI_Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A modular, full-stack platform for multimodal AI experimentation, document intelligence, and interactive media processing.**  
Developed as a capstone Diploma project for **Advanced AI at INSTANT**, this studio integrates vision-language models, audio processing, text analytics, and rigorous evaluation pipelines to streamline end-to-end multimodal workflows.

---

## 👤 Author & Contact

**Ahmed Hamed Attallah**  
Advanced AI Diploma Candidate, INSTANT  
- 📧 **Email**: [ahmedhamedattallah@gmail.com](mailto:ahmedhamedattallah@gmail.com)  
- 💼 **LinkedIn**: [linkedin.com/in/ahmed-hamed-attallah](https://www.linkedin.com/in/ahmed-hamed-attallah/)  
- 💻 **GitHub**: [github.com/ahmed-hamed-attallah](https://github.com/ahmed-hamed-attallah)  

---

## 🌟 Executive Summary

The **INSTANT-Multimodal_AI_Studio** is designed to bridge the gap between experimental AI research and production-ready applications. It provides a unified environment to ingest, preprocess, infer, and evaluate multimodal data (Image, Audio, Text, Documents). 

Built with a strong emphasis on **reproducible machine learning workflows**, the architecture separates concerns into distinct modules: preprocessing, modeling, pipelines, inference, and evaluation. This ensures that every stage—from feature engineering and imputation to clustering-based embedding analysis and driver analysis of model metrics—is transparent, modular, and scalable.

---

## 🔑 Key Features

- **🧠 Multimodal Ingestion & Processing**: Seamlessly parse and process text, audio (WAV/MP3), images (PNG/JPG), and complex documents (PDFs with layout analysis).
- **🔍 RAG & Vector Search**: Embedded retrieval-augmented generation pipeline supporting hybrid search across structured metadata and vector embeddings (ChromaDB/Qdrant).
- **📊 Interactive Studio UI & Dashboards**: Multi-page interactive dashboard (Streamlit/FastAPI) for testing prompts, comparing model outputs, and visualizing intermediate embeddings and performance metrics via pivot-based views.
- **⚙️ Extensible Pipeline Registry**: Modular design allowing easy integration of custom agentic workflows, external APIs, and local Python tools.
- **📈 Rigorous Model Evaluation**: Dedicated evaluation modules tracking industry-standard metrics (BLEU, ROUGE, WER, CER, F1-Score) with support for correlation and driver analysis to link operational metrics with business outcomes.
- **🚀 Streaming & Low-Latency API**: Asynchronous FastAPI backend for real-time model completions and low-latency interaction.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           INTERACTIVE UI & API LAYER                                │
│  ┌──────────────────────────────┐      WebSocket / REST      ┌──────────────────┐   │
│  │   Streamlit Studio UI        │ <──────────────────────>   │  FastAPI Backend │   │
│  │ (Pages: Captioning, VQA, OCR)│                            │ (Routes & Schemas)│   │
│  └──────────────────────────────┘                            └────────┬─────────┘   │
└───────────────────────────────────────────────────────────────────────┼─────────────┘
                                                                        │
┌───────────────────────────────────────────────────────────────────────┼─────────────┐
│                           CORE ORCHESTRATION LAYER                    │             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────┴──────────┐  │
│  │  Vision Pipeline     │  │  Audio Pipeline      │  │  OCR & Doc Pipeline      │  │
│  │ (BLIP, ViT, CLIP)    │  │ (Whisper, VAD)       │  │ (LayoutLM, Tesseract)     │  │
│  └──────────┬───────────┘  └──────────┬───────────┘  └───────────────┬──────────┘  │
│             │                         │                              │             │
│  ┌──────────┴───────────┐  ┌──────────┴───────────┐  ┌───────────────┴──────────┐  │
│  │  Preprocessing Hub   │  │  Inference Engine    │  │  Evaluation & Analytics  │  │
│  │ (Resize, Normalize,  │  │ (Batched, Async,     │  │ (WER, CER, BLEU, Driver  │  │
│  │  Resample, Clean)    │  │  Quantized)          │  │  Analysis, Clustering)   │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                                        │
┌───────────────────────────────────────────────────────────────────────┼─────────────┐
│                           DATA & STORAGE LAYER                        │             │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────┴──────────┐  │
│  │  Local/Cloud Data    │  │  Artifact Registry   │  │  Vector Database         │  │
│  │  (Raw, Processed)    │  │  (Models, Tokenizers)│  │  (ChromaDB / Qdrant)     │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technologies & Libraries |
| :--- | :--- |
| **Frontend / UI** | Streamlit (Multi-page app), React/Next.js (Optional external UI), Tailwind CSS |
| **Backend / API** | Python 3.11+, FastAPI, Uvicorn, Pydantic, WebSockets |
| **AI / Models** | Hugging Face `transformers`, `diffusers`, OpenAI Whisper, BLIP/BLIP-2, LayoutLMv3, Tesseract OCR |
| **Data & Vector** | Pandas, NumPy, ChromaDB / Qdrant, LangChain / LlamaIndex |
| **Evaluation** | NLTK, JiWER, ROUGE, Custom clustering/driver analysis scripts |
| **DevOps / MLOps** | Docker, Docker Compose, Hydra/OmegaConf (`configs/`), `pytest` |

---

## 📂 Project Structure Deep Dive

The repository is meticulously organized to enforce separation of concerns, reproducibility, and scalability:

```text
INSTANT-Multimodal_AI_Studio/
├── app/                    # Interactive UI Layer (e.g., Streamlit)
│   ├── main.py             # Application entry point
│   ├── config.py           # UI-specific configurations
│   ├── components/         # Reusable UI widgets (sidebar, file_uploader, viewers)
│   ├── pages/              # Multi-page routing (01_Image_Captioning.py to 07_About.py)
│   └── services/           # UI-to-backend service connectors
│
├── models/                 # Core AI Model Definitions & Wrappers
│   ├── vision/             # Captioning, Visual QA, Image Processing
│   ├── audio/              # Whisper model wrapper, transcription logic
│   ├── ocr/                # Text detection, recognition, layout analysis
│   └── text/               # Summarization and text processing models
│
├── pipelines/              # Orchestration of preprocessing -> inference steps
│   ├── vision_pipeline.py  # End-to-end image processing workflow
│   ├── audio_pipeline.py   # Audio loading -> resampling -> transcription
│   └── ocr_pipeline.py     # Document ingestion -> layout -> text extraction
│
├── preprocessing/          # Rigorous data preparation (EDA, cleaning, scaling)
│   ├── image/              # Resize, normalization, utility functions
│   ├── audio/              # Audio loading, resampling, noise reduction
│   ├── text/               # Text cleaning, normalization, tokenization
│   └── documents/          # PDF parsing, page-level processing
│
├── inference/              # Optimized inference engines (batching, async, quantization)
│   ├── vision_inference.py # Optimized vision model forwarding
│   └── ...                 # (audio, ocr, text, qs_inference)
│
├── evaluation/             # Metrics and analytics (Aligns with advanced ML workflows)
│   ├── vision_metrics.py   # e.g., CLIP score, BLEU for captions
│   ├── audio_metrics.py    # e.g., Word Error Rate (WER), Character Error Rate (CER)
│   └── text_metrics.py     # e.g., ROUGE, BERTScore
│
├── data/                   # Data versioning and storage
│   ├── raw/                # Immutable original data
│   ├── processed/          # Cleaned, feature-engineered data ready for modeling
│   └── samples/            # Small datasets for quick testing and CI/CD
│
├── artifacts/              # Saved models, tokenizers, processors, and configs
├── notebooks/              # Jupyter notebooks for EDA, experimentation, and prototyping
├── configs/                # Centralized configuration management (YAML)
├── scripts/                # Utility scripts (download_models.py, prepare_data.py)
├── tests/                  # Unit and integration tests (pytest)
├── api/                    # FastAPI REST & WebSocket endpoints
│   ├── main.py             # FastAPI app instance
│   ├── routes/             # Endpoint definitions (vision, audio, ocr, multimodal)
│   └── schemas/            # Pydantic data validation models
├── logs/                   # Application and inference logging
├── outputs/                # Generated artifacts (captions, transcriptions, summaries)
├── Dockerfile              # Containerization blueprint
├── docker-compose.yml      # Multi-container orchestration
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: `3.10` or higher
- **Package Manager**: `pip` or `poetry`
- **Docker**: (Optional, but highly recommended for consistent environments)
- **API Keys**: OpenAI, Gemini, or Hugging Face tokens (for specific advanced models)

### 1. Clone the Repository
```bash
git clone https://github.com/ahmed-hamed-attallah/INSTANT-Multimodal_AI_Studio.git
cd INSTANT-Multimodal_AI_Studio
```

### 2. Environment Configuration
```bash
cp .env.example .env
```
Edit `.env` with your credentials:
```env
# API Keys
HUGGINGFACE_TOKEN=your_hf_token
OPENAI_API_KEY=your_openai_key

# Vector Store & Storage
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma

# App Configuration
API_PORT=8000
UI_PORT=8501
```

### 3. Local Installation (Virtual Environment)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download required model artifacts (optional script)
python scripts/download_models.py
```

---

## 💻 Running the Application

### Option A: Local Development (Modular)
Run the backend API and the interactive UI in separate terminals:

**1. Start FastAPI Backend:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Access interactive API docs at:* `http://localhost:8000/docs`

**2. Start Interactive UI (Streamlit):**
```bash
streamlit run app/main.py --server.port 8501
```
*Access the Studio Dashboard at:* `http://localhost:8501`

### Option B: Docker Compose (Recommended)
Build and run the entire stack (API, UI, and Vector Database) in isolated containers:
```bash
docker compose up --build
```
To stop and clean up:
```bash
docker compose down
```

---

## 📊 Advanced Analytics & Evaluation

True to advanced machine learning standards, this project goes beyond simple inference. The `evaluation/` and `notebooks/` directories are designed for deep analytical workflows:

1. **Comprehensive Metrics**: Automated calculation of WER/CER for audio, BLEU/ROUGE for text, and layout accuracy for OCR.
2. **Driver Analysis**: Scripts to correlate preprocessing choices (e.g., audio resampling rates, image normalization techniques) with final model performance metrics.
3. **Embedding Clustering**: The `notebooks/06_Multimodal_Integration.ipynb` demonstrates how to extract, cluster, and visualize multimodal embeddings to identify data patterns and edge cases.
4. **Pivot-Based Monitoring**: The `06_Model_Evaluation.py` UI page provides dashboard views to monitor business and operational trends across different model runs.

---

## 🔌 API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/vision/caption` | Generate descriptive captions for uploaded images |
| `POST` | `/api/v1/vision/vqa` | Answer natural language questions about an image |
| `POST` | `/api/v1/audio/transcribe` | Convert speech to text with timestamp alignment |
| `POST` | `/api/v1/ocr/analyze` | Extract text and structural layout from documents (PDF/Image) |
| `POST` | `/api/v1/multimodal/rag` | Query the vector database using combined text/image prompts |
| `GET`  | `/health` | Application and model loading health check |

---

## 🧪 Testing & Quality Assurance

Ensure all pipelines and services are functioning correctly before deployment:
```bash
# Run all unit and integration tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=app --cov=models --cov=pipelines
```

---

## 🗺️ Roadmap & Future Enhancements

- [ ] **Advanced Clustering Integration**: Add unsupervised clustering pipelines to automatically group similar document layouts or audio anomalies for targeted model fine-tuning.
- [ ] **Real-time Streaming UI**: Enhance the frontend to support token-by-token streaming for LLM responses.
- [ ] **Distributed Inference**: Integrate Ray or Celery for horizontal scaling of heavy inference tasks (e.g., batch document processing).
- [ ] **CI/CD Pipeline**: Add GitHub Actions for automated testing, linting, and Docker image building on push.

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. 
1. Fork the Project (`https://github.com/ahmed-hamed-attallah/INSTANT-Multimodal_AI_Studio`)
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

> **Built with ❤️ and advanced AI principles by Ahmed Hamed Attallah for the INSTANT Advanced AI Diploma Program.**
```