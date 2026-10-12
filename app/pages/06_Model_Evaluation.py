import os
import tempfile
import time
from pathlib import Path
from PIL import Image
import pandas as pd
import streamlit as st
import torch
from app.services.audio_service import AudioService
from app.services.ocr_service import OCRService
from pipelines.vision_pipeline import VisionPipeline
from utils.metrics import (
    compute_wer,
    compute_cer,
    compute_exact_match,
    compute_token_f1,
    compute_rouge_l,
)

st.set_page_config(page_title="Model Evaluation & Benchmarks", layout="wide")

st.markdown("### 📊 Model Performance & Evaluation Benchmark")
st.caption("Measure inference latency, compute engine performance, and quality metrics against ground-truth references.")
st.divider()

# Display engine metadata in a modern KPI row
device_str = "CUDA (GPU) ⚡" if torch.cuda.is_available() else "CPU 🐌"
mem_info = "N/A"
if torch.cuda.is_available():
    try:
        mem_info = f"{torch.cuda.memory_allocated() / (1024**2):.1f} MB"
    except Exception:
        pass

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Active Hardware", device_str)
kpi2.metric("VRAM Allocated", mem_info)
kpi3.metric("Framework", "PyTorch / Streamlit")

st.divider()

task = st.selectbox(
    "🎯 Select Modality / Task to Evaluate",
    [
        "Speech-to-Text (Whisper)",
        "Audio Summarization (Whisper + Seq2Seq)",
        "Image Captioning (BLIP)",
        "Visual Question Answering (BLIP-VQA)",
        "Optical Character Recognition (Tesseract)",
    ],
)

st.markdown("---")

# ----------------- 1. SPEECH-TO-TEXT -----------------
if task == "Speech-to-Text (Whisper)":
    st.subheader("🎙️ Evaluate Whisper ASR")
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        audio_file = st.file_uploader("Upload Test Audio", type=["wav", "mp3", "flac", "ogg", "m4a"])
        ground_truth = st.text_area("Ground Truth Transcript", placeholder="Paste reference text here for WER/CER calculation...", height=150)
        run_btn = st.button("🚀 Run STT Benchmark", type="primary", use_container_width=True)
        
    with col_out:
        if audio_file and run_btn:
            service = AudioService()
            suffix = Path(audio_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            
            try:
                with st.spinner("⏳ Transcribing audio..."):
                    start_time = time.perf_counter()
                    res = service.transcribe(tmp_path)
                    latency = time.perf_counter() - start_time
                
                hypothesis = res.get("transcript", "")
                
                with st.container(border=True):
                    st.markdown("**Model Output:**")
                    st.write(hypothesis or "*(Empty)*")
                
                if ground_truth.strip():
                    wer = compute_wer(ground_truth, hypothesis)
                    cer = compute_cer(ground_truth, hypothesis)
                    
                    st.markdown("##### 📈 Quality Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Word Error Rate (WER)", f"{wer * 100:.2f}%")
                    m2.metric("Character Error Rate (CER)", f"{cer * 100:.2f}%")
                    m3.metric("Latency", f"{latency:.3f}s")
                else:
                    st.info("💡 Provide a Ground Truth Transcript above to calculate WER and CER metrics.")
                    st.metric("Latency", f"{latency:.3f}s")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            st.info("👆 Upload an audio file and click 'Run STT Benchmark' to see results.")

# ----------------- 2. AUDIO SUMMARIZATION -----------------
elif task == "Audio Summarization (Whisper + Seq2Seq)":
    st.subheader("📝 Evaluate Audio Summarization Pipeline")
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        audio_file = st.file_uploader("Upload Audio for Summarization", type=["wav", "mp3", "flac", "ogg", "m4a"])
        reference_summary = st.text_area("Reference Summary", placeholder="Paste reference summary here for ROUGE-L calculation...", height=150)
        run_btn = st.button("🚀 Run Summarization Benchmark", type="primary", use_container_width=True)
        
    with col_out:
        if audio_file and run_btn:
            service = AudioService()
            suffix = Path(audio_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            
            try:
                with st.spinner("⏳ Transcribing and summarizing..."):
                    start_time = time.perf_counter()
                    res = service.summarize_audio(tmp_path)
                    latency = time.perf_counter() - start_time
                
                summary = res.get("summary", "")
                transcript = res.get("transcript", "")
                
                with st.container(border=True):
                    st.markdown("**Generated Summary:**")
                    st.markdown(f"> *{summary}*")
                
                with st.expander("📄 View Intermediate Transcript"):
                    st.write(transcript)
                
                if reference_summary.strip():
                    rouge = compute_rouge_l(reference_summary, summary)
                    f1 = compute_token_f1(reference_summary, summary)
                    
                    st.markdown("##### 📈 Quality Metrics")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("ROUGE-L Score", f"{rouge:.4f}")
                    m2.metric("Token F1", f"{f1:.4f}")
                    m3.metric("End-to-End Latency", f"{latency:.3f}s")
                else:
                    st.info("💡 Provide a Reference Summary above to calculate ROUGE-L and Token F1 metrics.")
                    st.metric("End-to-End Latency", f"{latency:.3f}s")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            st.info("👆 Upload an audio file and click 'Run Summarization Benchmark' to see results.")

# ----------------- 3. IMAGE CAPTIONING -----------------
elif task == "Image Captioning (BLIP)":
    st.subheader("🖼️ Evaluate BLIP Captioning")
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
        ref_caption = st.text_input("Reference Caption", placeholder="Optional: for ROUGE / F1 calculation")
        run_btn = st.button("🚀 Run Captioning Benchmark", type="primary", use_container_width=True)
        
    with col_out:
        if img_file and run_btn:
            pipeline = VisionPipeline()
            image = Image.open(img_file).convert("RGB")
            
            with st.spinner("⏳ Generating caption..."):
                start_time = time.perf_counter()
                res = pipeline.generate_caption(image)
                latency = time.perf_counter() - start_time
            
            caption = res.get("caption", "")
            
            with st.container(border=True):
                st.markdown("**Predicted Caption:**")
                st.markdown(f"**{caption}**")
                st.image(image, caption="Input Image", use_container_width=True)
            
            if ref_caption.strip():
                rouge = compute_rouge_l(ref_caption, caption)
                f1 = compute_token_f1(ref_caption, caption)
                
                st.markdown("##### 📈 Quality Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("ROUGE-L", f"{rouge:.4f}")
                m2.metric("Token F1", f"{f1:.4f}")
                m3.metric("Latency", f"{latency:.3f}s")
            else:
                st.info("💡 Provide a Reference Caption above to calculate ROUGE-L and Token F1 metrics.")
                st.metric("Latency", f"{latency:.3f}s")
        else:
            st.info("👆 Upload an image and click 'Run Captioning Benchmark' to see results.")

# ----------------- 4. VISUAL QA -----------------
elif task == "Visual Question Answering (BLIP-VQA)":
    st.subheader("❓ Evaluate Visual Question Answering")
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
        question = st.text_input("Question", placeholder="e.g., What animal is shown?")
        ref_answer = st.text_input("Expected Ground-Truth Answer", placeholder="Optional: for Exact Match / F1 calculation")
        run_btn = st.button("🚀 Run VQA Benchmark", type="primary", use_container_width=True)
        
    with col_out:
        if img_file and question and run_btn:
            pipeline = VisionPipeline()
            image = Image.open(img_file).convert("RGB")
            
            with st.spinner("⏳ Analyzing image and generating answer..."):
                start_time = time.perf_counter()
                res = pipeline.answer_question(image, question)
                latency = time.perf_counter() - start_time
            
            answer = res.get("answer", "")
            
            with st.container(border=True):
                st.markdown("**Predicted Answer:**")
                st.markdown(f"`{answer}`")
                st.image(image, caption="Input Image", use_container_width=True)
            
            if ref_answer.strip():
                em = compute_exact_match(ref_answer, answer)
                f1 = compute_token_f1(ref_answer, answer)
                
                st.markdown("##### 📈 Quality Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Exact Match (EM)", f"{int(em)}")
                m2.metric("Token F1", f"{f1:.4f}")
                m3.metric("Latency", f"{latency:.3f}s")
            else:
                st.info("💡 Provide an Expected Ground-Truth Answer above to calculate Exact Match and Token F1 metrics.")
                st.metric("Latency", f"{latency:.3f}s")
        elif img_file and not question:
            st.warning("⚠️ Please enter a question to evaluate.")
        else:
            st.info("👆 Upload an image, enter a question, and click 'Run VQA Benchmark' to see results.")

# ----------------- 5. OCR -----------------
elif task == "Optical Character Recognition (Tesseract)":
    st.subheader("📄 Evaluate OCR Layout & Text Extraction")
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        img_file = st.file_uploader("Upload Document / Image", type=["png", "jpg", "jpeg"])
        ground_truth_ocr = st.text_area("Ground Truth Text", placeholder="Paste reference text here for WER/CER calculation...", height=150)
        run_btn = st.button("🚀 Run OCR Benchmark", type="primary", use_container_width=True)
        
    with col_out:
        if img_file and run_btn:
            service = OCRService()
            image = Image.open(img_file).convert("RGB")
            
            with st.spinner("⏳ Extracting text and analyzing layout..."):
                start_time = time.perf_counter()
                res = service.extract_text(image)
                latency = time.perf_counter() - start_time
            
            pred_text = res.get("text", "")
            
            with st.container(border=True):
                st.markdown("**Extracted OCR Text:**")
                st.text_area("OCR Output", pred_text, height=180, label_visibility="collapsed")
            
            if ground_truth_ocr.strip():
                wer = compute_wer(ground_truth_ocr, pred_text)
                cer = compute_cer(ground_truth_ocr, pred_text)
                
                st.markdown("##### 📈 Quality Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Word Error Rate (WER)", f"{wer * 100:.2f}%")
                m2.metric("Character Error Rate (CER)", f"{cer * 100:.2f}%")
                m3.metric("Latency", f"{latency:.3f}s")
            else:
                st.info("💡 Provide Ground Truth Text above to calculate WER and CER metrics.")
                st.metric("Latency", f"{latency:.3f}s")
        else:
            st.info("👆 Upload a document/image and click 'Run OCR Benchmark' to see results.")