import os
import sys
from pathlib import Path
import torch
import streamlit as st

# 1. Fix ModuleNotFoundError: Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Fix cuDNN "GET was unable to find an engine to execute this computation"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe import for sidebar
try:
    from app.components.sidebar import render_sidebar
    render_sidebar()
except ImportError:
    st.sidebar.title("🤖 Multimodal AI Studio")
    st.sidebar.info("Sidebar component not found. Using default navigation.")

# Hardware Engine Status
device_name = "CUDA (GPU) ⚡" if torch.cuda.is_available() else "CPU 🐌"
vram_status = "N/A"
if torch.cuda.is_available():
    try:
        vram_status = f"{torch.cuda.memory_allocated() / (1024**2):.1f} MB"
    except Exception:
        pass

# Modern Sidebar Status Panel
st.sidebar.divider()
st.sidebar.markdown("##### 🖥️ System Status")
st.sidebar.metric("Active Engine", device_name)
if torch.cuda.is_available():
    st.sidebar.metric("VRAM Allocated", vram_status)

# --- MAIN UI ---
st.markdown("# 🤖 Welcome to Multimodal AI Studio")
st.markdown(
    "A modular, full-stack platform for multimodal AI experimentation, "
    "document intelligence, and interactive media processing."
)
st.divider()

# System KPIs Row
k1, k2, k3 = st.columns(3)
k1.metric("🖥️ Active Hardware", device_name)
k2.metric("💾 VRAM Allocated", vram_status)
k3.metric("🏗️ Framework", "Streamlit / PyTorch")

st.divider()

# Navigation Guide & Feature Cards
st.markdown("#### 🚀 Getting Started")
st.markdown("Select a module from the **left navigation sidebar** to begin exploring our multimodal capabilities.")

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("##### 👁️ Vision Models")
        st.markdown("Generate captions and answer visual questions using state-of-the-art BLIP architectures.")
with c2:
    with st.container(border=True):
        st.markdown("##### 🎙️ Audio Intelligence")
        st.markdown("Transcribe speech, summarize long audio files, and interact with transcripts via Q&A.")
with c3:
    with st.container(border=True):
        st.markdown("##### 📄 Document OCR")
        st.markdown("Extract text and analyze structural layouts from complex documents and images.")

st.divider()
st.caption("Developed for the Advanced AI Diploma at INSTANT | Built by Ahmed Hamed Attallah")