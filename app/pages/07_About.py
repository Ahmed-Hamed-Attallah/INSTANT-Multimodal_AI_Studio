import streamlit as st
from app.components.sidebar import render_sidebar

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
render_sidebar()

st.markdown("### ℹ️ About Multimodal AI Studio")
st.markdown(
    "A modular, full-stack platform designed for multimodal AI experimentation, "
    "document intelligence, and interactive media processing. Developed as part of the "
    "**Advanced AI Diploma at INSTANT**."
)
st.divider()

st.markdown("#### 🏗️ Architecture Overview")
st.markdown("The studio is built on a robust, decoupled architecture leveraging state-of-the-art open-source models:")

# Modern grid layout for architecture components
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("##### 👁️ Vision Backbone")
        st.markdown("- **Image Captioning**: Salesforce BLIP")
        st.markdown("- **Visual QA**: ViLT / BLIP-VQA")
        st.markdown("- **Processing**: PIL, TorchVision")

with col2:
    with st.container(border=True):
        st.markdown("##### 🎙️ Audio & Language Backbone")
        st.markdown("- **Speech-to-Text**: OpenAI Whisper")
        st.markdown("- **Summarization**: BART Large CNN / Seq2Seq")
        st.markdown("- **Processing**: Librosa, PyTorch Audio")

col3, col4 = st.columns(2)
with col3:
    with st.container(border=True):
        st.markdown("##### 📄 OCR & Document Intelligence")
        st.markdown("- **Text Extraction**: EasyOCR / Tesseract")
        st.markdown("- **Layout Analysis**: PyPDFium2 / LayoutLM")
        
with col4:
    with st.container(border=True):
        st.markdown("##### 💻 Frontend & Orchestration")
        st.markdown("- **UI Framework**: Streamlit Multi-Page")
        st.markdown("- **State Management**: Streamlit Session State")
        st.markdown("- **Evaluation**: Custom WER, CER, ROUGE, F1 metrics")

st.divider()

st.markdown("#### 👤 Author & Contact")
st.markdown(
    "**Ahmed Hamed Attallah**  \n"
    "Advanced AI Diploma Candidate, INSTANT  \n"
    "📧 ahmedhamedattallah@gmail.com | "
    "💼 [LinkedIn](https://www.linkedin.com/in/ahmed-hamed-attallah/) | "
    "💻 [GitHub](https://github.com/ahmed-hamed-attallah)"
)