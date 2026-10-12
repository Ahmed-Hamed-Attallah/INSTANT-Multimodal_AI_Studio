import os
import tempfile
from pathlib import Path
import streamlit as st
from app.services.audio_service import AudioService
from utils.output_saver import save_output

st.markdown("### 🎙️ Speech to Text & Audio Q&A")
st.markdown("Transcribe audio files and interact with the transcript using AI.")
st.divider()

@st.cache_resource
def get_audio_service():
    return AudioService()

try:
    service = get_audio_service()
    device_name = getattr(service.pipeline, "device", None) or getattr(service.pipeline.audio_engine, "device", "cpu")
    hardware_label = "CUDA (GPU) ⚡" if device_name == "cuda" else "CPU 🐌"
    st.info(f"🖥️ **Audio Engine Status:** Running on {hardware_label}")
except Exception as e:
    st.error(f"🚨 Failed to initialize audio service: {e}")
    st.stop()

# Initialize session state for persistent transcript
if "audio_transcript" not in st.session_state:
    st.session_state["audio_transcript"] = ""

SUPPORTED_AUDIO_FORMATS = ["wav", "flac", "mp3", "m4a", "ogg", "aac", "opus", "wma"]

col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown("#### 📤 Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=SUPPORTED_AUDIO_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS).upper()}"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        if st.button("🎙️ Transcribe Audio", type="primary", use_container_width=True):
            with st.spinner("🧠 Transcribing audio (this may take a moment)..."):
                uploaded_file.seek(0)
                suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    result = service.transcribe(tmp_path)
                    if isinstance(result, dict) and result.get("success", False):
                        st.session_state["audio_transcript"] = result.get("transcript", "")
                        
                        # --- SAVE OUTPUT TO DISK ---
                        saved_path = save_output(
                            task_name="speech_to_text",
                            results={"transcript": result.get("transcript", ""), "segments": result.get("segments", [])},
                            input_file_name=uploaded_file.name
                        )
                        st.success("✅ Transcription complete!")
                        st.caption(f"💾 Output saved to: `{saved_path}`")
                    else:
                        err = result.get("error") if isinstance(result, dict) else str(result)
                        st.error(f"🚫 Transcription error: {err}")
                except Exception as e:
                    st.error(f"🚫 Execution error: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

with col_output:
    st.markdown("#### 📜 Transcript & Q&A")
    # Display transcript and Q&A if transcript is saved
    if st.session_state["audio_transcript"]:
        with st.container(border=True):
            st.markdown("**Transcription:**")
            st.write(st.session_state["audio_transcript"])
        
        st.divider()
        
        st.markdown("**💬 Ask a Question About the Transcript**")
        question = st.text_input(
            "Your question:", 
            placeholder="e.g., What was the main topic discussed?",
            key="audio_qa_input"
        )
        
        if st.button("🔍 Submit Question", type="secondary", use_container_width=True):
            if question.strip():
                with st.spinner("🧠 Finding answer in transcript..."):
                    qa_res = service.ask_transcript(question, st.session_state["audio_transcript"])
                    if qa_res.get("success"):
                        st.success(f"**Answer:** {qa_res.get('answer')}")
                    else:
                        st.error(f"🚫 QA Error: {qa_res.get('error')}")
            else:
                st.warning("⚠️ Please type a question first.")
    else:
        st.info("👆 Transcribe an audio file to view the transcript and ask questions.")