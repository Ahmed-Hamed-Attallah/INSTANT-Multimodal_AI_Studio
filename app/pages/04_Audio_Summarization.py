import os
import streamlit as st
from app.services.audio_service import AudioService
import sys
from pathlib import Path

# Add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Now import your utility
from utils.output_saver import save_output

st.markdown("### 📝 Audio Summarization")
st.markdown("Transcribe and condense long audio files into concise, actionable summaries.")
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

col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown("#### 📤 Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose an audio file to summarize", 
        type=["mp3", "wav", "m4a", "ogg"],
        help="Supported formats: MP3, WAV, M4A, OGG"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        st.markdown("#### ⚙️ Summary Settings")
        max_words = st.slider(
            "Max Summary Length (words)", 
            min_value=30, 
            max_value=250, 
            value=100,
            help="Controls the verbosity of the generated summary."
        )
        
        if st.button("📝 Summarize Audio", type="primary", use_container_width=True):
            with st.spinner("🧠 Transcribing and summarizing speech..."):
                uploaded_file.seek(0)
                data = uploaded_file.read()
                suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
                
                try:
                    res = service.summarize_audio(data, suffix=suffix, max_summary_words=max_words)
                    if res.get("success", False):
                        st.session_state["summary_result"] = res
                        st.success("✅ Summarization complete!")
                        
                        # --- SAVE OUTPUT TO DISK ---
                        # Note: Fixed variable references from original snippet to prevent NameError
                        saved_path = save_output(
                            task_name="speech_to_text",
                            results={"transcript": res.get("transcript", ""), "segments": res.get("segments", [])},
                            input_file_name=uploaded_file.name
                        )
                        st.caption(f"💾 Output saved to: `{saved_path}`")
                    else:
                        st.error(f"🚫 Processing Error: {res.get('error', 'Unknown error occurred.')}")
                except Exception as e:
                    st.error(f"🚫 Execution Error: {e}")

with col_output:
    st.markdown("#### 📜 Results")
    if "summary_result" in st.session_state and st.session_state["summary_result"].get("success", False):
        res = st.session_state["summary_result"]
        
        with st.container(border=True):
            st.markdown("**Summary:**")
            st.write(res.get("summary", "No summary produced."))
            
        with st.expander("📄 View Full Transcript", expanded=False):
            st.text(res.get("transcript", ""))
    else:
        st.info("👆 Upload an audio file and click 'Summarize' to see results here.")