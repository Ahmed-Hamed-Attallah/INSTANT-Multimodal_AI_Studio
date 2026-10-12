import streamlit as st
from PIL import Image
from app.services.ocr_service import OCRService
import sys
from pathlib import Path

# Add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Now import your utility
from utils.output_saver import save_output

st.markdown("### 📄 OCR & Layout Analysis")
st.markdown("Extract text and analyze the structural layout of documents and images.")
st.divider()

@st.cache_resource
def get_ocr_service():
    return OCRService()

try:
    service = get_ocr_service()
except Exception as e:
    st.error(f"🚨 Failed to initialize OCR service: {e}")
    st.stop()

col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown("#### 📤 Upload Document or Image")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        help="Supported formats: PNG, JPG, JPEG, WEBP, PDF"
    )
    
    image_to_process = None
    if uploaded_file is not None:
        # Handle PDFs or Images gracefully (assuming OCRService handles PDF conversion, 
        # but PIL open is kept as per your original logic for images)
        try:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            st.image(image_to_process, caption="Document Preview", use_container_width=True)
        except Exception:
            st.warning("⚠️ Could not render preview for this file type, but processing will continue.")

with col_output:
    st.markdown("#### 📜 Extracted Data")
    if image_to_process is not None:
        if st.button("🔍 Extract Text & Layout", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing document layout and extracting text..."):
                result = service.extract_layout(image_to_process)
                
                if result.get("success", False):
                    with st.container(border=True):
                        st.markdown("**Raw Extracted Text:**")
                        st.text_area(
                            "OCR Output", 
                            result.get("raw_text", ""), 
                            height=250,
                            label_visibility="collapsed"
                        )
                    
                    # --- SAVE OUTPUT TO DISK ---
                    saved_path = save_output(
                        task_name="ocr",
                        results={
                            "text": result.get("raw_text"),
                            "blocks": result.get("blocks", [])
                        },
                        input_file_name=uploaded_file.name if uploaded_file else "ocr_image.png",
                        image_to_save=image_to_process
                    )
                    st.caption(f"💾 Output saved to: `{saved_path}`")
                    
                    with st.expander("📐 Detected Word Blocks & Coordinates (JSON)", expanded=False):
                        st.json(result.get("blocks", []))
                else:
                    st.error(f"🚫 OCR Error: {result.get('error')}")
    else:
        st.info("👆 Upload an image or document to begin extraction.")