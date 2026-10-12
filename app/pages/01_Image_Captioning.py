import io
import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
from pathlib import Path

# Add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Now import your utility
import sys
from pathlib import Path

# Add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Now import your utility
from utils.output_saver import save_output

# Prevent cuDNN benchmark failures
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

st.markdown("### 🖼️ Automated Image Captioning")
st.markdown("Upload an image to generate a descriptive, AI-powered caption.")
st.divider()

# Device selection with safe fallback
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_caption_model():
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return processor, model

# Modern 2-column layout
col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown("#### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    image_to_process = None
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        st.image(image_to_process, caption="Uploaded Preview", use_container_width=True)

with col_output:
    st.markdown("#### 📝 Generated Caption")
    if image_to_process is not None:
        if st.button("✨ Generate Caption", type="primary", use_container_width=True):
            with st.spinner("🧠 Processing image and generating caption..."):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    processor, model = load_caption_model()
                    inputs = processor(images=image_to_process, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=50)
                    
                    caption = processor.decode(out[0], skip_special_tokens=True).capitalize()
                    
                    st.success(f"**{caption}**")
                    
                    # --- SAVE OUTPUT TO DISK ---
                    saved_path = save_output(
                        task_name="captioning",
                        results={"caption": caption},
                        input_file_name=uploaded_file.name if uploaded_file else "sample_image.png",
                        image_to_save=image_to_process
                    )
                    st.caption(f"💾 Output saved to: `{saved_path}`")
                    
                except RuntimeError as e:
                    # Automatic fallback if GPU engine still throws an error
                    st.warning("⚠️ GPU execution failed, falling back to CPU...")
                    processor, model = load_caption_model()
                    model.to("cpu")
                    inputs = processor(images=image_to_process, return_tensors="pt").to("cpu")
                    
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=50)
                    
                    caption = processor.decode(out[0], skip_special_tokens=True).capitalize()
                    st.success(f"**{caption}**")
    else:
        st.info("👆 Please upload an image to begin.")