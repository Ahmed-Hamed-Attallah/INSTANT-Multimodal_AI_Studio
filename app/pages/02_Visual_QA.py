import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import sys
from pathlib import Path

# Add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Now import your utility
from utils.output_saver import save_output

st.markdown("### 🔍 Visual Question Answering")
st.markdown("Ask natural language questions about your uploaded images.")
st.divider()

# Let user toggle or automatically fallback to CPU if GPU kernel is incompatible
use_gpu = st.sidebar.checkbox("⚡ Use GPU (if compatible)", value=False)
device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

@st.cache_resource
def load_vqa_model(target_device: str):
    model_id = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id)
    model.to(target_device)
    model.eval()
    return processor, model

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
    st.markdown("#### ❓ Ask a Question")
    question = st.text_input(
        "Your question:", 
        placeholder="e.g., What is the main object in this photo?",
        disabled=(image_to_process is None)
    )
    
    if image_to_process is not None and question:
        if st.button("🚀 Submit Question", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing image and generating answer..."):
                try:
                    processor, model = load_vqa_model(device)
                    inputs = processor(images=image_to_process, text=question, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        out = model.generate(**inputs)
                    
                    answer = processor.decode(out[0], skip_special_tokens=True)
                    st.success(f"**Answer:** {answer}")
                    
                    # --- SAVE OUTPUT TO DISK ---
                    saved_path = save_output(
                        task_name="vqa",
                        results={"question": question, "answer": answer},
                        input_file_name=uploaded_file.name if uploaded_file else "vqa_image.png",
                        image_to_save=image_to_process
                    )
                    st.caption(f"💾 Output saved to: `{saved_path}`")
                    
                except RuntimeError as e:
                    # If cudaErrorNoKernelImageForDevice triggers, catch it and run on CPU
                    st.warning(f"⚠️ GPU kernel mismatch detected. Falling back to CPU...")
                    processor, model = load_vqa_model("cpu")
                    inputs = processor(images=image_to_process, text=question, return_tensors="pt").to("cpu")
                    
                    with torch.no_grad():
                        out = model.generate(**inputs)
                    
                    answer = processor.decode(out[0], skip_special_tokens=True)
                    st.success(f"**Answer:** {answer}")
    else:
        st.info("👆 Please upload an image and type a question to begin.")