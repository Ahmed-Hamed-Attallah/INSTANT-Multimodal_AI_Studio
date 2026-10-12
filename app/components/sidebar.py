import streamlit as st
import torch

def render_sidebar():
    with st.sidebar:
        # --- Branding & Header ---
        st.markdown("### 🤖 Multimodal AI Studio")
        st.caption("v1.0.0 | Advanced AI Diploma @ INSTANT")
        st.divider()
        
        # --- System Status Panel ---
        st.markdown("##### 🖥️ System Status")
        device = "CUDA (GPU) ⚡" if torch.cuda.is_available() else "CPU 🐌"
        st.metric(label="Active Engine", value=device)
        
        # Dynamically show VRAM if GPU is active
        if torch.cuda.is_available():
            try:
                vram_status = f"{torch.cuda.memory_allocated() / (1024**2):.1f} MB"
                st.metric(label="VRAM Allocated", value=vram_status)
            except Exception:
                pass
                
        st.divider()
        
        # --- Core Modalities ---
        st.markdown("##### 🧠 Core Modalities")
        st.markdown(
            """
            👁️ **Vision:** Captioning & VQA  
            🎙️ **Audio:** Speech-to-Text & Summaries  
            📄 **OCR:** Document Layout & Extraction  
            🔗 **Unified:** Cross-modal Insights  
            """
        )
        
        st.divider()
        
        # --- Utilities ---
        st.markdown("##### ⚙️ Utilities")
        if st.button(
            "🗑️ Clear App Cache", 
            use_container_width=True, 
            help="Clears all cached models and resources from memory to free up VRAM/RAM."
        ):
            st.cache_resource.clear()
            st.success("✅ Cache cleared successfully!")
            
        st.divider()
        
        # --- Footer / Author Credits ---
        st.markdown(
            "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
            "Built by <b>Ahmed Hamed Attallah</b><br>"
            "<a href='https://www.linkedin.com/in/ahmed-hamed-attallah/' target='_blank'>LinkedIn</a> • "
            "<a href='https://github.com/ahmed-hamed-attallah' target='_blank'>GitHub</a>"
            "</div>",
            unsafe_allow_html=True
        )