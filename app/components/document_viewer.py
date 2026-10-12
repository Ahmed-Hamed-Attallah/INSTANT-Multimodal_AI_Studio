import streamlit as st

def render_ocr_blocks(blocks: list[dict]):
    with st.expander("Detected Text Blocks & Confidence", expanded=False):
        for idx, block in enumerate(blocks, start=1):
            text = block.get("text", "")
            conf = block.get("confidence", 0.0)
            st.markdown(f"**{idx}.** {text} `(conf: {conf:.2f})`")