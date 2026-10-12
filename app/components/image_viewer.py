import streamlit as st
from PIL import Image

def render_image(image: Image.Image, caption: str = "Uploaded Image", use_container_width: bool = True):
    st.image(image, caption=f"{caption} ({image.width}x{image.height}px)", use_container_width=use_container_width)