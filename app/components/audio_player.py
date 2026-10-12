import streamlit as st

def render_audio(audio_bytes: bytes, format: str = "audio/wav"):
    st.audio(audio_bytes, format=format)