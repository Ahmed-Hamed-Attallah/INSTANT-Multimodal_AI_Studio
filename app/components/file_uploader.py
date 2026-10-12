import streamlit as st
from pathlib import Path

def render_file_uploader(label: str, accepted_types: list[str], sample_dir: Path | None = None) -> tuple[any, str | None]:
    """
    Renders an upload widget alongside a dropdown selector for project samples if available.
    Returns: (uploaded_file_or_bytes, filename)
    """
    col1, col2 = st.columns([2, 1])
    selected_sample = None

    with col2:
        if sample_dir and sample_dir.exists():
            sample_files = ["None"] + [f.name for f in sample_dir.glob("*.*") if f.suffix[1:] in accepted_types]
            choice = st.selectbox(f"Or pick sample {label}", sample_files)
            if choice != "None":
                selected_sample = sample_dir / choice

    with col1:
        uploaded_file = st.file_uploader(f"Upload {label}", type=accepted_types)

    if uploaded_file is not None:
        return uploaded_file.read(), uploaded_file.name
    elif selected_sample is not None:
        return selected_sample.read_bytes(), selected_sample.name

    return None, None