import streamlit as st

def render_result_box(title: str, text: str, alert_type: str = "info"):
    st.subheader(title)
    if alert_type == "success":
        st.success(text)
    elif alert_type == "warning":
        st.warning(text)
    else:
        st.info(text)