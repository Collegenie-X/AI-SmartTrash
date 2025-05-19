"""
Main application file for the Smart Trash Classification App
"""

import streamlit as st
from components.header import render_header
from components.prediction_section import PredictionSection
from components.camera_section import CameraSection


def main():
    """Main application entry point"""
    # Configure page
    st.set_page_config(
        page_title="스마트 분리수거 도우미", page_icon="♻️", layout="wide"
    )

    # Render header
    render_header()

    # Create tabs for different modes
    tab1, tab2 = st.tabs(["이미지 업로드", "실시간 카메라"])

    with tab1:
        # Initialize and render prediction section
        prediction_section = PredictionSection()
        prediction_section.render()

    with tab2:
        # Initialize and render camera section
        camera_section = CameraSection()
        camera_section.render()


if __name__ == "__main__":
    main()
