"""
Header component for the Smart Trash Classification App
"""

import streamlit as st
from config.settings import APP_TITLE, APP_DESCRIPTION


def render_header():
    """Render the app header with title and description"""
    st.title(APP_TITLE)
    st.markdown("---")
    st.write(APP_DESCRIPTION)
    st.markdown("---")
