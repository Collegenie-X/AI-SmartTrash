"""
Environment settings and configuration management
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Application Settings
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Model Settings
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.h5")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

# Streamlit Settings
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

# Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# Categories for trash classification
CATEGORIES = {0: "병", 1: "캔", 2: "철", 3: "유리", 4: "일반"}

# UI Configuration
APP_TITLE = "스마트 분리수거 도우미"
APP_DESCRIPTION = "이미지를 업로드하여 쓰레기 종류를 분류해보세요!"
