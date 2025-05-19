"""
Configuration constants for the Smart Trash Classification App
"""

# Model configuration
MODEL_PATH = "models/model.h5"
IMAGE_SIZE = (96, 96)

# Categories
CATEGORIES = {0: "병", 1: "캔", 2: "철", 3: "유리", 4: "일반"}

# UI Configuration
APP_TITLE = "스마트 분리수거 도우미"
APP_DESCRIPTION = "이미지를 업로드하여 쓰레기 종류를 분류해보세요!"
