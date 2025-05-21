"""
Configuration constants for the Smart Trash Classification App
"""

# File Upload Settings
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE_MB = 200

# Image Processing
IMAGE_SIZE = (96, 96)
IMAGE_CHANNELS = 1  # Grayscale

# Model Settings
MODEL_INPUT_SIZE = (96, 96)
MODEL_FILE_NAME = "model.tflite"
LABELS_FILE_NAME = "labels.txt"

# Categories with Korean translations
CATEGORIES = {
    "BOTTLE": "병",
    "CAN": "캔",
    "METAL": "철",
    "GLASS": "유리",
    "GENERAL_WASTE": "일반 쓰레기",
    "BACKGROUND": "배경",
    "ETC": "기타"
}

# Confidence Thresholds
CONFIDENCE_LEVELS = {
    "HIGH": 0.5,    # Above 50%: Green
    "MEDIUM": 0.4,  # Above 40%: Yellow
    "LOW": 0.4      # Below 40%: Red
}

# Recycling Guidelines
RECYCLING_GUIDES = {
    "BOTTLE": ["내용물 비우기", "라벨 제거", "분리수거함 배출"],
    "CAN": ["내용물 비우기", "부피 감소", "분리수거함 배출"],
    "METAL": ["이물질 제거", "크기 조절", "분리수거함 배출"],
    "GLASS": ["안전 처리", "내용물 비우기", "분리수거함 배출"],
    "GENERAL_WASTE": ["일반 쓰레기봉투 사용", "음식물 분리"],
    "ETC": ["품목 확인 후 적절한 방법으로 배출"]
}

# UI Configuration
APP_TITLE = "스마트 분리수거 도우미"
APP_DESCRIPTION = "AI를 활용하여 실시간으로 쓰레기를 분류하고 적절한 분리수거 방법을 안내하는 시스템입니다."

# Camera Settings
CAMERA_CONFIG = {
    "brightness": 0,
    "contrast": 1.0,
    "saturation": 1.0,
    "interval": 1.0,
    "confidence_threshold": 0.7
}

# Result Display Settings
SECOND_PREDICTION_THRESHOLD = 0.2  # Show second prediction if confidence difference is within 20%
