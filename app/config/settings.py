"""
Environment settings and configuration management
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Application Settings
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Model Settings
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# 환경 변수에서 모델 경로와 라벨 경로 가져오기, 없으면 기본 경로 사용
MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH is None:
    MODEL_PATH = str(MODELS_DIR / "model.tflite")
    logging.getLogger(__name__).info(
        f"환경 변수 MODEL_PATH가 설정되지 않았습니다. 기본 경로 사용: {MODEL_PATH}"
    )
else:
    logging.getLogger(__name__).info(
        f"환경 변수에서 모델 경로를 가져왔습니다: {MODEL_PATH}"
    )

LABELS_PATH = os.getenv("LABELS_PATH")
if LABELS_PATH is None:
    LABELS_PATH = str(MODELS_DIR / "labels.txt")
    logging.getLogger(__name__).info(
        f"환경 변수 LABELS_PATH가 설정되지 않았습니다. 기본 경로 사용: {LABELS_PATH}"
    )
else:
    logging.getLogger(__name__).info(
        f"환경 변수에서 라벨 경로를 가져왔습니다: {LABELS_PATH}"
    )

# 모델 파일과 라벨 파일 존재 여부 확인
if not os.path.exists(MODEL_PATH):
    logging.getLogger(__name__).error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    model_dir = os.path.dirname(MODEL_PATH)
    if os.path.exists(model_dir):
        logging.getLogger(__name__).info(
            f"모델 디렉토리의 파일 목록: {os.listdir(model_dir)}"
        )
else:
    logging.getLogger(__name__).info(f"모델 파일이 존재합니다: {MODEL_PATH}")

if not os.path.exists(LABELS_PATH):
    logging.getLogger(__name__).error(f"라벨 파일을 찾을 수 없습니다: {LABELS_PATH}")
else:
    logging.getLogger(__name__).info(f"라벨 파일이 존재합니다: {LABELS_PATH}")

IMAGE_SIZE = (96, 96)  # TFLite 모델의 입력 크기
IMAGE_CHANNELS = 1  # 그레이스케일 이미지

# Streamlit Settings
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

# Logging Settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "app.log"))

# Categories for trash classification
CATEGORIES = {0: "병", 1: "캔", 2: "철", 3: "유리", 4: "일반"}

# UI Configuration
APP_TITLE = "스마트 분리수거 도우미"
APP_DESCRIPTION = "이미지를 업로드하여 쓰레기 종류를 분류해보세요!"

# Camera Settings
CAMERA_SETTINGS = {
    "brightness": 0,  # 밝기 (-100 ~ 100)
    "contrast": 1.0,  # 대비 (0.0 ~ 2.0)
    "saturation": 1.0,  # 채도 (0.0 ~ 2.0)
    "interval": 1.0,  # 처리 간격 (초)
    "confidence_threshold": 0.7,  # 신뢰도 임계값
}
