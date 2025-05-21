"""
Environment settings and configuration management
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from .constants import MODEL_FILE_NAME, LABELS_FILE_NAME

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

# Application Settings
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Model Paths
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / MODEL_FILE_NAME))
LABELS_PATH = os.getenv("LABELS_PATH", str(MODELS_DIR / LABELS_FILE_NAME))

# Server Settings
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "app.log"))

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Validate critical paths
def validate_paths():
    """Validate existence of critical files and directories"""
    if not MODELS_DIR.exists():
        logger.error(f"모델 디렉토리를 찾을 수 없습니다: {MODELS_DIR}")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"모델 디렉토리를 생성했습니다: {MODELS_DIR}")

    if not os.path.exists(MODEL_PATH):
        logger.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        model_dir = os.path.dirname(MODEL_PATH)
        if os.path.exists(model_dir):
            logger.info(f"모델 디렉토리의 파일 목록: {os.listdir(model_dir)}")
    else:
        logger.info(f"모델 파일이 존재합니다: {MODEL_PATH}")

    if not os.path.exists(LABELS_PATH):
        logger.error(f"라벨 파일을 찾을 수 없습니다: {LABELS_PATH}")
    else:
        logger.info(f"라벨 파일이 존재합니다: {LABELS_PATH}")

validate_paths()

IMAGE_SIZE = (96, 96)  # TFLite 모델의 입력 크기
IMAGE_CHANNELS = 1  # 그레이스케일 이미지

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
