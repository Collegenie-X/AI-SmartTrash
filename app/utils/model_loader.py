"""
Model loader utility for the Smart Trash Classification App
"""

import numpy as np
import tensorflow as tf
import os
import logging
import warnings
from typing import Tuple, List, Dict
import cv2

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TFLite Interpreter 경고 메시지 필터링
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow.lite.python.interpreter')


class TFLiteModelLoader:
    """TFLite 모델 로더 클래스

    Teachable Machine에서 생성된 TFLite 모델을 로드하고 예측을 수행합니다.
    """

    def __init__(self, model_path: str, labels_path: str):
        """모델 로더 초기화

        Args:
            model_path (str): TFLite 모델 파일 경로
            labels_path (str): 라벨 파일 경로
        """
        self.model_path = model_path
        logger.info(f"모델 경로: {self.model_path}")
        logger.info(f"라벨 경로: {labels_path}")
        logger.info(f"현재 작업 디렉토리: {os.getcwd()}")

        # 모델 파일 존재 여부 확인
        if not os.path.exists(self.model_path):
            logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            # 모델 디렉토리의 파일 목록 확인
            model_dir = os.path.dirname(self.model_path)
            if os.path.exists(model_dir):
                logger.info(f"모델 디렉토리의 파일 목록: {os.listdir(model_dir)}")

        self.labels = self._load_labels(labels_path)
        self.interpreter = self._load_model()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def _load_labels(self, labels_path: str) -> List[str]:
        """라벨 파일 로드

        Args:
            labels_path (str): 라벨 파일 경로

        Returns:
            List[str]: 라벨 리스트
        """
        with open(labels_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def _load_model(self) -> tf.lite.Interpreter:
        """TFLite 모델 로드

        Returns:
            tf.lite.Interpreter: TFLite 인터프리터
        """
        logger.info(f"모델 로딩 시도: {self.model_path}")

        # 절대 경로인지 확인
        if not os.path.isabs(self.model_path):
            # 상대 경로인 경우 현재 디렉토리 기준으로 절대 경로로 변환
            abs_path = os.path.abspath(self.model_path)
            logger.info(f"상대 경로를 절대 경로로 변환: {abs_path}")
            self.model_path = abs_path

        # 파일 존재 확인
        if not os.path.exists(self.model_path):
            # 모델 파일이 없는 경우 대안 경로 시도
            alternatives = [
                os.path.join(os.getcwd(), "models", "model.tflite"),
                os.path.join(os.getcwd(), "..", "models", "model.tflite"),
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "..",
                    "models",
                    "model.tflite",
                ),
            ]

            logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
            logger.info(f"대안 경로 시도 중...")

            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    logger.info(f"대안 경로에서 모델 파일을 찾았습니다: {alt_path}")
                    self.model_path = alt_path
                    break

            # 여전히 파일을 찾을 수 없는 경우
            if not os.path.exists(self.model_path):
                # 모델 디렉토리의 파일 목록 확인
                model_dir = os.path.dirname(self.model_path)
                if os.path.exists(model_dir):
                    files = os.listdir(model_dir)
                    logger.info(f"모델 디렉토리의 파일 목록: {files}")

                    # .tflite 확장자를 가진 파일이 있는지 확인
                    tflite_files = [f for f in files if f.endswith(".tflite")]
                    if tflite_files:
                        # 첫 번째 .tflite 파일 사용
                        self.model_path = os.path.join(model_dir, tflite_files[0])
                        logger.info(
                            f".tflite 파일을 찾았습니다. 이 파일을 사용합니다: {self.model_path}"
                        )

                # 여전히 파일을 찾을 수 없는 경우 예외 발생
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # 모델 로딩
        try:
            logger.info(f"모델 파일 로딩: {self.model_path}")
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            logger.info("모델 로딩 성공!")
            return interpreter
        except Exception as e:
            logger.error(f"모델 로딩 중 오류 발생: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리

        Args:
            image (np.ndarray): 입력 이미지 (BGR 또는 그레이스케일)

        Returns:
            np.ndarray: 전처리된 이미지 (1채널)
        """
        try:
            # 입력 이미지 검증
            if image is None:
                raise ValueError("입력 이미지가 None입니다.")
            
            if not isinstance(image, np.ndarray):
                raise ValueError(f"잘못된 이미지 타입: {type(image)}, numpy.ndarray가 필요합니다.")
            
            if image.size == 0:
                raise ValueError("이미지가 비어 있습니다.")

            print(f"모델 전처리 입력 이미지 형태: {image.shape}")
            
            # 이미 전처리된 이미지인지 확인
            if len(image.shape) == 4 and image.shape == (1, 96, 96, 1):
                print("이미 전처리된 이미지입니다.")
                return image

            # 이미지 전처리
            target_size = (96, 96)
            
            # BGR에서 그레이스케일로 변환
            if len(image.shape) == 3 and image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 크기 조정 (필요한 경우)
            if image.shape[:2] != target_size:
                image = cv2.resize(image, target_size)
            
            # 정규화 (0-1 범위로)
            image = image.astype(np.float32) / 255.0

            # 채널 차원 추가 (H,W) -> (H,W,1)
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            # 배치 차원 추가 (H,W,1) -> (1,H,W,1)
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)

            print(f"모델 전처리 출력 이미지 형태: {image.shape}")

            # 입력 텐서 형태 확인
            input_shape = self.input_details[0]['shape']
            if image.shape != tuple(input_shape):
                raise ValueError(f"잘못된 입력 차원: got {image.shape}, expected {tuple(input_shape)}")

            return image

        except Exception as e:
            print(f"모델 전처리 중 오류 발생: {str(e)}")
            raise

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """이미지 분류 예측

        Args:
            image (np.ndarray): 입력 이미지

        Returns:
            Tuple[str, float]: (예측 클래스, 신뢰도)
        """
        # 이미지 전처리
        input_data = self.preprocess_image(image)

        # 입력 텐서 설정
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        # 추론 실행
        self.interpreter.invoke()

        # 출력 텐서 가져오기
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        # 예측 결과 처리
        predicted_class = np.argmax(output_data[0])
        confidence = float(output_data[0][predicted_class])

        return self.labels[predicted_class], confidence

    def get_labels(self) -> List[str]:
        """라벨 리스트 반환

        Returns:
            List[str]: 라벨 리스트
        """
        return self.labels
