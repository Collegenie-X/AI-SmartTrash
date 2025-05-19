"""
Image processing utility for the Smart Trash Classification App
"""

import cv2
import numpy as np
from PIL import Image
from config.settings import IMAGE_SIZE, IMAGE_CHANNELS
from typing import Tuple, Optional


class ImageProcessor:
    """이미지 처리 유틸리티 클래스

    이미지 전처리 및 변환을 담당합니다.
    """

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """이미지 전처리

        Args:
            image (np.ndarray): 입력 이미지

        Returns:
            np.ndarray: 전처리된 이미지
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 크기 조정
        image = cv2.resize(image, IMAGE_SIZE)

        # 채널 차원 추가 (필요한 경우)
        if IMAGE_CHANNELS == 1:
            image = np.expand_dims(image, axis=-1)

        return image

    @staticmethod
    def apply_camera_settings(
        image: np.ndarray,
        brightness: int = 0,
        contrast: float = 1.0,
        saturation: float = 1.0,
    ) -> np.ndarray:
        """카메라 설정 적용

        Args:
            image (np.ndarray): 입력 이미지
            brightness (int, optional): 밝기 조정값. Defaults to 0.
            contrast (float, optional): 대비 조정값. Defaults to 1.0.
            saturation (float, optional): 채도 조정값. Defaults to 1.0.

        Returns:
            np.ndarray: 처리된 이미지
        """
        # 밝기 조정
        if brightness != 0:
            image = cv2.add(image, brightness)

        # 대비 조정
        if contrast != 1.0:
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

        # 채도 조정 (컬러 이미지인 경우에만)
        if len(image.shape) == 3 and saturation != 1.0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return image

    @staticmethod
    def read_image(file_path: str) -> Optional[np.ndarray]:
        """이미지 파일 읽기

        Args:
            file_path (str): 이미지 파일 경로

        Returns:
            Optional[np.ndarray]: 읽은 이미지 또는 None
        """
        try:
            image = cv2.imread(file_path)
            if image is None:
                return None
            return image
        except Exception as e:
            print(f"Error reading image: {str(e)}")
            return None

    @staticmethod
    def convert_to_display(image: np.ndarray) -> np.ndarray:
        """표시용 이미지로 변환

        Args:
            image (np.ndarray): 입력 이미지

        Returns:
            np.ndarray: 표시용 이미지
        """
        # 그레이스케일 이미지를 BGR로 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
