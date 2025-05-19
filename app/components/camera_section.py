"""
Camera section component for real-time trash classification
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
from utils.model_loader import TFLiteModelLoader
from utils.image_processor import ImageProcessor
from config.settings import CAMERA_SETTINGS


class CameraSection:
    def __init__(self):
        # 환경 변수에서 직접 경로 가져오기
        model_path = os.environ.get("MODEL_PATH")
        labels_path = os.environ.get("LABELS_PATH")

        # 디버그 출력
        print(f"CameraSection - 모델 경로: {model_path}")
        print(f"CameraSection - 라벨 경로: {labels_path}")

        self.model_loader = TFLiteModelLoader(model_path, labels_path)
        self.image_processor = ImageProcessor()
        self.camera = None
        self.last_prediction_time = 0

        # Initialize camera settings in session state
        if "camera_settings" not in st.session_state:
            st.session_state.camera_settings = {
                "brightness": 0,
                "contrast": 1.0,
                "saturation": 1.0,
                "process_interval": 2.0,
                "confidence_threshold": 0.5,
            }

    def start_camera(self):
        """Start the camera capture"""
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                st.error("카메라를 열 수 없습니다.")
                return False
            # Apply initial camera settings
            self._apply_camera_settings()
        return True

    def stop_camera(self):
        """Stop the camera capture"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def _apply_camera_settings(self):
        """Apply camera settings"""
        if self.camera is not None:
            self.camera.set(
                cv2.CAP_PROP_BRIGHTNESS, st.session_state.camera_settings["brightness"]
            )
            self.camera.set(
                cv2.CAP_PROP_CONTRAST, st.session_state.camera_settings["contrast"]
            )
            self.camera.set(
                cv2.CAP_PROP_SATURATION, st.session_state.camera_settings["saturation"]
            )

    def _adjust_frame(self, frame):
        """Adjust frame based on camera settings"""
        # Convert to float32 for calculations
        frame_float = frame.astype(np.float32)

        # Apply brightness
        frame_float += st.session_state.camera_settings["brightness"]

        # Apply contrast
        frame_float = (frame_float - 128) * st.session_state.camera_settings[
            "contrast"
        ] + 128

        # Apply saturation
        hsv = cv2.cvtColor(frame_float.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * st.session_state.camera_settings["saturation"]
        frame_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Clip values to valid range
        frame_float = np.clip(frame_float, 0, 255)

        return frame_float.astype(np.uint8)

    def render(self):
        """Render the camera interface"""
        st.markdown("### 실시간 분류")

        # Camera settings
        with st.expander("카메라 설정", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.session_state.camera_settings["brightness"] = st.slider(
                    "밝기", -100, 100, st.session_state.camera_settings["brightness"]
                )
                st.session_state.camera_settings["contrast"] = st.slider(
                    "대비", 0.0, 2.0, st.session_state.camera_settings["contrast"], 0.1
                )

            with col2:
                st.session_state.camera_settings["saturation"] = st.slider(
                    "채도",
                    0.0,
                    2.0,
                    st.session_state.camera_settings["saturation"],
                    0.1,
                )
                st.session_state.camera_settings["process_interval"] = st.slider(
                    "처리 간격 (초)",
                    0.5,
                    5.0,
                    st.session_state.camera_settings["process_interval"],
                    0.5,
                )
                st.session_state.camera_settings["confidence_threshold"] = st.slider(
                    "신뢰도 임계값",
                    0.0,
                    1.0,
                    st.session_state.camera_settings["confidence_threshold"],
                    0.05,
                )

        # Camera control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("카메라 시작"):
                if self.start_camera():
                    st.session_state.camera_active = True

        with col2:
            if st.button("카메라 중지"):
                self.stop_camera()
                st.session_state.camera_active = False

        # Initialize session state
        if "camera_active" not in st.session_state:
            st.session_state.camera_active = False
        if "last_prediction" not in st.session_state:
            st.session_state.last_prediction = None
        if "last_confidence" not in st.session_state:
            st.session_state.last_confidence = None

        # Camera view and prediction
        if st.session_state.camera_active and self.camera is not None:
            # Create placeholder for camera feed
            camera_placeholder = st.empty()
            result_placeholder = st.empty()

            try:
                while st.session_state.camera_active:
                    # Read frame from camera
                    ret, frame = self.camera.read()
                    if not ret:
                        st.error("카메라에서 프레임을 읽을 수 없습니다.")
                        break

                    # Convert frame to RGB and apply adjustments
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb = self._adjust_frame(frame_rgb)

                    # Display the frame
                    camera_placeholder.image(
                        frame_rgb, channels="RGB", use_column_width=True
                    )

                    # Process frame based on interval setting
                    if (
                        "last_process_time" not in st.session_state
                        or time.time() - st.session_state.last_process_time
                        > st.session_state.camera_settings["process_interval"]
                    ):
                        # Process the frame
                        processed_image = self.image_processor.preprocess_image(
                            frame_rgb
                        )

                        # Make prediction
                        predicted_class, confidence = self.model_loader.predict(
                            processed_image
                        )

                        # Only update if confidence is above threshold
                        if (
                            confidence
                            >= st.session_state.camera_settings["confidence_threshold"]
                        ):
                            # Update session state
                            st.session_state.last_prediction = predicted_class
                            st.session_state.last_confidence = confidence
                            st.session_state.last_process_time = time.time()

                            # Display results
                            result_placeholder.markdown("### 분류 결과")
                            result_placeholder.success(f"분류: {predicted_class}")
                            result_placeholder.info(f"신뢰도: {confidence:.2%}")

                            # Display information
                            self._display_info(predicted_class, result_placeholder)
                        else:
                            result_placeholder.markdown("### 분류 결과")
                            result_placeholder.warning(
                                "신뢰도가 낮아 분류할 수 없습니다."
                            )

                    time.sleep(0.1)  # Prevent high CPU usage

            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                self.stop_camera()
                st.session_state.camera_active = False

    def _display_info(self, category: str, placeholder):
        """Display information based on the predicted category

        Args:
            category (str): 분류된 카테고리
            placeholder: Streamlit placeholder for displaying information
        """
        placeholder.markdown("### 분류 정보")

        if category == "Jongphil":
            placeholder.write(
                """
            ### Jongphil 분류
            - 분류된 객체: Jongphil
            - 추가 정보: 이 이미지는 Jongphil로 분류되었습니다.
            """
            )
        else:
            placeholder.write(
                """
            ### 배경
            - 분류된 객체: 배경
            - 추가 정보: 이 이미지는 배경으로 분류되었습니다.
            """
            )
