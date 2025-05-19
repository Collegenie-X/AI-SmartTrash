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

# 전역 카메라 객체 초기화
if "cap" not in st.session_state:
    st.session_state.cap = None

# 프레임 처리 관련 세션 상태 초기화
if "frame" not in st.session_state:
    st.session_state.frame = None
if "last_process_time" not in st.session_state:
    st.session_state.last_process_time = 0


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
        
        # 세션 상태의 카메라 객체 참조
        self.camera = st.session_state.cap
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
        try:
            print("카메라 초기화 시작...")
            
            # macOS 카메라 접근 설정
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "capture_device_index=0"
            os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
            
            # 전역 카메라 객체가 이미 있는지 확인
            if st.session_state.cap is not None:
                print("기존 카메라 연결이 있습니다. 재사용합니다.")
                self.camera = st.session_state.cap
                
                # 카메라가 여전히 열려있는지 확인
                if not self.camera.isOpened():
                    print("기존 카메라가 닫혀있습니다. 새로 연결합니다.")
                    st.session_state.cap = None
                else:
                    print("기존 카메라가 정상입니다.")
                    return True
            
            # 새 카메라 객체 생성 (가장 간단한 방법으로)
            print("새 카메라 연결 시도...")
            
            # Windows/Linux와 macOS에서 모두 작동하는지 확인하기 위해 먼저 기본 방식 시도
            st.session_state.cap = cv2.VideoCapture(0)
            self.camera = st.session_state.cap
            
            # 카메라 상태 확인
            if self.camera.isOpened():
                print("카메라 연결 성공!")
                
                # 일부 테스트 프레임 읽기 시도
                ret, frame = self.camera.read()
                if ret:
                    print(f"프레임 캡처 성공! 프레임 크기: {frame.shape}")
                    return True
                else:
                    print("프레임 캡처 실패. 카메라는 열렸지만 프레임을 읽을 수 없습니다.")
            else:
                print("카메라 연결 실패. 다른 방법을 시도합니다.")
                
                # 대체 캡처 방법 시도
                st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
                self.camera = st.session_state.cap
                
                if self.camera.isOpened():
                    print("대체 방법으로 카메라 연결 성공!")
                    ret, frame = self.camera.read()
                    if ret:
                        print(f"대체 방법으로 프레임 캡처 성공! 프레임 크기: {frame.shape}")
                        return True
                
                print("모든 카메라 연결 방법이 실패했습니다.")
                self.camera = None
                st.session_state.cap = None
                return False
            
        except Exception as e:
            print(f"카메라 초기화 중 예외 발생: {str(e)}")
            self.stop_camera()
            return False

    def stop_camera(self):
        """Stop the camera capture"""
        try:
            print("카메라 중지 시도...")
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
                print("카메라 연결 해제 완료")
            self.camera = None
            st.session_state.camera_initialized = False
            st.session_state.camera_active = False
            print("카메라 중지 완료")
        except Exception as e:
            print(f"카메라 중지 중 오류 발생: {str(e)}")
            st.session_state.camera_active = False
            st.session_state.camera_initialized = False
            st.session_state.cap = None

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

        # Initialize session states
        if "camera_active" not in st.session_state:
            st.session_state.camera_active = False
        if "camera_initialized" not in st.session_state:
            st.session_state.camera_initialized = False
            
        # 세션에서 카메라 객체 참조 업데이트
        self.camera = st.session_state.cap

        # Create placeholders
        status_placeholder = st.empty()
        camera_placeholder = st.empty()
        result_placeholder = st.empty()

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
                    "채도", 0.0, 2.0, st.session_state.camera_settings["saturation"], 0.1
                )
                st.session_state.camera_settings["process_interval"] = st.slider(
                    "처리 간격 (초)", 0.5, 5.0, st.session_state.camera_settings["process_interval"], 0.5
                )

        # Camera control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("카메라 시작"):
                with st.spinner("카메라 초기화 중..."):
                    if self.start_camera():
                        st.session_state.camera_active = True
                        st.session_state.camera_initialized = True
                        status_placeholder.success("카메라가 시작되었습니다.")
                    else:
                        status_placeholder.error("카메라를 시작할 수 없습니다.")
                        st.session_state.camera_active = False
                        st.session_state.camera_initialized = False

        with col2:
            if st.button("카메라 중지"):
                self.stop_camera()
                camera_placeholder.empty()
                status_placeholder.info("카메라가 중지되었습니다.")

        # Camera view and prediction
        if st.session_state.camera_active and st.session_state.cap is not None:
            try:
                # 카메라 피드 처리 함수 정의
                def process_camera_feed():
                    # 카메라 객체 상태 확인
                    if not st.session_state.cap.isOpened():
                        status_placeholder.error("카메라가 열려있지 않습니다.")
                        return
                        
                    # 프레임 처리 루프
                    stop_button_pressed = st.button("중지", key="stop_feed")
                    
                    # 프레임 표시를 위한 이미지 컨테이너
                    img_container = camera_placeholder.empty()
                    
                    while st.session_state.camera_active and not stop_button_pressed:
                        ret, frame = st.session_state.cap.read()
                        if not ret or frame is None:
                            status_placeholder.error("프레임을 읽을 수 없습니다.")
                            break
                            
                        # BGR에서 RGB로 변환
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # 프레임 조정 적용 (설정에 따라)
                        adjusted_frame = self._adjust_frame(rgb_frame)
                        
                        # 이미지 컨테이너 업데이트
                        img_container.image(
                            adjusted_frame,
                            channels="RGB",
                            caption="카메라 피드",
                            use_container_width=True
                        )
                        
                        # 모델 예측 처리 (일정 간격으로)
                        current_time = time.time()
                        if (current_time - st.session_state.last_process_time 
                            > st.session_state.camera_settings["process_interval"]):
                            
                            # 모델 입력을 위한 이미지 처리
                            frame_resized = cv2.resize(frame, (96, 96))
                            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                            processed_image = self.image_processor.preprocess_image(gray_frame)
                            
                            # 예측 실행
                            predicted_class, confidence = self.model_loader.predict(processed_image)
                            
                            # 마지막 처리 시간 업데이트
                            st.session_state.last_process_time = current_time
                            
                            # 결과 표시
                            result_placeholder.markdown(f"""
                            ### 분류 결과
                            - 분류: {predicted_class}
                            - 신뢰도: {confidence:.2%}
                            """)
                        
                        # 일정 시간 대기 (프레임 속도 제한)
                        time.sleep(0.033)  # ~30 FPS
                
                # 카메라 피드 처리 실행
                process_camera_feed()
                
            except Exception as e:
                print(f"카메라 작동 중 오류 발생: {str(e)}")
                status_placeholder.error(f"카메라 작동 중 오류가 발생했습니다: {str(e)}")
                # 카메라 객체 상태 초기화가 필요하면 재설정
                if "카메라가 열려있지 않습니다" in str(e) or "프레임을 읽을 수 없습니다" in str(e):
                    self.stop_camera()

    def _display_info(self, category: str, placeholder):
        """Display information based on the predicted category

        Args:
            category (str): 분류된 카테고리
            placeholder: Streamlit placeholder for displaying information
        """
        placeholder.markdown("### 분류 정보")

        # 카테고리 이름을 그대로 사용
        placeholder.write(
            f"""
            ### {category} 분류
            - 분류된 객체: {category}
            - 추가 정보: 이 이미지는 {category}로 분류되었습니다.
            """
        )
