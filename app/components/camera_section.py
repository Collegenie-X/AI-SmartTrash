"""
Camera section component for real-time trash classification
"""

import streamlit as st
import numpy as np
from PIL import Image
import time
import os
from utils.model_loader import TFLiteModelLoader
from utils.image_processor import ImageProcessor
from config.settings import CAMERA_SETTINGS
import cv2

class CameraSection:
    def __init__(self):
        print("CameraSection 초기화 시작")
        
        # Initialize session state
        if "camera_state" not in st.session_state:
            st.session_state["camera_state"] = {
                "is_active": False,  # 카메라 활성화 상태
                "last_prediction": None,
                "prediction_count": 0,
                "settings": {
                    "confidence_threshold": 0.5,
                }
            }
        
        # 환경 변수에서 직접 경로 가져오기
        model_path = os.path.join(os.getcwd(), "models", "model.tflite")
        labels_path = os.path.join(os.getcwd(), "models", "labels.txt")

        print(f"모델 파일 경로: {model_path}")
        print(f"라벨 파일 경로: {labels_path}")

        # 파일 존재 확인
        print(f"모델 파일 존재: {os.path.exists(model_path)}")
        print(f"라벨 파일 존재: {os.path.exists(labels_path)}")

        # 라벨 파일 로드
        self.labels = self._load_labels(labels_path)
        print(f"로드된 라벨: {self.labels}")

        self.model_loader = TFLiteModelLoader(model_path, labels_path)
        self.image_processor = ImageProcessor()
        print("CameraSection 초기화 완료")

    def _load_labels(self, labels_path):
        """라벨 파일을 로드하는 함수"""
        try:
            print(f"라벨 파일 로드 시도: {labels_path}")
            with open(labels_path, 'r', encoding='utf-8') as f:
                labels = []
                for line in f.readlines():
                    # Remove any whitespace and newlines
                    label = line.strip()
                    if not label:  # Skip empty lines
                        continue
                    # If the line starts with a number and space, remove it
                    if ' ' in label and label.split(' ')[0].isdigit():
                        label = label.split(' ', 1)[1]
                    labels.append(label)
                print(f"라벨 파일 로드 성공: {labels}")
                return labels if labels else ["병", "캔", "철", "유리", "일반"]  # Fallback to default labels if empty
        except Exception as e:
            print(f"라벨 파일 로드 오류: {str(e)}")
            # Return default labels if there's an error
            default_labels = ["병", "캔", "철", "유리", "일반"]
            print(f"기본 라벨 사용: {default_labels}")
            return default_labels

    def process_image(self, image):
        """Process a single image and return prediction"""
        try:
            print("이미지 처리 시작")
            print(f"입력 이미지 크기: {image.size}")
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            print(f"NumPy 배열 변환 후 형태: {img_array.shape}")
            
            # Convert RGB to BGR (if needed)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                print("RGB에서 BGR로 변환 완료")
            
            # Convert to grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            print(f"그레이스케일 변환 후 형태: {img_array.shape}")
            
            # Resize to 96x96
            img_resized = cv2.resize(img_array, (96, 96))
            print("이미지 리사이징 완료: 96x96")
            
            # Normalize to 0-1 range
            img_normalized = img_resized.astype(np.float32) / 255.0
            print("이미지 정규화 완료")
            
            # Add channel dimension
            img_processed = np.expand_dims(img_normalized, axis=-1)
            print(f"채널 차원 추가 후 형태: {img_processed.shape}")
            
            # Add batch dimension
            img_processed = np.expand_dims(img_processed, axis=0)
            print(f"배치 차원 추가 후 최종 형태: {img_processed.shape}")
            
            # Get prediction
            print("모델 예측 시작")
            predicted_class_idx, confidence, all_predictions = self.model_loader.predict(img_processed)
            print(f"예측 완료 - 인덱스: {predicted_class_idx}, 신뢰도: {confidence:.2%}")
            
            # 라벨에서 인덱스 제거하고 실제 클래스 이름만 추출
            raw_label = self.labels[predicted_class_idx]
            if ' ' in raw_label:
                # "0 Bottle" -> "Bottle" 형태로 변환
                class_name = raw_label.split(' ', 1)[1]
            else:
                class_name = raw_label

            # 두 번째로 높은 확률의 클래스 찾기
            sorted_indices = np.argsort(all_predictions)[::-1]
            second_class_idx = sorted_indices[1]
            second_confidence = float(all_predictions[second_class_idx])
            
            # 두 번째 클래스의 라벨 처리
            raw_second_label = self.labels[second_class_idx]
            if ' ' in raw_second_label:
                second_class_name = raw_second_label.split(' ', 1)[1]
            else:
                second_class_name = raw_second_label

            # 한글 클래스명으로 변환
            korean_names = {
                'Bottle': '병',
                'Can': '캔',
                'Metal': '철',
                'Glass': '유리',
                'General Waste': '일반',
                'Background': '배경',
            }
            
            predicted_class = korean_names.get(class_name, class_name)
            second_predicted_class = korean_names.get(second_class_name, second_class_name)
            
            print(f"최종 예측 결과 - 클래스: {predicted_class}, 신뢰도: {confidence:.2%}")
            print(f"두 번째 예측 결과 - 클래스: {second_predicted_class}, 신뢰도: {second_confidence:.2%}")
            
            return {
                "class": predicted_class,
                "confidence": confidence,
                "second_class": second_predicted_class,
                "second_confidence": second_confidence,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return None

    def render(self):
        """Render the camera interface"""
        st.markdown("### 실시간 분류")
        
        # 카메라 컨트롤
        col1, col2 = st.columns(2)
        with col1:
            if st.button("카메라 시작", disabled=st.session_state["camera_state"]["is_active"]):
                st.session_state["camera_state"]["is_active"] = True
                st.success("카메라가 시작되었습니다.")
                st.rerun()
                
        with col2:
            if st.button("카메라 중지", disabled=not st.session_state["camera_state"]["is_active"]):
                st.session_state["camera_state"]["is_active"] = False
                if "camera_input" in st.session_state:
                    del st.session_state["camera_input"]
                st.info("카메라가 중지되었습니다.")
                st.rerun()
        
        # 카메라 상태 표시
        if not st.session_state["camera_state"]["is_active"]:
            st.warning("카메라가 비활성화되어 있습니다. '카메라 시작' 버튼을 눌러 활성화하세요.")
            return

        # 결과 표시를 위한 컨테이너들 미리 생성
        camera_container = st.container()
        log_container = st.container()  # 로그를 위한 새 컨테이너
        result_container = st.container()
        guide_container = st.container()
        
        # 카메라 입력 (카메라가 활성화된 경우에만)
        with camera_container:
            camera_image = st.camera_input(
                "쓰레기를 카메라에 비춰주세요",
                key="camera_input",
                disabled=not st.session_state["camera_state"]["is_active"]
            )
        
        # 이미지가 캡처되면 처리
        if camera_image is not None:
            try:
                # 이미지 열기
                image = Image.open(camera_image)
                
                # 진행 상태 표시
                with st.spinner("이미지 분석 중..."):
                    # 이미지 처리 및 예측
                    prediction = self.process_image(image)
                
                if prediction:
                    # 로그 정보 표시
                    with log_container:
                        st.markdown("### 📝 분석 로그")
                        st.code(f"""
이미지 분석 시작
예측된 클래스: {prediction['class']} (신뢰도: {prediction['confidence']:.2%})
두 번째 예측: {prediction['second_class']} (신뢰도: {prediction['second_confidence']:.2%})
분석 시각: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(prediction['timestamp']))}
누적 분석 횟수: {st.session_state["camera_state"]["prediction_count"]}
                        """)
                    
                    # 세션 상태 업데이트
                    st.session_state["camera_state"]["last_prediction"] = prediction
                    st.session_state["camera_state"]["prediction_count"] += 1
                    
                    # 분류 가능 여부 확인
                    recyclable_classes = ["병", "캔", "철", "유리", "일반"]
                    non_recyclable = ['배경', 'Jongphil', 'Background']
                    
                    # 결과를 별도의 컨테이너에 표시
                    with result_container:
                        if prediction['class'] in non_recyclable:
                            st.warning(f"⚠️ '{prediction['class']}'- 분리수거 가능한 물체가 감지되지 않았습니다. 다시 시도해주세요.")
                            return
                            
                        st.success("✅ 이미지 분석이 완료되었습니다!")
                        
                        # 분류 결과를 마크다운 형식으로 표시
                        st.markdown("""
                        ### 📊 분석 결과
                        ---
                        """)
                        
                        # 결과 카드 스타일 적용
                        st.markdown("""
                        <style>
                        .result-metric {
                            font-size: 1.2em;
                            color: #0066cc;
                            font-weight: bold;
                            padding: 10px;
                            background-color: #f8f9fa;
                            border-radius: 5px;
                            margin: 5px 0;
                        }
                        .confidence-high {
                            color: #00cc66;
                        }
                        .confidence-medium {
                            color: #ffcc00;
                        }
                        .confidence-low {
                            color: #ff6666;
                        }
                        .result-details {
                            margin-top: 10px;
                            padding: 10px;
                            background-color: #f8f9fa;
                            border-radius: 5px;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # 신뢰도에 따른 색상 클래스 결정
                        confidence_class = (
                            "confidence-high" if prediction['confidence'] > 0.5
                            else "confidence-medium" if prediction['confidence'] > 0.4
                            else "confidence-low"
                        )
                        
                        # 결과 표시
                        st.markdown(f"""
                        #### 🎯 인식 결과
                        <div class="result-metric">
                            감지된 객체: {prediction['class']}
                        </div>
                        
                        #### 📊 신뢰도
                        <div class="result-metric {confidence_class}">
                            {prediction['confidence']:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # 분리수거 가이드를 별도의 컨테이너에 표시
                    with guide_container:
                        # 신뢰도 차이가 작은 경우 (20% 이내) 두 클래스 모두 표시
                        if abs(prediction['confidence'] - prediction['second_confidence']) < 0.2:
                            st.markdown("---")
                            st.info(f"""
                            ℹ️ 두 가지 항목이 비슷한 확률로 감지되었습니다:
                            1. {prediction['class']} ({prediction['confidence']:.1%})
                            2. {prediction['second_class']} ({prediction['second_confidence']:.1%})
                            """)
                            
                            # 두 클래스 모두에 대해 분리수거 가이드 표시
                            for cls in [prediction['class'], prediction['second_class']]:
                                if cls in recyclable_classes:
                                    st.markdown(f"### {cls}의 분리수거 방법")
                                    self._display_disposal_guide(cls)

                        else:
                            # 단일 클래스에 대한 기존 로직
                            if prediction['class'] in recyclable_classes:
                                st.markdown("---")
                                self._display_disposal_guide(prediction['class'])
                            
                else:
                    st.warning("⚠️ 이미지를 분석할 수 없습니다. 다시 시도해주세요.")
            except Exception as e:
                st.error(f"❌ 이미지 처리 중 오류가 발생했습니다: {str(e)}")

    def _display_info(self, category: str, placeholder):
        """Display information based on the predicted category"""
        placeholder.markdown("### 분류 정보")
        placeholder.write(
            f"""
            ### {category} 분류
            - 분류된 객체: {category}
            - 추가 정보: 이 이미지는 {category}로 분류되었습니다.
            """
        )

    def _display_disposal_guide(self, waste_class):
        """Display disposal guide for the classified waste"""
        guides = {
            "병": "1. 내용물을 비우고 물로 헹궈주세요.\n2. 라벨을 제거해주세요.\n3. 병만 분리수거함에 넣어주세요.",
            "캔": "1. 내용물을 비우고 물로 헹궈주세요.\n2. 찌그러뜨려 부피를 줄여주세요.\n3. 캔 분리수거함에 넣어주세요.",
            "철": "1. 이물질을 제거해주세요.\n2. 크기가 큰 경우 적당한 크기로 잘라주세요.\n3. 철 분리수거함에 넣어주세요.",
            "유리": "1. 깨진 유리는 신문지에 싸서 배출해주세요.\n2. 내용물을 비우고 물로 헹궈주세요.\n3. 유리 분리수거함에 넣어주세요.",
            "일반": "1. 일반 쓰레기봉투에 넣어주세요.\n2. 음식물이 묻은 경우 음식물 쓰레기로 분리해주세요."
        }
        
        if waste_class in guides:
            st.markdown("### 분리수거 방법")
            st.info(guides[waste_class])
        else:
            st.warning(f"'{waste_class}' 해당 분류에 대한 분리수거 가이드가 없습니다.")
