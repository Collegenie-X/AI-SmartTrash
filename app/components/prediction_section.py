"""
Prediction section component for the Smart Trash Classification App
"""

import streamlit as st
import os
from utils.model_loader import TFLiteModelLoader
from utils.image_processor import ImageProcessor
import time
from PIL import Image
import numpy as np
import cv2
import io


class PredictionSection:
    def __init__(self):
        # Initialize session state
        if "prediction_state" not in st.session_state:
            st.session_state["prediction_state"] = {
                "last_prediction": None,
                "prediction_count": 0,
                "settings": {
                    "confidence_threshold": 0.5,
                }
            }
        
        # 환경 변수에서 직접 경로 가져오기
        model_path = os.path.join(os.getcwd(), "models", "model.tflite")
        labels_path = os.path.join(os.getcwd(), "models", "labels.txt")

        print(f"PredictionSection - 모델 경로: {model_path}")
        print(f"PredictionSection - 라벨 경로: {labels_path}")

        self.model_loader = TFLiteModelLoader(model_path, labels_path)
        self.image_processor = ImageProcessor()

    def _validate_image(self, file):
        """Validate the uploaded image file"""
        try:
            # Check file size (200MB limit)
            MAX_SIZE = 200 * 1024 * 1024  # 200MB in bytes
            file_size = len(file.getvalue())
            if file_size > MAX_SIZE:
                return False, f"파일 크기가 너무 큽니다. 최대 200MB까지 가능합니다. (현재: {file_size/1024/1024:.1f}MB)"

            # Verify image format
            try:
                img = Image.open(io.BytesIO(file.getvalue()))
                img.verify()  # Verify it's a valid image
                return True, None
            except Exception as e:
                return False, f"유효하지 않은 이미지 파일입니다: {str(e)}"

        except Exception as e:
            return False, f"파일 검증 중 오류가 발생했습니다: {str(e)}"

    def render(self):
        """Render the prediction interface"""
        st.header("이미지 업로드")

        # 결과 표시를 위한 컨테이너들 미리 생성
        upload_container = st.container()
        log_container = st.container()
        result_container = st.container()
        guide_container = st.container()

        # 이미지 업로드 섹션
        with upload_container:
            st.markdown("""
            ### 📤 이미지 업로드
            - 지원 형식: JPG, JPEG, PNG
            - 최대 파일 크기: 200MB
            """)
            
            uploaded_file = st.file_uploader(
                "이미지를 업로드하세요",
                type=["jpg", "jpeg", "png"],
                help="JPG, JPEG, PNG 형식의 이미지 파일을 선택해주세요. (최대 200MB)"
            )

            if uploaded_file is not None:
                try:
                    # Validate the uploaded file
                    is_valid, error_message = self._validate_image(uploaded_file)
                    
                    if not is_valid:
                        st.error(f"⚠️ {error_message}")
                        return

                    # Reset file pointer after validation
                    uploaded_file.seek(0)
                    
                    # 이미지 열기 및 표시
                    image = Image.open(uploaded_file)
                    st.image(image, caption="업로드된 이미지", use_container_width=True)

                    # 분류 버튼을 중앙에 배치
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        classify_button = st.button(
                            "🔍 분류하기",
                            help="이미지를 분석하여 쓰레기 종류를 분류합니다.",
                            use_container_width=True
                        )

                    if classify_button:
                        # 진행 상태 표시
                        with st.spinner("🔄 이미지 분석 중..."):
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
누적 분석 횟수: {st.session_state["prediction_state"]["prediction_count"]}
                                """)

                            # 세션 상태 업데이트
                            st.session_state["prediction_state"]["last_prediction"] = prediction
                            st.session_state["prediction_state"]["prediction_count"] += 1

                            # 분류 가능 여부 확인
                            recyclable_classes = ["병", "캔", "철", "유리", "일반"]
                            non_recyclable = ['배경', 'Jongphil', 'Background']

                            # 결과를 별도의 컨테이너에 표시
                            with result_container:
                                if prediction['class'] in non_recyclable:
                                    st.warning("⚠️ 분리수거 가능한 물체가 감지되지 않았습니다. 다시 시도해주세요.")
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
                    print(f"상세 오류: {str(e)}")
                    import traceback
                    print(f"스택 트레이스: {traceback.format_exc()}")

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
            if len(img_array.shape) == 3:  # Color image
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
            print(f"전체 예측값: {all_predictions}")
            
            # 두 번째로 높은 확률의 클래스 찾기
            sorted_indices = np.argsort(all_predictions)[::-1]
            second_class_idx = sorted_indices[1]
            second_confidence = float(all_predictions[second_class_idx])
            
            # 라벨 가져오기
            labels = self.model_loader.get_labels()
            print(f"사용 가능한 라벨: {labels}")
            
            # 한글 클래스명으로 변환
            korean_names = {
                'Bottle': '병',
                'Can': '캔',
                'Metal': '철',
                'Glass': '유리',
                'General Waste': '일반',
                'Background': '배경',
                'Jongphil': 'Jongphil'
            }
            
            # 라벨에서 숫자와 공백을 제거하고 실제 클래스 이름만 추출
            def clean_label(label):
                # If the label starts with a number and space, remove it
                if ' ' in label and label.split(' ')[0].isdigit():
                    return label.split(' ', 1)[1]
                return label
            
            # 예측된 클래스 처리
            predicted_label = clean_label(labels[predicted_class_idx])
            second_label = clean_label(labels[second_class_idx])
            
            predicted_class = korean_names.get(predicted_label, predicted_label)
            second_predicted_class = korean_names.get(second_label, second_label)
            
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
            st.warning("해당 분류에 대한 분리수거 가이드가 없습니다.")
