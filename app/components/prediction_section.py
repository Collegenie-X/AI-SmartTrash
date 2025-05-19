"""
Prediction section component for the Smart Trash Classification App
"""

import streamlit as st
import os
from utils.model_loader import TFLiteModelLoader
from utils.image_processor import ImageProcessor


class PredictionSection:
    def __init__(self):
        # 환경 변수에서 직접 경로 가져오기
        model_path = os.environ.get("MODEL_PATH")
        labels_path = os.environ.get("LABELS_PATH")

        # 디버그 출력
        print(f"PredictionSection - 모델 경로: {model_path}")
        print(f"PredictionSection - 라벨 경로: {labels_path}")

        self.model_loader = TFLiteModelLoader(model_path, labels_path)
        self.image_processor = ImageProcessor()

    def render(self):
        """Render the prediction interface"""
        st.header("이미지 업로드")

        uploaded_file = st.file_uploader(
            "이미지를 업로드하세요", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = self.image_processor.read_image(uploaded_file)
            if image is not None:
                st.image(image, caption="업로드된 이미지", use_container_width=True)

                if st.button("분류하기"):
                    try:
                        # Process the image
                        processed_image = self.image_processor.preprocess_image(image)

                        # Make prediction
                        predicted_class, confidence = self.model_loader.predict(
                            processed_image
                        )

                        # Display results
                        st.subheader("분류 결과")
                        st.write(f"분류: {predicted_class}")
                        st.write(f"신뢰도: {confidence:.2%}")

                        # Display information
                        self._display_info(predicted_class)

                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {str(e)}")

    def _display_info(self, category: str):
        """Display information based on the predicted category

        Args:
            category (str): 분류된 카테고리
        """
        st.subheader("분류 정보")

        # 카테고리별 정보 표시
        st.write(
            f"""
            ### {category} 분류
            - 분류된 객체: {category}
            - 추가 정보: 이 이미지는 {category}로 분류되었습니다.
            """
        )

        # 카테고리별 추가 정보 표시
        category_info = {
            "병": "- 내용물을 비우고 물로 헹궈서 배출해주세요.\n- 라벨은 제거하여 별도 배출해주세요.",
            "캔": "- 내용물을 비우고 물로 헹궈서 배출해주세요.\n- 담배꽁초 등 이물질을 넣지 마세요.",
            "철": "- 이물질이 섞이지 않도록 해주세요.\n- 크기가 큰 철제품은 대형폐기물로 배출해주세요.", 
            "유리": "- 깨진 유리는 신문지에 싸서 배출해주세요.\n- 형광등, 전구류는 전용 수거함에 배출해주세요."
        }

        if category in category_info:
            st.write("### 배출 방법")
            st.write(category_info[category])
