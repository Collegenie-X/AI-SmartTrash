"""
Prediction section component for the Smart Trash Classification App
"""

import streamlit as st
from utils.model_loader import ModelLoader
from utils.image_processor import ImageProcessor
from config.settings import CATEGORIES


class PredictionSection:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.image_processor = ImageProcessor()

    def render(self):
        """Render the prediction interface"""
        uploaded_file = st.file_uploader(
            "이미지를 업로드하세요", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="업로드된 이미지", use_column_width=True)

            if st.button("분류하기"):
                try:
                    # Process the image
                    processed_image = self.image_processor.preprocess_image(
                        uploaded_file
                    )

                    # Make prediction
                    predicted_class, confidence = self.model_loader.predict(
                        processed_image
                    )

                    # Display results
                    st.markdown("### 분류 결과")
                    st.success(f"분류: {CATEGORIES[predicted_class]}")
                    st.info(f"신뢰도: {confidence:.2%}")

                    # Display recycling information
                    self._display_recycling_info(predicted_class)

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")

    def _display_recycling_info(self, category):
        """Display recycling information based on the predicted category"""
        recycling_info = {
            0: "병은 깨끗이 씻어서 분리수거해주세요.",
            1: "캔은 내용물을 비우고 깨끗이 씻어서 분리수거해주세요.",
            2: "철은 다른 재활용품과 분리하여 배출해주세요.",
            3: "유리는 깨끗이 씻어서 분리수거해주세요.",
            4: "일반쓰레기는 종량제 봉투에 담아 배출해주세요.",
        }

        st.markdown("### 분리수거 방법")
        st.write(recycling_info[category])
