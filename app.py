import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path
import tempfile
import cv2
from PIL import Image
import io
import pandas as pd


class ModelManager:
    def __init__(self):
        self.model = None
        self.labels = []
        self.interpreter = None

    def load_model(self, model_file):
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".tflite"
            ) as tmp_file:
                tmp_file.write(model_file.getvalue())
                tmp_file_path = tmp_file.name

            # TFLite 인터프리터 초기화
            self.interpreter = tf.lite.Interpreter(model_path=tmp_file_path)
            self.interpreter.allocate_tensors()

            # 모델 정보 가져오기
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            st.success("모델이 성공적으로 로드되었습니다!")
            return True
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False

    def load_labels(self, labels_file):
        try:
            content = labels_file.getvalue().decode("utf-8")
            self.labels = [
                label.strip() for label in content.split("\n") if label.strip()
            ]
            st.success("라벨 파일이 성공적으로 로드되었습니다!")
            return True
        except Exception as e:
            st.error(f"라벨 파일 로드 중 오류 발생: {str(e)}")
            return False

    def preprocess_image(self, image):
        # 이미지를 모델 입력 크기로 조정
        image = tf.image.resize(image, (224, 224))
        # 정규화
        image = image / 255.0
        return image

    def predict(self, input_data):
        if self.interpreter is None or not self.labels:
            st.error("모델과 라벨을 먼저 로드해주세요.")
            return None

        try:
            input_data = np.array(input_data, dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0)

            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

            predictions = output_data[0]
            max_index = np.argmax(predictions)
            confidence = float(predictions[max_index])
            predicted_label = self.labels[max_index]

            return {
                "label": predicted_label,
                "confidence": confidence,
                "predictions": list(zip(self.labels, predictions.tolist())),
            }
        except Exception as e:
            st.error(f"예측 중 오류 발생: {str(e)}")
            return None


def process_image_file(image_file, model_manager):
    image = Image.open(image_file)
    image_array = np.array(image)
    processed_image = model_manager.preprocess_image(image_array)
    return processed_image


def process_camera_frame(frame, model_manager):
    # BGR을 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = model_manager.preprocess_image(frame_rgb)
    return processed_frame


def display_prediction_results(result):
    if result:
        st.write("### 🎯 분류 결과")

        # 주요 예측 결과
        col1, col2 = st.columns(2)
        with col1:
            st.metric("분류", result["label"])
        with col2:
            st.metric("신뢰도", f"{result['confidence']*100:.1f}%")

        # 진행률 바로 신뢰도 표시
        st.progress(float(result["confidence"]))

        # 모든 클래스에 대한 예측 확률을 차트로 표시
        st.write("### 📊 전체 예측 확률")
        predictions_df = pd.DataFrame(
            result["predictions"], columns=["클래스", "확률"]
        ).sort_values("확률", ascending=False)

        st.bar_chart(predictions_df.set_index("클래스"))


def main():
    st.title("🤖 AI 쓰레기 분류 시스템")
    st.write("TFLite 모델과 라벨 파일을 업로드하여 쓰레기 분류를 시작해보세요.")

    # 세션 상태 초기화
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager()

    # 사이드바에 파일 업로드 위젯 배치
    with st.sidebar:
        st.header("모델 & 라벨 업로드")

        model_file = st.file_uploader(
            "TFLite 모델 파일 선택 (.tflite)",
            type=["tflite"],
            help="학습된 TFLite 모델 파일을 업로드하세요.",
        )

        labels_file = st.file_uploader(
            "라벨 파일 선택 (.txt)",
            type=["txt"],
            help="클래스 라벨이 포함된 텍스트 파일을 업로드하세요.",
        )

        if model_file and labels_file:
            if st.button("모델 & 라벨 로드"):
                model_loaded = st.session_state.model_manager.load_model(model_file)
                labels_loaded = st.session_state.model_manager.load_labels(labels_file)

                if model_loaded and labels_loaded:
                    st.success(
                        "모델과 라벨이 준비되었습니다! 이제 분류를 시작할 수 있습니다."
                    )

    # 메인 영역
    if (
        st.session_state.model_manager.interpreter is not None
        and st.session_state.model_manager.labels
    ):
        # 탭 생성
        tab1, tab2 = st.tabs(["📸 이미지 업로드", "📹 실시간 카메라"])

        # 이미지 업로드 탭
        with tab1:
            st.header("이미지 파일로 분류하기")
            image_file = st.file_uploader(
                "분류할 이미지 선택", type=["jpg", "jpeg", "png"]
            )

            if image_file:
                st.image(image_file, caption="업로드된 이미지", use_column_width=True)

                if st.button("이미지 분류 시작"):
                    processed_image = process_image_file(
                        image_file, st.session_state.model_manager
                    )
                    result = st.session_state.model_manager.predict(processed_image)
                    display_prediction_results(result)

        # 실시간 카메라 탭
        with tab2:
            st.header("실시간 카메라로 분류하기")
            camera_placeholder = st.empty()

            if st.button("카메라 시작/정지", key="camera_toggle"):
                cap = cv2.VideoCapture(0)

                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("카메라를 찾을 수 없습니다.")
                            break

                        # 프레임 처리 및 예측
                        processed_frame = process_camera_frame(
                            frame, st.session_state.model_manager
                        )
                        result = st.session_state.model_manager.predict(processed_frame)

                        # 프레임에 결과 표시
                        if result:
                            label = result["label"]
                            confidence = result["confidence"]
                            cv2.putText(
                                frame,
                                f"{label} ({confidence*100:.1f}%)",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2,
                            )

                        # 프레임 표시
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB")

                        # 스트리밍 중지 확인
                        if not st.session_state.get("run_camera", True):
                            break

                finally:
                    cap.release()
                    st.session_state.run_camera = False
    else:
        st.info("👈 사이드바에서 모델과 라벨 파일을 먼저 업로드해주세요.")


if __name__ == "__main__":
    main()
