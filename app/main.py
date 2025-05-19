"""
스마트 분리수거 분류 애플리케이션 메인 파일
"""

import streamlit as st
import os
from pathlib import Path
from components.header import render_header
from components.prediction_section import PredictionSection
from components.camera_section import CameraSection
from config.settings import APP_TITLE, APP_DESCRIPTION


def main():
    """메인 애플리케이션"""
    # 모델 경로 설정 (절대 경로 사용)
    base_dir = Path(__file__).resolve().parent.parent
    model_path = str(base_dir / "models" / "model.tflite")
    labels_path = str(base_dir / "models" / "labels.txt")

    # 디버그 출력
    print(f"모델 파일 경로: {model_path}")
    print(f"라벨 파일 경로: {labels_path}")
    print(
        f"파일 존재 여부 - 모델: {os.path.exists(model_path)}, 라벨: {os.path.exists(labels_path)}"
    )

    # 환경 변수 설정
    os.environ["MODEL_PATH"] = model_path
    os.environ["LABELS_PATH"] = labels_path

    # 페이지 설정
    st.set_page_config(page_title=APP_TITLE, page_icon="♻️", layout="wide")

    # 헤더 렌더링
    render_header()

    # 탭 생성
    tab1, tab2 = st.tabs(["이미지 업로드", "실시간 카메라"])

    # 이미지 업로드 탭
    with tab1:
        prediction_section = PredictionSection()
        prediction_section.render()

    # 실시간 카메라 탭
    with tab2:
        camera_section = CameraSection()
        camera_section.render()


if __name__ == "__main__":
    main()
