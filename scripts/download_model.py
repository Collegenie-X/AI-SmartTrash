import tensorflow as tf
import tensorflow_hub as hub
import os
from pathlib import Path

def download_and_convert_model():
    print("모델 다운로드 및 변환 시작...")
    
    # 앱 디렉토리 경로 설정
    app_dir = Path(__file__).resolve().parent.parent
    models_dir = app_dir / "app" / "models"
    
    # models 디렉토리가 없으면 생성
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # MobileNetV2 모델 다운로드
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # 모델 구조 생성
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation='softmax')  # 5개 클래스 (병, 캔, 철, 유리, 일반)
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("TFLite 모델로 변환 중...")
    
    # TFLite 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()
    
    # 모델 저장 경로
    model_path = models_dir / "model.tflite"
    
    # 모델 저장
    print(f"모델 저장 중... 저장 경로: {model_path}")
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"변환 완료! 모델이 {model_path}에 저장되었습니다.")
    print("이제 app/models/model.tflite와 app/models/labels.txt 파일을 사용할 수 있습니다.")

if __name__ == "__main__":
    download_and_convert_model() 