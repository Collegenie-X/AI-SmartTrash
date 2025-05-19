import tensorflow as tf
import tensorflow_hub as hub
import os


def download_and_convert_model():
    print("모델 다운로드 중...")

    # MobileNetV2 모델 다운로드
    model_url = (
        "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/4"
    )
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(model_url, trainable=False),
            tf.keras.layers.Dense(8, activation="softmax"),  # 8개 클래스에 맞게 조정
        ]
    )

    # 모델 컴파일
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # 입력 shape 설정
    input_shape = (1, 96, 96, 3)
    model.build(input_shape)

    print("TFLite 모델로 변환 중...")

    # TFLite 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # 모델 저장
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)

    print("변환 완료! 'model.tflite' 파일이 생성되었습니다.")
    print(
        "이제 'model.tflite'와 'labels.txt' 파일을 Streamlit 앱에서 사용할 수 있습니다."
    )


if __name__ == "__main__":
    download_and_convert_model()
