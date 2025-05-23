# 스마트 분리수거 도우미

Teachable Machine으로 학습된 모델을 활용한 스마트 분리수거 분류 애플리케이션입니다.

## 기능

- 이미지 업로드를 통한 쓰레기 분류
- 5가지 카테고리 분류 (병, 캔, 철, 유리, 일반)
- 분류 결과에 따른 분리수거 방법 안내
- 실시간 분류 신뢰도 표시

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/AI-SmartTrash.git
cd AI-SmartTrash
```

2. 가상환경 생성 및 활성화
```bash
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

1. 모델 파일 준비
- Teachable Machine에서 학습된 모델(.tflite)을 `app/models/model.tflite` 경로에 저장
- Teachable Machine에서 학습된 모델(.tflite)을 `app/models/model.labels` 경로에 저장

2. 애플리케이션 실행 (Streamlit Run)

```bash
cd app 
streamlit run main.py
```

## 프로젝트 구조
```
AI-SmartTrash/
├── app/
│   ├── components/     # UI 컴포넌트
│   ├── utils/         # 유틸리티 함수
│   ├── config/        # 설정 파일
│   └── main.py        # 메인 애플리케이션
├── models/            # 학습된 모델
├── requirements.txt   # 의존성 목록
└── README.md         # 프로젝트 문서
``` 


OpenCV에서 카메라 접근 권한과 관련된 에러가 발생하고 있네요. macOS에서 카메라 접근 권한이 필요한 상황입니다. 에러 메시지를 보면 두 가지 해결 방법이 있습니다:
macOS의 시스템 환경설정에서 카메라 접근 권한을 허용하는 방법
환경 변수 OPENCV_AVFOUNDATION_SKIP_AUTH=1를 설정하는 방법
먼저 환경 변수를 설정하여 문제를 해결
```
export OPENCV_AVFOUNDATION_SKIP_AUTH=1
```

환경 변수를 설정했습니다. 이제 프로그램을 다시 실행해보시기 바랍니다.
만약 여전히 문제가 발생한다면, macOS의 시스템 환경설정에서 카메라 접근 권한을 확인해주세요:
Apple 메뉴 > 시스템 설정 열기
개인 정보 보호 및 보안 > 카메라 메뉴로 이동
해당 애플리케이션에 대한 카메라 접근 권한을 허용