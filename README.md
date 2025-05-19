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
- Teachable Machine에서 학습된 모델(.h5)을 `models/model.h5` 경로에 저장

2. 애플리케이션 실행
```bash
streamlit run app/main.py
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