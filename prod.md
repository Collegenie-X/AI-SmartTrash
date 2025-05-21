# 스마트 분리수거 도우미 (Smart Recycling Assistant)

## 프로젝트 개요

실시간 컴퓨터 비전을 활용하여 쓰레기를 자동으로 분류하고 적절한 분리수거 방법을 안내하는 AI 기반 애플리케이션입니다.

## 주요 기능

1. **이미지 기반 분류**
   - 이미지 업로드를 통한 쓰레기 분류
   - 실시간 카메라 촬영을 통한 분류
   - 96x96 그레이스케일 이미지 처리

2. **분류 카테고리**
   - 병 (Bottle)
   - 캔 (Can)
   - 철 (Metal)
   - 유리 (Glass)
   - 일반 쓰레기 (General Waste)
   - 기타 (Background, ETC)

3. **실시간 분석 기능**
   - 신뢰도 점수 표시
   - 다중 예측 결과 표시 (상위 2개)
   - 실시간 카메라 피드백

4. **사용자 인터페이스**
   - 직관적인 탭 기반 인터페이스
   - 실시간 분석 결과 표시
   - 상세한 분리수거 가이드 제공

## 기술 스택

- **프레임워크**: Streamlit
- **AI/ML**: TensorFlow Lite
- **이미지 처리**: OpenCV, Pillow
- **기타**: NumPy, Python 3.10+

## 시스템 요구사항

- Python 3.10 이상
- macOS/Linux/Windows
- 웹캠 (실시간 분류 기능 사용 시)
- 최소 4GB RAM

## 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/yourusername/AI-SmartTrash.git
cd AI-SmartTrash
```

2. **가상환경 설정**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

4. **환경 변수 설정 (macOS 카메라 권한)**
```bash
export OPENCV_AVFOUNDATION_SKIP_AUTH=1
```

## 실행 방법

1. **애플리케이션 실행**
```bash
streamlit run app/main.py
```

2. **모델 및 라벨 파일 준비**
   - `models/model.tflite` 파일 확인
   - `models/labels.txt` 파일 확인

## 프로젝트 구조

```
AI-SmartTrash/
├── app/
│   ├── components/          # UI 컴포넌트
│   │   ├── camera_section.py    # 카메라 관련 컴포넌트
│   │   ├── header.py           # 헤더 컴포넌트
│   │   └── prediction_section.py # 예측 관련 컴포넌트
│   ├── config/             # 설정 파일
│   │   ├── constants.py        # 상수 정의
│   │   └── settings.py         # 환경 설정
│   ├── utils/              # 유틸리티 함수
│   └── main.py            # 메인 애플리케이션
├── models/                # 모델 파일
├── requirements.txt      # 의존성 목록
└── README.md            # 프로젝트 문서
```

## 최근 업데이트 및 개선사항

1. **카메라 처리 개선**
   - macOS 카메라 권한 문제 해결
   - 프레임 캡처 및 처리 최적화
   - 카메라 해상도 조정 (640x480)

2. **이미지 처리 파이프라인 개선**
   - 그레이스케일 변환 프로세스 최적화
   - 이미지 크기 조정 (96x96)
   - 채널 수 관리 개선 (3채널 → 1채널)

3. **UI/UX 개선**
   - 실시간 피드백 강화
   - 에러 메시지 개선
   - 분석 로그 표시 기능 추가

4. **성능 최적화**
   - 메모리 사용량 최적화
   - 프레임 처리 속도 개선
   - 에러 핸들링 강화

## 알려진 이슈

1. macOS에서 카메라 접근 시 권한 관련 이슈
   - 해결방법: `OPENCV_AVFOUNDATION_SKIP_AUTH=1` 환경변수 설정

2. 높은 해상도 카메라에서 초기 설정 이슈
   - 해결방법: 카메라 해상도를 640x480으로 자동 조정

## 향후 계획

1. 모델 성능 개선
   - 더 많은 쓰레기 카테고리 추가
   - 정확도 향상을 위한 모델 재학습

2. 사용자 경험 개선
   - 다국어 지원
   - 커스텀 테마 지원
   - 분석 히스토리 저장 기능

3. 시스템 안정성 강화
   - 자동 에러 복구 기능
   - 성능 모니터링 도구 추가

## 문제 해결 가이드

### 일반적인 문제

1. **카메라가 작동하지 않는 경우**
   - 카메라 권한 설정 확인
   - 환경 변수 설정 확인
   - 다른 프로그램의 카메라 사용 여부 확인

2. **분류 결과가 부정확한 경우**
   - 조명 상태 확인
   - 객체와 카메라 거리 조정
   - 배경 간섭 최소화

### 개발자를 위한 팁

1. **디버깅**
   - 로그 확인: `app.log` 파일 참조
   - 환경 변수 확인
   - 메모리 사용량 모니터링

2. **성능 최적화**
   - 이미지 크기 조정
   - 프레임 처리 간격 조정
   - 메모리 캐시 관리

## 라이선스

MIT License

## 연락처

프로젝트 관련 문의: [이메일 주소] 