# RabiGuard macOS 시현용 데모 가이드 (RabiGuard Mac Demo)

이 디렉토리는 Raspberry Pi 5 및 Hailo NPU(Ai Hat+2) 환경에 맞게 제작된 RabiGuard 프로젝트를 **일반 macOS 환경(MacBook 등)에서 성공적으로 실행하고 시각적으로 보여주기 위해 특제 제작된 데모 패키지**입니다.

하드웨어(NPU) 의존성이 높은 부분을 macOS 가용 자원(CPU/MPS)으로 대체하고, 관람자에게 각 추론 과정을 한눈에 쉽게 전달할 수 있도록 **미래지향적인 유리 모핑(Glassmorphism) 다크 모드 스타일의 실시간 대시보드 웹 UI**를 동봉하고 있습니다.

---

## 🌟 주요 시각화 피처 (Key Visual Features)

1. **3분할 실시간 뷰포트 (Three-Viewport Grid)**
   - **메인 감시 카메라 (YOLO Tracking)**: 맥 웹캠 피드 위에 YOLOv8 인물 트래킹 바운딩 박스, 트랙 ID, 활성화된 침입 감지 구역(다각형)을 실시간 합성하여 출력합니다.
   - **가구 세그멘테이션 (YOLOE Segmentation)**: 로컬 `yoloe-26n-seg-pf.pt` 모델을 사용해 침대, 의자, 소파 등 구역 설정 후보가 될 가구들의 영역 마스크를 화려한 색상으로 분할 렌더링합니다.
   - **단안 뎁스 맵 (Monocular Depth Heatmap)**: 로컬 PyTorch 기반 `MiDaS_small` 모델을 구동하여 대상과의 거리를 실시간 뎁스 맵(Jet Colormap 열지도) 형태로 화려하게 시각화합니다.

2. **마우스 커스텀 구역 그리기 (Interactive Polygon Drawer)**
   - 화면 우측 상단의 **"Draw Zone"**을 누른 후 메인 카메라 뷰를 마우스로 클릭하여 원하는 형태의 다각형 감지 구역을 직접 그릴 수 있습니다.
   - 구역 저장 시 팝업을 통해 감지 구역 명칭과 체류 시간 임계값(Stay Threshold)을 설정할 수 있으며, 즉시 백엔드 엔진에 동적 반영됩니다.

3. **입체적 뎁스 검증 및 침입 알림 (Spatial Verification & Alerts Feed)**
   - 인물이 구역 내에 진입했을 때, 인물의 깊이(p_depth)와 바닥 구역의 깊이(z_depth) 차이가 0.5m 이하일 때만 입체적으로 참값(True Intrusion)으로 판단합니다. (평면 사진 스포핑 방지)
   - 침입이 확정되면 우측 **Alerts Feed**에 침입 순간의 스냅샷 이미지 크롭샷과 상세 측정 수치(인물 깊이, 구역 깊이, 오차 범위)가 실시간 카드로 차곡차곡 슬라이드인 됩니다.

---

## 📁 구성 파일 일람 (Folder Structure)

- `setup_env.sh`: 맥 환경용 가상환경(`.venv`)을 생성하고 PyTorch, OpenCV, Ultralytics, Timm, Flask 등 모든 패키지를 자동 설치하는 스크립트.
- `depth_estimator.py`: MiDaS 모델을 이용해 뎁스를 추정하고 이를 미터(m) 단위 보정 및 Colormap 이미지로 반환하는 래퍼 클래스.
- `app.py`: 웹캠 프레임 캡처, YOLO 트래킹, YOLOE 세그멘테이션, 뎁스 보정 연산, 구역 침입 검증 스레드를 종합 제어하고 API 및 SSE 스트림을 제공하는 Flask 웹 서버.
- `templates/index.html`: 데모 대시보드의 메인 HTML 구조 정의 파일.
- `static/css/style.css`: 고급스러운 유리 모핑 다크 테마 및 다양한 네온 상태 라이팅 효과가 구현된 스타일시트.
- `static/js/app.js`: 실시간 SSE 텔레메트리 연동, 침입 경보 컴포넌트 렌더링, 마우스 폴리곤 드로잉 및 다이얼로그 모달 스크립트.

---

## 🚀 데모 실행 방법 (Quick Start)

### 1. 가상환경 및 패키지 셋업 (최초 1회)
터미널을 열고 `rafour-app` 디렉토리로 진입한 뒤, 환경 빌드 스크립트를 실행합니다:
```bash
chmod +x demo/setup_env.sh
./demo/setup_env.sh
```

### 2. 데모 서버 실행
설치된 가상환경 바이너리를 사용해 Flask 서버를 즉시 구동합니다:
```bash
./demo/.venv/bin/python demo/app.py
```
 위해 반드시 **허용**해 주세요.*

### 3. 대시보드 브라우저 접속
서버가 시작되면 웹 브라우저(Safari 또는 Chrome)를 열고 다음 주소에 접속합니다:
👉 **`http://localhost:5001`**

### (1). yolo, yoloe, depth(임시) 결과물 이미지 도출
test_image에 테스트할 이미지 한 개 넣고, 아래 명령어 실행
```bash
python3 demo/image_processor.py
```

---

## 🎮 시현 추천 시나리오 (Demonstration Steps)

1. **대시보드 구경**: 세 개의 카메라 뷰포트가 실시간으로 동작하며, 프레임 레이트(FPS) 및 실시간 추적 정보가 헤더에 물 흐르듯 렌더링되는 프리미엄 UI를 어필합니다.
2. **구역 추천(YOLOE) 확인**: YOLOE Segmentation 뷰에 방 안의 의자나 침대 영역이 형형색색의 마스크로 추출되는 것을 관람객에게 보여주며 기기의 공간 인지 능력을 소개합니다.
3. **구역 그리기**: 대시보드에서 **"Draw Zone"**을 활성화한 후 마우스로 가구가 있는 주변 공간을 4번 클릭해 다각형을 완성하고, `Zone_Bed_1` 등으로 저장합니다.
4. **인물 추적 및 검증**: 카메라 앵글 속으로 사람이 들어가서 방금 그린 다각형 내부에 서서 대기합니다.
5. **침입 경보(Alarm)**: 2초의 체류 임계값이 지나고 뎁스 오차 0.5m 이하 조건이 완벽히 검증되면, 화면 상단 상태 표시줄이 **"INTRUSION ALARM"**으로 붉게 점멸하고, 우측 피드에 침입 순간 찍힌 스냅샷 이미지 카드가 실시간으로 밀려 들어오는 연출을 감상합니다.
