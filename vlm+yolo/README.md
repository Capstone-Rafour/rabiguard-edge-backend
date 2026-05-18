# VLM + YOLO (NPU & CPU Hybrid Architecture)

이 디렉토리는 Hailo NPU(SCDepthV3, VLM)와 CPU(YOLO26n NCNN)를 혼합하여 실시간으로 객체를 감지하고, 깊이(Depth)를 검증한 뒤, 이상 상황을 VLM으로 분석하는 아키텍처 구현체들을 포함하고 있습니다.

## 📂 파일 구성 및 설명

### 1. `arch1_parallel.py` (권장 - 실시간 병렬 처리)
*   **방식**: 카메라 프레임을 GStreamer 기반 NPU 파이프라인(SCDepthV3)에 태워 전체 화면의 깊이(Depth Map)를 실시간으로 추출함과 동시에, CPU에서 YOLO를 돌려 객체를 추적합니다.
*   **특징**: 성능 저하(딜레이)가 가장 적고 로직이 단순하며 속도가 빠릅니다. (가장 추천하는 방식)
*   **실행**: `python3 arch1_parallel.py` (웹캠 기본 사용)

### 2. `arch1_parallel_video.py` (비디오 입력 테스트용)
*   **방식**: `arch1_parallel.py`와 완벽히 동일한 구조이나, 라이브 웹캠 대신 로컬 비디오 파일을 입력으로 사용합니다.
*   **입력 파일**: `../_inputs/test_video_1.MP4`
*   **실행**: `python3 arch1_parallel_video.py`

### 3. `npu_vlm_camera_ncnn.py` (기본 연동 버전)
*   **방식**: YOLO 객체 감지(CPU)와 VLM 모델(NPU)만을 연동한 기본적인 상황 인식 스크립트입니다. Depth 판별 로직은 포함되어 있지 않습니다.
*   **특징**: `arch1`, `arch2`로 넘어가기 전 기능 검증용으로 구현된 기본 파이프라인 형태입니다.
*   **실행**: `python3 npu_vlm_camera_ncnn.py`

### 4. `arch2_sequential.py` (순차적 트리거 처리)
*   **방식**: 평소에는 CPU에서 YOLO만 실행하며 프레임을 분석합니다. 객체가 특정 구역에 3초 이상 머무르는 **이벤트가 발생할 때만** 해당 프레임을 NPU(SCDepthV3)로 넘겨 깊이를 계산합니다.
*   **특징**: 평상시 NPU 전력을 아낄 수 있으나, 이벤트 발생 시 NPU 모델 간 컨텍스트 스위칭 및 연산으로 인한 약간의 초기 지연(Delay)이 발생할 수 있습니다.
*   **실행**: `python3 arch2_sequential.py` (웹캠 기본 사용)

---

## 🛠️ 요구 사항 (Requirements)

1.  **Hailo-apps 및 Hailo 플랫폼**: 시스템 내에 Hailo NPU 환경(`hailo_platform`) 및 `hailo-apps` 패키지가 정상적으로 구성되어 있어야 합니다.
2.  **YOLO NCNN 모델**: 상위 디렉토리에 `yolo26n_ncnn_model` 폴더가 존재해야 합니다.
3.  **Ultralytics NCNN 패키지**: 파이썬 환경에 ncnn 지원 ultralytics가 설치되어야 합니다.

## ⚙️ 로직 흐름

1.  **YOLO 탐지**: 사람(Class 0)을 감지하고 Bounding Box 중앙점이 지정된 ROI(관심 구역) 내에 있는지 확인합니다.
2.  **시간 검증**: 해당 객체가 ROI 구역 내에 `3초` 이상 머무르는지(Tracking) 확인합니다.
3.  **Depth 검증**: 해당 객체의 Depth(거리)와 ROI 구역 자체의 평균 Depth를 비교하여, 오차가 `0.5m` 이내인지 확인합니다. (단순히 화면에 비치는 것이 아니라 실제로 그 구역 위에 있는지 입체적으로 확인)
4.  **VLM 분석**: 모든 조건을 통과하면, 강조 표시된 프레임을 NPU의 VLM(Vision Language Model)으로 넘겨 상황을 15단어 이내로 요약/알림합니다.