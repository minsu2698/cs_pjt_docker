# CS_PJT: Real-Time YOLO + SED Edge Inference System

## 📌 프로젝트 개요

이 저장소는 X+AI 기반 스마트 보안 시스템의 **Edge 추론 시스템**을 구성하는 코드입니다.  
실내 CCTV(웹캠) 및 마이크로부터 실시간으로 **총기 객체**와 **총성 음원**을 감지하고, 이를 Core 시스템으로 전송하여  
Fusion 분석 및 지도 기반 **실시간 경보 서비스를 제공**합니다.

- Edge 시스템은 도커 기반으로 구동되며, 클라우드 없이 독립적으로 실행됩니다.
- Core 시스템은 Windows 기반 Anaconda 환경에서 운영되며, 별도 저장소에서 관리됩니다.

> 💡 Core 서버 저장소:  
👉 https://github.com/minsu2698/cs_pjt_receiver_win

---

## 🧩 서비스 개요 (Edge 역할 중심)

- YOLOv5 기반 객체 탐지와 EfficientNet 기반 SED를 각각 컨테이너에서 독립적으로 수행
- 감지 기준을 만족하면 `sender_api`를 통해 감지된 이미지/음성을 Core로 전송
- Core는 이벤트를 통합 분석하여 경보를 판단하고 팝업 알림을 제공합니다

---

## 🎯 실제 작동 시나리오

1. **YOLO 컨테이너** (Webcam):
   - `armed`, `gun` 클래스의 confidence ≥ 0.4일 경우,
   - 감지 이미지 + 메타데이터(JSON) → sender-api에 POST 요청

2. **SED 컨테이너** (Mic):
   - 10초 단위 오디오 수집 후 `SED.onnx` 모델로 gunshot 감지
   - confidence ≥ 0.5일 경우 WAV + JSON 메타 전송

3. **Sender API 컨테이너** (FastAPI 서버):
   - `/upload` 수신 → `received_data/`에 저장
   - 동시에 외부 Core 서버에 `/yolo` 또는 `/receive_audio`로 Relay 전송

4. **Core Fusion Logic**:
   - YOLO 또는 SED 감지 시 이벤트 기록 (각 device에 대해 개별 저장)
   - 두 감지 모두 발생 시 Fusion Level 판단 → 긴급 알림 이미지 생성 + 팝업 전송 (JPG, Wave 포함) 및 Pop up display


---

## 📂 디렉토리 구조

```
cs_pjt_docker/
├── yolo/ # YOLOv5 객체 감지 모듈
│ ├── run_yolo.py
│ └── Dockerfile
├── sed/ # Sound Event Detection (총성 감지)
│ ├── run_sed.py
│ ├── model_utils.py
│ ├── SED.onnx
│ └── Dockerfile
├── sender_api/ # FastAPI 서버 (데이터 전송)
│ ├── main.py
│ └── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## ⚙️ 실행 방법

### ✅ 전체 실행

```bash
docker-compose up --build
```

> YOLO 및 SED 컨테이너가 자동 실행되며, sender_api가 8000 포트에서 수신 대기합니다.

---

### 🔄 캐시 제거 후 새로 빌드

```bash
docker compose up --build --no-cache -d
```

### 🛑 종료
```bash
docker compose down
```

### 🧪 개별 디버깅 명령

```bash
docker compose up yolo
docker compose up sed
docker compose up sender-api
```

### 📋 개별 로그 확인

```bash
docker logs -f yolo-container
docker logs -f sed-container
docker logs -f sender-api
```

### 🔇 SED 로그에서 ONNX 출력 제거

```bash
docker logs -f sed-container 2>&1 | grep -v "graph.cc"
```

---
## 🔁 전송 구조 요약

```
[YOLO / SED 컨테이너]
        ↓ (이미지/WAV + JSON)
[sender_api 컨테이너]
        ↓ (Relay POST 요청)
[Core 수신 서버]
        ↓
Fusion 판단 → Alert 생성 → 팝업 알림

```

- `/yolo`: YOLO 실시간 추론결과, Level, Time Stamep등 Meta data POST
- `/receive_audio`: SED 실시간 추론결과, Level, Time Stamep등 Meta data POST

---

## 데이터 저장 구조 (sender_api 기준)

```
sender_api/
└── received_data/
    ├── image/     ← YOLO 이미지 저장
        ├── metadata_image/     ← YOLO meta data
    ├── audio/     ← SED 오디오 저장
        ├── metadata_audio/     ← SED meta data
```

---

## 📄 기타 참고사항

- `.gitignore`를 통해 `sender_api/received_data/{audio,video}` 내부 파일은 Git에 포함되지 않음
- `.gitkeep`로 디렉토리 구조만 유지
- 외부 수신 측 FastAPI 서버는 별도 구성 필요

---

## ✍️ 작성자

- GitHub: [@minsu2698](https://github.com/minsu2698)
- Project by: 김민수
- Core Fusion 서버: https://github.com/minsu2698/cs_pjt_receiver_win
