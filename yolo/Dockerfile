FROM python:3.9-slim

WORKDIR /app

# 시스템 패키지 설치 (주석은 줄 끝 또는 새 줄로)
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx git curl wget \
    v4l-utils && \
    rm -rf /var/lib/apt/lists/*

# YOLOv5 코드 다운로드
RUN git clone https://github.com/ultralytics/yolov5 /app/yolov5

# PyTorch, OpenCV 등 설치
WORKDIR /app/yolov5
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio \
    opencv-python-headless requests -r requirements.txt

# ✅ gdown 설치 및 weight 다운로드
RUN pip install --no-cache-dir gdown
RUN gdown --id 1fKWFZD9DJPUaby3JEEjtchSxBSWrvA29 -O best.pt

# 사용자 추론 스크립트 복사
WORKDIR /app
COPY run_yolo.py .

CMD ["python", "run_yolo.py"]

