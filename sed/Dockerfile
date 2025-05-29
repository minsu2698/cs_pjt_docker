FROM python:3.9-slim

WORKDIR /app

# PyAudio 빌드를 위한 필수 패키지 추가 설치
RUN apt-get update && apt-get install -y \
    alsa-utils \
    build-essential \
    python3-dev \
    portaudio19-dev \
    libasound2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY run_sed.py .

# SED반영 KMS250528
COPY model_utils.py .
COPY SED.onnx .

CMD ["python", "run_sed.py"]

