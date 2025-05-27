######################################################################
######################################################################
######################################################################
# 실시간 Streaming yolov5 finetuning(잘되는 base version)
######################################################################
######################################################################


# from fastapi import FastAPI, UploadFile, File, Request
# from fastapi.responses import HTMLResponse
# import uvicorn
# import cv2
# import numpy as np
# import base64
# import os
# import time
# from datetime import datetime
# import io
# import wave
# import requests  # ✅ 추가: 노트북으로 전송을 위한 라이브러리

# app = FastAPI()

# # 📁 저장 경로
# VIDEO_DIR = "received_data/video"
# AUDIO_DIR = "received_data/audio"
# os.makedirs(VIDEO_DIR, exist_ok=True)
# os.makedirs(AUDIO_DIR, exist_ok=True)

# # 📡 노트북 FastAPI 수신 서버 주소
# NOTEBOOK_SERVER_URL = "http://192.168.67.61:8000"  # ✅ 여기에 노트북 IP 입력

# @app.get("/", response_class=HTMLResponse)
# async def root():
#     return "<h2>✅ Sender API is running</h2>"

# @app.post("/stream")
# async def receive_video_stream(request: Request):
#     """
#     video frame을 base64 인코딩된 상태로 전달받음
#     """
#     data = await request.json()
#     frame_data = data.get("frame")

#     if frame_data is None:
#         return {"error": "No frame received"}

#     # base64 디코딩 → numpy 배열 → 이미지 디코딩
#     img_bytes = base64.b64decode(frame_data)
#     np_arr = np.frombuffer(img_bytes, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     if frame is not None:
#         # ✅ 저장
#         save_path = f"{VIDEO_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#         cv2.imwrite(save_path, frame)

#         print("🖼️ 프레임 수신됨:", len(img_bytes), "bytes → 저장:", save_path)

#         # # ✅ 노트북으로 프레임 전송
#         # try:
#         #     res = requests.post(f"{NOTEBOOK_SERVER_URL}/stream", json={"frame": frame_data})
#         #     print("📤 노트북에 전송 완료 /stream → 상태:", res.status_code)
#         # except Exception as e:
#         #     print("⚠️ 노트북 전송 실패:", e)

#         return {"status": "frame_saved", "path": save_path}
#     else:
#         return {"error": "Invalid frame data"}

# @app.post("/sed")
# async def receive_audio(file: UploadFile = File(...)):
#     try:
#         data = await file.read()
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         save_path = f"{AUDIO_DIR}/{timestamp}.wav"

#         # 저장
#         with open(save_path, "wb") as f:
#             f.write(data)
#         print(f"🔊 저장된 오디오: {save_path}")

#         # 예시 샘플 출력 (payload만 추출)
#         wav_buffer = io.BytesIO(data)
#         with wave.open(wav_buffer, 'rb') as wf:
#             frames = wf.readframes(wf.getnframes())
#             audio_np = np.frombuffer(frames, dtype=np.int16)

#         print(f"🎧 수신된 오디오 길이: {len(audio_np)}, 예시 값: {audio_np[:10]}")

#         # # ✅ 노트북으로 오디오 전송
#         # try:
#         #     files = {'file': (f"audio_{timestamp}.wav", data, 'audio/wav')}
#         #     res = requests.post(f"{NOTEBOOK_SERVER_URL}/sed", files=files)
#         #     print("📤 노트북에 전송 완료 /sed → 상태:", res.status_code)
#         # except Exception as e:
#         #     print("⚠️ 노트북 전송 실패:", e)

#         return {
#             "status": "received",
#             "length": len(audio_np),
#             "timestamp": time.time()
#         }

#     except Exception as e:
#         print(f"❌ 오류 발생: {e}")
#         return {"status": "error", "detail": str(e)}

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################



######################################################################
######################################################################
######################################################################
# threshold(level) 설정 및 trigger 발생시 전송 : meta 정보 + 이미지
######################################################################
######################################################################

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
import os
import shutil
import json
import io
import wave
import numpy as np
from datetime import datetime
import time
import requests

from datetime import datetime
import pytz

KST = pytz.timezone("Asia/Seoul")  # ✅ 추가

app = FastAPI()

# 📁 저장 경로 설정
BASE_DIR = "received_data"
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
IMAGE_DIR = os.path.join(BASE_DIR, "image")
META_DIR = os.path.join(BASE_DIR, "metadata")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)



# 📡 노트북 FastAPI 수신 서버 주소
NOTEBOOK_SERVER_URL = "http://192.168.67.61:8000"

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h2>✅ Sender API (Trigger-based) is running</h2>"

# ✅ 1. SED 오디오 수신
@app.post("/sed")
async def receive_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
        save_path = f"{AUDIO_DIR}/{timestamp}.wav"

        with open(save_path, "wb") as f:
            f.write(data)
        print(f"🔊 저장된 오디오: {save_path}")

        # 오디오 분석 로그
        wav_buffer = io.BytesIO(data)
        with wave.open(wav_buffer, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_np = np.frombuffer(frames, dtype=np.int16)

        print(f"🎧 수신된 오디오 길이: {len(audio_np)}, 예시: {audio_np[:10]}")

        # # ✅ 노트북으로 전송
        # try:
        #     files = {'file': (f"audio_{timestamp}.wav", data, 'audio/wav')}
        #     res = requests.post(f"{NOTEBOOK_SERVER_URL}/sed", files=files)
        #     print("📤 노트북으로 오디오 전송 완료 → 상태:", res.status_code)
        # except Exception as e:
        #     print("⚠️ 노트북 전송 실패 (/sed):", e)

        return {
            "status": "received",
            "length": len(audio_np),
            "timestamp": time.time()
        }

    except Exception as e:
        print(f"❌ 오류 발생 (/sed): {e}")
        return {"status": "error", "detail": str(e)}
    
# ✅ 2. YOLO Trigger 이벤트 수신
@app.post("/yolo")
async def receive_yolo_trigger(json_str: str = Form(...), image: UploadFile = File(...)):
    try:
        payload = json_str.strip()
        data = json.loads(payload)

        # 필수 필드 검증
        required_fields = ["event_time", "event_type", "level", "class", "device_id"]
        for field in required_fields:
            if field not in data:
                return JSONResponse(status_code=400, content={"error": f"Missing field: {field}"})

        # 저장 경로 설정
        event_time = data["event_time"].replace(":", "-").replace(".", "-")
        device_id = data["device_id"]
        level = data["level"]
        cls = data["class"]
        filename_prefix = f"{event_time}_{device_id}_{cls}_{level}"

        img_path = os.path.join(IMAGE_DIR, f"{filename_prefix}.jpg")
        meta_path = os.path.join(META_DIR, f"{filename_prefix}.json")

        # 이미지 저장
        image.file.seek(0)  # 🔥 파일 포인터 초기화 필수
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # 🔥 저장된 이미지 용량 로그 출력
        print(f"📸 저장된 이미지 파일: {img_path}, 크기: {os.path.getsize(img_path)} bytes")


        # JSON 저장
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"📦 YOLO 이벤트 저장 완료: {img_path}, {meta_path}")

        # # ✅ 노트북으로 전송 (옵션)
        # try:
        #     files = {"image": open(img_path, "rb")}
        #     json_data = {"json": json.dumps(data)}
        #     notebook_res = requests.post(f"{NOTEBOOK_SERVER_URL}/yolo", data=json_data, files=files)
        #     print("📤 노트북으로 이벤트 전송 완료 → 상태:", notebook_res.status_code)
        # except Exception as e:
        #     print("⚠️ 노트북 전송 실패 (/yolo):", e)

        return {
            "status": "success",
            "filename": filename_prefix,
            "image_path": img_path,
            "metadata_path": meta_path
        }

    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON format in 'json' field"})
    except Exception as e:
        print(f"❌ 예외 발생 (/yolo): {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################    