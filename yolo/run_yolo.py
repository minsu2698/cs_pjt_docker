import torch
import cv2
import os
import time
import requests
import base64
from pathlib import Path
from datetime import datetime

# 🚀 YOLOv5 모델 로드
print("YOLOv5 모델 로드 중...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# ✅ 사용 가능한 /dev/video* 장치 탐색
def find_working_camera():
    print("📷 사용 가능한 웹캠 탐색 중...")
    for i in range(5):
        dev_path = f"/dev/video{i}"
        if not Path(dev_path).exists():
            continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ 사용 가능한 웹캠: {dev_path}")
                return cap
            cap.release()
    raise RuntimeError("❌ 사용 가능한 웹캠을 찾을 수 없습니다.")

# 📡 프레임 전송 함수
def send_frame_to_api(frame):
    print("📤 전송 함수 호출됨")
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        encoded = base64.b64encode(buffer).decode('utf-8')
        print("📦 base64 인코딩 완료, 길이:", len(encoded))
        
        response = requests.post("http://sender-api:8000/stream", json={"frame": encoded})
        print("📡 전송 결과:", response.status_code)
        print("📡 응답 내용:", response.text)
        
    except Exception as e:
        print("❌ 전송 실패:", repr(e))

cap = find_working_camera()

print("YOLO 추론 시작")
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 수신 실패")
        break

    results = model(frame)
    annotated_frame = results.render()[0]

    # 전송
    send_frame_to_api(annotated_frame)

    # 저장 (선택)
    save_path = f"output_{datetime.now().strftime('%H%M%S')}.jpg"
    cv2.imwrite(save_path, annotated_frame)

    print("탐지 결과:", results.pandas().xyxy[0].to_dict(orient="records"))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


