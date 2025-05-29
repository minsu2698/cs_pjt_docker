######################################################################
######################################################################
######################################################################
# 실시간 Streaming yolov5 finetuning(잘되는 base version)
######################################################################
######################################################################


# import torch
# import cv2
# import os
# import time
# import requests
# import base64
# from pathlib import Path
# from datetime import datetime


# # 🚀 YOLOv5 모델 로드
# print("YOLOv5 모델 로드 중...")
# #기존코드(잘동작)
# #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# #변경코드(finetuning)
# model = torch.hub.load('./yolov5', 'custom', path='yolov5/best.pt', source='local')

# # ✅ 사용 가능한 /dev/video* 장치 탐색
# def find_working_camera():
#     print("📷 사용 가능한 웹캠 탐색 중...")
#     for i in range(5):
#         dev_path = f"/dev/video{i}"
#         if not Path(dev_path).exists():
#             continue
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 print(f"✅ 사용 가능한 웹캠: {dev_path}")
#                 return cap
#             cap.release()
#     raise RuntimeError("❌ 사용 가능한 웹캠을 찾을 수 없습니다.")

# # 📡 프레임 전송 함수
# def send_frame_to_api(frame):
#     print("📤 전송 함수 호출됨")
#     try:
#         _, buffer = cv2.imencode('.jpg', frame)
#         encoded = base64.b64encode(buffer).decode('utf-8')
#         print("📦 base64 인코딩 완료, 길이:", len(encoded))
        
#         response = requests.post("http://sender-api:8000/stream", json={"frame": encoded})
#         print("📡 전송 결과:", response.status_code)
#         print("📡 응답 내용:", response.text)
        
#     except Exception as e:
#         print("❌ 전송 실패:", repr(e))

# cap = find_working_camera()

# print("YOLO 추론 시작")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("프레임 수신 실패")
#         break

#     results = model(frame)
#     annotated_frame = results.render()[0]

#     # 전송
#     send_frame_to_api(annotated_frame)

#     # 저장 (선택)
#     save_path = f"output_{datetime.now().strftime('%H%M%S')}.jpg"
#     cv2.imwrite(save_path, annotated_frame)

#     print("탐지 결과:", results.pandas().xyxy[0].to_dict(orient="records"))

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


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

import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import cv2
import os
import time
import requests
import base64
from pathlib import Path
from datetime import datetime
import json
import pytz

# 맨 위에서 monkey patch
import functools
print = functools.partial(print, flush=True)

KST = pytz.timezone("Asia/Seoul")



# 🔧 설정
ALERT_CLASSES = ['gun', 'armed']
CONF_THRESH = 0.6
DEVICE_ID = "edge-A1"

# 🚀 YOLOv5 모델 로드
print("YOLOv5 모델 로드 중...")
model = torch.hub.load('./yolov5', 'custom', path='yolov5/best.pt', source='local')

# ✅ 웹캠 자동 탐색
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

# ✅ 대표 클래스 및 confidence 추출
def get_representative_class(detections):
    filtered = [d for d in detections if d['name'] in ALERT_CLASSES and d['confidence'] >= CONF_THRESH]
    if not filtered:
        return None
    sorted_classes = sorted(filtered, key=lambda d: (ALERT_CLASSES.index(d['name']), -d['confidence']))
    rep = sorted_classes[0]
    return rep['name'], rep['confidence']

# ✅ confidence → level 매핑
def map_confidence_to_level(conf):
    if conf >= 0.95:
        return "Level5"
    elif conf >= 0.85:
        return "Level4"
    elif conf >= 0.75:
        return "Level3"
    elif conf >= 0.65:
        return "Level2"
    else:
        return "Level1"

# ✅ Trigger 발생 시 전송 (JSON + 이미지)
def send_trigger_event(frame, rep_class, confidence):
    try:
        now = datetime.now(KST)
        timestamp = now.strftime('%Y%m%d_%H%M%S_%f')
        filename = f"event_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        time.sleep(0.05)  # ✅ 파일 저장 대기

        level = map_confidence_to_level(confidence)

        json_payload = {
            "event_time": now.isoformat(),
            "event_type": "YOLO",
            "level": level,
            "class": rep_class,
            "device_id": DEVICE_ID
        }

        with open(filename, "rb") as f:  # ✅ 안정적인 파일 핸들 처리
            files = {"image": f}
            data = {"json_str": json.dumps(json_payload)}

            print("📤 Trigger Event 전송:", json_payload)
            response = requests.post("http://sender-api:8000/yolo", data=data, files=files)
            print("📡 응답:", response.status_code, response.text)

    except Exception as e:
        print("❌ Trigger Event 전송 실패:", repr(e))


# ✅ 실행 시작
cap = find_working_camera()
print("🎬 YOLO 추론 시작")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 수신 실패")
        break

    # YOLO 추론
    results = model(frame)
    annotated_frame = results.render()[0]

    # 탐지 결과 분석
    detection_data = results.pandas().xyxy[0].to_dict(orient="records")
    rep_result = get_representative_class(detection_data)
    if rep_result:
        rep_class, confidence = rep_result
        send_trigger_event(annotated_frame, rep_class, confidence)

    # 종료 조건 (Q 키 입력 시 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


######################################################################
######################################################################
######################################################################
######################################################################
######################################################################