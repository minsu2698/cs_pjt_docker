# #################################################################
# ################# SED 통합전 Device에서 오디오 수신 및 송신############
# ################### 구동 잘됨 Test ㅇk ############################


# import pyaudio
# import numpy as np
# import time
# import requests
# import wave
# import io

# # 기본 설정
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RECORD_SECONDS = 10
# CANDIDATE_RATES = [44100, 48000, 16000]
# TARGET_KEYWORDS = ["USB", "LifeChat", "Headset", "LX-3000"]
# #TARGET_KEYWORDS = ["LifeChat", "Headset", "LX-3000"]

# # PyAudio 초기화
# p = pyaudio.PyAudio()

# # 🎤 USB 마이크 탐색 (최초 1개만)
# target_device_index = None
# print("🎤 연결된 오디오 장치:")
# for i in range(p.get_device_count()):
#     dev = p.get_device_info_by_index(i)
#     name = dev['name']
#     channels = dev.get('maxInputChannels')
#     print(f"  [{i}] {name} | ch={channels}")

#     if target_device_index is None and any(k in name for k in TARGET_KEYWORDS) and channels > 0:
#         target_device_index = i
#         print(f"✅ USB 마이크 선택됨: index={i}, name='{name}'")

# if target_device_index is None:
#     print("❌ USB 마이크를 찾을 수 없습니다. 종료합니다.")
#     exit(1)

# # 🎧 샘플레이트 자동 선택
# selected_rate = None
# for rate in CANDIDATE_RATES:
#     try:
#         print(f"🔍 샘플레이트 테스트 중: {rate}Hz")
#         test_stream = p.open(format=FORMAT, channels=CHANNELS, rate=rate,
#                              input=True, input_device_index=target_device_index,
#                              frames_per_buffer=CHUNK)
#         test_stream.close()
#         selected_rate = rate
#         print(f"✅ 성공: {rate}Hz")
#         break
#     except Exception as e:
#         print(f"❌ 실패: {rate}Hz / {e}")

# if selected_rate is None:
#     selected_rate = 44100
#     print("⚠️ fallback으로 44100Hz 사용")

# print(f"\n🎙️ 최종 설정: device_index={target_device_index}, rate={selected_rate}Hz")
# print("🎧 10초 단위로 수집 시작...")

# # 🎙️ 마이크 스트림 열기
# try:
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=selected_rate,
#                     input=True, input_device_index=target_device_index,
#                     frames_per_buffer=CHUNK)
# except Exception as e:
#     print("❌ 마이크 열기 실패:", e)
#     exit(1)

# # 🔁 메인 루프
# while True:
#     try:
#         print("⏳ 녹음 중 (10초)...")
#         frames = []

#         for _ in range(int(selected_rate / CHUNK * RECORD_SECONDS)):
#             data = stream.read(CHUNK, exception_on_overflow=False)
#             np_data = np.frombuffer(data, dtype=np.int16)
#             print("🧪 샘플 예시:", np_data[:10])
#             frames.append(data)

#         audio_bytes = b''.join(frames)

#         # 메모리에서 WAV 변환
#         wav_buffer = io.BytesIO()
#         wf = wave.open(wav_buffer, 'wb')
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(p.get_sample_size(FORMAT))
#         wf.setframerate(selected_rate)
#         wf.writeframes(audio_bytes)
#         wf.close()
#         wav_buffer.seek(0)

#         print(f"📤 WAV 전송 크기: {len(wav_buffer.getvalue())} bytes")

#         try:
#             response = requests.post(
#                 "http://sender-api:8000/sed",
#                 files={"file": ("audio.wav", wav_buffer, "audio/wav")}
#             )
#             print("📡 전송 결과:", response.status_code)
#         except Exception as e:
#             print("❌ 전송 실패:", e)

#     except Exception as e:
#         print("❌ 녹음 중 오류:", e)
#         break


# #################################################################
# #################################################################
# #################################################################



# #################################################################
# ################# SED 통합 코드 준비중 ##############################

import pyaudio
import numpy as np
import time
import wave
import os
import io
import requests
from datetime import datetime
import os
import pytz
import json
from model_utils import run_sed_model

import functools
print = functools.partial(print, flush=True)

os.environ["ORT_LOG_LEVEL"] = "ERROR"

# 📌 설정값
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 10
CANDIDATE_RATES = [44100, 48000, 16000]
#TARGET_KEYWORDS = ["USB", "LifeChat", "Headset", "LX-3000"]
TARGET_KEYWORDS = ["LifeChat", "Headset", "LX-3000"]
WAV_DIR = "temp_audio"  # 🔐 SED 컨테이너 내부 임시 저장 디렉토리


CONF_THRESH = 0.2
DEVICE_ID = "edge-A2"




os.makedirs(WAV_DIR, exist_ok=True)

# ⏰ 한국 시간 설정
KST = pytz.timezone("Asia/Seoul")

# 🎤 마이크 초기화
p = pyaudio.PyAudio()
target_device_index = None
print("🎤 연결된 오디오 장치:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    name = dev['name']
    channels = dev.get('maxInputChannels')
    print(f"  [{i}] {name} | ch={channels}")
    if target_device_index is None and any(k in name for k in TARGET_KEYWORDS) and channels > 0:
        target_device_index = i
        print(f"✅ USB 마이크 선택됨: index={i}, name='{name}'")

if target_device_index is None:
    print("❌ USB 마이크를 찾을 수 없습니다. 종료합니다.")
    exit(1)

# 🎧 샘플레이트 선택
selected_rate = None
for rate in CANDIDATE_RATES:
    try:
        print(f"🔍 샘플레이트 테스트 중: {rate}Hz")
        test_stream = p.open(format=FORMAT, channels=CHANNELS, rate=rate,
                             input=True, input_device_index=target_device_index,
                             frames_per_buffer=CHUNK)
        test_stream.close()
        selected_rate = rate
        print(f"✅ 성공: {rate}Hz")
        break
    except Exception as e:
        print(f"❌ 실패: {rate}Hz / {e}")

if selected_rate is None:
    selected_rate = 44100
    print("⚠️ fallback으로 44100Hz 사용")

print(f"\n🎙️ 최종 설정: device_index={target_device_index}, rate={selected_rate}Hz")
print("🎧 10초 단위로 수집 시작...")

# 🎙️ 마이크 스트림 시작
try:
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=selected_rate,
                    input=True, input_device_index=target_device_index,
                    frames_per_buffer=CHUNK)
    # stream = p.open(format=FORMAT, channels=CHANNELS, rate=selected_rate,
    #                 input=True, input_device_index=target_device_index,
    #                 frames_per_buffer=CHUNK)
    # stream.start_stream()
    # time.sleep(0.2)  # 🔁 마이크 워밍업 시간 (200ms)

except Exception as e:
    print("❌ 마이크 열기 실패:", e)
    exit(1)


def map_confidence_to_level(conf):
    if conf >= 0.65:
        return "Level5"
    elif conf >= 0.55:
        return "Level4"
    elif conf >= 0.45:
        return "Level3"
    elif conf >= 0.35:
        return "Level2"
    else:
        return "Level1"


# 🔁 메인 루프 시작
while True:
    try:
        print("⏳ 녹음 중 (10초)...")
        frames = []
        for _ in range(int(selected_rate / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            np_data = np.frombuffer(data, dtype=np.int16)
            #print("🧪 샘플 예시:", np_data[:10])
            # _ = np_data[:10]  # 배열을 평가하지만 출력은 하지 않음
            frames.append(data)

        audio_bytes = b''.join(frames)
        # # 🔍 디버깅용 로그 추가
        # print(f"🔁 수집된 프레임 수: {len(frames)}")
        # print(f"🔢 수집된 총 샘플 수: {len(audio_bytes) // 2} (bytes: {len(audio_bytes)})")
        # expected_samples = selected_rate * RECORD_SECONDS
        # print(f"🎯 기대 샘플 수: {expected_samples} → 차이: {expected_samples - (len(audio_bytes) // 2)}")


        now = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(WAV_DIR, f"{now}.wav")

        # 💾 WAV 파일로 저장
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(selected_rate)
            wf.writeframes(audio_bytes)

        # print(f"💾 저장 완료: {wav_path}")
        # print(f"🔁 프레임 수집 완료: {len(frames)} blocks")
        # print(f"🔢 총 샘플 수: {len(audio_bytes)//2}")
        # print(f"🎯 기대 샘플 수: {selected_rate * RECORD_SECONDS}")

        # 🧠 모델 추론
        try:
            confidence = run_sed_model(wav_path)
            print(f"🔊 Gunshot Confidence: {confidence * 100:.2f}%")
        except Exception as e:
            print(f"❌ 모델 추론 실패: {e}")
            continue


        # 📡 confidence 기준 초과 시 전송
        if confidence > CONF_THRESH :
            print("🚨 confidence 기준 초과 → sender-api 전송 시작")
            try:
                level = map_confidence_to_level(confidence)
                event_time = datetime.now(KST).isoformat()

                json_payload = {
                    "event_time": event_time,
                    "event_type": "SED",
                    #"device_id": "edge-B1",           # 필요 시 변수화
                    "level": level,                   # "Level3" 등
                    "class": "gunshot",               # 고정
                    "device_id" : DEVICE_ID
                    #"confidence": round(confidence, 4)
                }

                # WAV 파일을 메모리로 읽어 BytesIO로 감쌈
                with open(wav_path, "rb") as f:
                    wav_buffer = io.BytesIO(f.read())
                    wav_buffer.seek(0)

                files = {
                    "file": (f"{now}.wav", wav_buffer, "audio/wav")
                }
                data = {
                    "json_str": json.dumps(json_payload)
                }

                # 전송 전 로그 출력
                print("📤 Trigger Event 전송:", json_payload)
                response = requests.post("http://sender-api:8000/sed", data=data, files=files)
                print("📡 응답:", response.status_code, response.text)

            except Exception as e:
                print(f"❌ 전송 실패: {e}")
        else:
            print("🔕 confidence 기준 미달 → 전송 생략")

    except Exception as e:
        print("❌ 루프 중 예외 발생:", e)
        break
