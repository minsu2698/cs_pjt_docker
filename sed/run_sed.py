import pyaudio
import numpy as np
import time
import requests
import wave
import io

# 기본 설정
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 10
CANDIDATE_RATES = [44100, 48000, 16000]
TARGET_KEYWORDS = ["USB", "LifeChat", "Headset", "LX-3000"]

# PyAudio 초기화
p = pyaudio.PyAudio()

# 🎤 USB 마이크 탐색 (최초 1개만)
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

# 🎧 샘플레이트 자동 선택
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

# 🎙️ 마이크 스트림 열기
try:
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=selected_rate,
                    input=True, input_device_index=target_device_index,
                    frames_per_buffer=CHUNK)
except Exception as e:
    print("❌ 마이크 열기 실패:", e)
    exit(1)

# 🔁 메인 루프
while True:
    try:
        print("⏳ 녹음 중 (10초)...")
        frames = []

        for _ in range(int(selected_rate / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            np_data = np.frombuffer(data, dtype=np.int16)
            print("🧪 샘플 예시:", np_data[:10])
            frames.append(data)

        audio_bytes = b''.join(frames)

        # 메모리에서 WAV 변환
        wav_buffer = io.BytesIO()
        wf = wave.open(wav_buffer, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(selected_rate)
        wf.writeframes(audio_bytes)
        wf.close()
        wav_buffer.seek(0)

        print(f"📤 WAV 전송 크기: {len(wav_buffer.getvalue())} bytes")

        try:
            response = requests.post(
                "http://sender-api:8000/sed",
                files={"file": ("audio.wav", wav_buffer, "audio/wav")}
            )
            print("📡 전송 결과:", response.status_code)
        except Exception as e:
            print("❌ 전송 실패:", e)

    except Exception as e:
        print("❌ 녹음 중 오류:", e)
        break








