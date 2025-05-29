import os
os.environ["ORT_LOG_LEVEL"] = "ERROR"  # ✅ 이 줄을 반드시 import 전에

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import onnxruntime as ort

def stft_normalization(stft, mean=0.808805, variance=9.343969):
    if not isinstance(stft, torch.Tensor):
        stft = torch.tensor(stft)
    eps = 1e-6
    mean_t = torch.tensor(mean, device=stft.device, dtype=stft.dtype)
    var_t = torch.tensor(variance, device=stft.device, dtype=stft.dtype)
    std = torch.sqrt(var_t + eps)
    return (stft - mean_t) / (2.0 * std)

def nonmelSpectrogram(y, n_window, hop_length):
    return np.abs(librosa.stft(y, n_fft=n_window, hop_length=hop_length))

def get_inference_model(modelpath: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        modelpath,
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
            "AzureExecutionProvider"
        ]
    )

def read_audio(wav_filepath) -> np.ndarray:
    target_length = 160000
    wavs, sr = sf.read(wav_filepath)

    if wavs.ndim == 2:
        wavs = wavs.mean(axis=1)
    if sr != 16000:
        wavs = librosa.resample(wavs, orig_sr=sr, target_sr=16000)
    if len(wavs) < target_length:
        wavs = np.pad(wavs, (0, target_length - len(wavs)), mode='constant')
    else:
        wavs = wavs[:target_length]

    return wavs

def run_sed_model(wav_path, model_path="SED.onnx"):
    wavs = read_audio(wav_path)
    mels = nonmelSpectrogram(wavs, n_window=512, hop_length=160)
    logmels = stft_normalization(mels)
    logmels = torch.tensor(logmels)
    logmels = F.pad(logmels.transpose(0, 1), (0, 0, 0, 24)).unsqueeze(0).unsqueeze(0)
    logmels_padded = F.pad(logmels.float(), (0, 0, 0, 2))
    
    model = get_inference_model(model_path)  # ✅ 여기로 대체
    output = model.run(None, {'input': logmels_padded.numpy()})[0]
    return float(output[0][0])
    print(f"Gunshot probability: {output[0][0] * 100:.2f}%")