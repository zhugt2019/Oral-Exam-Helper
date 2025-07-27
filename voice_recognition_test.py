import numpy as np
from faster_whisper import WhisperModel
import soundfile as sf
import os
import resampy # Import resampy

# 替换为你的音频文件路径
# 例如：'path/to/your/english_audio.wav'
AUDIO_FILE_PATH = 'D:/harvard.wav' # 请替换为实际路径

def test_transcription(audio_file_path):
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print("GPU (CUDA) detected for STT model.")
        else:
            device = "cpu"
            compute_type = "int8"
            print("No GPU detected, STT model will run on CPU.")
    except ImportError:
        device = "cpu"
        compute_type = "int8"
        print("Torch not installed or CUDA not available, STT model will run on CPU.")

    print(f"Loading faster-whisper model: base on {device} ({compute_type})")
    model = WhisperModel("base", device=device, compute_type=compute_type)
    print("Faster-Whisper model loaded.")

    try:
        # 使用 soundfile 读取音频，确保转换为 float32
        audio, samplerate = sf.read(audio_file_path, dtype='float32')

        # 如果音频是多声道，转换为单声道（faster-whisper期望单声道）
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # faster-whisper期望16000 Hz，如果不是，则进行重采样
        if samplerate != 16000:
            print(f"Resampling audio from {samplerate} Hz to 16000 Hz...")
            audio = resampy.resample(audio, samplerate, 16000)
            samplerate = 16000 # Update samplerate after resampling
            print("Resampling complete.")

        print(f"Read audio file: {audio_file_path}")
        print(f"Audio shape: {audio.shape}, Samplerate: {samplerate}")

        print("Starting transcription...")
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            language="en", # 明确指定英文
            vad_filter=True # 禁用VAD (Voice Activity Detection)
        )

        full_text = []
        for segment in segments:
            full_text.append(segment.text)
        
        recognized_text = "".join(full_text).strip()

        if recognized_text:
            print(f"\n--- Recognized Text ---\n'{recognized_text}'")
        else:
            print("\n--- No text recognized (empty or whitespace). ---")
            print("Possible reasons: low volume, heavy noise, non-speech audio, or model limitations.")

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transcription(AUDIO_FILE_PATH)