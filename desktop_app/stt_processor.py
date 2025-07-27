# desktop_app/stt_processor.py
from faster_whisper import WhisperModel
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
import time
import collections
import resampy
import traceback

class STTProcessorWorker(QThread):
    text_recognized = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, audio_capture_worker_instance, model_size="small", device="cpu", compute_type="int8"):
        super().__init__()
        self.audio_capture_worker_instance = audio_capture_worker_instance
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._running = True
        self.audio_buffer = collections.deque()
        
        self.target_stt_samplerate = 16000
        self.buffer_duration_sec = 2

    def run(self):
        try:
            print(f"Loading faster-whisper model: {self.model_size} on {self.device} ({self.compute_type})")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("Faster-Whisper model loaded.")
            print("DEBUG STT: STT processor running, waiting for audio chunks...")

            capture_samplerate = self.audio_capture_worker_instance.samplerate
            print(f"DEBUG STT: Audio captured at {capture_samplerate} Hz, target STT samplerate is {self.target_stt_samplerate} Hz.")

            while self._running:
                audio_chunk = self.audio_capture_worker_instance.get_audio_chunk()
                
                if audio_chunk is not None and audio_chunk.size > 0:
                    self.audio_buffer.append(audio_chunk.astype(np.float32))

                current_buffer_samples = sum(len(chunk) for chunk in self.audio_buffer)
                
                if current_buffer_samples / capture_samplerate >= self.buffer_duration_sec - 0.01:
                    print(f"DEBUG STT: Buffer threshold reached ({current_buffer_samples / capture_samplerate:.2f}s). Starting transcription...")
                    
                    if not self.audio_buffer:
                        print("DEBUG STT: Audio buffer is empty despite threshold being met. Skipping transcription.")
                        continue

                    audio_data = np.concatenate(list(self.audio_buffer))
                    self.audio_buffer.clear()

                    print(f"DEBUG STT: Concatenated audio data initial shape: {audio_data.shape}, total samples: {len(audio_data)}")

                    if audio_data.ndim > 1:
                        if audio_data.shape[1] > 1:
                            audio_data = audio_data.mean(axis=1)
                            print(f"DEBUG STT: Converted multi-channel to mono. New shape: {audio_data.shape}")
                        else:
                            audio_data = audio_data.flatten()
                            print(f"DEBUG STT: Flattened 2D single-channel array to 1D. New shape: {audio_data.shape}")

                    if len(audio_data) < self.target_stt_samplerate * 0.1:
                        print(f"DEBUG STT: WARNING: Processed audio data too short ({len(audio_data)} samples) for meaningful transcription. Skipping this batch.")
                        continue

                    print(f"DEBUG STT: Audio data shape before resampling: {audio_data.shape}")

                    if capture_samplerate != self.target_stt_samplerate:
                        print(f"DEBUG STT: Resampling audio from {capture_samplerate} Hz to {self.target_stt_samplerate} Hz for transcription...")
                        audio_data = np.ascontiguousarray(audio_data)
                        audio_data = resampy.resample(audio_data, capture_samplerate, self.target_stt_samplerate)
                        print("DEBUG STT: Resampling complete.")
                    
                    print(f"DEBUG STT: Audio data shape after resampling: {audio_data.shape}")

                    segments, info = self.model.transcribe(
                        audio_data,
                        beam_size=5,
                        language="en",
                        vad_filter=True
                    )
                    
                    full_text = []
                    for segment in segments:
                        full_text.append(segment.text)
                    
                    recognized_text = "".join(full_text).strip()
                    
                    if recognized_text:
                        print(f"DEBUG STT: Recognized text: '{recognized_text}'")
                        self.text_recognized.emit(recognized_text)
                    else:
                        print("DEBUG STT: Recognized text is empty or only whitespace (might be silence or non-speech).")
                
                time.sleep(0.01)

        except Exception as e:
            self.error_occurred.emit(f"STT处理错误: {e}")
            print(f"ERROR STT: STT processor error: {e}")
            traceback.print_exc()
        finally:
            print("STT processor stopped.")
            self._running = False

    def stop(self):
        self._running = False
        self.wait()