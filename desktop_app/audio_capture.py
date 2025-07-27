# desktop_app/audio_capture.py
import sounddevice as sd
import numpy as np
import queue
from PyQt6.QtCore import QThread, pyqtSignal
import time # 确保导入了 time 模块

class AudioCaptureWorker(QThread):
    # 信号用于向主线程发送音频数据块
    audio_data_available = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, device_id=None, samplerate=16000, channels=1, chunk_size=1024):
        super().__init__()
        self.device_id = device_id
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_size = chunk_size
        self._running = True
        self.q = queue.Queue()

    def run(self):
        try:
            # 尝试找到默认的循环回放设备 (Loopback device)
            if self.device_id is None:
                devices = sd.query_devices()
                default_output_device_idx = sd.default.device[1]
                default_output_device_name = devices[default_output_device_idx]['name']
                print(f"DEBUG: Default output device: {default_output_device_name}")

                loopback_device_id = None
                for i, dev in enumerate(devices):
                    if 'max_input_channels' in dev and dev['max_input_channels'] > 0:
                        name = dev['name'].lower()
                        if "loopback" in name or "立体声混音" in name or "stereo mix" in name or "cable output" in name:
                            loopback_device_id = i
                            print(f"DEBUG: Found potential loopback device: {dev['name']} (ID: {i})")
                            break
                
                if loopback_device_id is not None:
                    self.device_id = loopback_device_id
                else:
                    self.error_occurred.emit("未找到合适的Loopback音频设备。请确保已启用'立体声混音'或安装虚拟声卡并将其作为Teams的输出。")
                    self._running = False
                    print("DEBUG: Device ID not found, setting _running to False and returning.")
                    return # 在没有找到设备时直接返回

            # 添加详细的设备信息打印，以便调试
            print(f"Starting audio capture from device ID: {self.device_id}, Samplerate: {self.samplerate}, Channels: {self.channels}, Chunk Size: {self.chunk_size}")
            
            # --- START DEBUG LOGS ---
            print(f"DEBUG: Attempting to open audio stream for device ID: {self.device_id}")
            device_info = sd.query_devices(self.device_id)
            print(f"DEBUG: Selected device info: {device_info}")
            # --- END DEBUG LOGS ---

            with sd.InputStream(
                device=self.device_id,
                samplerate=self.samplerate,
                channels=self.channels,
                callback=self._audio_callback,
                blocksize=self.chunk_size # 设置回调函数的块大小
            ):
                # --- START DEBUG LOGS ---
                print("DEBUG: Audio input stream successfully opened and entered 'with' block. Entering capture loop.")
                # --- END DEBUG LOGS ---
                while self._running:
                    # 在此处可以进行一些UI更新或其他轻量级操作，或者只是等待
                    sd.sleep(100) # 避免忙等待，减少CPU使用
                # --- START DEBUG LOGS ---
                print("DEBUG: Audio capture loop exited.")
                # --- END DEBUG LOGS ---

        except sd.PortAudioError as pa_e:
            # 专门捕获 PortAudio 错误
            self.error_occurred.emit(f"音频端口错误: {pa_e}. 请检查设备ID、采样率和通道设置。")
            print(f"ERROR: PortAudioError: {pa_e}")
        except Exception as e:
            self.error_occurred.emit(f"音频捕获错误: {e}")
            print(f"ERROR: General Audio Capture Error: {e}", exc_info=True) # 打印完整的异常堆栈
        finally:
            print("Audio capture stopped.")
            self._running = False

    def _audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(f"Audio callback status: {status}", flush=True)
        # print(f"DEBUG: Received audio data block: {indata.shape}") # 注意: 这个会非常频繁，只在必要时打开
        # 将音频数据放入队列，供其他线程处理
        self.q.put(indata.copy())

    def stop(self):
        self._running = False
        # 清空队列以确保线程可以干净退出
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.wait() # 等待线程结束

    def get_audio_chunk(self):
        """从队列中获取音频数据块"""
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None
