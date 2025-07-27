# desktop_app/main_gui.py
import sys
import os
import torch
import collections
import numpy as np
import requests
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QTextEdit, QPushButton, QComboBox, QLabel,
    QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QTextCharFormat, QSyntaxHighlighter, QTextDocument, QFont

from .audio_capture import AudioCaptureWorker
from .stt_processor import STTProcessorWorker
from .rag_client import RAGClientWorker
from .config_desktop import BACKEND_URL

class CustomTextHighlighter(QSyntaxHighlighter):
    def __init__(self, parent: QTextDocument):
        super().__init__(parent)
        self.highlight_format = QTextCharFormat()
        self.highlight_format.setBackground(QColor("#aaddff"))
        self.highlight_ranges = []

    def highlightBlock(self, text: str):
        for start, length in self.highlight_ranges:
            self.setFormat(start, length, self.highlight_format)

    def setHighlightRanges(self, ranges):
        self.highlight_ranges = ranges
        self.rehighlight()

class InterviewAssistantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Interview Assistant - Desktop")
        self.setGeometry(100, 100, 1000, 800)

        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
            print("✅ GPU detected for STT. Using 'cuda' and 'float16'.")
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            print("⚠️ No GPU detected, STT will run on 'cpu' and 'int8'.")

        self.init_ui()
        self.init_workers()
        self.connect_signals()
        self.start_backend_check_timer()

        self.is_capturing = False
        self.stt_buffer_queue = collections.deque()
        self.current_stt_text = ""
        self.model_provider = "gemini"

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        
        self.llm_provider_combo = QComboBox(self)
        self.llm_provider_combo.addItem("qwen")
        self.llm_provider_combo.addItem("gemini")
        self.llm_provider_combo.currentIndexChanged.connect(self.on_llm_provider_changed)
        control_layout.addWidget(QLabel("LLM Provider:"))
        control_layout.addWidget(self.llm_provider_combo)

        self.start_capture_button = QPushButton("开始音频捕获")
        self.start_capture_button.clicked.connect(self.start_audio_capture)
        control_layout.addWidget(self.start_capture_button)

        self.stop_capture_button = QPushButton("停止音频捕获")
        self.stop_capture_button.clicked.connect(self.stop_audio_capture)
        self.stop_capture_button.setEnabled(False)
        control_layout.addWidget(self.stop_capture_button)

        self.clear_history_button = QPushButton("清除识别历史")
        self.clear_history_button.clicked.connect(self.clear_recognition_history)
        control_layout.addWidget(self.clear_history_button)

        self.backend_status_label = QLabel("后端状态: 检查中...")
        control_layout.addWidget(self.backend_status_label)

        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        self.transcript_label = QLabel("实时转录:")
        main_layout.addWidget(self.transcript_label)
        self.transcript_text_edit = QTextEdit()
        self.transcript_text_edit.setReadOnly(False)
        self.transcript_text_edit.setFont(QFont("Arial", 12))
        self.transcript_text_edit.setPlaceholderText("等待音频转录...")
        main_layout.addWidget(self.transcript_text_edit)

        self.query_button = QPushButton("使用选择的文本提问 RAG")
        self.query_button.clicked.connect(self.send_selected_text_to_rag)
        main_layout.addWidget(self.query_button)

        self.answer_label = QLabel("RAG 回答:")
        main_layout.addWidget(self.answer_label)
        self.answer_text_edit = QTextEdit()
        self.answer_text_edit.setReadOnly(True)
        self.answer_text_edit.setFont(QFont("Arial", 12))
        self.answer_text_edit.setPlaceholderText("RAG 回答将显示在这里...")
        main_layout.addWidget(self.answer_text_edit)

    def init_workers(self):
        self.audio_capture_worker = AudioCaptureWorker(samplerate=44100)
        self.stt_processor_worker = STTProcessorWorker(
            audio_capture_worker_instance=self.audio_capture_worker,
            device=self.device,
            compute_type=self.compute_type
        )
        self.rag_client_worker = RAGClientWorker()

    def connect_signals(self):
        self.audio_capture_worker.audio_data_available.connect(self.on_audio_data_available)
        self.audio_capture_worker.error_occurred.connect(self.on_worker_error)

        self.stt_processor_worker.text_recognized.connect(self.on_text_recognized)
        self.stt_processor_worker.error_occurred.connect(self.on_worker_error)

        self.rag_client_worker.rag_response_received.connect(self.on_rag_response_received)
        self.rag_client_worker.error_occurred.connect(self.on_worker_error)

    def start_backend_check_timer(self):
        self.backend_check_timer = QTimer(self)
        self.backend_check_timer.timeout.connect(self.check_backend_status)
        self.backend_check_timer.start(5000)

    def check_backend_status(self):
        try:
            response = requests.get(f"{BACKEND_URL}/status", timeout=2)
            if response.status_code == 200:
                self.backend_status_label.setText("后端状态: <font color='green'>已连接</font>")
                self.query_button.setEnabled(True)
            else:
                self.backend_status_label.setText(f"后端状态: <font color='red'>错误 ({response.status_code})</font>")
                self.query_button.setEnabled(False)
        except requests.exceptions.RequestException:
            self.backend_status_label.setText("后端状态: <font color='red'>未连接</font>")
            self.query_button.setEnabled(False)
        except Exception as e:
            self.backend_status_label.setText(f"后端状态: <font color='red'>未知错误 ({e})</font>")
            self.query_button.setEnabled(False)

    def start_audio_capture(self):
        if not self.is_capturing:
            devices = sd.query_devices()
            
            input_devices = [
                (i, dev) for i, dev in enumerate(devices) 
                if 'max_input_channels' in dev and dev['max_input_channels'] > 0
            ]
            
            if not input_devices:
                QMessageBox.warning(self, "无可用音频输入设备", "未找到任何可用的音频输入设备。请检查您的系统设置。")
                return

            display_list_for_dialog = []
            selected_device_id = None
            for i, dev in input_devices:
                display_name = dev['name']
                display_name = display_name.replace(" (Input)", "").replace(" (Output)", "").strip()
                display_list_for_dialog.append(f"{i}: {display_name} (输入: {dev['max_input_channels']} 通道)")

                name_lower = dev['name'].lower()
                if "loopback" in name_lower or "stereo mix" in name_lower or "cable output" in name_lower:
                    if selected_device_id is None:
                        selected_device_id = i
                    elif dev['max_input_channels'] > devices[selected_device_id]['max_input_channels']:
                        selected_device_id = i

            if selected_device_id is None:
                item, ok = QInputDialog.getItem(
                    self, 
                    "选择音频设备", 
                    "请选择用于捕获系统音频的设备 (如 '立体声混音' 或 虚拟声卡):\n" + "\n".join(display_list_for_dialog), 
                    display_list_for_dialog, 
                    0, 
                    False
                )
                if ok and item:
                    try:
                        selected_device_id = int(item.split(":")[0])
                    except ValueError:
                        QMessageBox.warning(self, "无效选择", "请选择一个有效的设备。")
                        return
                else:
                    return
            
            if not any(dev_id == selected_device_id for dev_id, _ in input_devices):
                QMessageBox.warning(self, "无效设备", "所选设备不是有效的输入设备。")
                return

            self.audio_capture_worker.device_id = selected_device_id
            self.audio_capture_worker.start()
            self.stt_processor_worker.start()
            self.is_capturing = True
            self.start_capture_button.setEnabled(False)
            self.stop_capture_button.setEnabled(True)
            self.transcript_text_edit.setPlaceholderText("正在捕获和转录音频...")

    def stop_audio_capture(self):
        if self.is_capturing:
            self.audio_capture_worker.stop()
            self.stt_processor_worker.stop()
            self.is_capturing = False
            self.start_capture_button.setEnabled(True)
            self.stop_capture_button.setEnabled(False)
            self.transcript_text_edit.setPlaceholderText("音频捕获已停止。")

    def on_audio_data_available(self, audio_data: np.ndarray):
        pass

    def on_text_recognized(self, text: str):
        self.current_stt_text += text + " "
        self.transcript_text_edit.setText(self.current_stt_text.strip())
        self.transcript_text_edit.verticalScrollBar().setValue(self.transcript_text_edit.verticalScrollBar().maximum())

    def on_llm_provider_changed(self, index):
        self.model_provider = self.llm_provider_combo.currentText()
        QMessageBox.information(self, "模型提供者", f"LLM 模型提供者已切换为: {self.model_provider}")

    def send_selected_text_to_rag(self):
        selected_text = self.transcript_text_edit.textCursor().selectedText()
        if not selected_text.strip():
            QMessageBox.warning(self, "无选择", "请先选择一段文本作为提问。")
            return

        print(f"User selected text: '{selected_text}'")
        self.answer_text_edit.setPlaceholderText("正在向RAG服务提问...")
        self.rag_client_worker.send_question(selected_text, self.model_provider)

    def on_rag_response_received(self, response: dict):
        answer = response.get("answer", "N/A")
        sources = response.get("sources", "无来源")
        full_response = f"回答:\n{answer}\n\n来源:\n{sources}"
        self.answer_text_edit.setText(full_response)

    def on_worker_error(self, message: str):
        QMessageBox.critical(self, "错误", message)
        print(f"Worker Error: {message}")
        if "音频捕获错误" in message:
            self.stop_audio_capture()
        elif "STT处理错误" in message:
            self.stt_processor_worker.stop()
            self.stop_audio_capture()

    def clear_recognition_history(self):
        self.current_stt_text = ""
        self.transcript_text_edit.clear()
        self.transcript_text_edit.setPlaceholderText("等待音频转录...")
        self.answer_text_edit.clear()
        self.answer_text_edit.setPlaceholderText("RAG 回答将显示在这里...")
        QMessageBox.information(self, "历史记录已清除", "实时转录和RAG回答的历史记录已清空。")

    def closeEvent(self, event):
        print("Closing application. Stopping workers...")
        self.stop_audio_capture()
        self.rag_client_worker.stop()
        self.backend_check_timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "desktop_app":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

    app = QApplication(sys.argv)
    window = InterviewAssistantGUI()
    window.show()
    sys.exit(app.exec())