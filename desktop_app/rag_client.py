# desktop_app/rag_client.py
import requests
from PyQt6.QtCore import QThread, pyqtSignal
from .config_desktop import BACKEND_URL

class RAGClientWorker(QThread):
    # 信号用于向主线程发送RAG结果
    rag_response_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = True

    def send_question(self, question: str, model_provider: str):
        """非阻塞地发送RAG请求"""
        self.question = question
        self.model_provider = model_provider
        # 启动线程执行网络请求
        if not self.isRunning():
            self.start()

    def run(self):
        try:
            print(f"Sending RAG question: '{self.question}' with model: '{self.model_provider}'")
            data = {"question": self.question, "model_provider": self.model_provider}
            response = requests.post(f"{BACKEND_URL}/chat/text", data=data, timeout=120)

            if response.status_code == 200:
                self.rag_response_received.emit(response.json())
            else:
                self.error_occurred.emit(f"RAG请求错误: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(f"RAG请求网络错误: {e}")
        except Exception as e:
            self.error_occurred.emit(f"RAG请求未知错误: {e}")
        finally:
            self.quit() # 请求完成后退出线程

    def stop(self):
        self._running = False
        if self.isRunning():
            self.quit()
            self.wait()