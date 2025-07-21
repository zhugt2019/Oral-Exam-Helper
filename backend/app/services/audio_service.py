import whisper
import os
import tempfile
import torch

class AudioService:
    def __init__(self):
        # 检查是否存在可用的NVIDIA GPU和正确的CUDA环境
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("+"*50)
        if self.device == "cuda":
            print("✅ GPU detected! AudioService will run on 'cuda'.")
            print(f"   CUDA Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ GPU not detected. AudioService will fall back to 'cpu'.")
            print("   Note: Speech-to-text performance will be significantly slower.")
        print("+"*50)

        # 可选模型包括：tiny, base, small, medium, large
        # 更多信息请参考：https://github.com/openai/whisper/blob/main/README.md
        self.model = whisper.load_model("base")

    def transcribe_audio(self, audio_file) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        try:
            # fp16 (半精度浮点数) 可以大幅提升在现代GPU上的推理速度并减少显存占用
            # 但在CPU模式下不支持，所以我们只在cuda模式下启用它
            use_fp16 = self.device == "cuda"
            
            result = self.model.transcribe(tmp_path, fp16=use_fp16)
            
            return result["text"]
        finally:
            os.remove(tmp_path)

audio_service = AudioService()
