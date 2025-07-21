from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    google_api_key: str = os.getenv("GEMINI_API_KEY")

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ChromaDB
    chroma_server_host: str = os.getenv("CHROMA_SERVER_HOST", "localhost")
    chroma_server_http_port: int = int(os.getenv("CHROMA_SERVER_HTTP_PORT", 8000))

    class Config:
        case_sensitive = True

settings = Settings()