from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from ..core.config import settings

def get_llm(model_provider: str):
    """Factory function to get a language model instance."""
    if model_provider == "gemini":
        return ChatGoogleGenerativeAI(google_api_key=settings.google_api_key, model="gemini-1.5-flash")
    elif model_provider == "qwen":
        return ChatOllama(model="qwen3:1.7b", base_url=settings.ollama_base_url)
    else:
        raise ValueError("Unsupported model provider.")