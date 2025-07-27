from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from ..services.rag_service import rag_service
from ..services.audio_service import audio_service
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class ChatResponse(BaseModel):
    answer: str
    sources: str

@router.post("/chat/text", response_model=ChatResponse)
async def chat_with_text(
    question: str = Form(...),
    model_provider: str = Form("gemini") # 'gemini', 'qwen'
):
    """Handles text-based questions."""
    try:
        logger.info(f"Received text question: '{question}' with model: '{model_provider}'")
        result = await rag_service.invoke_chain(question, model_provider)
        logger.info(f"Successfully processed text question. Answer length: {len(result.get('answer', ''))}")
        return result
    except Exception as e:
        logger.error(f"Error processing text question: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backend processing error: {str(e)}")

@router.post("/chat/audio", response_model=ChatResponse)
async def chat_with_audio(
    audio_file: UploadFile = File(...),
    model_provider: str = Form("gemini")
):
    """Handles audio-based questions."""
    logger.info(f"Received audio request with model: '{model_provider}' and file type: {audio_file.content_type}")
    if not audio_file.content_type.startswith("audio/"):
        logger.warning(f"Invalid audio file type received: {audio_file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid audio file.")

    try:
        # Transcribe audio to text
        transcribed_text = audio_service.transcribe_audio(audio_file.file)
        logger.info(f"Audio transcribed to text: '{transcribed_text}'")
        if not transcribed_text.strip():
            logger.warning("Transcribed text is empty or whitespace.")
            return ChatResponse(answer="Could not understand the audio. Please try again.", sources="")

        # Use the transcribed text to query RAG service
        result = await rag_service.invoke_chain(transcribed_text, model_provider)
        logger.info(f"Successfully processed audio question. Answer length: {len(result.get('answer', ''))}")
        return result
    except Exception as e:
        logger.error(f"Error processing audio question: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backend processing error: {str(e)}")

@router.get("/status")
def get_status():
    return {"status": "ok", "message": "Backend is running"}