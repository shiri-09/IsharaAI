from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from ..services.tts import synthesize

router = APIRouter(prefix="/api", tags=["tts"])


@router.get("/tts")
async def tts(text: str = Query(min_length=1, max_length=1500), lang: str = "en"):
    try:
        audio = await run_in_threadpool(synthesize, text, lang)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # gTTS network errors etc.
        raise HTTPException(502, f"TTS failed: {e}")
    return Response(content=audio, media_type="audio/mpeg")
