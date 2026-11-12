import base64

from google import genai
from google.genai import types

from ..config import settings
from .base import ChatMessage, strip_thinking


class GeminiProvider:
    name = "gemini"
    multimodal = True

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None

    def available(self) -> bool:
        return self._client is not None

    async def chat(self, messages: list[ChatMessage], system: str) -> str:
        if self._client is None:
            raise RuntimeError("GEMINI_API_KEY not set")
        contents = []
        for m in messages:
            parts = [types.Part(text=m["content"])]
            if m.get("image"):  # base64 JPEG webcam snapshot of the signer
                parts.insert(0, types.Part.from_bytes(data=base64.b64decode(m["image"]), mime_type="image/jpeg"))
            contents.append(types.Content(role="model" if m["role"] == "assistant" else "user", parts=parts))
        resp = await self._client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.4),
        )
        return strip_thinking(resp.text or "")
