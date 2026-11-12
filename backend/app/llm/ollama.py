import httpx

from ..config import settings
from .base import ChatMessage, strip_thinking


class OllamaProvider:
    name = "ollama"

    @property
    def multimodal(self) -> bool:
        return settings.ollama_vision

    def available(self) -> bool:
        return bool(settings.ollama_model)

    async def chat(self, messages: list[ChatMessage], system: str) -> str:
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            out = {"role": m["role"], "content": m["content"]}
            if m.get("image") and settings.ollama_vision:
                out["images"] = [m["image"]]  # Ollama takes base64 images for vision models
            msgs.append(out)
        payload = {
            "model": settings.ollama_model,
            "messages": msgs,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        if settings.ollama_send_think:
            # Qwen3.x-style thinking models: disable thinking for fast replies (~1s vs >90s)
            payload["think"] = settings.ollama_think
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            r = await client.post(f"{settings.ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            return strip_thinking(r.json()["message"]["content"])
