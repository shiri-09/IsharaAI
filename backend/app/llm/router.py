import asyncio
import logging

from ..config import settings
from .base import ChatMessage, LLMProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider

log = logging.getLogger("ishara.llm")


class LLMRouter:
    """Tries the primary provider, falls back to the other on any error/timeout."""

    def __init__(self) -> None:
        providers: dict[str, LLMProvider] = {"gemini": GeminiProvider(), "ollama": OllamaProvider()}
        primary = settings.llm_primary if settings.llm_primary in providers else "gemini"
        self.order = [providers[primary]] + [p for k, p in providers.items() if k != primary]

    async def chat(self, messages: list[ChatMessage], system: str) -> tuple[str, str]:
        errors = []
        for p in self.order:
            if not p.available():
                errors.append(f"{p.name}: not configured")
                continue
            try:
                text = await asyncio.wait_for(p.chat(messages, system), timeout=settings.llm_timeout)
                if text:
                    return text, p.name
                errors.append(f"{p.name}: empty response")
            except Exception as e:  # noqa: BLE001 - fall back on anything
                log.warning("LLM provider %s failed: %r", p.name, e)
                errors.append(f"{p.name}: {e!r}")
        raise RuntimeError("All LLM providers failed -> " + " | ".join(errors))


llm = LLMRouter()
