import re
from typing import Protocol

# A chat message: {"role": "user" | "assistant", "content": str}
ChatMessage = dict[str, str]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text).strip()


class LLMProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    async def chat(self, messages: list[ChatMessage], system: str) -> str: ...
