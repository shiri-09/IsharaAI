import json
import re

from ..llm.router import llm

LANGUAGES = {"hi": "Hindi", "kn": "Kannada", "ta": "Tamil", "te": "Telugu"}

_SYSTEM = (
    "You are a precise translator. Translate the given English text into each requested Indian language, "
    "using the native script. Keep it simple and natural. Reply with ONLY a JSON object mapping the "
    "language code to the translation, no markdown, no extra text."
)


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return {k: str(v) for k, v in data.items()} if isinstance(data, dict) else {}


async def translate_all(text: str, langs: list[str]) -> dict[str, str]:
    langs = [lg for lg in langs if lg in LANGUAGES]
    if not langs or not text.strip():
        return {}
    wanted = ", ".join(f'"{lg}" ({LANGUAGES[lg]})' for lg in langs)
    prompt = f"Languages: {wanted}\n\nEnglish text:\n{text}"
    try:
        out, _ = await llm.chat([{"role": "user", "content": prompt}], _SYSTEM)
    except RuntimeError:
        return {}
    data = _parse_json(out)
    return {lg: data[lg] for lg in langs if lg in data}
