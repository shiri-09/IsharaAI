import json
import re

from ..llm.router import llm

LANGUAGES = {"hi": "Hindi", "kn": "Kannada", "ta": "Tamil", "te": "Telugu"}

_SYSTEM = (
    "You are a precise translator. Translate the given English text into each requested Indian language, "
    "using the native script (Devanagari for Hindi, Kannada script, Tamil script, Telugu script), never Latin letters. Keep it simple and natural. Reply with ONLY a JSON object mapping the "
    "language code to the translation, no markdown, no extra text."
)


SCRIPTS = {"hi": ("Devanagari", 0x0900, 0x097F), "kn": ("Kannada", 0x0C80, 0x0CFF), "ta": ("Tamil", 0x0B80, 0x0BFF),
           "te": ("Telugu", 0x0C00, 0x0C7F)}


def _right_script(text: str, lang: str) -> bool:
    """At least half of the letters are in the language's own script (numbers, names and references may be Latin)."""
    _, lo, hi = SCRIPTS[lang]
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(lo <= ord(c) <= hi for c in letters) >= 0.5 * len(letters)


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
    result = {lg: data[lg] for lg in langs if lg in data and _right_script(data[lg], lg)}
    # local models sometimes answer in Latin letters (e.g. "Ho gaya hai"): retry those languages one by one
    for lg in [x for x in langs if x not in result]:
        try:
            one, _ = await llm.chat(
                [{"role": "user", "content": text}],
                f"Translate the user's English text into {LANGUAGES[lg]}. Write it ONLY in the {SCRIPTS[lg][0]} script, "
                f"never in Latin letters. Reply with only the translation.",
            )
        except RuntimeError:
            continue
        one = one.strip().strip('"')
        if _right_script(one, lg):
            result[lg] = one
    return result
