import io
from functools import lru_cache

from gtts import gTTS

SUPPORTED = {"en", "hi", "kn", "ta", "te"}


@lru_cache(maxsize=256)
def _synth(text: str, lang: str) -> bytes:
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()


def synthesize(text: str, lang: str) -> bytes:
    if lang not in SUPPORTED:
        raise ValueError(f"Unsupported language: {lang}")
    return _synth(text, lang)
