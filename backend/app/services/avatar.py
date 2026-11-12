"""Hearing -> Deaf direction: text (from speech or typing, any of 5 languages) -> ISL gloss -> avatar motion.

Motion sources, best first:
1. A known sentence: replay a real signer's keypoint clip from the corpus (data/external, if present).
2. Word signs: key poses from real signers (data/signs, built by scripts/import_word_signs.py). Each word has
   several photos of different signers; we use the medoid pose, the one most like all the others.
3. Fingerspelling: a letter the user taught on the Fingerspelling page (data/letters) if there is one, otherwise
   the built-in ISL alphabet (data/alphabet, real two-handed ISL letters from photos of 4 signers, built by
   scripts/import_alphabet.py). Every letter A-Z is therefore signed.

All poses are body-normalised (centre between the shoulders, shoulder width = 1), x/y only, so the browser can
draw them on any avatar size.
"""

import json
import re
import threading
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from ..config import settings
from ..llm.router import llm
from ..ml.dataset import EXTERNAL_DIR, load_sentences
from ..ml.features import LH, N_POINTS, RH, normalize

SIGNS_DIR: Path = settings.data_dir / "signs"
LETTERS_DIR: Path = settings.data_dir / "letters"
ALPHABET_FILE: Path = settings.data_dir / "alphabet" / "alphabet.json"

STOP = {"A", "AN", "THE", "IS", "ARE", "AM", "WAS", "WERE", "BE", "TO", "OF", "FOR", "AND", "IT", "THAT", "THIS"}
# Small function words ISL usually leaves out. Dropped only when there is no sign for them and they were not
# explicitly marked for fingerspelling, so they are never spelled letter by letter by accident.
FUNCTION_WORDS = {"AT", "IN", "ON", "BY", "WITH", "AS", "OR", "BUT", "SO", "IF", "THEN", "HAS", "HAVE", "HAD", "DO",
                  "DOES", "DID", "WILL", "SHALL", "WOULD", "SHOULD", "COULD", "BEEN", "BEING", "INTO", "ONTO", "UPON",
                  "ITS", "THAN", "VERY", "JUST", "ALSO", "PLEASE"}

GLOSS_PROMPT = (
    "Convert the user's message into Indian Sign Language (ISL) gloss for a signing avatar.\n"
    "1. If it is not in English, translate it to English first.\n"
    "2. Split it into sentences. For each sentence write ISL gloss as a list of SINGLE UPPERCASE WORDS in ISL order "
    "(time first, then topic, then comment; question words at the end); drop articles and forms of 'to be'.\n"
    "3. Use words from VOCABULARY whenever one fits, choosing a synonym from it if needed. A name, place or other "
    "word with no sign in VOCABULARY must be written as FS:WORD so it is finger-spelled, for example FS:RAVI.\n"
    'Reply with ONLY JSON: {{"english": "...", "parts": [{{"english": "...", "gloss": ["WORD", "WORD"]}}]}}\n\n'
    "VOCABULARY: {vocab}"
)

_JSON = re.compile(r"\{.*\}", re.S)
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
_WORDS = re.compile(r"[A-Za-z']+")


def _xy(frame: np.ndarray) -> list:
    """(95,3) normalised -> [[x, y] or None] per point (None = not visible)."""
    return [None if not np.any(p) else [round(float(p[0]), 3), round(float(p[1]), 3)] for p in frame]


def _medoid(frames: np.ndarray) -> np.ndarray:
    """frames: (n, 95, 3) normalised. The frame closest to all others (arms + hands), preferring visible hands."""
    hands = np.array([np.abs(f[LH]).sum() + np.abs(f[RH]).sum() > 0 for f in frames])
    cand = frames[hands] if hands.any() else frames
    if len(cand) == 1:
        return cand[0]
    sel = np.r_[11:23, LH.start : RH.stop]
    flat = cand[:, sel, :2].reshape(len(cand), -1)
    d = ((flat[:, None, :] - flat[None, :, :]) ** 2).sum(-1)
    return cand[int(d.sum(1).argmin())]


class AvatarEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.poses: dict[str, list] = {}
        self.vocab: list[str] = []
        self.alphabet: dict[str, list] = {}
        self._loaded = False

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            idx_file = SIGNS_DIR / "index.json"
            if idx_file.exists():
                index = json.loads(idx_file.read_text())
                cache: dict[str, list] = {}
                for word, fname in index.items():
                    if fname not in cache:
                        raw = np.load(SIGNS_DIR / fname).astype(np.float32)
                        cache[fname] = _xy(_medoid(normalize(raw).reshape(len(raw), N_POINTS, 3)))
                    self.poses[word.upper()] = cache[fname]
            self.vocab = sorted(self.poses)
            if ALPHABET_FILE.exists():
                self.alphabet = json.loads(ALPHABET_FILE.read_text())
            self._loaded = True

    # ---- fingerspelling letters ------------------------------------------------------------------------------------
    @staticmethod
    def taught_pose(ch: str) -> list | None:
        """The user's own taught letter (Fingerspelling page), if any."""
        folder = LETTERS_DIR / ch.upper()
        files = sorted(folder.glob("*.npy")) if folder.exists() else []
        if not files:
            return None
        frames = np.stack([np.load(f).astype(np.float32).reshape(-1, N_POINTS * 3).mean(0) for f in files])
        return _xy(_medoid(normalize(frames).reshape(len(frames), N_POINTS, 3)))

    def letter_pose(self, ch: str) -> tuple[list | None, str]:
        """(pose, source): the user's taught letter first, then the built-in ISL alphabet."""
        pose = self.taught_pose(ch)
        if pose is not None:
            return pose, "taught"
        self.load()
        pose = self.alphabet.get(ch.upper())
        return (pose, "builtin") if pose is not None else (None, "none")

    # ---- sentence clips (real motion) ------------------------------------------------------------------------------
    @staticmethod
    def sentence_clip(text: str) -> list | None:
        best, score = None, 0.0
        for s in load_sentences():
            r = SequenceMatcher(None, s["text"].lower().strip(" .?!"), text.lower().strip(" .?!")).ratio()
            if r > score:
                best, score = s, r
        if best is None or score < 0.86:
            return None
        return AvatarEngine.clip_for(best["id"])

    @staticmethod
    def clip_for(sid: str) -> list | None:
        """Keypoint motion of a real signer for sentence `sid` (the most complete recording)."""
        files = sorted((EXTERNAL_DIR / sid).glob("*.npy"))
        if not files:
            return None
        clips = [np.load(f).astype(np.float32) for f in files]
        # largest share of frames with a hand tracked = clearest recording
        raw = max(clips, key=lambda c: float(np.mean([np.any(fr[LH.start * 3 : RH.stop * 3]) for fr in c])))
        norm = normalize(raw).reshape(len(raw), N_POINTS, 3)
        step = max(1, len(norm) // 60)  # keep it light: at most ~60 frames
        return [_xy(f) for f in norm[::step]]

    # ---- text -> gloss ---------------------------------------------------------------------------------------------
    def _tokens(self, items: list) -> list[str]:
        """Split phrases into single glosses, keeping multi-word signs (e.g. 'A LOT', 'ON THE WAY') together."""
        words = []
        for it in items:
            for w in str(it).upper().replace(",", " ").split():
                if w.startswith("FS:"):
                    words.append(w)
                else:
                    w = w.strip(".?!;:")
                    if _WORDS.fullmatch(w):
                        words.append(w)
        out, i = [], 0
        while i < len(words):
            for n in (3, 2, 1):
                phrase = " ".join(words[i : i + n])
                if n == 1 or phrase in self.poses:
                    out.append(phrase)
                    i += n
                    break
        return [w for w in out if w.startswith("FS:") or w in self.poses or (w not in STOP and w not in FUNCTION_WORDS)]

    async def to_gloss(self, text: str) -> dict:
        self.load()
        try:
            out, _ = await llm.chat([{"role": "user", "content": text}], GLOSS_PROMPT.format(vocab=", ".join(self.vocab)))
            m = _JSON.search(out)
            data = json.loads(m.group(0)) if m else {}
        except (RuntimeError, json.JSONDecodeError):
            data = {}
        parts = [p for p in data.get("parts", []) if isinstance(p, dict) and p.get("english") and p.get("gloss")]
        if not parts and not text.isascii():  # non-English input the gloss step could not handle: translate first
            try:
                eng, _ = await llm.chat(
                    [{"role": "user", "content": text}], "Translate this into simple English. Reply with only the translation."
                )
                if eng.strip() and eng.strip() != text:
                    return await self.to_gloss(eng.strip())
            except RuntimeError:
                pass
        if not parts:  # fallback without an LLM: one part per sentence, words as glosses
            pieces = [x.strip() for x in _SENT_SPLIT.split(text) if x.strip()]
            parts = [{"english": x, "gloss": _WORDS.findall(x)} for x in pieces]
        return {"english": data.get("english") or text, "parts": parts}

    def _segments(self, gloss: list[str]) -> list[dict]:
        segs = []
        for g in gloss:
            word = g[3:] if g.startswith("FS:") else g
            if not g.startswith("FS:") and word in self.poses:
                segs.append({"type": "sign", "label": word, "pose": self.poses[word]})
                continue
            for ch in re.sub(r"[^A-Z]", "", word):
                pose, src = self.letter_pose(ch)
                segs.append({"type": "letter", "label": ch, "pose": pose, "source": src, "word": word})
        return segs

    async def plan(self, text: str, spell: bool = False) -> dict:
        if spell:  # no LLM, no sentence clip: every word is finger-spelled exactly as written
            words = [w.upper() for w in re.findall(r"[A-Za-z]+", text)]
            data = {"english": text, "parts": [{"english": text, "gloss": [f"FS:{w}" for w in words]}]} if words else {"english": text, "parts": []}
        else:
            data = await self.to_gloss(text)
        parts, segments, all_gloss = [], [], []
        for p in data["parts"]:
            gloss = self._tokens(p.get("gloss") or [])
            clip = None if spell else self.sentence_clip(p["english"])
            segs = [{"type": "clip", "label": p["english"], "frames": clip}] if clip else self._segments(gloss)
            parts.append({"english": p["english"], "gloss": gloss, "real_motion": bool(clip)})
            segments.extend(segs)
            all_gloss.extend(gloss)
        return {
            "english": data["english"],
            "gloss": all_gloss,
            "parts": parts,
            "segments": segments,
            "coverage": {
                "clips": sum(1 for s in segments if s["type"] == "clip"),
                "signed": sum(1 for s in segments if s["type"] == "sign"),
                "letters": sum(1 for s in segments if s["type"] == "letter"),
                "letters_taught": sum(1 for s in segments if s["type"] == "letter" and s.get("source") == "taught"),
                "letters_signed": sum(1 for s in segments if s["type"] == "letter" and s["pose"]),
            },
        }


avatar = AvatarEngine()
