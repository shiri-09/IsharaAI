import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..auth import current_user
from ..ml.features import FRAME_DIM
from ..models import User
from ..services.avatar import LETTERS_DIR, avatar

router = APIRouter(prefix="/api", tags=["avatar and fingerspelling"])


class PlanIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    spell: bool = False  # True: finger-spell every word, letter by letter (names, short forms)


@router.post("/avatar/plan")
async def plan(body: PlanIn, _: User = Depends(current_user)):
    await run_in_threadpool(avatar.load)
    return await avatar.plan(body.text.strip(), spell=body.spell)


@router.get("/avatar/vocab")
async def vocab():
    await run_in_threadpool(avatar.load)
    return {"words": avatar.vocab, "count": len(avatar.vocab)}


# ---------------------------------------------------------------- fingerspelling letter templates
class LetterIn(BaseModel):
    letter: str = Field(pattern=r"^[A-Za-z]$")
    frames: list[list[float]] = Field(min_length=3, max_length=120)


def _letter_counts() -> dict:
    if not LETTERS_DIR.exists():
        return {}
    return {d.name: len(list(d.glob("*.npy"))) for d in sorted(LETTERS_DIR.iterdir()) if d.is_dir()}


@router.get("/letters")
def letters(_: User = Depends(current_user)):
    return _letter_counts()


@router.get("/letters/templates")
async def letter_templates(_: User = Depends(current_user)):
    """One frame per sample for the browser's nearest-neighbour letter classifier: the user's own taught samples, plus the built-in
    ISL alphabet (real signers' hand shapes) for every letter the user has not taught, so spelling works out of the box."""
    await run_in_threadpool(avatar.load)
    out = []
    taught = set()
    if LETTERS_DIR.exists():
        for d in sorted(LETTERS_DIR.iterdir()):
            for f in sorted(d.glob("*.npy")):
                arr = np.load(f).astype(np.float32)
                out.append({"letter": d.name, "frame": [round(float(v), 4) for v in arr.mean(0)], "builtin": False})
                taught.add(d.name)
    for ch, pose in avatar.alphabet.items():
        if ch in taught or not pose:
            continue
        frame = []
        for pt in pose[:95]:
            frame += [float(pt[0]), float(pt[1]), 0.0] if pt else [0.0, 0.0, 0.0]
        if len(frame) == FRAME_DIM:
            out.append({"letter": ch, "frame": [round(v, 4) for v in frame], "builtin": True})
    return out


@router.post("/letters")
def add_letter(body: LetterIn, _: User = Depends(current_user)):
    arr = np.asarray(body.frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != FRAME_DIM:
        raise HTTPException(400, f"frames must be (T, {FRAME_DIM})")
    folder = LETTERS_DIR / body.letter.upper()
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{int(time.time() * 1000)}.npy", arr)
    return _letter_counts()


@router.delete("/letters/{letter}")
def clear_letter(letter: str, _: User = Depends(current_user)):
    folder = LETTERS_DIR / letter.upper()[:1]
    if folder.exists():
        for f in folder.glob("*.npy"):
            f.unlink()
    return _letter_counts()


# ---------------------------------------------------------------- the ISL alphabet library
PHOTOS = LETTERS_DIR.parent / "alphabet" / "photos"


@router.get("/alphabet")
async def alphabet_list(_: User = Depends(current_user)):
    """A to Z for the Sign library: the avatar's handshape for each letter, whether a real signer's photo exists, and whether
    the user has taught their own version."""
    await run_in_threadpool(avatar.load)
    taught = _letter_counts()
    out = []
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        pose = avatar.alphabet.get(ch)
        hands = sum(1 for h0 in (33, 54) if pose and pose[h0]) if pose else 0
        out.append({"letter": ch, "pose": pose, "hands": hands, "has_photo": (PHOTOS / f"{ch}.jpg").exists(), "taught": taught.get(ch, 0)})
    return out


@router.get("/alphabet/{letter}/photo")
def alphabet_photo(letter: str):
    """A real signer showing the letter (CC BY 4.0, see data/alphabet/SOURCE.md). Public so an <img> tag can load it."""
    f = PHOTOS / f"{letter.upper()[:1]}.jpg"
    if not f.exists():
        raise HTTPException(404, "No photo for this letter")
    return FileResponse(f, media_type="image/jpeg")
