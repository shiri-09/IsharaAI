import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
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
def letter_templates(_: User = Depends(current_user)):
    """Mean frame of every taught sample, for the browser's nearest-neighbour letter classifier."""
    out = []
    if LETTERS_DIR.exists():
        for d in sorted(LETTERS_DIR.iterdir()):
            for f in sorted(d.glob("*.npy")):
                arr = np.load(f).astype(np.float32)
                out.append({"letter": d.name, "frame": [round(float(v), 4) for v in arr.mean(0)]})
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
