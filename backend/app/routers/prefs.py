from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import get_db
from ..models import User, UserPrefs

router = APIRouter(prefix="/api/prefs", tags=["preferences"])


class PrefsIn(BaseModel):
    mode: Literal["isl", "aac"] | None = None
    onboarded: bool | None = None
    voice: Literal["en", "hi", "kn", "ta", "te"] | None = None
    text_size: Literal["normal", "large", "xlarge"] | None = None
    scan_ms: int | None = Field(default=None, ge=600, le=5000)
    avatar_speed: Literal["slow", "normal", "fast"] | None = None
    auto_speak: bool | None = None
    high_contrast: bool | None = None


def _out(p: UserPrefs | None) -> dict:
    return {
        "mode": p.mode if p else "",
        "onboarded": bool(p and p.onboarded),
        "voice": (p.voice if p else None) or "hi",
        "text_size": (p.text_size if p else None) or "normal",
        "scan_ms": (p.scan_ms if p else None) or 1600,
        "avatar_speed": (p.avatar_speed if p else None) or "normal",
        "auto_speak": bool(p.auto_speak) if p and p.auto_speak is not None else True,
        "high_contrast": bool(p and p.high_contrast),
    }


@router.get("")
def get_prefs(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return _out(db.get(UserPrefs, user.id))


@router.put("")
def put_prefs(body: PrefsIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.get(UserPrefs, user.id)
    if p is None:
        p = UserPrefs(user_id=user.id)
        db.add(p)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(p, field, int(value) if isinstance(value, bool) else value)
    db.commit()
    return _out(p)
