from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import get_db
from ..models import User, UserPrefs

router = APIRouter(prefix="/api/prefs", tags=["preferences"])


class PrefsIn(BaseModel):
    mode: Literal["isl", "aac"] | None = None
    onboarded: bool | None = None
    voice: Literal["en", "hi", "kn", "ta", "te"] | None = None


def _out(p: UserPrefs | None) -> dict:
    return {"mode": p.mode if p else "", "onboarded": bool(p and p.onboarded), "voice": p.voice if p else "hi"}


@router.get("")
def get_prefs(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return _out(db.get(UserPrefs, user.id))


@router.put("")
def put_prefs(body: PrefsIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.get(UserPrefs, user.id)
    if p is None:
        p = UserPrefs(user_id=user.id)
        db.add(p)
    if body.mode is not None:
        p.mode = body.mode
    if body.onboarded is not None:
        p.onboarded = int(body.onboarded)
    if body.voice is not None:
        p.voice = body.voice
    db.commit()
    return _out(p)
