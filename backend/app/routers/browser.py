"""Browser agent API. The agent runs in a background thread; the app polls this API to show it live and to answer its questions."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import AuthError, current_user, user_id_from_token
from ..db import get_db
from ..models import AgentMemory, User
from ..services import browser_agent as ba

router = APIRouter(prefix="/api/browser", tags=["browser agent"])


class RunIn(BaseModel):
    goal: str = Field(min_length=3, max_length=600)
    start_url: str = ""
    headless: bool = False


class AnswerIn(BaseModel):
    ask_id: str
    answer: str = Field(max_length=500)


class MemoryIn(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    value: str = Field(max_length=300)
    label: str = ""


def _own(user: User, rid: str) -> ba.BrowserRun:
    r = ba.RUNS.get(rid)
    if r is None or r.user_id != user.id:
        raise HTTPException(404, "Run not found")
    return r


@router.post("/runs")
def start(body: RunIn, user: User = Depends(current_user)):
    url = body.start_url.strip()
    if url and not ba.SAFE_NAV.match(url):
        raise HTTPException(422, "The start page must begin with http:// or https://")
    return ba.start_run(user.id, body.goal.strip(), url, body.headless).state()


@router.get("/runs/{rid}")
def get(rid: str, user: User = Depends(current_user)):
    return _own(user, rid).state()


@router.post("/runs/{rid}/answer")
def answer(rid: str, body: AnswerIn, user: User = Depends(current_user)):
    r = _own(user, rid)
    if not r.ask or r.ask["id"] != body.ask_id:
        raise HTTPException(409, "That question is no longer open")
    r.answers.put({"ask_id": body.ask_id, "answer": body.answer})
    return {"ok": True}


@router.post("/runs/{rid}/stop")
def stop(rid: str, user: User = Depends(current_user)):
    r = _own(user, rid)
    r.stop.set()
    return {"ok": True}


@router.get("/runs/{rid}/screenshot")
def screenshot(rid: str, token: str = Query(""), t: int = 0):
    """The latest browser picture. The token travels in the query string because an <img> tag cannot send a header."""
    try:
        uid = user_id_from_token(token)
    except AuthError:
        raise HTTPException(401, "Not authenticated")
    r = ba.RUNS.get(rid)
    if r is None or r.user_id != uid or not r.shot:
        raise HTTPException(404, "No picture yet")
    return Response(r.shot, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- what the agent has learned about the user
@router.get("/memory")
def memory(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(AgentMemory).where(AgentMemory.user_id == user.id).order_by(AgentMemory.key)).all()
    return [{"key": m.key, "label": m.label, "value": m.value, "uses": m.uses} for m in rows]


@router.put("/memory")
def memory_put(body: MemoryIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if ba.is_sensitive(body.key, body.label):
        raise HTTPException(422, "Passwords, card numbers, OTPs and PINs are never stored.")
    key = ba.canon(body.key)
    m = db.scalar(select(AgentMemory).where(AgentMemory.user_id == user.id, AgentMemory.key == key))
    if m:
        m.value, m.label = body.value, body.label or m.label
    else:
        db.add(AgentMemory(user_id=user.id, key=key, label=body.label, value=body.value))
    db.commit()
    return {"ok": True}


@router.delete("/memory/{key}")
def memory_delete(key: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    m = db.scalar(select(AgentMemory).where(AgentMemory.user_id == user.id, AgentMemory.key == key))
    if m:
        db.delete(m)
        db.commit()
    return {"ok": True}
