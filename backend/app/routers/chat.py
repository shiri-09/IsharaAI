import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import get_db
from ..llm.router import llm
from ..models import Conversation, Message, User
from ..services.translate import LANGUAGES, translate_all

router = APIRouter(prefix="/api", tags=["chat"])

SYSTEM_PROMPT = (
    "You are Ishara AI, a warm and patient conversation partner for Deaf and hard-of-hearing people in India. "
    "The user's messages usually come from an Indian Sign Language (ISL) recogniser, so they may be short, "
    "telegraphic, or missing grammar words. Infer the intended meaning from context and the conversation history. "
    "Reply in simple, clear English in 1-3 short sentences. If the meaning is genuinely unclear, ask one short "
    "clarifying question. Never mention that you are reading recogniser output."
)
HISTORY_TURNS = 12


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    source: str = "text"  # "sign" | "text"
    languages: list[str] = Field(default_factory=lambda: list(LANGUAGES))


def _msg_out(m: Message) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "source": m.source,
        "translations": json.loads(m.translations or "{}"),
        "provider": m.provider,
        "created_at": m.created_at.isoformat(),
    }


def _own_conversation(db: Session, user: User, cid: int) -> Conversation:
    conv = db.get(Conversation, cid)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    return conv


@router.get("/languages")
def languages():
    return LANGUAGES


@router.get("/conversations")
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    convs = db.scalars(
        select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.id.desc())
    ).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in convs]


@router.get("/conversations/{cid}")
def get_conversation(cid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    conv = _own_conversation(db, user, cid)
    return {"id": conv.id, "title": conv.title, "messages": [_msg_out(m) for m in conv.messages]}


@router.delete("/conversations/{cid}")
def delete_conversation(cid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    db.delete(_own_conversation(db, user, cid))
    db.commit()
    return {"ok": True}


@router.post("/chat")
async def chat(body: ChatIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.conversation_id is not None:
        conv = _own_conversation(db, user, body.conversation_id)
    else:
        conv = Conversation(user_id=user.id, title=body.text.strip()[:60])
        db.add(conv)
        db.commit()

    user_msg = Message(conversation_id=conv.id, role="user", content=body.text.strip(), source=body.source)
    db.add(user_msg)
    db.commit()

    history = db.scalars(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.id.desc()).limit(HISTORY_TURNS)
    ).all()[::-1]
    messages = [{"role": m.role, "content": m.content} for m in history]

    try:
        reply, provider = await llm.chat(messages, SYSTEM_PROMPT)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    translations = await translate_all(reply, body.languages)
    bot_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=reply,
        translations=json.dumps(translations, ensure_ascii=False),
        provider=provider,
    )
    db.add(bot_msg)
    db.commit()
    return {"conversation_id": conv.id, "user_message": _msg_out(user_msg), "reply": _msg_out(bot_msg)}
