import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import get_db
from ..models import Conversation, ConversationMemory, Message, User
from ..services.dialogue import dialogue_graph
from ..services.translate import LANGUAGES

router = APIRouter(prefix="/api", tags=["chat"])

HISTORY_TURNS = 12
MAX_IMAGE_B64 = 1_500_000  # ~1.1 MB JPEG


class Candidate(BaseModel):
    text: str
    p: float = 0.0


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    source: str = "text"  # "sign" | "text"
    languages: list[str] = Field(default_factory=lambda: list(LANGUAGES))
    candidates: list[Candidate] = Field(default_factory=list, max_length=5)  # recogniser top-k for signs
    image: str | None = None  # base64 JPEG (data URL prefix allowed) of the signer, for the multimodal model


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
    conv = _own_conversation(db, user, cid)
    mem = db.get(ConversationMemory, cid)  # explicit: SQLite does not enforce FK cascades by default
    if mem:
        db.delete(mem)
    db.delete(conv)
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

    image = body.image.split(",", 1)[-1] if body.image else None
    if image and len(image) > MAX_IMAGE_B64:
        raise HTTPException(413, "Snapshot too large")

    history = db.scalars(
        select(Message).where(Message.conversation_id == conv.id).order_by(Message.id.desc()).limit(HISTORY_TURNS)
    ).all()[::-1]
    memory = db.get(ConversationMemory, conv.id)
    user_turns = db.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conv.id, Message.role == "user")
    )

    try:
        state = await dialogue_graph.ainvoke(
            {
                "history": [{"role": m.role, "content": m.content} for m in history],
                "text": body.text.strip(),
                "source": body.source,
                "candidates": [c.model_dump() for c in body.candidates],
                "image": image,
                "languages": body.languages,
                "summary": memory.summary if memory else "",
                "turn": (user_turns or 0) + 1,
            }
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    user_msg = Message(conversation_id=conv.id, role="user", content=state["interpreted"], source=body.source)
    db.add(user_msg)
    db.commit()
    bot_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=state["reply"],
        translations=json.dumps(state.get("translations", {}), ensure_ascii=False),
        provider=state["provider"],
    )
    db.add(bot_msg)
    if state.get("new_summary"):
        if memory is None:
            memory = ConversationMemory(conversation_id=conv.id)
            db.add(memory)
        memory.summary = state["new_summary"]
    db.commit()
    return {
        "conversation_id": conv.id,
        "user_message": _msg_out(user_msg),
        "reply": _msg_out(bot_msg),
        "intent": state.get("intent"),
        "interpreted_from": body.text if state["interpreted"] != body.text.strip() else None,
    }
