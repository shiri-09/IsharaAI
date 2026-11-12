import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import get_db
from ..models import AgentAction, AgentRun, Conversation, ConversationMemory, Message, User
from ..services.dialogue import CONTEXT_HINTS, booking_reference, describe, dialogue_graph
from ..services.translate import LANGUAGES, translate_all

router = APIRouter(prefix="/api", tags=["chat"])

HISTORY_TURNS = 12
MAX_IMAGE_B64 = 1_500_000  # ~1.1 MB JPEG
CONTEXTS = list(CONTEXT_HINTS)


class Candidate(BaseModel):
    text: str
    p: float = 0.0


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    context: str = "general"  # used when this message starts a new conversation
    source: str = "text"  # "sign" | "text" | "aac" | "voice"
    languages: list[str] = Field(default_factory=lambda: list(LANGUAGES))
    candidates: list[Candidate] = Field(default_factory=list, max_length=5)
    image: str | None = None


def _msg_out(m: Message) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "interpreted": m.interpreted or "",
        "meta": json.loads(m.meta or "{}"),
        "source": m.source,
        "translations": json.loads(m.translations or "{}"),
        "provider": m.provider,
        "created_at": m.created_at.isoformat(),
    }


def _action_out(a: AgentAction | None) -> dict | None:
    if a is None:
        return None
    payload = json.loads(a.payload or "{}")
    if a.kind != "book_appointment" or payload.get("run_id"):
        from ..services.orchestrator import describe_effect, reference

        return {"id": a.id, "kind": a.kind, "status": a.status, "fields": payload.get("fields", {}), "missing": [], "problem": "",
                "summary": describe_effect(a.kind, payload.get("fields", {})), "reference": reference(a.kind, a.id) if a.status == "done" else None,
                "result": a.result, "conversation_id": a.conversation_id, "created_at": a.created_at.isoformat()}
    return {
        "id": a.id,
        "kind": a.kind,
        "status": a.status,
        "fields": payload.get("fields", {}),
        "missing": payload.get("missing", []),
        "problem": payload.get("problem", ""),
        "summary": describe(payload.get("fields", {})),
        "reference": booking_reference(a.id) if a.status == "done" else None,
        "result": a.result,
        "conversation_id": a.conversation_id,
        "created_at": a.created_at.isoformat(),
    }


def _own_conversation(db: Session, user: User, cid: int) -> Conversation:
    conv = db.get(Conversation, cid)
    if conv is None or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    return conv


def _open_action(db: Session, conv_id: int) -> AgentAction | None:
    return db.scalar(
        select(AgentAction)
        .where(AgentAction.conversation_id == conv_id, AgentAction.status.in_(["draft", "awaiting_consent"]))
        .order_by(AgentAction.id.desc())
    )


@router.get("/languages")
def languages():
    return LANGUAGES


@router.get("/contexts")
def contexts():
    return CONTEXTS


@router.get("/conversations")
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    convs = db.scalars(select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.id.desc())).all()
    out = []
    for c in convs:
        last = db.scalar(select(Message.created_at).where(Message.conversation_id == c.id).order_by(Message.id.desc()))
        n = db.scalar(select(func.count(Message.id)).where(Message.conversation_id == c.id)) or 0
        out.append({"id": c.id, "title": c.title, "context": c.context or "general", "mode": c.mode or "ai", "messages": n,
                    "created_at": c.created_at.isoformat(), "updated_at": (last or c.created_at).isoformat()})
    return out


@router.get("/conversations/{cid}")
def get_conversation(cid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    conv = _own_conversation(db, user, cid)
    return {
        "id": conv.id,
        "title": conv.title,
        "context": conv.context or "general",
        "mode": conv.mode or "ai",
        "messages": [_msg_out(m) for m in conv.messages],
        "action": _action_out(_open_action(db, conv.id)),
    }


@router.delete("/conversations/{cid}")
def delete_conversation(cid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    conv = _own_conversation(db, user, cid)
    mem = db.get(ConversationMemory, cid)  # explicit: SQLite does not enforce FK cascades by default
    if mem:
        db.delete(mem)
    for a in db.scalars(select(AgentAction).where(AgentAction.conversation_id == cid)).all():
        db.delete(a)
    for r in db.scalars(select(AgentRun).where(AgentRun.conversation_id == cid)).all():
        db.delete(r)
    db.delete(conv)
    db.commit()
    return {"ok": True}


@router.post("/chat")
async def chat(body: ChatIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.conversation_id is not None:
        conv = _own_conversation(db, user, body.conversation_id)
    else:
        ctx = body.context if body.context in CONTEXTS else "general"
        conv = Conversation(user_id=user.id, title=body.text.strip()[:60], context=ctx, mode="ai")
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
    open_action = _open_action(db, conv.id)
    pending = None
    if open_action:
        pending = {**json.loads(open_action.payload or "{}"), "status": open_action.status}

    recognised = body.text.strip()
    try:
        state = await dialogue_graph.ainvoke(
            {
                # the agent reasons over what each turn meant; the stored transcript keeps what was signed
                "history": [{"role": m.role, "content": m.interpreted or m.content} for m in history],
                "text": recognised,
                "source": body.source,
                "candidates": [c.model_dump() for c in body.candidates],
                "image": image,
                "languages": body.languages,
                "summary": memory.summary if memory else "",
                "turn": (user_turns or 0) + 1,
                "context": conv.context or "general",
                "pending": pending,
            }
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    # ---- task agent output: create / update the draft, never execute -------------------------------------------
    action_row = open_action
    a = state.get("action")
    if a and a.get("kind"):
        if action_row is None:
            action_row = AgentAction(user_id=user.id, conversation_id=conv.id, kind=a["kind"])
            db.add(action_row)
        action_row.payload = json.dumps({k: a.get(k) for k in ("fields", "missing", "problem")})
        action_row.status = a.get("status", "draft")
    elif a and a.get("status") == "cancelled" and action_row is not None:
        action_row.status = "cancelled"

    interpreted = state.get("interpreted") or recognised
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=recognised,  # never rewritten
        interpreted=interpreted if interpreted != recognised else "",
        source=body.source,
        meta=json.dumps(
            {"candidates": [c.model_dump() for c in body.candidates][:3], "intent": state.get("intent")}, ensure_ascii=False
        ),
    )
    db.add(user_msg)
    db.commit()
    bot_msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=state["reply"],
        translations=json.dumps(state.get("translations", {}), ensure_ascii=False),
        provider=state["provider"],
        meta=json.dumps(
            {
                "sources": [{"title": k["title"], "source": k["source"]} for k in state.get("knowledge", [])],
                "action_id": action_row.id if action_row is not None else None,
            }
        ),
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
        "context": conv.context or "general",
        "user_message": _msg_out(user_msg),
        "reply": _msg_out(bot_msg),
        "intent": state.get("intent"),
        "action": _action_out(action_row) if action_row is not None and action_row.status != "cancelled" else None,
    }


# ------------------------------------------------------------------------------- consent-gated actions
class ConfirmIn(BaseModel):
    languages: list[str] = Field(default_factory=list)


@router.get("/actions")
def list_actions(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.scalars(
        select(AgentAction).where(AgentAction.user_id == user.id, AgentAction.status == "done").order_by(AgentAction.id.desc())
    ).all()
    return [_action_out(a) for a in rows]


def _own_action(db: Session, user: User, aid: int) -> AgentAction:
    a = db.get(AgentAction, aid)
    if a is None or a.user_id != user.id:
        raise HTTPException(404, "Action not found")
    return a


@router.post("/actions/{aid}/confirm")
async def confirm_action(aid: int, body: ConfirmIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = _own_action(db, user, aid)
    if a.status != "awaiting_consent":
        raise HTTPException(409, f"This action is {a.status}, not waiting for confirmation")
    fields = json.loads(a.payload or "{}").get("fields", {})
    # execute: in this MVP the booking is recorded in Ishara AI's own calendar table
    a.status = "done"
    a.result = f"Booked: {describe(fields)}. Reference {booking_reference(a.id)}."
    db.commit()
    # verify: read it back from storage before telling the user it is done
    check = db.get(AgentAction, aid)
    verified = check is not None and check.status == "done"
    text = (
        f"Done. {describe(fields)} is booked. Your reference is {booking_reference(a.id)}."
        if verified
        else "I could not confirm the booking. Please try again."
    )
    translations = await translate_all(text, [lg for lg in body.languages if lg != "en"]) if body.languages else {}
    msg = Message(
        conversation_id=a.conversation_id,
        role="assistant",
        content=text,
        translations=json.dumps(translations, ensure_ascii=False),
        provider="agent",
        meta=json.dumps({"action_id": a.id, "verified": verified}),
    )
    db.add(msg)
    db.commit()
    return {"action": _action_out(a), "verified": verified, "message": _msg_out(msg)}


@router.post("/actions/{aid}/cancel")
def cancel_action(aid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = _own_action(db, user, aid)
    if a.status in ("done", "cancelled"):
        raise HTTPException(409, f"This action is already {a.status}")
    a.status = "cancelled"
    msg = Message(conversation_id=a.conversation_id, role="assistant", content="Okay, I cancelled that. Nothing was booked.", provider="agent")
    db.add(msg)
    db.commit()
    return {"action": _action_out(a), "message": _msg_out(msg)}
