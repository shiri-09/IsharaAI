"""Real-time speech mode: a two-way interpreter between a signer and a hearing speaker. No AI replies.

  signer  -> recognised sentence -> shown as a caption, translated, spoken aloud for the hearing person
  speaker -> speech (browser Speech API, any of 5 languages) -> caption in English -> avatar signs it

Every turn is stored (conversation mode "live") so the whole exchange is in History.
"""

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import get_db
from ..llm.router import llm
from ..models import Conversation, Message, User
from ..services.translate import LANGUAGES, translate_all
from .chat import _msg_out, _own_conversation

router = APIRouter(prefix="/api/live", tags=["real-time speech"])


class TurnIn(BaseModel):
    who: str = Field(pattern="^(signer|speaker)$")
    text: str = Field(min_length=1, max_length=1000)
    conversation_id: int | None = None
    source: str = "text"  # sign | voice | text
    lang: str = "en"  # language the text is in (speaker side)
    languages: list[str] = Field(default_factory=list)  # signer side: languages to translate into
    candidates: list[dict] = Field(default_factory=list, max_length=5)


@router.post("/turn")
async def turn(body: TurnIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.conversation_id:
        conv = _own_conversation(db, user, body.conversation_id)
    else:
        conv = Conversation(user_id=user.id, title=("Live: " + body.text)[:80], context="general", mode="live")
        db.add(conv)
        db.flush()
    text = body.text.strip()
    english, translations = text, {}
    if body.who == "signer":
        translations = await translate_all(text, [lg for lg in body.languages if lg in LANGUAGES])
    elif body.lang != "en" and not text.isascii():  # typed or spoken in an Indian script: translate to English
        try:
            out, _ = await llm.chat([{"role": "user", "content": text}], "Translate this into simple English. Reply with only the translation.")
            english = out.strip().strip('"') or text
        except RuntimeError:
            english = text
        translations = {body.lang: text}
    msg = Message(
        conversation_id=conv.id,
        role=body.who,
        content=text,
        interpreted=english if english != text else "",
        source=body.source,
        translations=json.dumps(translations, ensure_ascii=False),
        meta=json.dumps({"lang": body.lang, "candidates": body.candidates}),
        provider="live",
    )
    db.add(msg)
    db.commit()
    return {"conversation_id": conv.id, "message": _msg_out(msg), "english": english}
