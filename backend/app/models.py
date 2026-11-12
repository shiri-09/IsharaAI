from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    context: Mapped[str] = mapped_column(String(24), default="general")  # general | campus | classroom | office | hospital | home
    mode: Mapped[str] = mapped_column(String(8), default="ai")  # ai (AI conversation) | live (real-time speech) | agent
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    user: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"; live mode: "signer" | "speaker"
    content: Mapped[str] = mapped_column(Text)  # for signed turns: exactly what the recogniser produced, never rewritten
    interpreted: Mapped[str] = mapped_column(Text, default="")  # meaning the agent acted on, when it differs from content
    meta: Mapped[str] = mapped_column(Text, default="{}")  # JSON: candidates, intent, sources, action id
    source: Mapped[str] = mapped_column(String(16), default="text")  # "sign" | "text"
    translations: Mapped[str] = mapped_column(Text, default="{}")  # JSON {lang: text}
    provider: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ConversationMemory(Base):
    """Long-term dialogue state kept by the LangGraph summarize node."""

    __tablename__ = "conversation_memory"
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class UserPrefs(Base):
    """Accessibility preferences chosen during first-time setup."""

    __tablename__ = "user_prefs"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    mode: Mapped[str] = mapped_column(String(8), default="")  # "isl" | "aac" | "" (not chosen yet)
    onboarded: Mapped[int] = mapped_column(Integer, default=0)
    voice: Mapped[str] = mapped_column(String(4), default="hi")
    text_size: Mapped[str] = mapped_column(String(8), default="normal")  # normal | large | xlarge
    scan_ms: Mapped[int] = mapped_column(Integer, default=1600)  # AAC scanning speed
    avatar_speed: Mapped[str] = mapped_column(String(8), default="normal")  # slow | normal | fast
    auto_speak: Mapped[int] = mapped_column(Integer, default=1)
    high_contrast: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class AgentAction(Base):
    """A task the accessibility agent prepared. It only runs after the user explicitly confirms."""

    __tablename__ = "agent_actions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # book_appointment
    payload: Mapped[str] = mapped_column(Text, default="{}")  # JSON fields
    status: Mapped[str] = mapped_column(String(16), default="draft")  # draft | awaiting_consent | done | cancelled
    result: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class AgentRun(Base):
    """One agentic orchestration run: the goal, the plan, every step's result, and the full trace."""

    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    goal: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(16), default="text")  # text | voice | sign
    # planning | needs_info | blocked | awaiting_consent | done | cancelled | failed | answered
    status: Mapped[str] = mapped_column(String(20), default="planning")
    steps: Mapped[str] = mapped_column(Text, default="[]")  # JSON list of {tool, args, why, status, result, ...}
    trace: Mapped[str] = mapped_column(Text, default="[]")  # JSON list of {t, node, detail}
    question: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    provider: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
