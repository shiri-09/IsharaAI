"""The Ishara AI accessibility agent: one LangGraph state machine that picks capabilities by intent.

    START ─► classify ─┬─(signed & unsure)─► interpret ─┐
                       └───────────────────────────────► retrieve ─┬─(task)─► plan_task ─┐
                                                                    └────────────────────► respond ─┬─► translate ─┐
                                                                                                     └─► summarize ─┴─► END

Capabilities (one central agent, specialised steps rather than separate LLMs):
- Accessibility Orchestrator   classify: intent (emergency, task, info, question, request, social) + the
                               conversation's context (campus, classroom, office, hospital, home).
- Communication Agent          interpret: resolves an unsure sign from conversation + filmstrip; respond/translate.
- Navigation / knowledge       retrieve: grounded answers from data/knowledge (RAG), with cited sources.
- Task Agent                   plan_task: fills a task across turns ("meet my professor tomorrow" ... "at 10 AM").
- Verification and Safety      plan_task checks the request against the rules it found and NEVER executes:
                               the API stores it as awaiting consent; it runs only after the user confirms.
- Adaptive Access              the API applies each user's saved preferences (voice, speed, text size).

The signer's recognised words are never rewritten: the API stores them apart from what the agent interpreted.
"""

import json
import re
from datetime import date, datetime, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..llm.base import ChatMessage
from ..llm.router import llm
from .knowledge import kb
from .translate import translate_all

SYSTEM_PROMPT = (
    "You are Ishara AI, an accessibility assistant for Deaf and hard-of-hearing people and for people who use AAC "
    "(they cannot speak or use their hands) in India. Messages may come from a sign language recogniser or an AAC "
    "phrase board, so they can be short or telegraphic. Use the conversation so far: a short message such as "
    "'At 10 AM' usually continues the previous request. If images of the signer are attached, you may use visible "
    "context such as facial expression, but never comment on appearance. Reply in simple, clear English in 1-3 "
    "short sentences. If the meaning is genuinely unclear, ask one short clarifying question. Never mention that "
    "you are reading recogniser output."
)

INTENT_RULES = {
    "emergency": "The user may be in an emergency. Respond calmly and first: tell them to call 112 (India's "
    "emergency number) or show this screen to someone nearby, then ask one short question about what they need.",
    "task": "The user wants something done. Follow the TASK STATUS below exactly.",
    "info": "The user wants information. Answer ONLY from the KNOWLEDGE below and name its section in brackets, "
    "for example [Library]. If the knowledge does not cover it, say you do not have that information.",
    "question": "The user asked a question. Answer it directly and simply. Use the KNOWLEDGE below when relevant.",
    "request": "The user is asking for something. Acknowledge it and say how you can help.",
    "social": "This is social conversation. Be friendly and keep the conversation going with a short question.",
}

CONTEXT_HINTS = {
    "general": "",
    "campus": "Setting: a college campus. The user may ask about places, offices, staff or schedules.",
    "classroom": "Setting: a classroom. Keep replies short; the user may talk to a teacher or classmates.",
    "office": "Setting: an office or administration desk. The user may need forms, appointments or directions.",
    "hospital": "Setting: a hospital or clinic. Be calm and clear; health and safety come first.",
    "home": "Setting: at home with family. Be warm and casual.",
}

_EMERGENCY = re.compile(
    r"\b(emergency|ambulance|police|pain|hurt|injur\w*|bleed\w*|fire|accident|afraid|scared|danger|unsafe|"
    r"faint\w*|call my family|help me)\b",
    re.I,
)
_TASK = re.compile(
    r"\b(meet|meeting|appointment|book|schedule|reschedule|see (the |my )?(professor|teacher|hod|doctor|sir|madam))\b",
    re.I,
)
_SLOT = re.compile(
    r"\b(\d{1,2}(:\d{2})?\s*(am|pm)|\d{1,2}:\d{2}|morning|afternoon|evening|today|tomorrow|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|next week|professor|prof|sir|madam|hod|yes|no)\b",
    re.I,
)
_INFO = re.compile(
    r"\b(where|when|timing|timings|open|close|hours|lift|ramp|washroom|toilet|library|canteen|office|department|"
    r"block|room|floor|class|timetable|exam|interpreter|medical|nurse|route|way to)\b",
    re.I,
)
_QUESTION = re.compile(r"(\?|^(what|where|when|why|how|who|which|can|could|do|does|did|is|are|will|would)\b)", re.I)
_REQUEST = re.compile(r"\b(please|i want|i need|give|bring|get|show|tell|call|repeat)\b", re.I)

INTERPRET_PROMPT = (
    "An Indian Sign Language recogniser produced several candidate sentences for what the user just signed, "
    "with probabilities. Using the conversation so far (and the signer's images if attached), pick the single "
    "most plausible candidate. Reply with ONLY the number of the chosen candidate."
)

TASK_PROMPT = (
    "You extract a task from a conversation for an accessibility assistant. Today is {today} ({weekday}). "
    "Supported task: book_appointment, with fields: with (who to meet, e.g. 'Professor of Data Science' or a name), "
    "date (YYYY-MM-DD), time (HH:MM, 24-hour), purpose (optional). Start from the CURRENT DRAFT and update it with "
    "anything new in the conversation; a short message like 'at 10 AM' fills the time. Resolve 'tomorrow' and "
    "weekday names to real dates. If the user cancels, set cancel to true. If the conversation is not about this "
    "task, set action to null. Reply with ONLY JSON: "
    '{{"action": "book_appointment" | null, "fields": {{"with": "", "date": "", "time": "", "purpose": ""}}, "cancel": false}}'
)

SUMMARY_PROMPT = (
    "Update the running summary of this conversation between a user and Ishara AI. Keep facts about the user "
    "(name, needs, preferences, plans) and open questions. Max 80 words. Reply with the summary only."
)

AMBIGUOUS_BELOW = 0.6
SUMMARY_EVERY = 6
REQUIRED = ("with", "date", "time")
SLOT_WINDOWS = [((10, 0), (13, 0)), ((14, 0), (16, 0))]  # from the knowledge base appointment rules


class DialogueState(TypedDict, total=False):
    history: list[ChatMessage]
    summary: str
    turn: int
    text: str
    source: str  # "sign" | "text" | "aac"
    candidates: list[dict]
    image: str | None
    languages: list[str]
    context: str
    pending: dict | None  # current task draft for this conversation
    intent: str
    interpreted: str
    knowledge: list[dict]
    action: dict | None
    reply: str
    provider: str
    translations: dict[str, str]
    new_summary: str


def _summary(s: DialogueState) -> str:
    return f"Summary of earlier conversation: {s['summary']}\n\n" if s.get("summary") else ""


# ---------------------------------------------------------------------------- orchestrator
def _intent_of(text: str, pending: dict | None) -> str:
    if _EMERGENCY.search(text):
        return "emergency"
    if _TASK.search(text) or (pending and pending.get("status") in ("draft", "awaiting_consent") and _SLOT.search(text)):
        return "task"
    if _INFO.search(text) and _QUESTION.search(text.strip()):
        return "info"
    if _QUESTION.search(text.strip()):
        return "question"
    if _REQUEST.search(text):
        return "request"
    return "social"


def classify(s: DialogueState) -> DialogueState:
    return {"intent": _intent_of(s["text"], s.get("pending"))}


def _after_classify(s: DialogueState) -> str:
    c = s.get("candidates") or []
    if s.get("source") == "sign" and len(c) > 1 and c[0].get("p", 1.0) < AMBIGUOUS_BELOW:
        return "interpret"
    return "retrieve"


# ---------------------------------------------------------------------------- communication agent
async def interpret(s: DialogueState) -> DialogueState:
    cands = s["candidates"]
    listing = "\n".join(f"{i + 1}. {c['text']} (p={c.get('p', 0):.2f})" for i, c in enumerate(cands))
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in s.get("history", [])[-6:]) or "(start of conversation)"
    msg: ChatMessage = {"role": "user", "content": f"{_summary(s)}Conversation so far:\n{convo}\n\nCandidates:\n{listing}"}
    if s.get("image"):
        msg["image"] = s["image"]
    try:
        out, _ = await llm.chat([msg], INTERPRET_PROMPT)
        m = re.search(r"\d+", out)
        k = int(m.group(0)) - 1 if m else 0
        chosen = cands[k]["text"] if 0 <= k < len(cands) else s["text"]
    except RuntimeError:
        chosen = s["text"]
    return {"interpreted": chosen, "intent": _intent_of(chosen, s.get("pending"))}


# ---------------------------------------------------------------------------- navigation / knowledge (RAG)
def retrieve(s: DialogueState) -> DialogueState:
    if s.get("intent") not in ("info", "question", "task", "request"):
        return {"knowledge": []}
    q = s.get("interpreted") or s["text"]
    hits = kb.search(q, k=3)
    return {"knowledge": [{"title": p.title, "source": p.source, "text": p.text, "score": round(sc, 3)} for p, sc in hits]}


def _after_retrieve(s: DialogueState) -> str:
    return "plan_task" if s.get("intent") == "task" else "respond"


# ---------------------------------------------------------------------------- task agent + verification
def _parse_json(out: str) -> dict:
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _check_slot(fields: dict) -> str:
    """Verification: is the requested time inside the allowed appointment windows on a weekday?"""
    try:
        d = date.fromisoformat(fields["date"])
        h, mnt = (int(x) for x in fields["time"].split(":")[:2])
    except (KeyError, ValueError):
        return ""
    if d < date.today():
        return "That date has already passed."
    if d.weekday() >= 5:
        return "Appointments are only on weekdays."
    minutes = h * 60 + mnt
    if not any(a[0] * 60 + a[1] <= minutes < b[0] * 60 + b[1] for a, b in SLOT_WINDOWS):
        return "Appointments are only between 10 AM and 1 PM, or 2 PM and 4 PM."
    return ""


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _rule_fields(text: str) -> dict:
    """Deterministic slot extraction, so a slow or failed LLM call never loses the user's details."""
    t = text.lower()
    out = {}
    who = re.search(r"\b(professor|prof\.?|teacher|hod|head of department|doctor|sir|madam)\b(?:\s+([a-z]+))?", t)
    if who:
        skip = {"tomorrow", "today", "at", "on", "for", "in", "next", "this"}
        name = who.group(2) if who.group(2) and who.group(2) not in skip else ""
        out["with"] = ("Professor " + name.title()).strip() if who.group(1).startswith("prof") else who.group(1).title()
    today = date.today()
    if "tomorrow" in t:
        out["date"] = (today + timedelta(days=1)).isoformat()
    elif re.search(r"\btoday\b", t):
        out["date"] = today.isoformat()
    else:
        for i, wd in enumerate(_WEEKDAYS):
            if re.search(rf"\b{wd}\b", t):
                ahead = (i - today.weekday()) % 7 or 7
                out["date"] = (today + timedelta(days=ahead)).isoformat()
                break
    tm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", t) or re.search(r"\b(\d{1,2}):(\d{2})\b()", t)
    if tm:
        hh, mm, ap = int(tm.group(1)), int(tm.group(2) or 0), (tm.group(3) or "").replace(".", "")
        hh = hh % 12 + (12 if ap == "pm" else 0) if ap else hh
        out["time"] = f"{hh:02d}:{mm:02d}"
    return out


async def plan_task(s: DialogueState) -> DialogueState:
    pending = s.get("pending") or {}
    draft = pending.get("fields") or {}
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in s.get("history", [])[-6:])
    user_text = s.get("interpreted") or s["text"]
    today = date.today()
    prompt = f"CURRENT DRAFT: {json.dumps(draft)}\n\nConversation:\n{convo}\nuser: {user_text}"
    try:
        out, _ = await llm.chat(
            [{"role": "user", "content": prompt}], TASK_PROMPT.format(today=today.isoformat(), weekday=today.strftime("%A"))
        )
        data = _parse_json(out)
    except RuntimeError:
        data = {}
    if data.get("cancel") or re.search(r"\b(cancel|don't book|do not book|never mind)\b", user_text, re.I):
        return {"action": {**pending, "status": "cancelled"}}
    rules = _rule_fields(user_text)
    if not data.get("action") and not rules and not pending and not _TASK.search(user_text):
        return {"action": None, "intent": "request"}
    llm_fields = {k: v for k, v in (data.get("fields") or {}).items() if v and str(v).lower() not in ("null", "none")}
    fields = {**draft, **llm_fields, **rules}
    # normalise times like "10:00 AM" the model may still return
    t = str(fields.get("time", "")).strip().lower()
    mt = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    if mt:
        hh, mm, ap = int(mt.group(1)), int(mt.group(2) or 0), mt.group(3)
        hh = hh % 12 + (12 if ap == "pm" else 0) if ap else hh
        fields["time"] = f"{hh:02d}:{mm:02d}"
    missing = [k for k in REQUIRED if not fields.get(k)]
    problem = "" if missing else _check_slot(fields)
    status = "awaiting_consent" if not missing and not problem else "draft"
    return {"action": {**pending, "kind": "book_appointment", "fields": fields, "missing": missing, "problem": problem, "status": status}}


def _task_status(a: dict | None) -> str:
    if not a:
        return ""
    if a.get("status") == "cancelled":
        return "TASK STATUS: the user cancelled the booking. Confirm it is cancelled."
    f = a.get("fields", {})
    known = ", ".join(f"{k}: {v}" for k, v in f.items() if v) or "nothing yet"
    if a.get("missing"):
        return (
            f"TASK STATUS: collecting details for an appointment. Known so far: {known}. Ask ONLY for: "
            f"{', '.join(a['missing'])}. Do NOT say it is set, scheduled or booked yet."
        )
    if a.get("problem"):
        return f"TASK STATUS: the requested slot is not allowed: {a['problem']} Known: {known}. Ask for another time."
    return (
        f"TASK STATUS: ready to book an appointment ({known}). Repeat the details in one sentence and ask the user "
        "to confirm with the Confirm button. Do NOT say it is booked: nothing is booked until they confirm."
    )


# ---------------------------------------------------------------------------- respond + translate + memory
async def respond(s: DialogueState) -> DialogueState:
    user_text = s.get("interpreted") or s["text"]
    latest: ChatMessage = {"role": "user", "content": user_text}
    if s.get("image"):
        latest["image"] = s["image"]
    parts = [SYSTEM_PROMPT, INTENT_RULES[s.get("intent", "social")], CONTEXT_HINTS.get(s.get("context") or "general", "")]
    if s.get("knowledge"):
        parts.append("KNOWLEDGE (sample campus data):\n" + "\n".join(f"[{k['title']}] {k['text']}" for k in s["knowledge"]))
    parts.append(_task_status(s.get("action")))
    parts.append(_summary(s))
    system = "\n\n".join(p for p in parts if p).strip()
    reply, provider = await llm.chat([*s.get("history", []), latest], system)
    return {"interpreted": user_text, "reply": reply, "provider": provider}


def _fan_out(s: DialogueState) -> list[str]:
    nxt = []
    if s.get("languages"):
        nxt.append("translate")
    if s.get("turn", 0) % SUMMARY_EVERY == 0:
        nxt.append("summarize")
    return nxt or [END]


async def translate(s: DialogueState) -> DialogueState:
    return {"translations": await translate_all(s["reply"], s["languages"])}


async def summarize(s: DialogueState) -> DialogueState:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in s.get("history", []))
    convo += f"\nuser: {s.get('interpreted') or s['text']}\nassistant: {s['reply']}"
    prompt = f"Previous summary: {s.get('summary') or '(none)'}\n\nRecent turns:\n{convo}"
    try:
        out, _ = await llm.chat([{"role": "user", "content": prompt}], SUMMARY_PROMPT)
        return {"new_summary": out.strip()[:800]}
    except RuntimeError:
        return {}


def build_graph():
    g = StateGraph(DialogueState)
    for name, fn in [("classify", classify), ("interpret", interpret), ("retrieve", retrieve), ("plan_task", plan_task),
                     ("respond", respond), ("translate", translate), ("summarize", summarize)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _after_classify, {"interpret": "interpret", "retrieve": "retrieve"})
    g.add_edge("interpret", "retrieve")
    g.add_conditional_edges("retrieve", _after_retrieve, {"plan_task": "plan_task", "respond": "respond"})
    g.add_edge("plan_task", "respond")
    g.add_conditional_edges("respond", _fan_out, ["translate", "summarize", END])
    g.add_edge("translate", END)
    g.add_edge("summarize", END)
    return g.compile()


dialogue_graph = build_graph()


def booking_reference(action_id: int) -> str:
    return f"ISH-{action_id:04d}"


def describe(fields: dict) -> str:
    try:
        d = datetime.strptime(f"{fields['date']} {fields['time']}", "%Y-%m-%d %H:%M")
        when = d.strftime("%A %d %B at %I:%M %p").replace(" 0", " ")
    except (KeyError, ValueError):
        when = f"{fields.get('date', '')} {fields.get('time', '')}".strip()
    return f"Appointment with {fields.get('with', 'the professor')} on {when}" + (
        f" ({fields['purpose']})" if fields.get("purpose") else ""
    )

