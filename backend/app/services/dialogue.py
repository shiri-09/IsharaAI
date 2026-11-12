"""Multi-turn dialogue as a LangGraph state machine.

    START ─► classify ─┬─(signed & ambiguous)─► interpret ─┐
                       └────────────────────────────────────► respond ─┬─► translate ─┐
                                                                        └─► summarize ─┴─► END

- classify:  intent routing (emergency / question / request / social). Emergencies get a safety-first reply.
- interpret: when the recogniser is unsure, a multimodal LLM picks the most plausible of its top-k candidates
             using the conversation, the rolling summary and the signer's webcam filmstrip.
- respond:   reply using rolling summary + recent turns + filmstrip (multimodal).
- translate: Hindi / Kannada / Tamil / Telugu (runs in parallel with summarize).
- summarize: keeps a rolling summary of long conversations (persisted per conversation by the API layer).
"""

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..llm.base import ChatMessage
from ..llm.router import llm
from .translate import translate_all

SYSTEM_PROMPT = (
    "You are Ishara AI, a warm and patient conversation partner for Deaf and hard-of-hearing people in India. "
    "The user's messages usually come from an Indian Sign Language (ISL) recogniser, so they may be short or "
    "telegraphic. Infer the intended meaning from context and the conversation history. If images of the signer "
    "are attached (a filmstrip of the signing), you may use visible context such as facial expression or "
    "surroundings, but never comment on the person's appearance. Reply in simple, clear English in 1-3 short "
    "sentences. If the meaning is genuinely unclear, ask one short clarifying question. Never mention that you are "
    "reading recogniser output."
)

INTENT_RULES = {
    "emergency": "The user may be in an emergency. Respond calmly and first: tell them to call 112 (India's "
    "emergency number) or show this screen to someone nearby, then ask one short question about what they need.",
    "question": "The user asked a question. Answer it directly and simply.",
    "request": "The user is asking for something. Acknowledge it and say how you can help.",
    "social": "This is social conversation. Be friendly and keep the conversation going with a short question.",
}

_EMERGENCY = re.compile(
    r"\b(help|emergency|hospital|doctor|ambulance|police|pain|hurt|injur\w*|bleed\w*|fire|accident|lost|"
    r"afraid|scared|danger|unsafe|sick|ill|faint\w*|call my family)\b",
    re.I,
)
_QUESTION = re.compile(r"(\?|^(what|where|when|why|how|who|which|can|could|do|does|did|is|are|will|would)\b)", re.I)
_REQUEST = re.compile(r"\b(please|i want|i need|give|bring|get|show|tell|call|repeat)\b", re.I)

INTERPRET_PROMPT = (
    "An Indian Sign Language recogniser produced several candidate sentences for what the user just signed, "
    "with probabilities. Using the conversation so far (and the signer's images if attached), pick the single "
    "most plausible candidate. Reply with ONLY the number of the chosen candidate."
)

SUMMARY_PROMPT = (
    "Update the running summary of this conversation between a Deaf user and Ishara AI. Keep facts about the user "
    "(name, needs, preferences, plans) and open questions. Max 80 words. Reply with the summary only."
)

AMBIGUOUS_BELOW = 0.6
SUMMARY_EVERY = 6  # refresh the rolling summary every N user turns


class DialogueState(TypedDict, total=False):
    history: list[ChatMessage]  # recent previous turns, oldest first
    summary: str  # rolling summary of older turns
    turn: int  # number of user turns so far, including this one
    text: str
    source: str  # "sign" | "text"
    candidates: list[dict]  # recogniser top-k: [{"text", "p"}]
    image: str | None  # base64 JPEG filmstrip of the signer
    languages: list[str]
    intent: str
    interpreted: str
    reply: str
    provider: str
    translations: dict[str, str]
    new_summary: str


def _context(s: DialogueState) -> str:
    return f"Summary of earlier conversation: {s['summary']}\n\n" if s.get("summary") else ""


def classify(s: DialogueState) -> DialogueState:
    texts = [s["text"], *[c["text"] for c in s.get("candidates") or []][:1]]
    joined = " ".join(texts)
    if _EMERGENCY.search(joined):
        intent = "emergency"
    elif _QUESTION.search(s["text"].strip()):
        intent = "question"
    elif _REQUEST.search(joined):
        intent = "request"
    else:
        intent = "social"
    return {"intent": intent}


def _route_after_classify(s: DialogueState) -> str:
    c = s.get("candidates") or []
    if s.get("source") == "sign" and len(c) > 1 and c[0].get("p", 1.0) < AMBIGUOUS_BELOW:
        return "interpret"
    return "respond"


async def interpret(s: DialogueState) -> DialogueState:
    cands = s["candidates"]
    listing = "\n".join(f"{i + 1}. {c['text']} (p={c.get('p', 0):.2f})" for i, c in enumerate(cands))
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in s.get("history", [])[-6:]) or "(start of conversation)"
    msg: ChatMessage = {"role": "user", "content": f"{_context(s)}Conversation so far:\n{convo}\n\nCandidates:\n{listing}"}
    if s.get("image"):
        msg["image"] = s["image"]
    try:
        out, _ = await llm.chat([msg], INTERPRET_PROMPT)
        m = re.search(r"\d+", out)
        k = int(m.group(0)) - 1 if m else 0
        chosen = cands[k]["text"] if 0 <= k < len(cands) else s["text"]
    except RuntimeError:
        chosen = s["text"]
    # re-run intent rules on the chosen sentence (e.g. a recognised emergency)
    return {"interpreted": chosen, "intent": classify({**s, "text": chosen, "candidates": []})["intent"]}


async def respond(s: DialogueState) -> DialogueState:
    user_text = s.get("interpreted") or s["text"]
    latest: ChatMessage = {"role": "user", "content": user_text}
    if s.get("image"):
        latest["image"] = s["image"]
    system = f"{SYSTEM_PROMPT}\n\n{INTENT_RULES[s.get('intent', 'social')]}\n\n{_context(s)}".strip()
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
    g.add_node("classify", classify)
    g.add_node("interpret", interpret)
    g.add_node("respond", respond)
    g.add_node("translate", translate)
    g.add_node("summarize", summarize)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _route_after_classify, {"interpret": "interpret", "respond": "respond"})
    g.add_edge("interpret", "respond")
    g.add_conditional_edges("respond", _fan_out, ["translate", "summarize", END])
    g.add_edge("translate", END)
    g.add_edge("summarize", END)
    return g.compile()


dialogue_graph = build_graph()
