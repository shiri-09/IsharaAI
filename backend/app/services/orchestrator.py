"""Agentic orchestration: goal -> plan of tool calls -> checks -> consent -> execute -> verify.

The planner is the local LLM (Ollama, through the LLM router) with a strict JSON contract. A deterministic rule
planner runs when the LLM is slow, unavailable or returns something unusable, so a run never fails because of the
model. Every step and decision is written to a trace that the UI shows and History keeps.

Safety rules, enforced in code (not left to the model):
- Read-only tools (lookup_info, check_slot, translate) run immediately.
- Tools that change something (book_appointment, set_reminder, request_interpreter, save_message) never run
  without the user's explicit confirmation, and only if every check before them passed.
- After executing, each effect is read back from storage (verification) before the user is told it is done.
"""

import json
import re
from datetime import date, datetime, timedelta

from ..llm.router import llm
from .dialogue import SLOT_WINDOWS, _check_slot, _rule_fields
from .knowledge import kb

READ_ONLY = {"lookup_info", "check_slot", "translate"}
EFFECTS = {"book_appointment", "set_reminder", "request_interpreter", "save_message"}
TOOLS = READ_ONLY | EFFECTS

TOOL_DOCS = {
    "lookup_info": "Answer a question from the campus knowledge base. args: {query}",
    "check_slot": "Check an appointment time is allowed and free. args: {with, date (YYYY-MM-DD), time (HH:MM)}",
    "book_appointment": "Book a meeting. args: {with, date, time, purpose}",
    "set_reminder": "Set a reminder for the user. args: {text, date, time}",
    "request_interpreter": "Request a sign language interpreter. args: {event, date, time, place}",
    "save_message": "Save a message to send to someone (kept in the outbox). args: {to, text}",
    "translate": "Translate text into Indian languages. args: {text, languages: [hi|kn|ta|te]}",
}

PLAN_PROMPT = """You are the planner of an accessibility agent for a Deaf or hard-of-hearing student on a college campus.
Today is {today} ({weekday}). Break the user's goal into the fewest tool calls that achieve it.

TOOLS:
{tools}

Rules:
- Use ONLY these tools. Dates as YYYY-MM-DD, times as 24-hour HH:MM. Resolve words like "tomorrow" or "Friday".
- Before book_appointment always add check_slot with the same with/date/time.
- If a required detail is missing (who, date or time), do not guess: put a short question in "ask".
- Do not invent facts; questions about the campus go to lookup_info.
Reply with ONLY JSON: {{"steps": [{{"tool": "...", "args": {{...}}, "why": "short reason"}}], "ask": ""}}"""

# An action is only allowed if the user's own words ask for that kind of action (the model may not add extras).
INTENT = {
    "book_appointment": re.compile(r"\b(meet|meeting|appointment|book|schedule|see|set up|arrange|slot)\b", re.I),
    "set_reminder": re.compile(r"\b(remind|reminder|forget|alert me)\b", re.I),
    "request_interpreter": re.compile(r"\binterpreter\b", re.I),
    "save_message": re.compile(r"\b(tell|message|inform|let \w+ know|write to|email|text)\b", re.I),
}
INTENT["check_slot"] = INTENT["book_appointment"]

_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)|\b(\d{1,2}):(\d{2})\b", re.I)
_QUESTION = re.compile(r"^\s*(where|what|when|which|who|how|is|are|can|do|does)\b|\?\s*$", re.I)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_json(out: str) -> dict:
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _time_of(text: str) -> str | None:
    m = _TIME.search(text)
    if not m:
        return None
    if m.group(1):
        hh, mm, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower().replace(".", "")
        hh = hh % 12 + (12 if ap == "pm" else 0)
    else:
        hh, mm = int(m.group(4)), int(m.group(5))
    return f"{hh:02d}:{mm:02d}" if hh < 24 and mm < 60 else None


_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DMY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
_MDY = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
_NUM = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")


def _explicit_date(text: str) -> str | None:
    """'16 November', 'Nov 16', '16/11' (day first, Indian style) -> ISO date, rolling into next year if past."""
    today = date.today()
    d = m = None
    if mm := _DMY.search(text):
        d, m = int(mm.group(1)), _MONTHS[mm.group(2).lower()[:3]]
    elif mm := _MDY.search(text):
        d, m = int(mm.group(2)), _MONTHS[mm.group(1).lower()[:3]]
    elif (mm := _NUM.search(text)) and not _TIME.search(mm.group(0)):
        d, m = int(mm.group(1)), int(mm.group(2))
    if d is None:
        return None
    try:
        out = date(today.year, m, d)
        return (out if out >= today else date(today.year + 1, m, d)).isoformat()
    except ValueError:
        return None


def _date_time(text: str) -> dict:
    f = _rule_fields(text)
    out = {k: f[k] for k in ("date", "time") if k in f}
    if "date" not in out and (d := _explicit_date(text)):
        out["date"] = d
    if "time" not in out and (t := _time_of(text)):
        out["time"] = t
    return out


# ---------------------------------------------------------------------------------------------- rule planner
def rule_plan(goal: str) -> dict:
    """Deterministic fallback planner. Handles the common goals reliably without any model."""
    g = goal.lower()
    steps: list[dict] = []
    when = _date_time(goal)
    fields = _rule_fields(goal)
    if INTENT["book_appointment"].search(g) and re.search(
        r"\b(professor|prof|teacher|hod|doctor|sir|madam|counsel+or|advisor|mentor)\b", g
    ):
        args = {"with": fields.get("with", "the professor"), **when}
        purpose = re.search(r"\b(?:about|for|to discuss)\s+(.+?)(?:\s+(?:on|at|tomorrow|today)\b|$)", goal, re.I)
        if purpose:
            args["purpose"] = purpose.group(1).strip(" .")
        steps += [{"tool": "check_slot", "args": dict(args), "why": "Make sure the time is allowed and free"},
                  {"tool": "book_appointment", "args": args, "why": "Book the meeting"}]
    if re.search(r"\binterpreter\b", g):
        ev = re.search(r"\bfor (?:the |a |my )?(.+?)(?:\s+(?:on|at|tomorrow|today|in)\b|$)", goal, re.I)
        place = re.search(r"\bin (?:the )?([A-Za-z0-9 ]+?(?:hall|room|lab|auditorium|block))\b", goal, re.I)
        steps.append({"tool": "request_interpreter", "why": "Ask the disability support office for an interpreter",
                      "args": {"event": ev.group(1).strip(" .") if ev else "Event", **when,
                               "place": place.group(1).strip() if place else ""}})
    if re.search(r"\bremind\b|\breminder\b", g):
        what = re.search(r"\bremind me (?:to )?(.+?)(?:\s+(?:on|at|tomorrow|today|by|this|next)\b|$)", goal, re.I)
        steps.append({"tool": "set_reminder", "why": "Save a reminder",
                      "args": {"text": (what.group(1) if what else goal).strip(" ."), **when}})
    msg = re.search(r"\b(?:tell|message|inform|let)\s+(?:my\s+)?([A-Za-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:know\s+)?(?:that\s+)?(.+)", goal, re.I)
    if msg and not steps:
        steps.append({"tool": "save_message", "why": "Prepare the message", "args": {"to": msg.group(1), "text": msg.group(2).strip(" .")}})
    if not steps and _QUESTION.search(goal):
        steps.append({"tool": "lookup_info", "args": {"query": goal}, "why": "Find the answer in the campus information"})
    return {"steps": steps, "ask": "" if steps else "What would you like me to do? For example: book a meeting, set a reminder, or request an interpreter."}


# ---------------------------------------------------------------------------------------------- validation
def _missing(tool: str, args: dict) -> list[str]:
    need = {"check_slot": ["date", "time"], "book_appointment": ["date", "time"], "set_reminder": ["text", "date"],
            "request_interpreter": ["date", "time"], "save_message": ["to", "text"], "lookup_info": ["query"],
            "translate": ["text"]}.get(tool, [])
    return [k for k in need if not str(args.get(k) or "").strip()]


def _clean_args(tool: str, args: dict, goal: str) -> dict:
    a = {k: str(v).strip() for k, v in (args or {}).items() if v not in (None, "") and not isinstance(v, (dict, list))}
    if tool == "translate":
        a["languages"] = [x for x in (args or {}).get("languages", ["hi", "kn", "ta", "te"]) if x in ("hi", "kn", "ta", "te")]
    # normalise date/time; fill from the goal's own words when the model left them out or wrote them oddly
    rules = _date_time(goal)
    if "date" in a:
        try:
            date.fromisoformat(a["date"])
        except ValueError:
            a.pop("date")
    if "time" in a and not re.fullmatch(r"\d{2}:\d{2}", a["time"]):
        a["time"] = _time_of(a["time"]) or ""
        if not a["time"]:
            a.pop("time")
    if tool in ("check_slot", "book_appointment", "set_reminder", "request_interpreter"):
        # dates and times come only from the user's own words: never invented by the model
        for k in ("date", "time"):
            if k in rules:
                a[k] = rules[k]
            else:
                a.pop(k, None)
    if tool == "book_appointment" and a.get("purpose"):
        words = [w for w in re.findall(r"[a-z]{4,}", a["purpose"].lower()) if w not in ("meeting", "appointment", "discussion", "discuss")]
        if not words or not any(w in goal.lower() for w in words):
            a.pop("purpose")  # a purpose the user never mentioned
    if tool in ("check_slot", "book_appointment"):
        who = (a.get("with") or _rule_fields(goal).get("with") or "").strip()
        if who.lower() in ("", "professor", "the professor", "prof", "teacher", "the teacher", "sir", "madam"):
            who = "your " + ("teacher" if "teacher" in who.lower() else "professor")
        a["with"] = who
    if tool == "set_reminder" and a.get("text"):
        t = re.sub(r"^(to|please)\s+", "", a["text"], flags=re.I)
        a["text"] = t[:1].lower() + t[1:] if t[:2] != t[:2].upper() else t
    return a


def normalise_plan(raw: dict, goal: str, rules: dict | None = None) -> dict:
    steps = []
    for s in raw.get("steps") or []:
        if not isinstance(s, dict) or s.get("tool") not in TOOLS:
            continue
        if s["tool"] in INTENT and not INTENT[s["tool"]].search(goal):
            continue  # an action the user did not ask for
        steps.append({"tool": s["tool"], "args": _clean_args(s["tool"], s.get("args") or {}, goal), "why": str(s.get("why") or "")[:160]})
    # a lookup tacked onto an action is noise unless the user actually asked a question
    if any(s["tool"] in EFFECTS for s in steps) and not _QUESTION.search(goal):
        steps = [s for s in steps if s["tool"] != "lookup_info"]
    # an action the user clearly asked for that the model left out is added from the rule planner
    for r in (rules or {}).get("steps", []):
        if r["tool"] in EFFECTS and not any(s["tool"] == r["tool"] for s in steps):
            steps.append(r)
    # safety: every booking is preceded by a slot check with the same details
    out = []
    for s in steps:
        if s["tool"] == "book_appointment" and not any(o["tool"] == "check_slot" for o in out):
            out.append({"tool": "check_slot", "args": {k: s["args"].get(k, "") for k in ("with", "date", "time")}, "why": "Make sure the time is allowed and free"})
        if s["tool"] == "check_slot" and any(o["tool"] == "check_slot" and o["args"] == s["args"] for o in out):
            continue
        out.append(s)
    return {"steps": out[:6], "ask": str(raw.get("ask") or "")}


async def make_plan(goal: str, trace: list) -> dict:
    today = date.today()
    prompt = PLAN_PROMPT.format(today=today.isoformat(), weekday=today.strftime("%A"),
                                tools="\n".join(f"- {k}: {v}" for k, v in TOOL_DOCS.items()))
    plan, provider = None, "rules"
    rules = normalise_plan(rule_plan(goal), goal)
    try:
        out, provider = await llm.chat([{"role": "user", "content": goal}], prompt)
        plan = normalise_plan(_parse_json(out), goal, rules)
        if not plan["steps"] and not plan["ask"]:
            plan = None
    except RuntimeError as e:
        trace.append({"t": _now_iso(), "node": "plan", "detail": f"LLM planner unavailable ({str(e)[:80]}); using rule planner"})
    if plan is None or (not plan["steps"] and rules["steps"]):
        plan, provider = rules, "rules"
    trace.append({"t": _now_iso(), "node": "plan", "detail": f"Planned {len(plan['steps'])} step(s) with {provider}",
                  "data": [s["tool"] for s in plan["steps"]]})
    return {**plan, "provider": provider}


# ---------------------------------------------------------------------------------------------- tools
def next_free_slot(day: date, taken: set[tuple[str, str]], after: str | None = None) -> tuple[str, str] | None:
    d = day
    for _ in range(14):
        if d.weekday() < 5 and d >= date.today():
            for (h0, m0), (h1, _m1) in SLOT_WINDOWS:
                for h in range(h0, h1):
                    t = f"{h:02d}:{m0:02d}"
                    if d == day and after and t <= after:
                        continue
                    if (d.isoformat(), t) not in taken and not (d == date.today() and t <= datetime.now().strftime("%H:%M")):
                        return d.isoformat(), t
        d += timedelta(days=1)
        after = None
    return None


def run_read_only(tool: str, args: dict, taken: set[tuple[str, str]]) -> dict:
    """-> {ok, result, data}"""
    if tool == "lookup_info":
        hits = kb.search(args["query"], k=2)
        if not hits:
            return {"ok": False, "result": "I could not find this in the campus information."}
        p = hits[0][0]
        return {"ok": True, "result": p.text[:500], "data": {"sources": [{"title": h.title, "source": h.source} for h, _ in hits]}}
    if tool == "check_slot":
        problem = _check_slot(args)
        if not problem and (args.get("date"), args.get("time")) in taken:
            problem = "That time is already booked."
        if problem:
            try:
                alt = next_free_slot(date.fromisoformat(args["date"]), taken, args.get("time"))
            except (KeyError, ValueError):
                alt = None
            extra = f" The next free slot is {_human(alt[0], alt[1])}." if alt else ""
            return {"ok": False, "result": problem + extra, "data": {"suggest": {"date": alt[0], "time": alt[1]} if alt else None}}
        return {"ok": True, "result": f"{_human(args['date'], args['time'])} is inside office hours and free."}
    return {"ok": True, "result": ""}


def _human(d: str, t: str | None = None) -> str:
    try:
        dd = datetime.strptime(f"{d} {t or '00:00'}", "%Y-%m-%d %H:%M")
    except ValueError:
        return f"{d} {t or ''}".strip()
    s = dd.strftime("%A %d %B").replace(" 0", " ")
    return s + (" at " + dd.strftime("%I:%M %p").lstrip("0") if t else "")


def describe_effect(tool: str, a: dict) -> str:
    when = _human(a.get("date", ""), a.get("time")) if a.get("date") else ""
    if tool == "book_appointment":
        return f"Book a meeting with {a.get('with', 'the professor')} on {when}" + (f" about {a['purpose']}" if a.get("purpose") else "")
    if tool == "set_reminder":
        return f"Remind you to {a.get('text', '')} on {when}"
    if tool == "request_interpreter":
        return f"Request an ISL interpreter for {a.get('event', 'the event')} on {when}" + (f" in {a['place']}" if a.get("place") else "")
    if tool == "save_message":
        return f"Save a message to {a.get('to', '')}: \"{a.get('text', '')}\" (kept in your outbox, not sent)"
    if tool == "lookup_info":
        return f"Look up: {a.get('query', '')}"
    if tool == "check_slot":
        return f"Check {when} with {a.get('with', 'the professor')}"
    if tool == "translate":
        return "Translate the text"
    return tool


REF_PREFIX = {"book_appointment": "ISH", "set_reminder": "REM", "request_interpreter": "INT", "save_message": "MSG"}


def reference(tool: str, action_id: int) -> str:
    return f"{REF_PREFIX.get(tool, 'ACT')}-{action_id:04d}"
