"""Agentic orchestration API. See services/orchestrator.py for the planner, tools and safety rules.

Lifecycle of a run:
  POST /api/agent/runs            goal -> plan -> read-only steps run -> needs_info | blocked | awaiting_consent | answered
  POST /api/agent/runs/{id}/reply more details from the user -> re-plan with goal + reply
  POST /api/agent/runs/{id}/confirm   user consent -> execute effects -> verify by reading back -> done
  POST /api/agent/runs/{id}/cancel    nothing is executed
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import get_db
from ..models import AgentAction, AgentRun, Conversation, Message, User
from ..services import orchestrator as orch
from ..services.translate import translate_all

router = APIRouter(prefix="/api/agent", tags=["agentic orchestration"])


class RunIn(BaseModel):
    goal: str = Field(min_length=1, max_length=1000)
    source: str = "text"
    languages: list[str] = Field(default_factory=list)


class ReplyIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class DecideIn(BaseModel):
    languages: list[str] = Field(default_factory=list)


def _out(r: AgentRun) -> dict:
    return {
        "id": r.id,
        "conversation_id": r.conversation_id,
        "goal": r.goal,
        "source": r.source,
        "status": r.status,
        "steps": json.loads(r.steps or "[]"),
        "trace": json.loads(r.trace or "[]"),
        "question": r.question,
        "summary": r.summary,
        "provider": r.provider,
        "created_at": r.created_at.isoformat(),
        "updated_at": (r.updated_at or r.created_at).isoformat(),
    }


def _taken(db: Session, user: User) -> set[tuple[str, str]]:
    """Times already booked by this user (so the agent never double-books)."""
    out = set()
    for a in db.scalars(select(AgentAction).where(AgentAction.user_id == user.id, AgentAction.kind == "book_appointment",
                                                  AgentAction.status == "done")).all():
        f = json.loads(a.payload or "{}").get("fields", {})
        if f.get("date") and f.get("time"):
            out.add((f["date"], f["time"]))
    return out


def _say(db: Session, run: AgentRun, text: str, role: str = "assistant", translations: dict | None = None) -> None:
    db.add(Message(conversation_id=run.conversation_id, role=role, content=text, provider="agent" if role == "assistant" else "",
                   source=run.source if role == "user" else "text", translations=json.dumps(translations or {}, ensure_ascii=False),
                   meta=json.dumps({"run_id": run.id})))


async def _plan_and_check(db: Session, user: User, run: AgentRun, goal: str, plan_steps: list | None = None) -> None:
    trace = json.loads(run.trace or "[]")
    if plan_steps is None:
        trace.append({"t": orch._now_iso(), "node": "goal", "detail": goal})
        plan = await orch.make_plan(goal, trace)
        run.provider = plan["provider"]
    else:
        plan = {"steps": plan_steps, "ask": ""}
        trace.append({"t": orch._now_iso(), "node": "replan", "detail": "Same plan with the updated details"})
    steps = [{"tool": s["tool"], "args": s["args"], "why": s.get("why", ""), "status": "pending", "result": "",
              "needs_consent": s["tool"] in orch.EFFECTS, "description": orch.describe_effect(s["tool"], s["args"])}
             for s in plan["steps"]]
    missing = {}
    for s in steps:
        m = orch._missing(s["tool"], s["args"])
        if m:
            missing[s["tool"]] = m
    if not steps or missing:
        run.status = "needs_info"
        if missing:
            effect = next((t for t in missing if t in orch.EFFECTS), next(iter(missing)))
            order = ["to", "text", "query", "date", "time"]
            fields = sorted({f for fs in missing.values() for f in fs}, key=lambda f: order.index(f) if f in order else 9)
            nice = {"date": "which day", "time": "what time", "to": "who it is for", "text": "what it should say", "query": "your question"}
            what = {"book_appointment": "book the meeting", "check_slot": "book the meeting", "set_reminder": "set the reminder",
                    "request_interpreter": "request the interpreter", "save_message": "prepare the message"}.get(effect, "do this")
            run.question = f"To {what}, I need to know {' and '.join(nice.get(f, f) for f in fields)}."
        else:
            run.question = plan["ask"] or "What would you like me to do?"
        trace.append({"t": orch._now_iso(), "node": "ask", "detail": run.question})
        run.steps, run.trace = json.dumps(steps), json.dumps(trace)
        return
    taken = _taken(db, user)
    blocked = False
    for s in steps:
        if s["tool"] in orch.READ_ONLY:
            res = orch.run_read_only(s["tool"], s["args"], taken)
            if s["tool"] == "translate":
                res = {"ok": True, "result": json.dumps(await translate_all(s["args"]["text"], s["args"].get("languages") or ["hi"]), ensure_ascii=False)}
            s["status"] = "done" if res["ok"] else "failed"
            s["result"] = res["result"]
            if res.get("data"):
                s["data"] = res["data"]
            trace.append({"t": orch._now_iso(), "node": f"tool:{s['tool']}", "detail": ("OK: " if res["ok"] else "FAILED: ") + res["result"][:200]})
            if not res["ok"] and s["tool"] == "check_slot":
                blocked = True
        elif blocked:
            s["status"] = "skipped"
            s["result"] = "Not done, because a check before it failed."
        else:
            s["status"] = "needs_consent"
    effects = [s for s in steps if s["status"] == "needs_consent"]
    if blocked:
        run.status = "blocked"
        fail = next(s for s in steps if s["status"] == "failed")
        run.question = fail["result"] + " Tell me another time, or say yes to use the suggested one." if fail.get("data", {}).get("suggest") else fail["result"]
        trace.append({"t": orch._now_iso(), "node": "guard", "detail": "Blocked: nothing will be executed"})
    elif effects:
        run.status = "awaiting_consent"
        run.question = "Shall I do this? " + "; ".join(s["description"] for s in effects) + "."
        trace.append({"t": orch._now_iso(), "node": "consent", "detail": "Waiting for the user's confirmation"})
    else:
        run.status = "answered"
        run.summary = " ".join(s["result"] for s in steps if s["status"] == "done")[:1500] or "I could not find an answer."
        trace.append({"t": orch._now_iso(), "node": "answer", "detail": "Answered from read-only steps"})
    run.steps, run.trace = json.dumps(steps), json.dumps(trace)


def _own(db: Session, user: User, rid: int) -> AgentRun:
    r = db.get(AgentRun, rid)
    if r is None or r.user_id != user.id:
        raise HTTPException(404, "Run not found")
    return r


@router.post("/runs")
async def start(body: RunIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    conv = Conversation(user_id=user.id, title=body.goal[:80], context="campus", mode="agent")
    db.add(conv)
    db.flush()
    run = AgentRun(user_id=user.id, conversation_id=conv.id, goal=body.goal, source=body.source)
    db.add(run)
    db.flush()
    _say(db, run, body.goal, role="user")
    await _plan_and_check(db, user, run, body.goal)
    _say(db, run, run.summary if run.status == "answered" else run.question)
    db.commit()
    return _out(run)


@router.post("/runs/{rid}/reply")
async def reply(rid: int, body: ReplyIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = _own(db, user, rid)
    if run.status not in ("needs_info", "blocked", "awaiting_consent"):
        raise HTTPException(409, f"This run is {run.status}")
    _say(db, run, body.text, role="user")
    steps = json.loads(run.steps or "[]")
    sug = next((st["data"]["suggest"] for st in steps if (st.get("data") or {}).get("suggest")), None)
    yes = orch.re.fullmatch(r"\s*(yes|ok|okay|sure|yes please|use it|fine|that works)\W*", body.text, orch.re.I)
    new_when = orch._date_time(body.text)
    timed = any(st["tool"] in ("check_slot", "book_appointment", "set_reminder", "request_interpreter") for st in steps)
    use_plan = timed and bool((yes and sug) or new_when)
    run.status, run.question = "planning", ""
    if use_plan:
        # same plan, new date/time: the suggested free slot ("yes") or the one the user just gave
        when = sug if yes and sug else new_when
        for st in steps:
            if st["tool"] in ("check_slot", "book_appointment", "set_reminder", "request_interpreter"):
                st["args"] = {**st["args"], **{k: v for k, v in when.items() if k in ("date", "time")}}
        run.goal = f"{run.goal} [updated: {orch._human(when.get('date', ''), when.get('time'))}]"
        await _plan_and_check(db, user, run, run.goal, plan_steps=steps)
    else:
        run.goal = f"{run.goal}. {body.text}"
        await _plan_and_check(db, user, run, run.goal)
    _say(db, run, run.summary if run.status == "answered" else run.question)
    db.commit()
    return _out(run)


@router.post("/runs/{rid}/confirm")
async def confirm(rid: int, body: DecideIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = _own(db, user, rid)
    if run.status != "awaiting_consent":
        raise HTTPException(409, f"This run is {run.status}, not waiting for confirmation")
    steps = json.loads(run.steps or "[]")
    trace = json.loads(run.trace or "[]")
    trace.append({"t": orch._now_iso(), "node": "consent", "detail": "User confirmed"})
    # re-check right before acting: the slot may have been taken since planning
    taken = _taken(db, user)
    for s in steps:
        if s["tool"] == "check_slot":
            res = orch.run_read_only("check_slot", s["args"], taken)
            if not res["ok"]:
                run.status, run.question = "blocked", res["result"]
                s["status"], s["result"] = "failed", res["result"]
                trace.append({"t": orch._now_iso(), "node": "guard", "detail": "Re-check failed: " + res["result"]})
                run.steps, run.trace = json.dumps(steps), json.dumps(trace)
                _say(db, run, res["result"])
                db.commit()
                return _out(run)
    done_lines = []
    for s in steps:
        if s["status"] != "needs_consent":
            continue
        a = AgentAction(user_id=user.id, conversation_id=run.conversation_id, kind=s["tool"],
                        payload=json.dumps({"fields": s["args"], "run_id": run.id}), status="done", result=s["description"])
        db.add(a)
        db.flush()
        ref = orch.reference(s["tool"], a.id)
        a.result = f"{s['description']}. Reference {ref}."
        db.commit()
        # verify: read it back from storage before reporting success
        check = db.get(AgentAction, a.id)
        ok = check is not None and check.status == "done" and json.loads(check.payload).get("fields") == s["args"]
        s["status"] = "done" if ok else "failed"
        s["reference"] = ref if ok else None
        s["verified"] = ok
        s["result"] = f"Done and verified. Reference {ref}." if ok else "Could not verify this step."
        trace.append({"t": orch._now_iso(), "node": f"execute:{s['tool']}", "detail": s["result"]})
        if ok:
            done_lines.append(f"{s['description'].replace('Book a', 'Booked a').replace('Remind you', 'Reminder set').replace('Request an', 'Requested an').replace('Save a', 'Saved a')} (reference {ref})")
    ok_all = all(s["status"] in ("done", "skipped") for s in steps)
    run.status = "done" if ok_all else "failed"
    run.summary = ("Done. " + ". ".join(done_lines) + ".") if ok_all else "Some steps could not be completed."
    trace.append({"t": orch._now_iso(), "node": "verify", "detail": "All effects verified in storage" if ok_all else "Verification failed"})
    run.steps, run.trace = json.dumps(steps), json.dumps(trace)
    tr = await translate_all(run.summary, [lg for lg in body.languages if lg != "en"]) if body.languages else {}
    _say(db, run, run.summary, translations=tr)
    db.commit()
    out = _out(run)
    out["translations"] = tr
    return out


@router.post("/runs/{rid}/cancel")
def cancel(rid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = _own(db, user, rid)
    if run.status in ("done", "cancelled"):
        raise HTTPException(409, f"This run is already {run.status}")
    steps = json.loads(run.steps or "[]")
    for s in steps:
        if s["status"] in ("needs_consent", "pending"):
            s["status"], s["result"] = "skipped", "Cancelled by you. Nothing was done."
    trace = json.loads(run.trace or "[]") + [{"t": orch._now_iso(), "node": "cancel", "detail": "User cancelled; nothing executed"}]
    run.status, run.summary, run.steps, run.trace = "cancelled", "Cancelled. Nothing was done.", json.dumps(steps), json.dumps(trace)
    _say(db, run, run.summary)
    db.commit()
    return _out(run)


@router.get("/runs")
def list_runs(user: User = Depends(current_user), db: Session = Depends(get_db)):
    runs = db.scalars(select(AgentRun).where(AgentRun.user_id == user.id).order_by(AgentRun.id.desc()).limit(100)).all()
    return [_out(r) for r in runs]


@router.get("/runs/{rid}")
def get_run(rid: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return _out(_own(db, user, rid))


@router.get("/tools")
def tools():
    return [{"name": k, "doc": v, "needs_consent": k in orch.EFFECTS} for k, v in orch.TOOL_DOCS.items()]
