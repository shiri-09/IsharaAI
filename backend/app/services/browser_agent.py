"""Browser agent: a real browser (Playwright driving the installed Chrome) operated step by step by the local LLM (Qwen via Ollama).

Loop:  look at the page -> the model picks ONE action -> safety checks -> do it in the browser, visibly -> repeat.
The person is in the loop the whole time:
  - choices that depend on taste (which dish, which time) are asked, with the options taken from the page;
  - anything that spends money or sends something is confirmed first, in one clear sentence;
  - personal details (name, phone, address, ...) are filled from MEMORY; if the agent has never seen one it asks once,
    types it, and remembers it for next time;
  - login, payment, passwords, card numbers, OTPs and PINs are NEVER typed by the agent or stored: the person does that in the
    browser window and presses Continue.
Every step is recorded as an event the app shows live. A screenshot is kept after each step so the app can show the browser too.
"""

import asyncio
import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import select

from ..config import settings
from ..db import SessionLocal
from ..llm.router import llm
from ..models import AgentMemory

log = logging.getLogger("ishara.browser")

MAX_STEPS = 40
ASK_TIMEOUT = 900  # seconds the agent waits for the person
PROFILE_DIR = settings.data_dir / "browser_profile"

SENSITIVE = re.compile(r"password|passcode|otp|one.?time|cvv|cvc|card.?number|credit|debit|pin\b|upi|aadhaar|aadhar|\bpan\b|account.?number|ifsc|security.?code", re.I)
RISKY_CLICK = re.compile(r"place\s*order|pay\b|pay\s*now|confirm\s*(order|booking|payment)|submit|book\s*now|buy\s*now|purchase|send\s*(message|request)|request\s*meeting|delete|cancel\s*(order|booking)|subscribe|sign\s*up|register", re.I)
STABLE = {"name", "phone", "email", "address"}  # remembered for next time; per-task answers (notes, reason, dish) are asked fresh
SAFE_NAV = re.compile(r"^https?://", re.I)

SYNONYMS = {
    "name": ["name", "full name", "your name", "customer name", "first name"],
    "phone": ["phone", "mobile", "contact number", "phone number", "mobile number", "telephone", "your phone number", "10 digit"],
    "email": ["email", "e-mail", "mail id"],
    "address": ["address", "delivery address", "block and room", "room", "hostel", "block", "landmark"],
    "notes": ["notes", "instructions", "delivery notes", "note", "special request"],
}


def canon(label: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", label.lower()).strip()
    for key, words in SYNONYMS.items():
        if any(w in t for w in words):
            return key
    return re.sub(r"\s+", "_", t)[:40] or "field"


def is_sensitive(*texts: str) -> bool:
    return any(t and SENSITIVE.search(t) for t in texts)


# ---------------------------------------------------------------------------------------------- page reading
OBSERVE_JS = r"""() => {
  const vis = (el) => { const r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    return r.width > 2 && r.height > 2 && cs.visibility !== 'hidden' && cs.display !== 'none' && r.bottom > -50 && r.top < innerHeight * 2; };
  document.querySelectorAll('[data-ia]').forEach(e => e.removeAttribute('data-ia'));
  const sel = 'a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=checkbox],[role=radio],[role=tab],[onclick]';
  const out = []; let id = 1;
  for (const el of document.querySelectorAll(sel)) {
    if (!vis(el) || el.type === 'hidden') continue;
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
    const text = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || '').trim().replace(/\s+/g, ' ').slice(0, 70);
    let label = '';
    if (el.id) { const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]'); if (l) label = l.innerText.trim().replace(/\s+/g, ' ').slice(0, 60); }
    if (!label) { const l = el.closest('label'); if (l) label = l.innerText.trim().replace(/\s+/g, ' ').split('\n')[0].slice(0, 60); }
    let ctx = '';
    if (tag === 'button' || tag === 'a' || el.getAttribute('role') === 'button') {
      // the card this button belongs to: climb until the parent would also hold another button with the same label
      let p = el;
      for (let i = 0; i < 4 && p.parentElement; i++) {
        const same = Array.from(p.parentElement.querySelectorAll(tag)).filter(b => (b.innerText || '').trim() === (el.innerText || '').trim());
        if (same.length > 1) break;
        p = p.parentElement;
      }
      const t = (p.innerText || '').trim().replace(/\s+/g, ' ');
      if (p !== el && t.length > text.length + 3) ctx = t.slice(0, 110);
    }
    el.setAttribute('data-ia', String(id));
    const r = el.getBoundingClientRect();
    out.push({ id, tag, type, text, label, placeholder: el.getAttribute('placeholder') || '', name: el.getAttribute('name') || '',
      value: (tag === 'input' || tag === 'textarea' || tag === 'select') ? (el.value || '').slice(0, 60) : '', checked: !!el.checked, disabled: !!el.disabled,
      ctx: ctx === text ? '' : ctx, options: tag === 'select' ? Array.from(el.options).map(o => o.text.trim()).filter(Boolean).slice(0, 20) : undefined,
      inView: r.top >= 0 && r.bottom <= innerHeight });
    id++; if (id > 80) break;
  }
  return { url: location.href, title: document.title, text: (document.body.innerText || '').trim().replace(/\n{2,}/g, '\n').slice(0, 1400),
           elements: out, atBottom: innerHeight + scrollY >= document.body.scrollHeight - 4 };
}"""

CURSOR_JS = r"""(() => {
  if (window.__iaCursor) return; window.__iaCursor = true;
  const make = () => {
    if (!document.body) return setTimeout(make, 50);
    const c = document.createElement('div');
    c.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;transition:transform .05s linear;transform:translate(-100px,-100px)';
    c.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M3 2 L3 18 L7.2 14.2 L10 20.5 L12.6 19.4 L9.9 13.2 L15.6 13.2 Z" fill="#f0a64a" stroke="#1a1208" stroke-width="1.2" stroke-linejoin="round"/></svg><span style="position:absolute;left:20px;top:20px;white-space:nowrap;background:#f0a64a;color:#000;font:600 10px system-ui;padding:1px 5px;border-radius:4px">Ishara agent</span>';
    document.body.appendChild(c);
    addEventListener('mousemove', e => { c.style.transform = 'translate(' + e.clientX + 'px,' + e.clientY + 'px)'; }, true);
    addEventListener('mousedown', e => { const r = document.createElement('div'); r.style.cssText = 'position:fixed;z-index:2147483646;pointer-events:none;width:30px;height:30px;border-radius:50%;border:3px solid #f0a64a;left:' + (e.clientX - 15) + 'px;top:' + (e.clientY - 15) + 'px;transition:all .5s ease-out;opacity:1'; document.body.appendChild(r); requestAnimationFrame(() => { r.style.transform = 'scale(2.2)'; r.style.opacity = '0'; }); setTimeout(() => r.remove(), 600); }, true);
  };
  make();
})();"""


def fmt_elements(els: list[dict]) -> str:
    lines = []
    for e in els:
        bits = [f"[{e['id']}]", e["tag"] + (f":{e['type']}" if e["type"] and e["type"] != "submit" else "")]
        name = e["text"] or e["label"] or e["placeholder"] or e["name"]
        if e["label"] and e["text"] and e["label"] != e["text"]:
            name = f"{e['label']} / {e['text']}"
        if name:
            bits.append(f'"{name}"')
        if e["value"]:
            bits.append(f'value="{e["value"]}"')
        if e["tag"] == "select" and e.get("options"):
            bits.append("options=" + " | ".join(e["options"]))
        if e["ctx"]:
            bits.append(f'(in: {e["ctx"]})')
        if e["checked"]:
            bits.append("[checked]")
        if e["disabled"]:
            bits.append("[disabled]")
        if not e["inView"]:
            bits.append("(offscreen)")
        lines.append(" ".join(bits))
    return "\n".join(lines)


SYSTEM = """You are Ishara's browser agent. You operate a real web browser for a person who may be Deaf or may use only eye gestures, so questions must be short and clear.
You see the page as numbered elements. Reply with ONE JSON object and nothing else:
{"thought": "<one short sentence>", "action": "click|type|select|goto|scroll|wait|ask_user|done|fail", "id": <element number>, "text": "<text to type or the option to select>", "field": "<name of the personal detail being typed: name, phone, address, email or notes; empty if it is not a personal detail>", "url": "<for goto>", "question": "<for ask_user>", "kind": "choice|text|confirm|continue", "options": ["<for choice>"], "message": "<for done or fail>"}
Rules:
- Work toward the goal one small step at a time. One action per reply.
- Never invent personal details. If a field needs one (name, phone, address, email, delivery notes), use action "type" with "field" set and leave "text" empty. The app fills it from memory or asks the person.
- If the choice depends on the person's taste (which dish, which professor, which time) and the goal did not already say it, use ask_user with kind "choice" and 2 to 5 options copied from the page. Ask once per decision. If the goal already names the choice, just do it.
- The app itself asks the person to confirm before any click that spends money, sends or books something (Place order, Pay, Submit, Request meeting). So do NOT ask for that confirmation yourself: just click the button when everything is filled in.
- When the page has the same button on every item (Add to cart on each dish, Book on each slot), do not ask in words: click one of them and the app shows the person the real options.
- Personal details are never asked with ask_user: use type with "field" instead.
- ask_user is only for a real decision or a missing piece of information, and it must contain a specific "question" written for the person (for example "Which dish would you like?"). Never ask a vague question such as "What would you like me to do?".
- After the person answers a question, act on the answer; never ask the same thing again.
- Never type passwords, card numbers, OTPs or UPI PINs. If login or payment is needed, use ask_user with kind "continue" asking the person to do it in the browser window.
- "select" is for dropdowns (a select element): text is the option to choose.
- When the goal is finished, use done with a short message (include any order or reference number you can see). Use fail only if it is impossible."""


# ---------------------------------------------------------------------------------------------- the run
@dataclass
class BrowserRun:
    id: str
    user_id: int
    goal: str
    start_url: str = ""
    headless: bool = False
    status: str = "starting"  # starting | thinking | acting | waiting_user | done | failed | stopped
    events: list[dict] = field(default_factory=list)
    ask: dict | None = None
    result: str = ""
    shot: bytes = b""
    shot_at: float = 0.0
    created: float = field(default_factory=time.time)
    answers: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    stop: threading.Event = field(default_factory=threading.Event)
    _n: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    conv_id: int | None = None

    def emit(self, kind: str, text: str, **data) -> None:
        with self._lock:
            self._n += 1
            self.events.append({"i": self._n, "t": time.strftime("%H:%M:%S"), "kind": kind, "text": text, **data})
            self.events = self.events[-300:]

    def state(self) -> dict:
        return {"id": self.id, "goal": self.goal, "status": self.status, "events": list(self.events), "ask": self.ask, "result": self.result,
                "headless": self.headless, "has_shot": bool(self.shot), "shot_at": self.shot_at}


RUNS: dict[str, BrowserRun] = {}
_counter = 0


def _llm_call(user: str):
    """Playwright's sync mode keeps an event loop alive in the agent thread, so the async LLM call runs in a thread of its own."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(llm.chat([{"role": "user", "content": user}], SYSTEM))).result()


def _parse_json(out: str) -> dict:
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {}
    txt = m.group(0)
    for cand in (txt, re.sub(r",\s*([}\]])", r"\1", txt)):
        try:
            d = json.loads(cand)
            return d if isinstance(d, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


class Agent:
    def __init__(self, run: BrowserRun):
        self.run = run
        self.page = None
        self.ctx = None
        self.history: list[str] = []
        self.confirmed: set[str] = set()
        self.repeat: dict[str, int] = {}
        self.answered: dict[str, str] = {}
        self.vague = 0

    # ---- memory (database, personal to this user)
    def remembered(self) -> dict[str, str]:
        with SessionLocal() as db:
            rows = db.scalars(select(AgentMemory).where(AgentMemory.user_id == self.run.user_id)).all()
            return {r.key: r.value for r in rows}

    def memory_get(self, key: str) -> str | None:
        with SessionLocal() as db:
            r = db.scalar(select(AgentMemory).where(AgentMemory.user_id == self.run.user_id, AgentMemory.key == key))
            if r and r.value:
                r.uses += 1
                db.commit()
                return r.value
        return None

    def memory_set(self, key: str, label: str, value: str) -> None:
        with SessionLocal() as db:
            r = db.scalar(select(AgentMemory).where(AgentMemory.user_id == self.run.user_id, AgentMemory.key == key))
            if r:
                r.value, r.label = value, label or r.label
            else:
                db.add(AgentMemory(user_id=self.run.user_id, key=key, label=label, value=value))
            db.commit()

    # ---- talking to the person
    def snap(self) -> None:
        try:
            if self.page and not self.page.is_closed():
                self.run.shot = self.page.screenshot(type="jpeg", quality=62)
                self.run.shot_at = time.time()
        except Exception:  # noqa: BLE001 - a missed frame is harmless
            pass

    def ask(self, kind: str, question: str, options: list[str] | None = None, field_name: str = "", sensitive: bool = False) -> str | None:
        """Block until the person answers. Returns None if the run was stopped."""
        self.snap()
        a = {"id": f"q{int(time.time() * 1000)}", "kind": kind, "question": question, "options": options or [], "field": field_name, "sensitive": sensitive}
        self.run.ask = a
        self.run.status = "waiting_user"
        self.run.emit("ask", question, ask=a)
        deadline = time.time() + ASK_TIMEOUT
        while time.time() < deadline and not self.run.stop.is_set():
            try:
                ans = self.run.answers.get(timeout=0.4)
            except queue.Empty:
                self.snap_if_stale()
                continue
            if ans.get("ask_id") != a["id"]:
                continue
            self.run.ask = None
            self.run.status = "thinking"
            text = str(ans.get("answer", "")).strip()
            self.run.emit("answer", text if not sensitive else "(done by the person)")
            return text
        self.run.ask = None
        return None

    def snap_if_stale(self) -> None:
        if time.time() - self.run.shot_at > 1.2:
            self.snap()

    # ---- the browser, visibly
    def center(self, loc):
        loc.scroll_into_view_if_needed(timeout=4000)
        box = loc.bounding_box()
        if not box:
            return None
        return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    def move_to(self, loc) -> bool:
        c = self.center(loc)
        if not c:
            return False
        try:
            loc.evaluate("e => { e.style.transition='outline .15s'; e.style.outline='3px solid #f0a64a'; e.style.outlineOffset='3px'; setTimeout(() => { e.style.outline=''; e.style.outlineOffset=''; }, 1400); }")
        except Exception:  # noqa: BLE001
            pass
        for k in range(3):
            self.page.mouse.move(c[0], c[1], steps=8)
            self.snap()
        return True

    def element(self, eid) -> dict | None:
        for e in self.obs["elements"]:
            if e["id"] == eid:
                return e
        return None

    def observe(self) -> dict:
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=4000)
        except Exception:  # noqa: BLE001
            pass
        self.obs = self.page.evaluate(OBSERVE_JS)
        self.snap()
        return self.obs

    # ---- the model decides
    def decide(self, note: str = "") -> dict:
        mem = self.remembered()
        known = ", ".join(sorted(mem)) or "nothing yet"
        user = (
            f"GOAL: {self.run.goal}\nI can fill these personal details from memory: {known}.\n"
            f"RECENT STEPS:\n" + ("\n".join(self.history[-8:]) or "(none yet)") + "\n\n"
            f"PAGE: {self.obs['title']} ({self.obs['url']})\n"
            f"VISIBLE TEXT:\n{self.obs['text'][:900]}\n\nELEMENTS:\n{fmt_elements(self.obs['elements'])}\n"
            + (f"\nNOTE: {note}\n" if note else "")
            + '\nReply with the JSON object only.'
        )
        for attempt in range(2):
            try:
                out, provider = _llm_call(user)
            except RuntimeError as e:
                return {"action": "fail", "message": f"The local AI model did not answer ({str(e)[:80]}). Is Ollama running?"}
            d = _parse_json(out)
            if d.get("action"):
                return d
            user += "\n\nYour last reply was not valid JSON. Reply with ONE JSON object only."
        return {"action": "fail", "message": "I could not decide the next step."}

    # ---- actions
    def do(self, d: dict) -> bool:
        """Perform one decided action. Returns False to end the run."""
        act = str(d.get("action", "")).lower()
        thought = str(d.get("thought", ""))[:160]
        eid = d.get("id")
        eid = int(eid) if isinstance(eid, (int, float, str)) and str(eid).lstrip("-").isdigit() else None
        fp = hash((self.obs["url"], self.obs["text"][:600], tuple((e["id"], e["value"], e["checked"]) for e in self.obs["elements"])))
        sig = f"{act}|{eid}|{d.get('text', '')}|{d.get('url', '')}"
        rk = f"{sig}|{fp}"  # the same action on a page that did not change
        self.repeat[rk] = self.repeat.get(rk, 0) + 1
        if self.repeat[rk] >= 3 and act not in ("done", "fail", "ask_user"):
            ans = self.ask("text", "I seem to be going around in circles. What should I do next?", field_name="help")
            if ans is None:
                return False
            self.history.append(f"Person said: {ans}")
            self.repeat.clear()
            return True

        if act == "done":
            self.run.result = str(d.get("message") or "Done.")
            self.run.emit("done", self.run.result)
            self.run.status = "done"
            return False
        if act == "fail":
            self.run.result = str(d.get("message") or "I could not finish this.")
            self.run.emit("error", self.run.result)
            self.run.status = "failed"
            return False
        if act == "ask_user":
            kind = d.get("kind") if d.get("kind") in ("choice", "text", "confirm", "continue") else ("choice" if d.get("options") else "text")
            opts = [str(o) for o in (d.get("options") or [])][:6]
            q = str(d.get("question") or "").strip()
            if len(q) < 8 or q.lower().startswith("what would you like me to do"):
                self.vague = getattr(self, "vague", 0) + 1
                self.history.append("Your ask_user had no specific question. Do the next step yourself, or ask a specific question.")
                if self.vague < 3:
                    return True
                q = str(d.get("thought") or "How should I continue?")
            if kind == "choice" and not opts:
                kind = "text"
            PERSONAL_Q = re.compile(r"(name|phone|mobile|e-?mail|address|room|block)", re.I)
            if kind == "text" and PERSONAL_Q.search(q) and not is_sensitive(q):
                ck = canon(q)
                val = self.memory_get(ck)
                if val:
                    self.run.emit("memory", f"Using your saved {ck}: {val}")
                    self.history.append(f'The person {ck} is "{val}". Type it into the right field with action type and field "{ck}".')
                    return True
                ans = self.ask("text", q, field_name=ck)
                if ans is None:
                    return False
                if ans and ck in STABLE:
                    self.memory_set(ck, q, ans)
                    self.run.emit("memory", f"Remembered your {ck} for next time")
                self.history.append(f'The person {ck} is "{ans}". Type it into the right field with action type and field "{ck}".')
                return True
            key = q.lower()
            if key in self.answered:
                self.history.append(f"Already asked '{q}' and the person said: {self.answered[key]}. Use that answer.")
                return True
            ans = self.ask(kind, q, opts, sensitive=False)
            if ans is None:
                return False
            self.answered[key] = ans
            if kind == "confirm" and ans.lower() not in ("yes", "y", "ok", "confirm", "go ahead"):
                self.history.append(f"Asked '{q}' -> the person said NO. Do not do it.")
                self.run.emit("info", "Okay, I will not do that.")
            else:
                self.history.append(f"Asked '{q}' -> person answered: {ans}")
            return True
        if act == "wait":
            self.page.wait_for_timeout(1200)
            self.history.append("waited")
            return True
        if act == "scroll":
            self.run.emit("action", "Scroll down", thought=thought)
            self.page.mouse.wheel(0, 600)
            self.page.wait_for_timeout(400)
            self.history.append("scrolled down")
            return True
        if act == "goto":
            url = str(d.get("url") or "")
            if not SAFE_NAV.match(url):
                self.history.append(f"refused to open {url}: only http and https pages")
                return True
            self.run.emit("action", f"Open {urlparse(url).netloc}{urlparse(url).path}", thought=thought)
            self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            self.history.append(f"opened {url}")
            return True

        el = self.element(eid) if eid is not None else None
        if el is None:
            self.history.append(f"element {eid} does not exist; choose one from the list")
            return True
        loc = self.page.locator(f'[data-ia="{eid}"]').first
        name = el["text"] or el["label"] or el["placeholder"] or el["name"] or el["tag"]

        if act == "click":
            # Several identical buttons (Add to cart on every dish, Book on every slot): it is a choice, so the person picks.
            group = [e for e in self.obs["elements"] if e["tag"] == el["tag"] and e["text"] == el["text"] and e["ctx"] and not e["disabled"]]
            gkey = f"group:{el['text']}:{len(group)}"
            if len(group) >= 3 and el["text"] and gkey not in self.answered:
                labels = []
                for e in group[:6]:
                    lab = re.sub(r"\s*" + re.escape(e["text"]) + r"\s*$", "", e["ctx"]).strip() or e["ctx"]
                    labels.append(lab[:70])
                goal_l = self.run.goal.lower()
                hits = [lab for lab in labels if " ".join(lab.lower().split()[:2]) in goal_l]
                if len(hits) == 1:  # the goal already names it ("order a masala dosa")
                    ans = hits[0]
                    self.run.emit("info", f'You asked for "{ans[:40]}", so I will pick it.')
                else:
                    ans = self.ask("choice", f'Which one would you like? ("{el["text"]}")', labels)
                if ans is None:
                    return False
                self.answered[gkey] = ans
                pick = next((e for e, lab in zip(group, labels) if lab.lower() == ans.lower()), None) or next(
                    (e for e, lab in zip(group, labels) if ans.lower() in lab.lower() or lab.lower() in ans.lower()), None)
                if pick is None:
                    self.history.append(f'The person wants "{ans}". Click the matching "{el["text"]}" button.')
                    return True
                el = pick
                eid = pick["id"]
                loc = self.page.locator(f'[data-ia="{eid}"]').first
                name = el["text"]
            if (RISKY_CLICK.search(name) or RISKY_CLICK.search(el["ctx"][:30])) and sig not in self.confirmed and el["tag"] in ("button", "a", "input"):
                ans = self.ask("confirm", f'I am about to click "{name}". Shall I go ahead?')
                if ans is None:
                    return False
                if ans.lower() not in ("yes", "y", "ok", "confirm", "go ahead"):
                    self.history.append(f'the person said NO to clicking "{name}"')
                    self.run.emit("info", "Okay, I will not click it.")
                    return True
                self.confirmed.add(sig)
            self.run.status = "acting"
            self.run.emit("action", f'Click "{name}"' + (f" ({el['ctx'][:50]})" if el["ctx"] else ""), thought=thought)
            if not self.move_to(loc):
                self.history.append(f'could not reach "{name}"')
                return True
            self.page.mouse.down()
            self.page.mouse.up()
            self.page.wait_for_timeout(500)
            self.history.append(f'clicked "{name}"' + (f" ({el['ctx'][:40]})" if el["ctx"] else ""))
            self.snap()
            return True

        if act == "select":
            self.run.status = "acting"
            want = str(d.get("text") or "")
            self.run.emit("action", f'Choose "{want}" in "{name}"', thought=thought)
            self.move_to(loc)
            try:
                loc.select_option(label=want, timeout=4000)
            except Exception:  # noqa: BLE001
                try:
                    loc.select_option(value=want, timeout=3000)
                except Exception:  # noqa: BLE001
                    self.history.append(f'could not choose "{want}" in "{name}"; options are {el.get("options")}')
                    return True
            self.history.append(f'chose "{want}" in "{name}"')
            self.snap()
            return True

        if act == "type":
            label = el["label"] or el["placeholder"] or el["name"] or name
            if el["type"] == "password" or is_sensitive(label, el["name"], el["placeholder"], el["type"]):
                self.run.emit("info", f'"{label}" is private. I will not type it.')
                ans = self.ask("continue", f'Please enter "{label}" yourself in the browser window, then press Continue.', sensitive=True)
                if ans is None:
                    return False
                self.history.append(f'the person entered "{label}" themselves')
                return True
            text = str(d.get("text") or "")
            fld = str(d.get("field") or "").strip()
            from_memory = False
            if fld or not text:
                key = canon(fld or label)
                val = self.memory_get(key)
                if val:
                    text, from_memory = val, True
                    self.run.emit("memory", f'Using your saved {fld or label}: {val}')
                else:
                    ans = self.ask("text", f'What should I put for "{label}"?', field_name=fld or label)
                    if ans is None:
                        return False
                    text = ans
                    if text:
                        if key in STABLE:
                            self.memory_set(key, label, text)
                            self.run.emit("memory", f'Remembered your {fld or label} for next time')
            if not text:
                self.history.append(f'nothing to type in "{label}"')
                return True
            self.run.status = "acting"
            self.run.emit("action", f'Type in "{label}"' + (" (from memory)" if from_memory else ""), thought=thought)
            self.move_to(loc)
            loc.click(timeout=4000)
            try:
                loc.fill("", timeout=3000)
            except Exception:  # noqa: BLE001
                pass
            for i in range(0, len(text), 4):  # visible typing
                self.page.keyboard.type(text[i : i + 4], delay=35)
                if (i // 4) % 3 == 0:
                    self.snap()
            self.history.append(f'typed into "{label}"')
            self.snap()
            return True

        self.history.append(f"unknown action {act}")
        return True

    # ---- the loop
    def run_loop(self) -> None:
        from playwright.sync_api import sync_playwright

        r = self.run
        try:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            with sync_playwright() as p:
                self.ctx = p.chromium.launch_persistent_context(
                    str(PROFILE_DIR / str(r.user_id)),
                    channel="chrome",
                    headless=r.headless,
                    viewport={"width": 1280, "height": 800},
                    args=["--window-position=40,40", "--disable-blink-features=AutomationControlled"],
                )
                self.ctx.add_init_script(CURSOR_JS)
                self.page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()
                r.emit("info", "Browser opened. You can watch it here and in the Chrome window.")
                if r.start_url:
                    self.page.goto(r.start_url, wait_until="domcontentloaded", timeout=25000)
                    r.emit("action", f"Open {urlparse(r.start_url).netloc}{urlparse(r.start_url).path}")
                else:
                    self.page.goto("about:blank")
                for step in range(MAX_STEPS):
                    if r.stop.is_set():
                        break
                    r.status = "thinking"
                    self.observe()
                    d = self.decide()
                    if d.get("thought"):
                        r.emit("thought", str(d["thought"])[:200])
                    try:
                        if not self.do(d):
                            break
                    except Exception as e:  # noqa: BLE001 - a failed click is not the end: tell the model and try again
                        log.warning("browser action failed: %r", e)
                        self.history.append(f"that action failed ({str(e)[:80]})")
                        r.emit("info", "That did not work, trying another way.")
                else:
                    r.result, r.status = "I ran out of steps before finishing.", "failed"
                    r.emit("error", r.result)
                if r.stop.is_set() and r.status not in ("done", "failed"):
                    r.status, r.result = "stopped", "Stopped."
                    r.emit("info", "Stopped.")
                self.snap()
                time.sleep(1.2)
                self.ctx.close()
        except Exception as e:  # noqa: BLE001
            log.exception("browser agent crashed")
            r.status, r.result = "failed", f"The browser could not run: {str(e)[:160]}"
            r.emit("error", r.result)
        finally:
            r.ask = None
            if r.status in ("starting", "thinking", "acting", "waiting_user"):
                r.status = "failed"
            self._persist()

    def _persist(self) -> None:
        """Keep a readable record in History."""
        from ..models import Conversation, Message

        try:
            with SessionLocal() as db:
                conv = Conversation(user_id=self.run.user_id, title=self.run.goal[:80], context="general", mode="agent")
                db.add(conv)
                db.flush()
                db.add(Message(conversation_id=conv.id, role="user", content=self.run.goal, source="text"))
                lines = [f"{e['t']}  {e['text']}" for e in self.run.events if e["kind"] in ("action", "ask", "answer", "memory", "info", "done", "error")]
                db.add(Message(conversation_id=conv.id, role="assistant", content=(self.run.result or self.run.status) + "\n\n" + "\n".join(lines), provider="browser-agent"))
                db.commit()
                self.run.conv_id = conv.id
        except Exception:  # noqa: BLE001
            log.exception("could not save the browser run to history")


def start_run(user_id: int, goal: str, start_url: str = "", headless: bool = False) -> BrowserRun:
    global _counter
    for r in RUNS.values():  # one live browser per person
        if r.user_id == user_id and r.status in ("starting", "thinking", "acting", "waiting_user"):
            r.stop.set()
    _counter += 1
    run = BrowserRun(id=f"br{_counter}", user_id=user_id, goal=goal, start_url=start_url, headless=headless)
    RUNS[run.id] = run
    for old in sorted(RUNS)[:-20]:
        RUNS.pop(old, None)
    threading.Thread(target=Agent(run).run_loop, daemon=True, name=f"browser-{run.id}").start()
    return run
