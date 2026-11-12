import { useEffect, useMemo, useRef, useState } from "react";
import {
  BellRing,
  Bot,
  CalendarCheck,
  CalendarClock,
  CalendarPlus,
  Check,
  Copy,
  Download,
  ExternalLink,
  Hand,
  Languages,
  Loader2,
  Lock,
  Mail,
  MessageCircle,
  Plus,
  Search,
  UtensilsCrossed,
  Video,
  X,
  type LucideIcon,
} from "lucide-react";
import type { Run } from "../pages/Agent";

/**
 * "Agent workspace": a browser-style window that shows the agent working, like a computer-use agent.
 * It follows the REAL run: when a step is running it opens that step's page, moves the cursor to each field, types the
 * real values and presses Save. The backend commits the real record at the same moment (it waits for this animation).
 * Everything is Ishara's own pages and data. Nothing here controls any other website.
 */

type Step = Run["steps"][number];
type Field = { id: string; label: string; value: string };
type Spec = { key: string; url: string; title: string; fields: Field[]; submit: string; icon: LucideIcon };

const TOOL_ICON: Record<string, LucideIcon> = {
  lookup_info: Search,
  check_slot: CalendarClock,
  book_appointment: CalendarCheck,
  set_reminder: BellRing,
  request_interpreter: Hand,
  save_message: Mail,
  translate: Languages,
  order_meal: UtensilsCrossed,
};

const human = (d?: string, t?: string) => {
  if (!d) return "";
  const dt = new Date(`${d}T${t || "00:00"}`);
  const day = dt.toLocaleDateString([], { weekday: "long", day: "numeric", month: "long" });
  return t ? `${day}, ${dt.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : day;
};

function specFor(s: Step, i: number): Spec {
  const a = s.args;
  const key = `${i}-${s.tool}`;
  switch (s.tool) {
    case "book_appointment":
      return { key, url: "ishara.ai/calendar/new-meeting", title: "New meeting", icon: CalendarCheck, submit: "Save meeting", fields: [
        { id: "with", label: "Who", value: a.with || "" }, { id: "when", label: "Date and time", value: human(a.date, a.time) },
        ...(a.purpose ? [{ id: "about", label: "About", value: a.purpose }] : []), { id: "room", label: "Online room", value: "Create a meeting link" },
      ] };
    case "set_reminder":
      return { key, url: "ishara.ai/reminders/new", title: "New reminder", icon: BellRing, submit: "Save reminder", fields: [
        { id: "text", label: "Remind me to", value: a.text || "" }, { id: "when", label: "Date and time", value: human(a.date, a.time) } ] };
    case "request_interpreter":
      return { key, url: "ishara.ai/support/interpreter", title: "Interpreter request", icon: Hand, submit: "Send request", fields: [
        { id: "event", label: "Event", value: a.event || "" }, { id: "when", label: "Date and time", value: human(a.date, a.time) },
        ...(a.place ? [{ id: "place", label: "Place", value: a.place }] : []) ] };
    case "save_message":
      return { key, url: "ishara.ai/outbox/new", title: "New message", icon: Mail, submit: "Save message", fields: [
        { id: "to", label: "To", value: a.to || "" }, { id: "text", label: "Message", value: a.text || "" } ] };
    case "order_meal":
      return { key, url: "ishara.ai/food/search", title: "Find food", icon: UtensilsCrossed, submit: "Add to my orders", fields: [
        { id: "item", label: "What would you like?", value: a.item || "" } ] };
    case "lookup_info":
      return { key, url: "ishara.ai/campus/search", title: "Campus search", icon: Search, submit: "Search", fields: [
        { id: "q", label: "Search campus information", value: a.query || "" } ] };
    case "check_slot":
      return { key, url: "ishara.ai/calendar/week", title: "Calendar", icon: CalendarClock, submit: "", fields: [] };
    default:
      return { key, url: "ishara.ai/agent", title: s.tool, icon: Bot, submit: "Run", fields: [] };
  }
}

function StepLinks({ links, reference }: { links: NonNullable<Step["links"]>; reference?: string | null }) {
  const [copied, setCopied] = useState(false);
  const btn = "inline-flex items-center gap-1.5 rounded-lg border border-white/15 bg-white/[0.05] px-3 py-1.5 text-xs text-zinc-200 hover:bg-white/[0.1]";
  const ics = () => {
    const url = URL.createObjectURL(new Blob([links.ics || ""], { type: "text/calendar" }));
    const el = document.createElement("a");
    el.href = url;
    el.download = `${reference || "event"}.ics`;
    el.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };
  return (
    <div className="flex flex-wrap justify-center gap-2">
      {links.google && <a className={btn} href={links.google} target="_blank" rel="noreferrer" data-gaze data-dwell="1500"><CalendarPlus size={13} /> Add to Google Calendar</a>}
      {links.ics && <button className={btn} onClick={ics} data-gaze data-dwell="1500"><Download size={13} /> Download .ics</button>}
      {links.meet && <a className={btn} href={links.meet} target="_blank" rel="noreferrer" data-gaze data-dwell="1500"><Video size={13} /> Join online room</a>}
      {links.meet && <button className={btn} onClick={() => { navigator.clipboard?.writeText(links.meet!); setCopied(true); setTimeout(() => setCopied(false), 1500); }}><Copy size={13} /> {copied ? "Copied" : "Copy link"}</button>}
      {links.swiggy && <a className={btn} href={links.swiggy} target="_blank" rel="noreferrer" data-gaze data-dwell="1500"><ExternalLink size={13} /> Open Swiggy</a>}
      {links.whatsapp && <a className={btn} href={links.whatsapp} target="_blank" rel="noreferrer" data-gaze data-dwell="1500"><MessageCircle size={13} /> WhatsApp</a>}
      {links.email && <a className={btn} href={links.email} data-gaze data-dwell="1500"><Mail size={13} /> Email</a>}
    </div>
  );
}

function Cursor({ x, y, pulse }: { x: number; y: number; pulse: number }) {
  return (
    <div className="pointer-events-none absolute left-0 top-0 z-30 transition-[transform] duration-[340ms] ease-[cubic-bezier(.4,.1,.2,1)]" style={{ transform: `translate(${x}px, ${y}px)` }}>
      <svg width="26" height="26" viewBox="0 0 24 24" className="drop-shadow-[0_2px_6px_rgba(0,0,0,0.6)]">
        <path d="M3 2 L3 18 L7.2 14.2 L10 20.5 L12.6 19.4 L9.9 13.2 L15.6 13.2 Z" fill="#f0a64a" stroke="#1a1208" strokeWidth="1.2" strokeLinejoin="round" />
      </svg>
      <span className="absolute left-5 top-5 whitespace-nowrap rounded-md bg-amber-400 px-1.5 py-0.5 text-[10px] font-semibold text-black">Ishara agent</span>
      {pulse > 0 && <span key={pulse} className="absolute -left-2 -top-2 h-6 w-6 animate-ping rounded-full bg-amber-400/70" />}
    </div>
  );
}

function StepPage({ step, spec, running, onCaption }: { step: Step; spec: Spec; running: boolean; onCaption: (c: string) => void }) {
  const box = useRef<HTMLDivElement>(null);
  const els = useRef<Record<string, HTMLElement | null>>({});
  const full = useMemo(() => Object.fromEntries(spec.fields.map((f) => [f.id, f.value])), [spec]);
  const [typed, setTyped] = useState<Record<string, string>>(running ? {} : full);
  const [focus, setFocus] = useState<string | null>(null);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [pulse, setPulse] = useState(0);
  const [hit, setHit] = useState(false);

  useEffect(() => {
    if (!running) {
      setTyped(full);
      setFocus(null);
      setCursor(null);
      return;
    }
    let dead = false;
    setTyped({});
    setHit(false);
    const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
    const moveTo = (id: string, fx = 0.15) => {
      const el = els.current[id];
      const b = box.current;
      if (!el || !b) return;
      const r = el.getBoundingClientRect();
      const br = b.getBoundingClientRect();
      setCursor({ x: r.left - br.left + Math.min(r.width * fx, 180), y: r.top - br.top + r.height * 0.55 });
    };
    (async () => {
      setCursor({ x: 24, y: 24 });
      await sleep(250);
      for (const f of spec.fields) {
        if (dead) return;
        onCaption(`Clicking “${f.label}”`);
        moveTo(f.id);
        await sleep(330);
        setFocus(f.id);
        onCaption(`Typing ${f.label.toLowerCase()}…`);
        const per = Math.max(8, Math.min(24, 620 / Math.max(1, f.value.length)));
        for (let i = 1; i <= f.value.length; i++) {
          if (dead) return;
          setTyped((t) => ({ ...t, [f.id]: f.value.slice(0, i) }));
          await sleep(per);
        }
        await sleep(110);
      }
      if (dead) return;
      setFocus(null);
      if (spec.submit) {
        onCaption(`Clicking “${spec.submit}”`);
        moveTo("__submit", 0.5);
        await sleep(420);
        if (dead) return;
        setPulse((p) => p + 1);
        setHit(true);
        onCaption("Saving…");
      } else {
        // calendar check: point at the slot
        moveTo("__slot", 0.5);
        onCaption("Looking at the time slot…");
        await sleep(500);
        setPulse((p) => p + 1);
      }
    })();
    return () => {
      dead = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, spec.key]);

  const done = step.status === "done";
  const failed = step.status === "failed";
  const a = step.args;
  const targetDay = a.date ? (new Date(`${a.date}T00:00`).getDay() + 6) % 7 : -1; // Monday = 0
  const targetHour = a.time ? Number(a.time.split(":")[0]) : -1;
  const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"];

  return (
    <div ref={box} className="relative h-full overflow-hidden p-5">
      <div className="mx-auto max-w-xl">
        <div className="mb-4 flex items-center gap-3">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-white/[0.06]"><spec.icon size={18} className="text-amber-300" /></span>
          <h3 className="text-lg font-semibold text-zinc-100">{spec.title}</h3>
        </div>

        {step.tool === "check_slot" ? (
          <div className="rounded-xl border border-white/10 bg-black/30 p-3">
            <div className="grid grid-cols-[44px_repeat(5,1fr)] gap-1 text-center text-[11px] text-zinc-500">
              <span />
              {DAYS.map((d) => <span key={d}>{d}</span>)}
              {[10, 11, 12, 13, 14, 15, 16].map((h) => (
                <div key={h} className="contents">
                  <span className="pr-1 text-right leading-7">{h > 12 ? h - 12 : h}{h >= 12 ? "pm" : "am"}</span>
                  {DAYS.map((_, di) => {
                    const t = di === targetDay && h === targetHour;
                    const closed = h === 13;
                    return (
                      <div key={di} ref={(el) => { if (t) els.current.__slot = el; }}
                        className={`h-7 rounded ${t ? (failed ? "border border-red-400 bg-red-400/25" : done ? "border border-emerald-400 bg-emerald-400/25" : "border border-amber-400 bg-amber-400/20") : closed ? "bg-white/[0.02]" : "bg-white/[0.05]"}`} />
                    );
                  })}
                </div>
              ))}
            </div>
            <p className="mt-2 text-xs text-zinc-500">{human(a.date, a.time)} with {a.with}</p>
            {(done || failed) && <p className={`mt-1 text-sm ${failed ? "text-red-300" : "text-emerald-300"}`}>{step.result}</p>}
          </div>
        ) : (
          <div className="space-y-3">
            {spec.fields.map((f) => {
              const v = typed[f.id] ?? "";
              const on = focus === f.id;
              return (
                <div key={f.id}>
                  <p className="mb-1 text-[11px] uppercase tracking-wider text-zinc-500">{f.label}</p>
                  <div ref={(el) => { els.current[f.id] = el; }}
                    className={`min-h-[40px] rounded-lg border px-3 py-2 text-[15px] text-zinc-100 transition ${on ? "border-amber-400/80 bg-amber-400/[0.06] shadow-[0_0_0_3px_rgba(240,166,74,0.15)]" : "border-white/12 bg-black/30"}`}>
                    {v}
                    {on && <span className="ml-px inline-block h-[1.05em] w-[2px] translate-y-[2px] animate-pulse bg-amber-300" />}
                  </div>
                </div>
              );
            })}
            {step.tool === "lookup_info" && done && (
              <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3 text-sm text-zinc-200">
                {step.result}
                {step.data?.sources?.length ? <p className="mt-1 text-xs text-zinc-500">Source: {step.data.sources.map((x) => x.title).join(", ")}</p> : null}
              </div>
            )}
            {spec.submit && (
              <div className="pt-1">
                <div ref={(el) => { els.current.__submit = el; }}
                  className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition ${hit || done ? "bg-amber-400 text-black" : "bg-white/10 text-zinc-200"}`}>
                  {running && hit ? <Loader2 size={14} className="animate-spin" /> : done ? <Check size={14} /> : <Plus size={14} />} {spec.submit}
                </div>
              </div>
            )}
          </div>
        )}

        {done && step.reference && (
          <div className="mt-5 rounded-xl border border-emerald-400/30 bg-emerald-400/[0.07] p-4 text-center">
            <p className="flex items-center justify-center gap-2 text-emerald-200"><Check size={16} /> Saved and verified <span className="rounded-md border border-emerald-400/40 px-2 font-mono text-xs">{step.reference}</span></p>
            {step.links && Object.keys(step.links).length > 0 && <div className="mt-3"><StepLinks links={step.links} reference={step.reference} /></div>}
          </div>
        )}
        {failed && step.tool !== "check_slot" && <p className="mt-4 flex items-center gap-2 text-sm text-red-300"><X size={14} /> {step.result}</p>}
      </div>
      {cursor && running && <Cursor x={cursor.x} y={cursor.y} pulse={pulse} />}
    </div>
  );
}

export default function AgentWorkspace({ run }: { run: Run | null }) {
  const [caption, setCaption] = useState("");
  const [picked, setPicked] = useState<number | null>(null);
  const steps = run?.steps ?? [];
  const runningIdx = steps.findIndex((s) => s.status === "running");
  const lastDone = steps.map((s, i) => (["done", "failed"].includes(s.status) ? i : -1)).filter((i) => i >= 0).pop() ?? -1;
  const auto = runningIdx >= 0 ? runningIdx : lastDone >= 0 ? lastDone : steps.length ? 0 : -1;
  useEffect(() => setPicked(null), [run?.id, runningIdx]);
  const idx = picked ?? auto;
  const step = idx >= 0 ? steps[idx] : null;
  const spec = step ? specFor(step, idx) : null;
  const working = !!run && (run.status === "planning" || run.status === "executing");
  const busyNow = !!step && step.status === "running";

  // caption of what the agent is doing when no cursor animation is active
  const status = !run ? "Ready" : run.status === "planning" && steps.length === 0 ? "Reading your request…" : run.status === "awaiting_consent" ? "Waiting for your OK" : run.status === "done" ? "Done" : working ? caption || "Working…" : run.status === "blocked" ? "Blocked by a check" : run.status === "needs_info" ? "Needs more details" : run.status === "answered" ? "Answered" : run.status;

  return (
    <div className="overflow-hidden rounded-2xl border border-white/12 bg-[#0c0d10] shadow-[0_20px_60px_-20px_rgba(0,0,0,0.8)]">
      {/* browser chrome */}
      <div className="flex items-center gap-2 border-b border-white/[0.07] bg-[#15161a] px-3 pt-2">
        <div className="mr-2 flex gap-1.5 pb-2"><span className="h-3 w-3 rounded-full bg-[#ff5f57]" /><span className="h-3 w-3 rounded-full bg-[#febc2e]" /><span className="h-3 w-3 rounded-full bg-[#28c840]" /></div>
        <div className="flex min-w-0 flex-1 items-end gap-1 overflow-x-auto">
          {steps.length === 0 && <div className="rounded-t-lg bg-[#0c0d10] px-3 py-2 text-xs text-zinc-300">New tab</div>}
          {steps.map((s, i) => {
            const Icon = TOOL_ICON[s.tool] ?? Bot;
            return (
              <button key={i} onClick={() => setPicked(i)} className={`flex max-w-[170px] items-center gap-1.5 truncate rounded-t-lg px-3 py-2 text-xs ${i === idx ? "bg-[#0c0d10] text-zinc-100" : "text-zinc-500 hover:text-zinc-300"}`}>
                {s.status === "running" ? <Loader2 size={12} className="animate-spin text-amber-300" /> : s.status === "done" ? <Check size={12} className="text-emerald-400" /> : <Icon size={12} />}
                <span className="truncate">{specFor(s, i).title}</span>
              </button>
            );
          })}
        </div>
        <span className="mb-2 flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 text-[11px] text-zinc-400">
          <span className={`h-1.5 w-1.5 rounded-full ${working ? "animate-pulse bg-amber-400" : "bg-zinc-600"}`} /> {working ? "Agent is working" : "Idle"}
        </span>
      </div>
      <div className="flex items-center gap-2 border-b border-white/[0.07] bg-[#0f1013] px-3 py-1.5">
        <div className="flex flex-1 items-center gap-2 rounded-full bg-black/40 px-3 py-1 text-xs text-zinc-400">
          <Lock size={11} className="text-emerald-400/80" /> <span className="truncate">{spec ? spec.url : "ishara.ai/agent"}</span>
        </div>
      </div>

      {/* page */}
      <div className="relative h-[560px] bg-[radial-gradient(ellipse_at_top,#17181d,#0a0b0d)]">
        {step && spec ? (
          <StepPage key={`${run?.id}-${spec.key}-${busyNow}`} step={step} spec={spec} running={busyNow} onCaption={setCaption} />
        ) : (
          <div className="grid h-full place-items-center p-6 text-center">
            <div className="space-y-3">
              <Bot size={34} className="mx-auto text-zinc-500" />
              {run && run.status === "planning" ? (
                <>
                  <p className="text-lg text-zinc-200">“{run.goal}”</p>
                  <p className="flex items-center justify-center gap-2 text-sm text-amber-200"><Loader2 size={14} className="animate-spin" /> Reading your request and planning the steps…</p>
                </>
              ) : (
                <>
                  <p className="chrome-text text-2xl font-semibold">Ishara agent workspace</p>
                  <p className="text-sm text-zinc-500">Ask for something and watch the agent open pages, fill them in and save, step by step.</p>
                </>
              )}
            </div>
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 border-t border-white/[0.07] bg-[#0f1013] px-4 py-2 text-xs text-zinc-400">
        <Bot size={13} className="text-amber-300" />
        <span className="font-medium text-zinc-300">Agent</span> <span className="text-zinc-600">›</span> <span>{status}</span>
        {steps.length > 0 && <span className="ml-auto font-mono text-zinc-600">step {Math.max(1, idx + 1)} of {steps.length}</span>}
      </div>
    </div>
  );
}
