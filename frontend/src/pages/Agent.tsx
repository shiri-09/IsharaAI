import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  AlertTriangle,
  BellRing,
  Bot,
  CalendarCheck,
  CalendarClock,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Hand,
  Languages,
  Loader2,
  Mail,
  Mic,
  MicOff,
  Search,
  Send,
  ShieldCheck,
  SkipForward,
  X,
  XCircle,
  UtensilsCrossed,
  type LucideIcon,
} from "lucide-react";
import CameraStage from "../components/CameraStage";
import { speak } from "../components/MessageBubble";
import { api, type Prediction } from "../lib/api";
import { usePrefs } from "../lib/prefs";
import { useSpeech } from "../lib/speech";
import { useSignReader } from "../lib/useSignReader";

type Step = {
  tool: string;
  args: Record<string, string>;
  why: string;
  status: "pending" | "done" | "failed" | "needs_consent" | "skipped";
  result: string;
  description: string;
  needs_consent: boolean;
  reference?: string | null;
  verified?: boolean;
  data?: { suggest?: { date: string; time: string } | null; sources?: { title: string; source: string }[] };
};
export type Run = {
  id: number;
  conversation_id: number;
  goal: string;
  source: string;
  status: "planning" | "needs_info" | "blocked" | "awaiting_consent" | "done" | "cancelled" | "failed" | "answered";
  steps: Step[];
  trace: { t: string; node: string; detail: string }[];
  question: string;
  summary: string;
  provider: string;
  created_at: string;
  translations?: Record<string, string>;
};

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
const TOOL_NAME: Record<string, string> = {
  lookup_info: "Look up campus info",
  check_slot: "Check the time",
  book_appointment: "Book appointment",
  set_reminder: "Set reminder",
  request_interpreter: "Request interpreter",
  save_message: "Save message",
  translate: "Translate",
  order_meal: "Order meal (demo)",
};

export const STATUS: Record<Run["status"], { text: string; cls: string }> = {
  planning: { text: "Planning", cls: "border-white/15 text-zinc-300" },
  needs_info: { text: "Needs details", cls: "border-sky-300/40 bg-sky-300/10 text-sky-200" },
  blocked: { text: "Blocked by a check", cls: "border-red-400/40 bg-red-400/10 text-red-200" },
  awaiting_consent: { text: "Waiting for your OK", cls: "border-amber-400/50 bg-amber-400/10 text-amber-200" },
  done: { text: "Done and verified", cls: "border-emerald-400/50 bg-emerald-400/10 text-emerald-200" },
  answered: { text: "Answered", cls: "border-emerald-400/50 bg-emerald-400/10 text-emerald-200" },
  cancelled: { text: "Cancelled", cls: "border-white/15 text-zinc-400" },
  failed: { text: "Failed", cls: "border-red-400/40 text-red-300" },
};

const EXAMPLES = [
  "Book a meeting with Professor Rao next Monday at 11 AM about my project",
  "I need a sign language interpreter for the seminar on Wednesday at 2 PM in the main auditorium",
  "Remind me to submit my assignment on Friday at 5 PM",
  "Tell Priya that I will be late to class",
  "Order lunch from Swiggy",
  "Where is the accessible washroom?",
  "Meet my professor",
];

function StepIcon({ s }: { s: Step }) {
  if (s.status === "done") return <CheckCircle2 size={18} className="text-emerald-400" />;
  if (s.status === "failed") return <XCircle size={18} className="text-red-400" />;
  if (s.status === "needs_consent") return <ShieldCheck size={18} className="text-amber-300" />;
  if (s.status === "skipped") return <SkipForward size={18} className="text-zinc-500" />;
  return <CircleDashed size={18} className="text-zinc-500" />;
}

/** Shows one run: plan, step results, the question / consent, outcome and the trace. Used here and in History. */
export function RunView({ run, busy, onReply, onConfirm, onCancel }: {
  run: Run;
  busy?: boolean;
  onReply?: (text: string) => void;
  onConfirm?: () => void;
  onCancel?: () => void;
}) {
  const [reply, setReply] = useState("");
  const [showTrace, setShowTrace] = useState(false);
  const st = STATUS[run.status];
  const suggest = run.steps.find((s) => s.data?.suggest)?.data?.suggest;
  const open = run.status === "needs_info" || run.status === "blocked";
  function send(e: FormEvent) {
    e.preventDefault();
    if (reply.trim() && onReply) {
      onReply(reply.trim());
      setReply("");
    }
  }
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start gap-3">
        <p className="flex-1 text-lg text-zinc-100">“{run.goal.replace(/ \[updated: .*?\]/g, "")}”</p>
        <span className={`rounded-full border px-3 py-1 text-xs ${st.cls}`}>{st.text}</span>
      </div>

      <ol className="space-y-2">
        {run.steps.map((s, i) => {
          const Icon = TOOL_ICON[s.tool] ?? Bot;
          return (
            <li key={i} className={`rounded-xl border p-3 ${s.status === "needs_consent" ? "border-amber-400/30 bg-amber-400/[0.04]" : "border-white/10 bg-white/[0.02]"}`}>
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-zinc-600">{i + 1}</span>
                <Icon size={16} className="text-zinc-400" />
                <span className="text-sm font-medium text-zinc-200">{TOOL_NAME[s.tool] ?? s.tool}</span>
                {s.needs_consent && <span className="rounded border border-amber-400/30 px-1.5 text-[10px] uppercase tracking-wider text-amber-300/80">needs consent</span>}
                <span className="ml-auto">
                  <StepIcon s={s} />
                </span>
              </div>
              <p className="mt-1.5 pl-7 text-sm text-zinc-400">{s.description}</p>
              {s.result && <p className={`mt-1 pl-7 text-sm ${s.status === "failed" ? "text-red-300" : "text-zinc-300"}`}>{s.result}</p>}
              {s.reference && (
                <p className="mt-1.5 pl-7">
                  <span className="rounded-md border border-emerald-400/40 bg-emerald-400/10 px-2 py-0.5 font-mono text-xs text-emerald-200">{s.reference}</span>
                  {s.verified && <span className="ml-2 text-xs text-emerald-300/80">verified in storage</span>}
                </p>
              )}
              {s.data?.sources && s.data.sources.length > 0 && (
                <p className="mt-1 pl-7 text-xs text-zinc-500">Source: {s.data.sources.map((x) => x.title).join(", ")}</p>
              )}
            </li>
          );
        })}
        {run.steps.length === 0 && <li className="text-sm text-zinc-500">No steps yet.</li>}
      </ol>

      {(run.question && (open || run.status === "awaiting_consent")) && (
        <div className={`rounded-xl border p-4 ${run.status === "blocked" ? "border-red-400/30 bg-red-400/[0.05]" : run.status === "awaiting_consent" ? "border-amber-400/40 bg-amber-400/[0.06]" : "border-sky-300/30 bg-sky-300/[0.05]"}`}>
          <div className="flex items-start gap-2">
            {run.status === "blocked" ? <AlertTriangle size={18} className="mt-0.5 text-red-300" /> : <Bot size={18} className="mt-0.5 text-zinc-300" />}
            <p className="text-zinc-100">{run.question}</p>
          </div>
          {run.status === "awaiting_consent" && onConfirm && (
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <button className="btn btn-chrome py-3 text-base" onClick={onConfirm} disabled={busy}>
                {busy ? <Loader2 size={16} className="animate-spin" /> : <Check size={16} />} Yes, do it
              </button>
              <button className="btn btn-ghost py-3 text-base" onClick={onCancel} disabled={busy}>
                <X size={16} /> No, cancel
              </button>
            </div>
          )}
          {open && onReply && (
            <form onSubmit={send} className="mt-3 flex flex-wrap gap-2">
              {suggest && (
                <button type="button" className="btn btn-chrome px-3 py-2 text-sm" disabled={busy} onClick={() => onReply("yes")}>
                  <Check size={14} /> Use the suggested time
                </button>
              )}
              <input className="input min-w-[200px] flex-1" placeholder={run.status === "blocked" ? "Or type another day and time" : "Type the details, e.g. Tuesday at 11 AM"} value={reply} onChange={(e) => setReply(e.target.value)} />
              <button className="btn btn-ghost px-3" disabled={busy || !reply.trim()}>
                {busy ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
              </button>
              {onCancel && (
                <button type="button" className="btn btn-ghost px-3 text-sm" onClick={onCancel} disabled={busy}>
                  Cancel
                </button>
              )}
            </form>
          )}
        </div>
      )}

      {run.summary && (
        <div className={`rounded-xl border p-4 ${run.status === "cancelled" ? "border-white/10" : "border-emerald-400/30 bg-emerald-400/[0.05]"}`}>
          <p className="text-zinc-100">{run.summary}</p>
          {run.translations && Object.values(run.translations).map((t, i) => <p key={i} className="mt-1 text-sm text-zinc-400">{t}</p>)}
        </div>
      )}

      <button className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300" onClick={() => setShowTrace((v) => !v)}>
        <ChevronDown size={14} className={showTrace ? "rotate-180 transition" : "transition"} /> Orchestration trace ({run.trace.length} events, planner: {run.provider || "n/a"})
      </button>
      {showTrace && (
        <ol className="space-y-1 border-l border-white/10 pl-4">
          {run.trace.map((e, i) => (
            <li key={i} className="text-xs">
              <span className="font-mono text-zinc-600">{e.t.slice(11, 19)}</span> <span className="font-mono text-amber-300/80">{e.node}</span>{" "}
              <span className="text-zinc-400">{e.detail}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export default function Agent() {
  const { prefs } = usePrefs();
  const voice = prefs?.voice || "hi";
  const [goal, setGoal] = useState("");
  const [run, setRun] = useState<Run | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [input, setInput] = useState<"type" | "sign">("sign");
  const [unsure, setUnsure] = useState<Prediction | null>(null);
  const reader = useSignReader((p) => {
    if (p.accepted) {
      setUnsure(null);
      setGoal(p.text);
      startRef.current(p.text, "sign");
    } else setUnsure(p);
  });
  const startRef = useRef<(t: string, s?: string) => void>(() => undefined);
  const speech = useSpeech((t) => {
    setGoal(t);
    start(t, "voice");
  });

  const loadRuns = () => api<Run[]>("/api/agent/runs").then(setRuns).catch(() => undefined);
  useEffect(() => {
    loadRuns();
  }, []);

  function announce(r: Run) {
    if (prefs?.auto_speak === false) return;
    const text = r.status === "done" || r.status === "answered" || r.status === "cancelled" ? r.summary : r.question;
    const local = r.translations?.[voice];
    if (local && voice !== "en") speak(local, voice).catch(() => undefined);
    else if (text) speak(text, "en").catch(() => undefined);
  }

  async function call(path: string, body: unknown) {
    setBusy(true);
    setError("");
    try {
      const r = await api<Run>(path, { body });
      setRun(r);
      announce(r);
      loadRuns();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function start(text: string, source = "text") {
    if (!text.trim()) return;
    setUnsure(null);
    call("/api/agent/runs", { goal: text.trim(), source, languages: [voice].filter((l) => l !== "en") });
  }

  startRef.current = start;

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
      <div className="space-y-5">
        <div className="flex flex-wrap items-center gap-3">
          <Bot className="text-amber-300" />
          <div>
            <h1 className="text-xl font-semibold text-zinc-100">Agentic orchestration</h1>
            <p className="text-sm text-zinc-500">
              Say what you need. The agent plans the steps, checks them, asks before it changes anything, then does it and verifies the result.
            </p>
          </div>
        </div>

        <div className="flex gap-1 rounded-xl border border-white/10 bg-black/40 p-1">
          {([["sign", "Sign it"], ["type", "Type or speak"]] as const).map(([k, label]) => (
            <button key={k} onClick={() => { if (k === "type") reader.cam.stop(); setInput(k); }}
              className={`flex-1 rounded-lg px-3 py-2 text-sm transition ${input === k ? "bg-white/10 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"}`}>
              {label}
            </button>
          ))}
        </div>

        {input === "sign" && (
          <div className="space-y-2">
            <CameraStage {...reader.cam} onStart={reader.cam.start} onStop={reader.cam.stop}
              overlay={<div className="absolute inset-x-3 bottom-3 rounded-xl border border-white/10 bg-black/60 px-4 py-2 text-sm text-zinc-300 backdrop-blur">
                {reader.signing ? "Reading your signs… lower your hands to finish" : "Raise your hands and sign what you need. The agent starts when you finish."}</div>} />
            {unsure && (
              <div className="glass space-y-2 p-3">
                <p className="text-sm text-zinc-400">Not sure. Which did you mean?</p>
                <div className="flex flex-wrap gap-2">
                  {unsure.top.map((t) => (
                    <button key={t.id} className="btn btn-ghost px-3 py-1.5 text-xs" onClick={() => { setGoal(t.text); start(t.text, "sign"); }}>
                      {t.text} <span className="font-mono text-zinc-600">{Math.round(t.p * 100)}%</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            <p className="px-1 text-xs text-zinc-500">The sign reader knows about 100 set sentences (see the Sign library). For anything else, such as a name or a date, switch to Type or speak.</p>
            {reader.error && <p className="text-xs text-red-400/90">{reader.error}</p>}
          </div>
        )}

        <div className="glass space-y-3 p-5">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              start(goal);
            }}
            className="flex flex-wrap gap-2"
          >
            <input className="input min-w-[240px] flex-1" placeholder="What do you need done?" value={speech.interim || goal} onChange={(e) => setGoal(e.target.value)} />
            <button type="button" className={`btn px-3 ${speech.listening ? "btn-chrome" : "btn-ghost"}`} onClick={() => speech.toggle("en")} title="Speak your request">
              {speech.listening ? <MicOff size={16} /> : <Mic size={16} />}
            </button>
            <button className="btn btn-chrome px-4" disabled={busy || !goal.trim()}>
              {busy ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />} Run
            </button>
          </form>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((e) => (
              <button key={e} className="rounded-lg border border-white/10 px-2.5 py-1 text-left text-xs text-zinc-400 hover:text-zinc-200" onClick={() => { setGoal(e); start(e); }} disabled={busy}>
                {e}
              </button>
            ))}
          </div>
          {(error || speech.error) && <p className="text-sm text-red-400/90">{error || speech.error}</p>}
        </div>

        <div className="glass min-h-[260px] p-5">
          {busy && !run && (
            <p className="flex items-center gap-2 text-zinc-400">
              <Loader2 size={16} className="animate-spin" /> Planning with the local model…
            </p>
          )}
          {run ? (
            <RunView
              run={run}
              busy={busy}
              onReply={(t) => call(`/api/agent/runs/${run.id}/reply`, { text: t })}
              onConfirm={() => call(`/api/agent/runs/${run.id}/confirm`, { languages: [voice].filter((l) => l !== "en") })}
              onCancel={() => call(`/api/agent/runs/${run.id}/cancel`, {})}
            />
          ) : (
            !busy && (
              <div className="grid min-h-[220px] place-items-center text-center text-sm text-zinc-500">
                <div className="space-y-2">
                  <p className="chrome-text text-2xl font-semibold">Plan → check → your OK → do → verify</p>
                  <p>Nothing that changes anything happens without your confirmation.</p>
                </div>
              </div>
            )
          )}
        </div>
      </div>

      <aside className="glass h-fit p-3">
        <p className="label px-2 pb-2 pt-1">Recent runs</p>
        {runs.length === 0 && <p className="px-2 py-3 text-sm text-zinc-600">None yet.</p>}
        <div className="max-h-[70vh] space-y-1 overflow-y-auto">
          {runs.map((r) => (
            <button key={r.id} onClick={() => setRun(r)} className={`w-full rounded-lg px-3 py-2 text-left transition ${run?.id === r.id ? "bg-white/[0.08]" : "hover:bg-white/[0.03]"}`}>
              <span className="line-clamp-2 text-sm text-zinc-300">{r.goal.replace(/ \[updated: .*?\]/g, "")}</span>
              <span className="mt-1 flex items-center gap-2">
                <span className={`rounded-full border px-2 py-px text-[10px] ${STATUS[r.status].cls}`}>{STATUS[r.status].text}</span>
                <span className="font-mono text-[10px] text-zinc-600">{new Date(r.created_at).toLocaleString()}</span>
              </span>
            </button>
          ))}
        </div>
      </aside>
    </div>
  );
}
