import { useCallback, useEffect, useRef, useState } from "react";
import { Brain, Check, CheckCircle2, Globe, Hand, Loader2, MousePointerClick, Square, Trash2, X, XCircle } from "lucide-react";
import { api, tokenStore } from "../lib/api";
import { speak } from "./MessageBubble";

export type BAsk = { id: string; kind: "choice" | "text" | "confirm" | "continue"; question: string; options: string[]; field: string; sensitive: boolean };
type BEvent = { i: number; t: string; kind: string; text: string };
export type BRun = {
  id: string;
  goal: string;
  status: "starting" | "thinking" | "acting" | "waiting_user" | "done" | "failed" | "stopped";
  events: BEvent[];
  ask: BAsk | null;
  result: string;
  has_shot: boolean;
  shot_at: number;
};
type Mem = { key: string; label: string; value: string; uses: number };

const LIVE = ["starting", "thinking", "acting", "waiting_user"];
const KIND_STYLE: Record<string, string> = {
  action: "text-amber-200",
  thought: "text-zinc-500",
  ask: "text-sky-300",
  answer: "text-zinc-200",
  memory: "text-emerald-300",
  done: "text-emerald-300",
  error: "text-red-300",
  info: "text-zinc-400",
};
const KIND_LABEL: Record<string, string> = { action: "Doing", thought: "Thinking", ask: "Asking you", answer: "You said", memory: "Memory", done: "Done", error: "Problem", info: "Note" };

/** A sensible first page for a goal when nobody typed one (the demo sites ship with the app). */
export function guessUrl(goal: string): string {
  const g = goal.toLowerCase();
  if (/(meeting|professor|appointment)/.test(g)) return "/demo/portal";
  if (/(lunch|food|meal|canteen|order|eat|dosa|biryani)/.test(g)) return "/demo/canteen";
  return "";
}

export default function BrowserAgent({
  launch,
  aac,
  onAsk,
  registerAnswer,
  onDone,
}: {
  launch: { goal: string; url?: string; n: number } | null;
  aac?: boolean;
  onAsk?: (a: BAsk | null) => void;
  registerAnswer?: (fn: ((text: string) => void) | null) => void;
  onDone?: () => void;
}) {
  const [run, setRun] = useState<BRun | null>(null);
  const [error, setError] = useState("");
  const [text, setText] = useState("");
  const [mem, setMem] = useState<Mem[]>([]);
  const [tick, setTick] = useState(0);
  const spoken = useRef("");
  const logRef = useRef<HTMLDivElement>(null);
  const runRef = useRef<BRun | null>(null);
  runRef.current = run;

  const loadMem = useCallback(() => api<Mem[]>("/api/browser/memory").then(setMem).catch(() => undefined), []);
  useEffect(() => {
    loadMem();
  }, [loadMem]);

  // start a run when the page above asks for one
  const lastN = useRef(0);
  useEffect(() => {
    if (!launch || launch.n === lastN.current) return;
    lastN.current = launch.n;
    setError("");
    setRun(null);
    api<BRun>("/api/browser/runs", { body: { goal: launch.goal, start_url: launch.url ? new URL(launch.url, location.origin).href : "" } })
      .then(setRun)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [launch]);

  // follow it live
  const id = run?.id;
  const live = !!run && LIVE.includes(run.status);
  useEffect(() => {
    if (!id || !live) return;
    let stop = false;
    const loop = async () => {
      while (!stop) {
        await new Promise((r) => setTimeout(r, 450));
        try {
          const s = await api<BRun>(`/api/browser/runs/${id}`);
          if (stop) return;
          setRun(s);
          setTick((n) => n + 1);
          if (!LIVE.includes(s.status)) {
            loadMem();
            onDone?.();
            return;
          }
        } catch (e) {
          if (!stop) setError(e instanceof Error ? e.message : String(e));
          return;
        }
      }
    };
    loop();
    return () => {
      stop = true;
    };
  }, [id, live, loadMem, onDone]);

  const ask = run?.ask ?? null;
  const answer = useCallback(
    async (a: string) => {
      const r = runRef.current;
      if (!r?.ask) return;
      const askId = r.ask.id;
      setRun({ ...r, ask: null });
      setText("");
      try {
        await api(`/api/browser/runs/${r.id}/answer`, { body: { ask_id: askId, answer: a } });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [],
  );

  useEffect(() => {
    onAsk?.(ask);
  }, [ask, onAsk]);
  useEffect(() => {
    registerAnswer?.(answer);
    return () => registerAnswer?.(null);
  }, [answer, registerAnswer]);

  // say each question aloud once
  useEffect(() => {
    if (ask && ask.id !== spoken.current) {
      spoken.current = ask.id;
      speak(ask.question, "en").catch(() => undefined);
    }
  }, [ask]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [run?.events.length]);

  async function stop() {
    if (run) await api(`/api/browser/runs/${run.id}/stop`, { body: {} }).catch(() => undefined);
  }
  async function forget(key: string) {
    await api(`/api/browser/memory/${encodeURIComponent(key)}`, { method: "DELETE" }).catch(() => undefined);
    loadMem();
  }

  const token = tokenStore.get() ?? "";
  const shotUrl = run?.has_shot ? `/api/browser/runs/${run.id}/screenshot?token=${encodeURIComponent(token)}&t=${Math.round(run.shot_at * 1000)}` : "";
  const BTN = aac ? "px-6 py-5 text-2xl" : "px-4 py-2.5 text-sm";

  return (
    <div className="flex flex-col gap-4">
      {/* the live browser */}
      <div className="order-2 overflow-hidden rounded-2xl border border-white/10 bg-black">
        <div className="flex items-center gap-2 border-b border-white/10 bg-white/[0.03] px-3 py-2">
          <span className="h-3 w-3 rounded-full bg-red-400/70" />
          <span className="h-3 w-3 rounded-full bg-amber-300/70" />
          <span className="h-3 w-3 rounded-full bg-emerald-400/70" />
          <div className="ml-2 flex flex-1 items-center gap-2 rounded-lg bg-black/50 px-3 py-1 text-xs text-zinc-400">
            <Globe size={12} /> {run ? (live ? "Ishara agent is using this browser" : run.status === "done" ? "Finished" : run.status) : "Live browser"}
          </div>
          {run && live && (
            <span className="flex items-center gap-1.5 text-xs text-zinc-500">
              <Loader2 size={12} className="animate-spin" /> {run.status === "waiting_user" ? "waiting for you" : run.status === "acting" ? "working" : "thinking"}
            </span>
          )}
          {run && live && (
            <button className="btn btn-ghost !px-2 !py-1 text-xs" data-gaze data-dwell="1500" data-no onClick={stop}>
              <Square size={12} /> Stop
            </button>
          )}
        </div>
        <div className="relative aspect-[16/10] w-full bg-[#08090b]">
          {shotUrl ? (
            <img key={tick} src={shotUrl} alt="What the agent sees" className="h-full w-full object-contain" />
          ) : (
            <div className="grid h-full place-items-center p-6 text-center text-sm text-zinc-500">
              <div className="space-y-2">
                <MousePointerClick className="mx-auto text-amber-300" />
                <p>{run ? "Opening the browser…" : "Tell the agent what to do. A real Chrome window opens on this computer and you watch it work here."}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* a question for the person */}
      {ask && (
        <div className="order-1 rounded-2xl border border-amber-400/40 bg-amber-400/[0.06] p-5">
          <p className={`${aac ? "text-3xl" : "text-lg"} font-medium text-zinc-100`}>{ask.question}</p>
          {ask.kind === "choice" && (
            <div className={`mt-4 grid gap-2 ${aac ? "grid-cols-1" : "sm:grid-cols-2"}`}>
              {ask.options.map((o) => (
                <button key={o} data-gaze data-dwell="1500" onClick={() => answer(o)} className={`rounded-xl border border-white/20 bg-white/[0.06] text-left text-zinc-100 hover:bg-white/[0.12] ${BTN}`}>
                  {o}
                </button>
              ))}
            </div>
          )}
          {ask.kind === "confirm" && (
            <div className="mt-4 grid grid-cols-2 gap-3">
              <button data-gaze data-yes data-dwell="1600" onClick={() => answer("yes")} className={`flex items-center justify-center gap-2 rounded-xl border border-emerald-400/50 bg-emerald-400/15 font-semibold text-emerald-100 ${BTN}`}>
                <Check size={aac ? 30 : 18} /> Yes, go ahead
              </button>
              <button data-gaze data-no data-dwell="1600" onClick={() => answer("no")} className={`flex items-center justify-center gap-2 rounded-xl border border-red-400/40 bg-red-400/10 font-semibold text-red-100 ${BTN}`}>
                <X size={aac ? 30 : 18} /> No
              </button>
            </div>
          )}
          {ask.kind === "continue" && (
            <div className="mt-4">
              <button data-gaze data-yes data-dwell="1600" onClick={() => answer("continue")} className={`flex items-center gap-2 rounded-xl border border-emerald-400/50 bg-emerald-400/15 font-semibold text-emerald-100 ${BTN}`}>
                <CheckCircle2 size={aac ? 30 : 18} /> I did it, continue
              </button>
            </div>
          )}
          {ask.kind === "text" && !aac && (
            <form
              className="mt-4 flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                answer(text);
              }}
            >
              <input autoFocus className="input flex-1" value={text} onChange={(e) => setText(e.target.value)} placeholder="Type your answer" />
              <button className="btn btn-chrome px-4">Send</button>
              <button type="button" className="btn btn-ghost px-3" onClick={() => answer("")}>Skip</button>
            </form>
          )}
          {ask.kind === "text" && aac && <p className="mt-3 text-lg text-amber-200">Type your answer with the keyboard below, then look at “Answer”.</p>}
          {ask.kind === "choice" && !aac && (
            <form
              className="mt-3 flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (text.trim()) answer(text);
              }}
            >
              <input className="input flex-1" value={text} onChange={(e) => setText(e.target.value)} placeholder="Or type something else" />
              <button className="btn btn-ghost px-3" disabled={!text.trim()}>Send</button>
            </form>
          )}
        </div>
      )}

      {run && !live && run.result && (
        <div className={`order-1 flex items-start gap-3 rounded-2xl border p-4 ${run.status === "done" ? "border-emerald-400/40 bg-emerald-400/[0.07]" : "border-red-400/40 bg-red-400/[0.07]"}`}>
          {run.status === "done" ? <CheckCircle2 className="mt-0.5 text-emerald-300" /> : <XCircle className="mt-0.5 text-red-300" />}
          <p className={`${aac ? "text-2xl" : "text-base"} text-zinc-100`}>{run.result}</p>
        </div>
      )}
      {error && <p className="order-1 text-sm text-red-400/90">{error}</p>}

      <div className={`order-3 grid gap-4 ${aac ? "" : "lg:grid-cols-[1.4fr_1fr]"}`}>
        {/* what it is doing */}
        <div className="glass p-4">
          <p className="label pb-2">What the agent did</p>
          <div ref={logRef} className="max-h-64 space-y-1.5 overflow-y-auto pr-1 font-mono text-[12px] leading-snug">
            {(run?.events ?? []).filter((e) => !(aac && e.kind === "thought")).map((e) => (
              <div key={e.i} className="flex gap-2">
                <span className="shrink-0 text-zinc-600">{e.t}</span>
                <span className={`w-20 shrink-0 ${KIND_STYLE[e.kind] ?? "text-zinc-400"}`}>{KIND_LABEL[e.kind] ?? e.kind}</span>
                <span className="text-zinc-300">{e.text}</span>
              </div>
            ))}
            {!run && <p className="text-zinc-600">Nothing yet.</p>}
          </div>
        </div>

        {/* what it remembers */}
        <div className="glass p-4">
          <p className="label flex items-center gap-2 pb-2"><Brain size={13} /> What I remember about you</p>
          {mem.length === 0 && <p className="text-sm text-zinc-600">Nothing yet. The first time a site asks for your name, phone or address, I ask once and remember it for next time. Passwords, card numbers and OTPs are never stored.</p>}
          <div className="space-y-1.5">
            {mem.map((m) => (
              <div key={m.key} className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.02] px-3 py-2">
                <div className="min-w-0 flex-1">
                  <p className="text-[11px] uppercase tracking-wider text-zinc-500">{m.key}</p>
                  <p className="truncate text-sm text-zinc-200">{m.value}</p>
                </div>
                <span className="font-mono text-[10px] text-zinc-600">used {m.uses}x</span>
                <button className="text-zinc-500 hover:text-red-300" title="Forget this" onClick={() => forget(m.key)}>
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
          <p className="mt-3 flex items-center gap-1.5 text-[11px] text-zinc-600"><Hand size={11} /> Signed answers work for the set sentences; for anything else use the buttons or type.</p>
        </div>
      </div>
    </div>
  );
}
