import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { useLocation } from "react-router-dom";
import type { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";
import { Brain, Circle, Loader2, Plus, Square } from "lucide-react";
import CameraStage from "../components/CameraStage";
import { api, type Sentence } from "../lib/api";
import { hasHands, toFrame } from "../lib/keypoints";
import { useHolistic } from "../lib/useHolistic";

type Phase = "idle" | "countdown" | "recording" | "saving";
type FtResult = {
  status: string;
  message?: string;
  new_takes?: number;
  added_sentences?: string[];
  skipped_too_few?: Record<string, number>;
  held_out?: number;
  held_out_accuracy_before?: number | null;
  held_out_accuracy_after?: number | null;
  seconds?: number;
  at?: string;
};
type TrainState = { running: boolean; log: string; returncode: number | null; mode?: string; result?: FtResult | null };
type ModelInfo = {
  ready: boolean;
  labels: { id: string; text: string }[];
  val_acc: number | null;
  test_acc: number | null;
  test_top3: number | null;
  n_test: number | null;
  new_takes?: { total: number; old_sentences: Record<string, number>; new_sentences: Record<string, number>; pending: Record<string, number> };
  last_finetune?: FtResult | null;
};

// Same segmentation rule as the live recogniser: start at first hands-up frame, stop after hands are down.
const END_GAP = 10;

export default function Collect() {
  const [sentences, setSentences] = useState<Sentence[]>([]);
  const loc = useLocation() as { state?: { sentenceId?: string } };
  const [selected, setSelected] = useState<string>(loc.state?.sentenceId ?? "");
  const [newText, setNewText] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [count, setCount] = useState(0);
  const [frames, setFrames] = useState(0);
  const [msg, setMsg] = useState("");
  const [train, setTrain] = useState<TrainState | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);

  const [guided, setGuided] = useState<{ ids: string[]; i: number; take: number } | null>(null);
  const [showAll, setShowAll] = useState(false);
  const guidedRef = useRef(guided);
  guidedRef.current = guided;
  const recordRef = useRef<() => void>(() => undefined);
  const TAKES_EACH = 3;

  const buf = useRef<number[][]>([]);
  const gap = useRef(0);
  const started = useRef(false);
  const phaseRef = useRef<Phase>("idle");
  phaseRef.current = phase;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;

  const load = useCallback(async () => {
    const s = await api<Sentence[]>("/api/sentences");
    setSentences(s);
    setSelected((cur) => cur || s[0]?.id || "");
  }, []);

  const loadModel = useCallback(() => api<ModelInfo>("/api/model").then(setModel), []);

  useEffect(() => {
    load();
    loadModel();
  }, [load, loadModel]);

  const save = useCallback(async () => {
    const data = buf.current.slice(0, buf.current.length - gap.current);
    buf.current = [];
    gap.current = 0;
    started.current = false;
    if (data.length < 8) {
      setPhase("idle");
      setMsg("Too short — no signing detected. Try again.");
      return;
    }
    setPhase("saving");
    try {
      const r = await api<{ samples: number }>("/api/samples", { body: { sentence_id: selectedRef.current, frames: data } });
      setSentences((s) => s.map((x) => (x.id === selectedRef.current ? { ...x, samples: r.samples } : x)));
      setMsg(`Saved ${data.length} frames.`);
      const g = guidedRef.current;
      if (g) {
        // guided mode: next take of this sentence, or the next sentence, and keep recording
        const takeNo = g.take + 1;
        const done = takeNo >= TAKES_EACH;
        const ni = done ? g.i + 1 : g.i;
        if (ni >= g.ids.length) {
          setGuided(null);
          setMsg("All sentences recorded. Press Fine-tune below to teach the model your signing.");
        } else {
          selectedRef.current = g.ids[ni];
          setSelected(g.ids[ni]);
          setGuided({ ...g, i: ni, take: done ? 0 : takeNo });
          window.setTimeout(() => recordRef.current(), 1800);
        }
      }
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setPhase("idle");
    }
  }, []);

  const onResult = useCallback(
    (r: HolisticLandmarkerResult) => {
      if (phaseRef.current !== "recording") return;
      const hands = hasHands(r);
      if (!started.current && !hands) return;
      started.current = true;
      buf.current.push(toFrame(r));
      gap.current = hands ? 0 : gap.current + 1;
      setFrames(buf.current.length);
      if (gap.current >= END_GAP) {
        phaseRef.current = "saving";
        save();
      }
    },
    [save],
  );

  const cam = useHolistic(onResult);

  function record() {
    if (!selectedRef.current || phaseRef.current !== "idle") return;
    setMsg("");
    buf.current = [];
    gap.current = 0;
    started.current = false;
    setFrames(0);
    setPhase("countdown");
    let n = 3;
    setCount(n);
    const t = setInterval(() => {
      n -= 1;
      setCount(n);
      if (n === 0) {
        clearInterval(t);
        setPhase("recording");
      }
    }, 700);
  }

  recordRef.current = record;

  function startGuided() {
    const ids = sentences.filter((x) => x.in_model && x.active !== false).map((x) => x.id);
    if (!ids.length || cam.status !== "running") {
      setMsg("Start the camera first.");
      return;
    }
    selectedRef.current = ids[0];
    setSelected(ids[0]);
    setGuided({ ids, i: 0, take: 0 });
    window.setTimeout(() => recordRef.current(), 600);
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" || (e.target as HTMLElement).tagName === "INPUT") return;
      e.preventDefault();
      if (phaseRef.current === "idle" && cam.status === "running") record();
      else if (phaseRef.current === "recording") save();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  async function addSentence(e: FormEvent) {
    e.preventDefault();
    if (!newText.trim()) return;
    const s = await api<Sentence>("/api/sentences", { body: { text: newText } });
    setNewText("");
    await load();
    setSelected(s.id);
  }

  async function startTraining(mode: "finetune" | "full") {
    if (mode === "full" && !confirm("Retrain from scratch on every recording? This takes several minutes and replaces the model.")) return;
    await api(`/api/model/train?mode=${mode}`, { method: "POST" });
    setTrain({ running: true, log: "", returncode: null, mode });
  }

  useEffect(() => {
    if (!train?.running) return;
    const t = setInterval(async () => {
      const s = await api<TrainState>("/api/model/train");
      setTrain(s);
      if (!s.running) loadModel();
    }, 2000);
    return () => clearInterval(t);
  }, [train?.running, loadModel]);

  const current = sentences.find((s) => s.id === selected);

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      <CameraStage
        {...cam}
        onStart={cam.start}
        onStop={cam.stop}
        overlay={
          <>
            {phase === "countdown" && (
              <div className="absolute inset-0 grid place-items-center bg-black/40">
                <span className="chrome-text text-8xl font-semibold">{count}</span>
              </div>
            )}
            {phase === "recording" && (
              <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-full border border-red-400/30 bg-black/70 px-3 py-1 font-mono text-xs text-red-300">
                <Circle size={8} className="animate-pulse fill-red-400" /> REC · {frames} frames ·{" "}
                {started.current ? "lower hands to finish" : "waiting for hands"}
              </div>
            )}
          </>
        }
        footer={
          <div className="space-y-3">
            {guided && (
              <div className="flex items-center gap-3 rounded-xl border border-amber-400/40 bg-amber-400/[0.07] px-4 py-2 text-sm text-amber-100">
                <span>
                  Sentence {guided.i + 1} of {guided.ids.length}, take {guided.take + 1} of {TAKES_EACH}. Sign it, then lower your hands.
                </span>
                <button className="btn btn-ghost ml-auto !px-2 !py-1 text-xs" onClick={() => setGuided(null)}>
                  Stop
                </button>
              </div>
            )}
            <p className="label">Now recording</p>
            <p className="text-lg text-zinc-100">{current?.text ?? "Pick a sentence →"}</p>
            {current?.has_video && (
              <div className="flex items-start gap-3">
                <video key={current.id} src={`/api/sentences/${current.id}/video`} muted loop autoPlay playsInline className="h-28 rounded-lg border border-white/10 bg-black" />
                <p className="text-xs text-zinc-500">How a real signer signs it. Copy this, so your takes match what the reader learned.</p>
              </div>
            )}
            <div className="flex flex-wrap items-center gap-3">
              {phase === "recording" ? (
                <button className="btn btn-ghost" onClick={save}>
                  <Square size={14} /> Stop &amp; save
                </button>
              ) : (
                <button className="btn btn-chrome" onClick={record} disabled={cam.status !== "running" || phase !== "idle"}>
                  {phase === "saving" ? <Loader2 size={14} className="animate-spin" /> : <Circle size={14} />} Record take
                </button>
              )}
              <span className="text-xs text-zinc-600">Space = record / stop. Sign the whole sentence, then lower your hands.</span>
            </div>
            {msg && <p className="text-sm text-zinc-400">{msg}</p>}
          </div>
        }
      />

      <div className="space-y-6">
        <div className="glass p-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-medium text-zinc-100">Sentences the reader knows</h2>
            <button className="text-xs text-zinc-500 hover:text-zinc-300" onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Show only the 20" : "Show all"}
            </button>
          </div>
          <button className="btn btn-chrome mb-3 w-full" onClick={startGuided} disabled={!!guided || cam.status !== "running"}>
            <Circle size={14} /> Teach my signing: record all 20, {TAKES_EACH} takes each (about 6 min)
          </button>
          <div className="max-h-[340px] space-y-1 overflow-y-auto pr-1">
            {sentences.filter((x) => showAll || (x.in_model && x.active !== false)).map((s) => (
              <button
                key={s.id}
                onClick={() => setSelected(s.id)}
                className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm transition ${
                  s.id === selected ? "bg-white/[0.08] text-zinc-100" : "text-zinc-400 hover:bg-white/[0.03]"
                }`}
              >
                <span>{s.text}</span>
                <span className={`font-mono text-xs ${s.samples >= 20 ? "text-emerald-400/80" : "text-zinc-600"}`}>
                  {s.samples}
                </span>
              </button>
            ))}
          </div>
          <form onSubmit={addSentence} className="mt-4 flex gap-2">
            <input className="input" placeholder="Add a sentence…" value={newText} onChange={(e) => setNewText(e.target.value)} />
            <button className="btn btn-ghost px-3">
              <Plus size={16} />
            </button>
          </form>
        </div>

        <div className="glass space-y-4 p-5">
          <div className="flex items-center gap-2">
            <Brain size={16} className="text-zinc-400" />
            <h2 className="font-medium text-zinc-100">Transformer model</h2>
            <span className="ml-auto font-mono text-xs text-zinc-500">
              {model?.ready
                ? `${model.labels.length} sentences · cross-validated ${
                    model.test_acc != null ? (model.test_acc * 100).toFixed(1) + "%" : "n/a"
                  }${model.test_top3 != null ? ` · top-3 ${(model.test_top3 * 100).toFixed(1)}%` : ""}`
                : "not trained"}
            </span>
          </div>
          {(() => {
            const nt = model?.new_takes;
            const total = nt?.total ?? 0;
            const sentences = Object.keys(nt?.old_sentences ?? {}).length + Object.keys(nt?.new_sentences ?? {}).length;
            const pending = Object.keys(nt?.pending ?? {}).length;
            return (
              <>
                <button className="btn btn-chrome w-full" onClick={() => startTraining("finetune")} disabled={train?.running || total === 0}>
                  {train?.running && train.mode === "finetune" ? <Loader2 size={14} className="animate-spin" /> : <Brain size={14} />}
                  {train?.running && train.mode === "finetune"
                    ? "Fine-tuning…"
                    : total > 0
                      ? `Fine-tune on ${total} new recording${total === 1 ? "" : "s"} (${sentences} sentence${sentences === 1 ? "" : "s"}), about 30 s`
                      : "Model is up to date. Record something new to fine-tune."}
                </button>
                {pending > 0 && (
                  <p className="text-xs text-amber-300/80">
                    {pending} new sentence{pending === 1 ? " needs" : "s need"} at least 8 takes before {pending === 1 ? "it is" : "they are"} added.
                  </p>
                )}
                <button className="w-full text-xs text-zinc-500 hover:text-zinc-300" onClick={() => startTraining("full")} disabled={train?.running}>
                  {train?.running && train.mode === "full" ? "Retraining from scratch…" : "Retrain from scratch on everything (slow, re-measures accuracy)"}
                </button>
              </>
            );
          })()}
          {(() => {
            const r = (train && !train.running ? train.result : null) ?? model?.last_finetune;
            if (!r || r.status !== "ok") return r?.message ? <p className="text-xs text-zinc-500">{r.message}</p> : null;
            const pc = (v?: number | null) => (v == null ? "n/a" : `${(v * 100).toFixed(0)}%`);
            return (
              <div className="rounded-lg border border-emerald-400/30 bg-emerald-400/[0.06] p-3 text-sm text-emerald-100">
                <p className="font-medium">Last fine-tune: learned {r.new_takes} new recordings in {r.seconds}s</p>
                {r.added_sentences && r.added_sentences.length > 0 && <p className="text-xs text-emerald-200/80">Added: {r.added_sentences.join(", ")}</p>}
                {r.held_out ? <p className="text-xs text-emerald-200/80">On {r.held_out} takes it never trained on: {pc(r.held_out_accuracy_before)} before, {pc(r.held_out_accuracy_after)} after.</p> : null}
              </div>
            );
          })()}
          {train?.log && (
            <pre className="max-h-48 overflow-auto rounded-lg bg-black/50 p-3 font-mono text-[11px] text-zinc-400">{train.log}</pre>
          )}
        </div>
      </div>
    </div>
  );
}
