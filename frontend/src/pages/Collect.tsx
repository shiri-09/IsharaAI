import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import type { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";
import { Brain, Circle, Loader2, Plus, Square } from "lucide-react";
import CameraStage from "../components/CameraStage";
import { api, type Sentence } from "../lib/api";
import { hasHands, toFrame } from "../lib/keypoints";
import { useHolistic } from "../lib/useHolistic";

type Phase = "idle" | "countdown" | "recording" | "saving";
type TrainState = { running: boolean; log: string; returncode: number | null };
type ModelInfo = { ready: boolean; labels: { id: string; text: string }[]; val_acc: number | null; n_train: number | null };

// Same segmentation rule as the live recogniser: start at first hands-up frame, stop after hands are down.
const END_GAP = 10;

export default function Collect() {
  const [sentences, setSentences] = useState<Sentence[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [newText, setNewText] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [count, setCount] = useState(0);
  const [frames, setFrames] = useState(0);
  const [msg, setMsg] = useState("");
  const [train, setTrain] = useState<TrainState | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);

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
    if (!selected || phase !== "idle") return;
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

  async function startTraining() {
    await api("/api/model/train", { method: "POST" });
    setTrain({ running: true, log: "", returncode: null });
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
  const ready = sentences.filter((s) => s.samples > 0).length;

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
            <p className="label">Now recording</p>
            <p className="text-lg text-zinc-100">{current?.text ?? "Pick a sentence →"}</p>
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
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-medium text-zinc-100">Sentences</h2>
            <span className="font-mono text-xs text-zinc-500">aim for 20+ takes each</span>
          </div>
          <div className="max-h-[340px] space-y-1 overflow-y-auto pr-1">
            {sentences.map((s) => (
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
                ? `${model.labels.length} classes · val ${model.val_acc != null ? (model.val_acc * 100).toFixed(1) + "%" : "n/a"}`
                : "not trained"}
            </span>
          </div>
          <button className="btn btn-chrome w-full" onClick={startTraining} disabled={train?.running || ready < 2}>
            {train?.running ? <Loader2 size={14} className="animate-spin" /> : <Brain size={14} />}
            {train?.running ? "Training…" : `Train on ${ready} recorded sentences`}
          </button>
          {train?.log && (
            <pre className="max-h-48 overflow-auto rounded-lg bg-black/50 p-3 font-mono text-[11px] text-zinc-400">{train.log}</pre>
          )}
        </div>
      </div>
    </div>
  );
}
