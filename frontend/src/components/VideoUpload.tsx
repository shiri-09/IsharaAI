import { useEffect, useRef, useState } from "react";
import { HolisticLandmarker } from "@mediapipe/tasks-vision";
import { FileVideo, Loader2, Send, Upload } from "lucide-react";
import { api, type Prediction } from "../lib/api";
import { toFrame } from "../lib/keypoints";
import { drawPoints, getLandmarker } from "../lib/useHolistic";

export type Segment = Prediction & { start: number; end: number };
type Phase = "empty" | "ready" | "reading" | "recognising" | "done" | "error";

const FPS = 25;
const MAX_FRAMES = 3000;

function waitSeek(v: HTMLVideoElement, t: number) {
  return new Promise<void>((res) => {
    const done = () => {
      v.removeEventListener("seeked", done);
      res();
    };
    v.addEventListener("seeked", done);
    v.currentTime = t;
  });
}

function snap(v: HTMLVideoElement, w = 256) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = Math.round((v.videoHeight / v.videoWidth) * w) || 192;
  c.getContext("2d")?.drawImage(v, 0, 0, c.width, c.height);
  return c;
}

function strip(frames: HTMLCanvasElement[]): string | null {
  if (!frames.length) return null;
  const out = document.createElement("canvas");
  out.width = frames.reduce((a, f) => a + f.width, 0);
  out.height = Math.max(...frames.map((f) => f.height));
  const ctx = out.getContext("2d")!;
  let x = 0;
  for (const f of frames) {
    ctx.drawImage(f, x, 0);
    x += f.width;
  }
  return out.toDataURL("image/jpeg", 0.75);
}

type Props = {
  /** Send one recognised sentence (with its top-3 and a filmstrip for the vision model) to the conversation. */
  onSend: (text: string, top: Prediction["top"], image: string | null) => Promise<void>;
  autoSend: boolean;
};

/** Upload a recorded signing video. Keypoints are read in the browser, frame by frame, then recognised by the same model as the live camera. */
export default function VideoUpload({ onSend, autoSend }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [phase, setPhase] = useState<Phase>("empty");
  const [progress, setProgress] = useState(0);
  const [split, setSplit] = useState(false);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [error, setError] = useState("");
  const [sent, setSent] = useState<Record<number, string>>({});
  const strips = useRef<Record<number, string | null>>({});
  const cancel = useRef(false);

  useEffect(() => () => {
    cancel.current = true;
    if (url) URL.revokeObjectURL(url);
  }, [url]);

  function pick(f: File | undefined) {
    if (!f) return;
    if (!f.type.startsWith("video/")) {
      setError("Please choose a video file (mp4, webm, mov).");
      return;
    }
    if (url) URL.revokeObjectURL(url);
    setUrl(URL.createObjectURL(f));
    setName(f.name);
    setSegments([]);
    setSent({});
    setError("");
    setPhase("ready");
  }

  async function run() {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    cancel.current = false;
    setError("");
    setSegments([]);
    setSent({});
    setPhase("reading");
    setProgress(0);
    try {
      const lm = await getLandmarker();
      if (!v.duration || !isFinite(v.duration)) await new Promise((r) => v.addEventListener("loadedmetadata", r, { once: true }));
      v.pause();
      const step = Math.max(1 / FPS, v.duration / MAX_FRAMES);
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      const ctx = c.getContext("2d")!;
      const frames: number[][] = [];
      const shots: HTMLCanvasElement[] = [];
      const every = Math.max(1, Math.round(0.35 / step));
      for (let t = 0, i = 0; t < v.duration; t += step, i++) {
        if (cancel.current) return;
        await waitSeek(v, t);
        const r = lm.detectForVideo(v, performance.now());
        frames.push(toFrame(r));
        if (i % every === 0) shots.push(snap(v));
        ctx.clearRect(0, 0, c.width, c.height);
        drawPoints(ctx, r.poseLandmarks[0], HolisticLandmarker.POSE_CONNECTIONS, c.width, c.height, "rgba(200,206,216,0.35)", 2);
        drawPoints(ctx, r.leftHandLandmarks[0], HolisticLandmarker.HAND_CONNECTIONS, c.width, c.height, "rgba(238,241,245,0.9)", 2.5);
        drawPoints(ctx, r.rightHandLandmarks[0], HolisticLandmarker.HAND_CONNECTIONS, c.width, c.height, "rgba(238,241,245,0.9)", 2.5);
        setProgress(Math.min(1, t / v.duration));
      }
      setPhase("recognising");
      const res = await api<{ frames: number; segments: Segment[] }>("/api/recognize", { body: { frames, split } });
      // filmstrip (start / middle / end) of each sentence, for the multimodal model
      const perShot = frames.length / Math.max(1, shots.length);
      res.segments.forEach((s, k) => {
        const a = Math.floor(s.start / perShot);
        const b = Math.max(a, Math.min(shots.length - 1, Math.floor((s.end - 1) / perShot)));
        strips.current[k] = strip([shots[a], shots[Math.floor((a + b) / 2)], shots[b]].filter(Boolean));
      });
      setSegments(res.segments);
      setPhase("done");
      if (autoSend) {
        for (let k = 0; k < res.segments.length; k++) {
          const s = res.segments[k];
          if (!s.accepted) continue;
          await onSend(s.text, s.top, strips.current[k]);
          setSent((m) => ({ ...m, [k]: s.text }));
        }
      }
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function sendOne(k: number, text: string) {
    await onSend(text, segments[k].top, strips.current[k]);
    setSent((m) => ({ ...m, [k]: text }));
  }

  const busy = phase === "reading" || phase === "recognising";

  return (
    <div className="glass overflow-hidden">
      <div
        className="relative aspect-[4/3] w-full bg-black"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          pick(e.dataTransfer.files?.[0]);
        }}
      >
        {url ? (
          <>
            <video ref={videoRef} src={url} playsInline muted controls={!busy} preload="auto" className="absolute inset-0 h-full w-full object-contain" />
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full object-contain" />
          </>
        ) : (
          <label className="absolute inset-0 grid cursor-pointer place-items-center">
            <div className="flex flex-col items-center gap-4 text-center">
              <div className="grid h-16 w-16 place-items-center rounded-2xl border border-white/10 bg-white/[0.03]">
                <Upload className="text-zinc-400" />
              </div>
              <span className="btn btn-chrome">Choose a video</span>
              <p className="max-w-xs text-xs text-zinc-500">or drop it here. mp4, webm or mov. Signer facing the camera, hands and upper body in view.</p>
            </div>
            <input type="file" accept="video/*" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
          </label>
        )}
        {busy && (
          <div className="absolute inset-x-3 bottom-3 rounded-xl border border-white/10 bg-black/70 px-4 py-3 backdrop-blur">
            <div className="mb-2 flex items-center gap-2 text-sm text-zinc-300">
              <Loader2 size={14} className="animate-spin" />
              {phase === "reading" ? `Reading hands and body… ${Math.round(progress * 100)}%` : "Recognising the sentence…"}
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-white/5">
              <div className="h-full bg-gradient-to-r from-zinc-500 to-zinc-100" style={{ width: `${(phase === "reading" ? progress : 1) * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      <div className="space-y-3 p-5">
        {url && (
          <div className="flex flex-wrap items-center gap-3">
            <FileVideo size={16} className="text-zinc-500" />
            <span className="min-w-0 flex-1 truncate text-sm text-zinc-300">{name}</span>
            <label className="btn btn-ghost cursor-pointer px-3 py-1.5 text-xs">
              Change
              <input type="file" accept="video/*" className="hidden" disabled={busy} onChange={(e) => pick(e.target.files?.[0])} />
            </label>
            <button className="btn btn-chrome px-4 py-1.5 text-sm" onClick={run} disabled={busy}>
              {busy ? <Loader2 size={14} className="animate-spin" /> : null} Recognise
            </button>
          </div>
        )}
        <label className="flex items-center gap-2 text-xs text-zinc-500">
          <input type="checkbox" checked={split} onChange={(e) => setSplit(e.target.checked)} disabled={busy} />
          The video has several sentences (split where the hands go down)
        </label>

        {segments.length > 0 && (
          <div className="space-y-3">
            {segments.map((s, k) => (
              <div key={k} className="space-y-2 rounded-xl border border-white/10 bg-white/[0.02] p-3">
                <div className="flex items-center justify-between">
                  <span className="label">{segments.length > 1 ? `Sentence ${k + 1}` : "Recognised"}</span>
                  <span className="font-mono text-xs text-zinc-500">{(s.confidence * 100).toFixed(0)}%</span>
                </div>
                <p className={`text-lg ${s.accepted ? "text-zinc-100" : "text-zinc-500"}`}>{s.text}</p>
                <div className="h-1 overflow-hidden rounded-full bg-white/5">
                  <div className="h-full bg-gradient-to-r from-zinc-500 to-zinc-100" style={{ width: `${s.confidence * 100}%` }} />
                </div>
                {sent[k] ? (
                  <p className="text-xs text-emerald-300/80">Sent to the conversation: {sent[k]}</p>
                ) : (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {s.top.map((t) => (
                      <button key={t.id} className="btn btn-ghost px-3 py-1.5 text-xs" onClick={() => sendOne(k, t.text)}>
                        <Send size={12} /> {t.text} <span className="font-mono text-zinc-600">{(t.p * 100).toFixed(0)}%</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        {phase === "empty" && <p className="text-sm text-zinc-600">Recognised sentences appear here. The video itself stays in your browser; only the hand and body points are sent for recognition.</p>}
        {error && <p className="text-sm text-red-400/90">{error}</p>}
      </div>
    </div>
  );
}
