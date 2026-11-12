import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Pause, Play, ScanFace } from "lucide-react";
import { speak } from "../components/MessageBubble";
import { api, LANGS, type Msg } from "../lib/api";
import { usePrefs } from "../lib/prefs";
import { useFaceSignals, type FaceSignals } from "../lib/useFaceSignals";

const PHRASES = [
  "Yes",
  "No",
  "Thank you",
  "I need water",
  "I am hungry",
  "I need the washroom",
  "I am in pain",
  "Please call my family",
  "I need help",
  "I am tired",
  "Can you repeat that?",
  "I am feeling cold",
];
const URGENT = new Set(["I am in pain", "I need help", "Please call my family"]);

const SCAN_MS = 1600;
const BROW_ON = 0.45; // eyebrow raise selects
const BROW_OFF = 0.25;
const JAW_ON = 0.5; // mouth open, held, pauses / resumes
const JAW_HOLD_MS = 700;

type ChatResp = { conversation_id: number; user_message: Msg; reply: Msg };

function Meter({ label, value, threshold }: { label: string; value: number; threshold: number }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-zinc-500">
        <span>{label}</span>
        <span className="font-mono">{Math.round(value * 100)}</span>
      </div>
      <div className="relative h-2 rounded-full bg-white/[0.06]">
        <div
          className={`h-full rounded-full ${value >= threshold ? "bg-amber-400" : "bg-zinc-400"}`}
          style={{ width: `${Math.min(100, value * 100)}%` }}
        />
        <div className="absolute inset-y-[-3px] w-px bg-white/40" style={{ left: `${threshold * 100}%` }} />
      </div>
    </div>
  );
}

export default function Aac() {
  const { prefs, save } = usePrefs();
  const voice = prefs?.voice || "hi";
  const [idx, setIdx] = useState(0);
  const [paused, setPaused] = useState(false);
  const [meters, setMeters] = useState({ brow: 0, jaw: 0, face: false });
  const [history, setHistory] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const conv = useRef<number | null>(null);
  const state = useRef({ idx, paused, busy, voice });
  state.current = { idx, paused, busy, voice };
  const gesture = useRef({ browArmed: true, lastSelect: 0, jawSince: 0, jawFired: false });

  const select = useCallback(async (i: number) => {
    const s = state.current;
    if (s.busy) return;
    const phrase = PHRASES[i];
    setBusy(true);
    setError("");
    speak(phrase, "en").catch(() => undefined);
    try {
      const r = await api<ChatResp>("/api/chat", {
        body: { text: phrase, conversation_id: conv.current, source: "aac", languages: [s.voice].filter((l) => l !== "en") },
      });
      conv.current = r.conversation_id;
      setHistory((h) => [...h, r.user_message, r.reply]);
      const spoken = s.voice === "en" ? r.reply.content : r.reply.translations[s.voice];
      if (spoken) await speak(spoken, s.voice).catch(() => undefined);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  const onFrame = useCallback(
    (f: FaceSignals) => {
      const g = gesture.current;
      const now = performance.now();
      // eyebrow raise (rising edge with hysteresis) selects the highlighted tile
      if (f.face && g.browArmed && f.brow > BROW_ON && now - g.lastSelect > 900) {
        g.browArmed = false;
        g.lastSelect = now;
        if (!state.current.paused) select(state.current.idx);
      }
      if (f.brow < BROW_OFF) g.browArmed = true;
      // mouth held open toggles pause
      if (f.face && f.jaw > JAW_ON) {
        if (!g.jawSince) g.jawSince = now;
        if (!g.jawFired && now - g.jawSince > JAW_HOLD_MS) {
          g.jawFired = true;
          setPaused((p) => !p);
        }
      } else {
        g.jawSince = 0;
        g.jawFired = false;
      }
      setMeters({ brow: f.brow, jaw: f.jaw, face: f.face });
    },
    [select],
  );

  const face = useFaceSignals(onFrame);

  useEffect(() => {
    face.start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // scanning highlight
  useEffect(() => {
    if (paused || busy) return;
    const h = window.setInterval(() => setIdx((i) => (i + 1) % PHRASES.length), SCAN_MS);
    return () => window.clearInterval(h);
  }, [paused, busy]);

  const lastReply = [...history].reverse().find((m) => m.role === "assistant");

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <ScanFace className="text-amber-300" />
          <h1 className="text-xl font-semibold text-zinc-100">AAC board</h1>
          <span className="text-sm text-zinc-500">Raise your eyebrows to choose the highlighted phrase. Hold your mouth open to pause.</span>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {PHRASES.map((p, i) => {
            const on = i === idx && !paused;
            return (
              <button
                key={p}
                onClick={() => select(i)}
                className={`relative min-h-[110px] rounded-2xl border p-4 text-left text-lg transition ${
                  on
                    ? "scale-[1.03] border-amber-400/70 bg-amber-400/15 text-zinc-50 shadow-[0_0_40px_-10px_rgba(240,166,74,0.6)]"
                    : URGENT.has(p)
                      ? "border-red-400/20 bg-red-400/[0.04] text-zinc-300"
                      : "border-white/10 bg-white/[0.03] text-zinc-300"
                }`}
              >
                {p}
                {URGENT.has(p) && <span className="absolute bottom-3 right-3 text-[10px] uppercase tracking-wider text-red-300/70">urgent</span>}
              </button>
            );
          })}
        </div>
        <div className="glass min-h-[120px] p-5">
          <p className="label mb-2">Reply</p>
          {busy && (
            <p className="flex items-center gap-2 text-zinc-400">
              <Loader2 size={16} className="animate-spin" /> Speaking and waiting for a reply…
            </p>
          )}
          {!busy && lastReply && (
            <div className="space-y-1">
              <p className="text-lg text-zinc-100">{lastReply.content}</p>
              {voice !== "en" && lastReply.translations[voice] && <p className="text-zinc-400">{lastReply.translations[voice]}</p>}
            </div>
          )}
          {!busy && !lastReply && <p className="text-zinc-600">Replies appear here and are spoken aloud.</p>}
          {error && <p className="mt-2 text-sm text-red-400/90">{error}</p>}
        </div>
      </div>

      <aside className="space-y-4">
        <div className="glass overflow-hidden">
          <div className="relative aspect-[4/3] bg-black">
            <video ref={face.videoRef} playsInline muted className="h-full w-full -scale-x-100 object-cover" />
            {face.status !== "running" && (
              <div className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-zinc-500">
                {face.status === "loading" ? <Loader2 className="animate-spin" /> : face.status === "error" ? face.error : "Starting camera…"}
              </div>
            )}
            <span
              className={`absolute left-3 top-3 rounded-full px-2.5 py-1 font-mono text-[11px] ${
                meters.face ? "bg-emerald-500/20 text-emerald-300" : "bg-black/60 text-zinc-500"
              }`}
            >
              {meters.face ? "FACE TRACKED" : "NO FACE"}
            </span>
          </div>
          <div className="space-y-3 p-4">
            <Meter label="Eyebrow raise (select)" value={meters.brow} threshold={BROW_ON} />
            <Meter label="Mouth open (pause)" value={meters.jaw} threshold={JAW_ON} />
            <button className="btn btn-ghost w-full" onClick={() => setPaused((p) => !p)}>
              {paused ? <Play size={16} /> : <Pause size={16} />} {paused ? "Resume scanning" : "Pause scanning"}
            </button>
          </div>
        </div>
        <div className="glass space-y-2 p-4">
          <p className="label">Reply voice</p>
          <div className="flex flex-wrap gap-2">
            {[["en", "English"], ...Object.entries(LANGS)].map(([k, v]) => (
              <button
                key={k}
                onClick={() => save({ voice: k })}
                className={`rounded-full border px-3 py-1 text-sm ${voice === k ? "border-zinc-300/40 bg-white/10 text-zinc-100" : "border-white/10 text-zinc-500"}`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
