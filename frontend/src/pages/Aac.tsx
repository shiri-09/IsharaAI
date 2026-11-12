import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Pause, Play, ScanFace } from "lucide-react";
import ActionCard from "../components/ActionCard";
import { speak } from "../components/MessageBubble";
import { api, LANGS, type AgentAction, type ChatResp, type Msg } from "../lib/api";
import { usePrefs } from "../lib/prefs";
import { useFaceSignals, type FaceSignals } from "../lib/useFaceSignals";

type Tile = { text: string; group: "needs" | "urgent" | "campus" | "consent" };

const PHRASES: Tile[] = [
  { text: "Yes", group: "needs" },
  { text: "No", group: "needs" },
  { text: "Thank you", group: "needs" },
  { text: "I need water", group: "needs" },
  { text: "I need the washroom", group: "needs" },
  { text: "Can you repeat that?", group: "needs" },
  { text: "I am in pain", group: "urgent" },
  { text: "I need help", group: "urgent" },
  { text: "Please call my family", group: "urgent" },
  { text: "I want to meet my professor", group: "campus" },
  { text: "Tomorrow", group: "campus" },
  { text: "At 10 AM", group: "campus" },
  { text: "At 2 PM", group: "campus" },
  { text: "Where is the accessible washroom?", group: "campus" },
  { text: "Where is the library?", group: "campus" },
  { text: "I need a sign language interpreter", group: "campus" },
];
const CONSENT: Tile[] = [
  { text: "Yes, confirm the booking", group: "consent" },
  { text: "No, cancel it", group: "consent" },
];

const BROW_ON = 0.45; // eyebrow raise selects
const BROW_OFF = 0.25;
const JAW_ON = 0.5; // mouth open, held, pauses / resumes
const JAW_HOLD_MS = 700;

function Meter({ label, value, threshold }: { label: string; value: number; threshold: number }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-zinc-500">
        <span>{label}</span>
        <span className="font-mono">{Math.round(value * 100)}</span>
      </div>
      <div className="relative h-2 rounded-full bg-white/[0.06]">
        <div className={`h-full rounded-full ${value >= threshold ? "bg-amber-400" : "bg-zinc-400"}`} style={{ width: `${Math.min(100, value * 100)}%` }} />
        <div className="absolute inset-y-[-3px] w-px bg-white/40" style={{ left: `${threshold * 100}%` }} />
      </div>
    </div>
  );
}

const TILE_STYLE: Record<Tile["group"], string> = {
  needs: "border-white/10 bg-white/[0.03] text-zinc-300",
  urgent: "border-red-400/20 bg-red-400/[0.04] text-zinc-300",
  campus: "border-sky-300/15 bg-sky-300/[0.03] text-zinc-300",
  consent: "border-amber-400/30 bg-amber-400/[0.06] text-zinc-100",
};

export default function Aac() {
  const { prefs, save } = usePrefs();
  const voice = prefs?.voice || "hi";
  const scanMs = prefs?.scan_ms || 1600;
  const [idx, setIdx] = useState(0);
  const [paused, setPaused] = useState(false);
  const [meters, setMeters] = useState({ brow: 0, jaw: 0, face: false });
  const [history, setHistory] = useState<Msg[]>([]);
  const [action, setAction] = useState<AgentAction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const conv = useRef<number | null>(null);

  const consent = action?.status === "awaiting_consent";
  const tiles = consent ? CONSENT : PHRASES;
  const state = useRef({ idx, paused, busy, voice, tiles, action });
  state.current = { idx, paused, busy, voice, tiles, action };
  const gesture = useRef({ browArmed: true, lastSelect: 0, jawSince: 0, jawFired: false });

  const say = async (m: Msg) => {
    const v = state.current.voice;
    const spoken = v === "en" ? m.content : m.translations?.[v];
    if (spoken && prefs?.auto_speak !== false) await speak(spoken, v).catch(() => undefined);
  };

  const select = useCallback(async (i: number) => {
    const s = state.current;
    if (s.busy) return;
    const tile = s.tiles[i];
    if (!tile) return;
    setBusy(true);
    setError("");
    try {
      if (tile.group === "consent" && s.action) {
        const kind = tile.text.startsWith("Yes") ? "confirm" : "cancel";
        speak(tile.text, "en").catch(() => undefined);
        const r = await api<{ action: AgentAction; message: Msg }>(`/api/actions/${s.action.id}/${kind}`, {
          body: { languages: [s.voice].filter((l) => l !== "en") },
        });
        setAction(r.action.status === "cancelled" ? null : r.action);
        setHistory((h) => [...h, r.message]);
        setIdx(0);
        await say(r.message);
        return;
      }
      speak(tile.text, "en").catch(() => undefined);
      const r = await api<ChatResp>("/api/chat", {
        body: {
          text: tile.text,
          conversation_id: conv.current,
          context: "campus",
          source: "aac",
          languages: [s.voice].filter((l) => l !== "en"),
        },
      });
      conv.current = r.conversation_id;
      setHistory((h) => [...h, r.user_message, r.reply]);
      setAction(r.action);
      setIdx(0);
      await say(r.reply);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onFrame = useCallback(
    (f: FaceSignals) => {
      const g = gesture.current;
      const now = performance.now();
      if (f.face && g.browArmed && f.brow > BROW_ON && now - g.lastSelect > 900) {
        g.browArmed = false;
        g.lastSelect = now;
        if (!state.current.paused) select(state.current.idx);
      }
      if (f.brow < BROW_OFF) g.browArmed = true;
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

  useEffect(() => {
    if (paused || busy) return;
    const h = window.setInterval(() => setIdx((i) => (i + 1) % tiles.length), consent ? Math.max(scanMs, 1800) : scanMs);
    return () => window.clearInterval(h);
  }, [paused, busy, tiles.length, scanMs, consent]);

  const lastReply = [...history].reverse().find((m) => m.role === "assistant");

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <ScanFace className="text-amber-300" />
          <h1 className="text-xl font-semibold text-zinc-100">{consent ? "Confirm the booking?" : "AAC board"}</h1>
          <span className="text-sm text-zinc-500">
            {consent ? "Raise your eyebrows on Yes to book it, or on No to cancel. Nothing is booked without your OK." : "Raise your eyebrows to choose the highlighted phrase. Hold your mouth open to pause."}
          </span>
        </div>
        <div className={`grid gap-3 ${consent ? "grid-cols-1 sm:grid-cols-2" : "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4"}`}>
          {tiles.map((t, i) => {
            const on = i === idx && !paused;
            return (
              <button
                key={t.text}
                onClick={() => select(i)}
                className={`relative rounded-2xl border p-4 text-left transition ${consent ? "min-h-[160px] text-2xl" : "min-h-[104px] text-lg"} ${
                  on ? "scale-[1.03] border-amber-400/70 bg-amber-400/15 text-zinc-50 shadow-[0_0_40px_-10px_rgba(240,166,74,0.6)]" : TILE_STYLE[t.group]
                }`}
              >
                {t.text}
                {t.group !== "needs" && t.group !== "consent" && (
                  <span className={`absolute bottom-3 right-3 text-[10px] uppercase tracking-wider ${t.group === "urgent" ? "text-red-300/70" : "text-sky-200/60"}`}>
                    {t.group}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        {action && <ActionCard action={action} busy={busy} highlight={consent ? (idx === 0 ? "confirm" : "cancel") : null} />}
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
              {voice !== "en" && lastReply.translations?.[voice] && <p className="text-zinc-400">{lastReply.translations[voice]}</p>}
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
            <span className={`absolute left-3 top-3 rounded-full px-2.5 py-1 font-mono text-[11px] ${meters.face ? "bg-emerald-500/20 text-emerald-300" : "bg-black/60 text-zinc-500"}`}>
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
