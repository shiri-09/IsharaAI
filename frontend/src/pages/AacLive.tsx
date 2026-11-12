import { useEffect, useRef, useState } from "react";
import { Ear, Loader2, Mic, MicOff, Plus, Radio } from "lucide-react";
import GazeKeyboard from "../components/GazeKeyboard";
import { speak } from "../components/MessageBubble";
import { api, type Msg } from "../lib/api";
import { usePrefs } from "../lib/prefs";
import { LOCALES, useSpeech } from "../lib/speech";

type Turn = Msg & { english?: string };
const QUICK = ["Hello", "Yes", "No", "Thank you", "Please repeat that", "Please speak slowly", "I need help", "One moment please"];

/**
 * Real-time speech for eye-tracking users: the user types or picks a phrase and it is spoken aloud; the other person's
 * reply is heard by the microphone and shown as large captions. No AI answers in between.
 */
export default function AacLive() {
  const { prefs } = usePrefs();
  const [hearLang, setHearLang] = useState(prefs?.voice && prefs.voice in LOCALES ? prefs.voice : "en");
  const [conv, setConv] = useState<number | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const convRef = useRef<number | null>(null);
  convRef.current = conv;
  const langRef = useRef(hearLang);
  langRef.current = hearLang;
  const listenRef = useRef<() => void>(() => undefined);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  async function turn(who: "signer" | "speaker", t: string, source: string) {
    const r = await api<{ conversation_id: number; message: Msg; english: string }>("/api/live/turn", {
      body: { who, text: t, source, conversation_id: convRef.current, lang: who === "speaker" ? langRef.current : "en", languages: [] },
    });
    setConv(r.conversation_id);
    setTurns((x) => [...x, { ...r.message, english: r.english }]);
  }

  async function speakOut(t: string) {
    if (!t.trim()) return;
    setText("");
    setError("");
    try {
      await turn("signer", t.trim(), "aac");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    await speak(t, "en").catch(() => undefined);
    window.setTimeout(() => listenRef.current(), 350); // now listen for the other person
  }

  const speech = useSpeech(async (heard) => {
    setBusy(true);
    try {
      await turn("speaker", heard.trim(), "voice");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  });
  listenRef.current = () => {
    if (!speech.listening) speech.toggle(langRef.current);
  };

  const last = [...turns].reverse().find((t) => t.role === "speaker");
  return (
    <div className="flex h-[calc(100vh-3.2rem)] min-h-[600px] flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3">
        <Radio className="text-amber-300" />
        <h1 className="text-xl font-semibold text-zinc-100">Real-time speech</h1>
        <span className="text-sm text-zinc-500">Your words are spoken aloud. Then the microphone opens for the other person's reply.</span>
        <button className="btn btn-ghost ml-auto px-3 py-2" data-gaze data-dwell="1500" onClick={() => { setTurns([]); setConv(null); }}>
          <Plus size={16} /> New
        </button>
      </div>

      <div className="glass min-h-[130px] flex-1 overflow-y-auto p-4">
        <p className="label mb-2 flex items-center gap-2">
          <Ear size={14} /> They said
        </p>
        {speech.interim && <p className="text-3xl italic text-zinc-400">{speech.interim}</p>}
        {!speech.interim && last && (
          <>
            <p className="text-4xl leading-tight text-zinc-50">{last.content}</p>
            {last.interpreted && <p className="mt-2 text-2xl text-zinc-400">English: {last.interpreted}</p>}
          </>
        )}
        {!speech.interim && !last && <p className="text-3xl text-zinc-600">{speech.listening ? "Listening…" : "Their words appear here in large letters."}</p>}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button data-gaze data-dwell="1400" onClick={() => speech.toggle(hearLang)} className={`btn px-5 py-3 text-lg ${speech.listening ? "btn-chrome" : "btn-ghost"}`}>
            {speech.listening ? <MicOff size={20} /> : <Mic size={20} />} {speech.listening ? "Listening… stop" : "Listen"}
          </button>
          {Object.entries(LOCALES).map(([k, v]) => (
            <button key={k} data-gaze data-dwell="1200" onClick={() => setHearLang(k)} className={`rounded-full border px-4 py-2 text-base ${hearLang === k ? "border-zinc-300/50 bg-white/10 text-zinc-100" : "border-white/10 text-zinc-500"}`}>
              {v.label}
            </button>
          ))}
          {busy && <Loader2 size={18} className="animate-spin text-zinc-500" />}
        </div>
        {speech.error && <p className="mt-2 text-sm text-red-400/90">{speech.error}</p>}
      </div>

      <div ref={scroller} className="glass max-h-[11vh] space-y-1 overflow-y-auto p-3">
        {turns.length === 0 && <p className="text-lg text-zinc-600">The whole conversation is saved to History.</p>}
        {turns.map((m) => (
          <p key={m.id} className={`text-xl ${m.role === "signer" ? "text-right text-amber-100" : "text-zinc-200"}`}>
            <span className="mr-2 text-xs uppercase tracking-wider text-zinc-600">{m.role === "signer" ? "You" : "They"}</span>
            {m.content}
          </p>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-1.5 md:grid-cols-4">
        {QUICK.slice(0, 4).map((q) => (
          <button key={q} data-gaze data-dwell="1300" onClick={() => speakOut(q)} className="min-h-[clamp(40px,5.2vh,54px)] rounded-xl border border-white/10 bg-white/[0.03] px-3 text-base text-zinc-200">
            {q}
          </button>
        ))}
      </div>
      <div className="glass space-y-1.5 p-3">
        <div className="min-h-[46px] rounded-xl border border-white/15 bg-black/50 px-4 py-1.5 text-2xl text-zinc-100">
          {text || <span className="text-zinc-600">Type what you want to say…</span>}
          <span className="animate-pulse text-amber-300">|</span>
        </div>
        <GazeKeyboard value={text} onChange={setText} submitLabel="Say it" onSubmit={() => speakOut(text)} />
      </div>
      {error && <p className="text-base text-red-400/90">{error}</p>}
    </div>
  );
}
