import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Ear, Hand, Loader2, Mic, MicOff, Plus, Radio, Send, Volume2 } from "lucide-react";
import AvatarStage from "../components/AvatarStage";
import CameraStage from "../components/CameraStage";
import { speak } from "../components/MessageBubble";
import TakeFeedback from "../components/TakeFeedback";
import { api, type Msg, type Prediction } from "../lib/api";
import { buildSteps, type Segment } from "../lib/avatar";
import { usePrefs } from "../lib/prefs";
import { LOCALES, useSpeech } from "../lib/speech";
import { useSignReader } from "../lib/useSignReader";

type TurnResp = { conversation_id: number; message: Msg; english: string };

/**
 * Real-time speech mode: a live two-way interpreter, no AI replies.
 * Signer -> sign reader -> caption + spoken aloud in the hearing person's language.
 * Hearing person -> speech -> caption + the avatar signs it.
 */
export default function Live() {
  const { prefs } = usePrefs();
  const [hearingLang, setHearingLang] = useState(prefs?.voice && prefs.voice in LOCALES ? prefs.voice : "en");
  const [conv, setConv] = useState<number | null>(null);
  const [turns, setTurns] = useState<(Msg & { english?: string })[]>([]);
  const [pending, setPending] = useState<Prediction | null>(null);
  const [typed, setTyped] = useState("");
  const [signTyped, setSignTyped] = useState("");
  const [segments, setSegments] = useState<Segment[] | null>(null);
  const [playKey, setPlayKey] = useState(0);
  const [busy, setBusy] = useState(false);
  const [autoMic, setAutoMic] = useState(true);
  const autoMicRef = useRef(true);
  autoMicRef.current = autoMic;
  const listenRef = useRef<(lang: string) => void>(() => undefined);
  const [error, setError] = useState("");
  const convRef = useRef<number | null>(null);
  convRef.current = conv;
  const langRef = useRef(hearingLang);
  langRef.current = hearingLang;
  const scroller = useRef<HTMLDivElement>(null);
  const speedK = prefs?.avatar_speed === "slow" ? 1.6 : prefs?.avatar_speed === "fast" ? 0.7 : 1;
  const steps = useMemo(() => (segments ? buildSteps(segments, speedK) : null), [segments, speedK]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [turns]);

  async function turn(who: "signer" | "speaker", text: string, source: string, extra: Record<string, unknown> = {}) {
    setError("");
    const lang = langRef.current;
    const r = await api<TurnResp>("/api/live/turn", {
      body: {
        who,
        text,
        source,
        conversation_id: convRef.current,
        lang: who === "speaker" ? lang : "en",
        languages: who === "signer" && lang !== "en" ? [lang] : [],
        ...extra,
      },
    });
    setConv(r.conversation_id);
    setTurns((t) => [...t, { ...r.message, english: r.english }]);
    return r;
  }

  // ---- signer side
  async function fromSigner(text: string, source: string, top: Prediction["top"] = []) {
    setPending(null);
    try {
      const r = await turn("signer", text, source, { candidates: top.map((c) => ({ text: c.text, p: c.p })) });
      const lang = langRef.current;
      const spoken = lang === "en" ? text : r.message.translations?.[lang] || text;
      // read the signer's sentence aloud, then open the hearing person's microphone for their reply
      await speak(spoken, lang === "en" || !r.message.translations?.[lang] ? "en" : lang).catch(() => undefined);
      if (autoMicRef.current) window.setTimeout(() => listenRef.current(langRef.current), 400);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }
  const [lastPred, setLastPred] = useState<Prediction | null>(null);
  const reader = useSignReader((p) => {
    setLastPred(p);
    if (p.accepted) fromSigner(p.text, "sign", p.top);
    else setPending(p);
  });

  // ---- hearing side
  async function fromSpeaker(text: string, source: string) {
    if (!text.trim()) return;
    setBusy(true);
    try {
      const r = await turn("speaker", text.trim(), source);
      const plan = await api<{ segments: Segment[] }>("/api/avatar/plan", { body: { text: r.english } });
      setSegments(plan.segments);
      setPlayKey((k) => k + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  const speech = useSpeech((t) => fromSpeaker(t, "voice"));
  listenRef.current = (lang) => {
    if (!speech.listening) speech.toggle(lang);
  };

  function newSession() {
    setConv(null);
    setTurns([]);
    setSegments(null);
    setPending(null);
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <Radio className="text-amber-300" />
        <div className="flex-1">
          <h1 className="text-xl font-semibold text-zinc-100">Real-time speech</h1>
          <p className="text-sm text-zinc-500">A live interpreter between a signer and a hearing person. Signs are spoken aloud; speech is signed by the avatar. Everything is saved to History.</p>
        </div>
        <button className="btn btn-ghost px-3 text-sm" onClick={newSession}>
          <Plus size={14} /> New session
        </button>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* signer */}
        <div className="space-y-3">
          <p className="label flex items-center gap-2">
            <Hand size={14} /> Signer
          </p>
          <CameraStage
            {...reader.cam}
            onStart={reader.cam.start}
            onStop={reader.cam.stop}
            overlay={
              <div className="absolute inset-x-3 bottom-3 rounded-xl border border-white/10 bg-black/60 px-4 py-2 text-sm text-zinc-300 backdrop-blur">
                {reader.signing ? "Reading your signs… lower your hands to finish" : "Raise your hands to sign a sentence"}
              </div>
            }
            footer={
              <div className="space-y-2">
                {pending && (
                  <div className="space-y-2">
                    <p className="text-sm text-zinc-400">Not sure. Which did you sign?</p>
                    <div className="flex flex-wrap gap-2">
                      {pending.top.map((t) => (
                        <button key={t.id} className="btn btn-ghost px-3 py-1.5 text-xs" onClick={() => fromSigner(t.text, "sign", pending.top)}>
                          {t.text} <span className="font-mono text-zinc-600">{Math.round(t.p * 100)}%</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
                <TakeFeedback pred={lastPred} />
                <form
                  onSubmit={(e: FormEvent) => {
                    e.preventDefault();
                    if (signTyped.trim()) fromSigner(signTyped.trim(), "text");
                    setSignTyped("");
                  }}
                  className="flex gap-2"
                >
                  <input className="input" placeholder="…or type for the signer" value={signTyped} onChange={(e) => setSignTyped(e.target.value)} />
                  <button className="btn btn-ghost px-3" disabled={!signTyped.trim()}>
                    <Volume2 size={16} />
                  </button>
                </form>
                {reader.error && <p className="text-xs text-red-400/90">{reader.error}</p>}
              </div>
            }
          />
        </div>

        {/* hearing person */}
        <div className="space-y-3">
          <p className="label flex items-center gap-2">
            <Ear size={14} /> Hearing person
          </p>
          <div className="glass flex flex-col items-center gap-3 p-4">
            <div className="flex w-full flex-wrap items-center gap-2">
              {Object.entries(LOCALES).map(([k, v]) => (
                <button
                  key={k}
                  onClick={() => setHearingLang(k)}
                  className={`rounded-full border px-3 py-1 text-xs ${hearingLang === k ? "border-zinc-300/40 bg-white/10 text-zinc-100" : "border-white/10 text-zinc-500"}`}
                >
                  {v.label}
                </button>
              ))}
              <label className="ml-auto flex items-center gap-1.5 text-xs text-zinc-500">
                <input type="checkbox" checked={autoMic} onChange={(e) => setAutoMic(e.target.checked)} />
                Open mic after the signer
              </label>
              {busy && <Loader2 size={14} className="animate-spin text-zinc-500" />}
            </div>
            <div className="w-full max-w-[380px]">
              <AvatarStage steps={steps} playKey={playKey} />
            </div>
            <button className={`btn w-full py-3 ${speech.listening ? "btn-chrome" : "btn-ghost"}`} onClick={() => speech.toggle(hearingLang)}>
              {speech.listening ? <MicOff size={18} /> : <Mic size={18} />} {speech.listening ? "Listening… tap to stop" : `Tap and speak in ${LOCALES[hearingLang].label}`}
            </button>
            {speech.interim && <p className="w-full text-sm italic text-zinc-400">{speech.interim}</p>}
            <form
              onSubmit={(e: FormEvent) => {
                e.preventDefault();
                fromSpeaker(typed, "text");
                setTyped("");
              }}
              className="flex w-full gap-2"
            >
              <input className="input" placeholder="…or type what the hearing person says" value={typed} onChange={(e) => setTyped(e.target.value)} />
              <button className="btn btn-ghost px-3" disabled={busy || !typed.trim()}>
                <Send size={16} />
              </button>
            </form>
            {speech.error && <p className="w-full text-xs text-red-400/90">{speech.error}</p>}
          </div>
        </div>
      </div>

      <div className="glass p-5">
        <p className="label mb-3">Live transcript</p>
        <div ref={scroller} className="max-h-[320px] space-y-3 overflow-y-auto">
          {turns.length === 0 && <p className="text-sm text-zinc-600">The conversation will appear here as captions.</p>}
          {turns.map((m) => {
            const signer = m.role === "signer";
            const local = Object.values(m.translations || {})[0];
            return (
              <div key={m.id} className={`flex ${signer ? "justify-start" : "justify-end"}`}>
                <div className={`max-w-[80%] rounded-2xl border px-4 py-2.5 ${signer ? "border-amber-400/25 bg-amber-400/[0.05]" : "border-white/10 bg-white/[0.04]"}`}>
                  <p className="text-[11px] uppercase tracking-wider text-zinc-500">
                    {signer ? "Signer" : "Hearing person"} · {m.source} · {new Date(m.created_at).toLocaleTimeString()}
                  </p>
                  <p className="text-lg text-zinc-100">{m.content}</p>
                  {!signer && m.interpreted && <p className="text-sm text-zinc-400">English: {m.interpreted}</p>}
                  {signer && local && <p className="text-sm text-zinc-400">{local}</p>}
                </div>
              </div>
            );
          })}
        </div>
        {error && <p className="mt-2 text-sm text-red-400/90">{error}</p>}
      </div>
    </div>
  );
}
