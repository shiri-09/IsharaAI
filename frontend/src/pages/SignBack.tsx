import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Link, useLocation } from "react-router-dom";
import { CaseUpper, Hand, Loader2, Mic, MicOff, Play, RotateCcw, Sparkles, Waves } from "lucide-react";
import AvatarStage from "../components/AvatarStage";
import { api } from "../lib/api";
import { buildSteps, type Segment, type Step } from "../lib/avatar";
import { usePrefs } from "../lib/prefs";

type Plan = {
  english: string;
  gloss: string[];
  parts: { english: string; gloss: string[]; real_motion: boolean }[];
  segments: Segment[];
  coverage: { clips: number; signed: number; letters: number; letters_taught: number; letters_signed?: number };
};

const LOCALES: Record<string, { label: string; code: string }> = {
  en: { label: "English", code: "en-IN" },
  hi: { label: "हिन्दी", code: "hi-IN" },
  kn: { label: "ಕನ್ನಡ", code: "kn-IN" },
  ta: { label: "தமிழ்", code: "ta-IN" },
  te: { label: "తెలుగు", code: "te-IN" },
};

const EXAMPLES = ["Are you free today?", "Please come to the class at 10 AM.", "मुझे पानी चाहिए", "My name is Ravi.", "Do not worry, I will help you."];

// Web Speech API (Chrome / Edge). Typed loosely because it is not in TypeScript's DOM types everywhere.
type Rec = { lang: string; interimResults: boolean; continuous: boolean; start(): void; stop(): void; onresult: ((e: any) => void) | null; onend: (() => void) | null; onerror: ((e: any) => void) | null };
const SR: (new () => Rec) | undefined = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

export default function SignBack() {
  const { prefs } = usePrefs();
  const [lang, setLang] = useState(prefs?.voice && prefs.voice in LOCALES ? prefs.voice : "en");
  const [text, setText] = useState("");
  const [listening, setListening] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [playKey, setPlayKey] = useState(0);
  const [stepIdx, setStepIdx] = useState(-1);
  const rec = useRef<Rec | null>(null);

  const speedK = prefs?.avatar_speed === "slow" ? 1.6 : prefs?.avatar_speed === "fast" ? 0.7 : 1;
  const steps: Step[] | null = useMemo(() => (plan ? buildSteps(plan.segments, speedK) : null), [plan, speedK]);
  const current = steps && stepIdx >= 0 ? steps[stepIdx] : null;

  useEffect(() => () => rec.current?.stop(), []);

  const loc = useLocation() as { state?: { text?: string; spell?: boolean } };
  useEffect(() => {
    const t = loc.state?.text;
    if (t) {
      setText(t);
      sign(t, !!loc.state?.spell);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function listen() {
    if (!SR) {
      setError("Speech recognition needs Chrome or Edge. You can type instead.");
      return;
    }
    if (listening) {
      rec.current?.stop();
      return;
    }
    const r = new SR();
    r.lang = LOCALES[lang].code;
    r.interimResults = true;
    r.continuous = false;
    r.onresult = (e: any) => {
      const t = Array.from(e.results as ArrayLike<any>).map((x: any) => x[0].transcript).join(" ");
      setText(t);
      if (e.results[e.results.length - 1].isFinal) sign(t);
    };
    r.onend = () => setListening(false);
    r.onerror = (e: any) => {
      setError(e.error === "not-allowed" ? "Microphone permission was denied." : `Speech recognition: ${e.error}`);
      setListening(false);
    };
    rec.current = r;
    setError("");
    setListening(true);
    r.start();
  }

  async function sign(t: string, spell = false) {
    if (!t.trim()) return;
    setBusy(true);
    setError("");
    try {
      const p = await api<Plan>("/api/avatar/plan", { body: { text: t, spell } });
      setPlan(p);
      setStepIdx(-1);
      setPlayKey((k) => k + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    sign(text);
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.05fr]">
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Waves className="text-amber-300" />
          <div>
            <h1 className="text-xl font-semibold text-zinc-100">Reply in sign</h1>
            <p className="text-sm text-zinc-500">For the hearing person: speak or type, and the avatar signs it to the Deaf user.</p>
          </div>
        </div>

        <div className="glass space-y-4 p-5">
          <div className="flex flex-wrap gap-2">
            {Object.entries(LOCALES).map(([k, v]) => (
              <button
                key={k}
                onClick={() => setLang(k)}
                className={`rounded-full border px-3 py-1 text-sm ${lang === k ? "border-amber-400/50 bg-amber-400/10 text-amber-100" : "border-white/10 text-zinc-400"}`}
              >
                {v.label}
              </button>
            ))}
          </div>
          <button
            onClick={listen}
            className={`flex w-full items-center justify-center gap-3 rounded-2xl border py-6 text-lg transition ${
              listening ? "border-red-400/50 bg-red-400/10 text-red-200" : "border-white/10 bg-white/[0.03] text-zinc-200 hover:border-amber-400/40"
            }`}
          >
            {listening ? <MicOff /> : <Mic />} {listening ? "Listening… tap to stop" : `Tap and speak in ${LOCALES[lang].label}`}
          </button>
          <form onSubmit={submit} className="flex gap-2">
            <input className="input" placeholder="…or type what you want to say" value={text} onChange={(e) => setText(e.target.value)} />
            <button className="btn btn-chrome whitespace-nowrap px-4" disabled={busy || !text.trim()}>
              {busy ? <Loader2 size={16} className="animate-spin" /> : <Hand size={16} />} Sign it
            </button>
            <button type="button" className="btn btn-ghost whitespace-nowrap px-3" disabled={busy || !text.trim()} onClick={() => sign(text, true)} title="Finger-spell every word letter by letter (names, short forms)">
              <CaseUpper size={16} /> Spell
            </button>
          </form>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((e) => (
              <button key={e} className="rounded-lg border border-white/10 px-2.5 py-1 text-xs text-zinc-400 hover:text-zinc-200" onClick={() => { setText(e); sign(e); }}>
                {e}
              </button>
            ))}
          </div>
          {error && <p className="text-sm text-red-400/90">{error}</p>}
        </div>

        {plan && (
          <div className="glass space-y-3 p-5">
            <p className="label">Meaning in English</p>
            <p className="text-zinc-100">{plan.english}</p>
            <p className="label pt-2">Signed in ISL order</p>
            {plan.parts.map((part, pi) => (
              <div key={pi} className="flex flex-wrap items-center gap-1.5">
                {part.real_motion && (
                  <span className="rounded-md bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-300" title="This sentence is replayed from a real signer's recorded motion">
                    real signer motion
                  </span>
                )}
                {part.gloss.map((g, gi) => {
                  const label = g.startsWith("FS:") ? g.slice(3) : g;
                  const on = current?.label === label || (part.real_motion && current?.label === part.english);
                  return (
                    <span
                      key={gi}
                      className={`rounded-md border px-2 py-0.5 font-mono text-xs transition ${
                        on ? "border-amber-400/70 bg-amber-400/15 text-amber-100" : g.startsWith("FS:") ? "border-dashed border-white/20 text-zinc-400" : "border-white/10 text-zinc-300"
                      }`}
                      title={g.startsWith("FS:") ? "Finger-spelled" : "Signed"}
                    >
                      {g.startsWith("FS:") ? `${label} (spelled)` : g}
                    </span>
                  );
                })}
              </div>
            ))}
            {plan.coverage.letters > 0 && (
              <p className="pt-1 text-xs text-zinc-500">
                Spelled words use the ISL two-handed alphabet, taken from photos of real signers. A letter you teach on{" "}
                <Link to="/spell" className="text-amber-300 underline">Fingerspelling</Link> replaces the built-in one.
                {(plan.coverage.letters_signed ?? plan.coverage.letters) < plan.coverage.letters && " Letters without a handshape are shown as letter cards."}
              </p>
            )}
          </div>
        )}
      </div>

      <div className="glass flex flex-col items-center gap-3 p-5">
        <div className="flex w-full items-center gap-2">
          <Sparkles size={16} className="text-zinc-400" />
          <span className="label">ISL avatar</span>
          {current && (
            <span className="ml-2 rounded-md border border-amber-400/40 px-2 py-0.5 font-mono text-xs text-amber-200">
              {current.letter ? `spelling ${current.label}: ${current.letter}` : current.label}
            </span>
          )}
          <button className="btn btn-ghost ml-auto px-3 py-1.5 text-xs" disabled={!steps} onClick={() => setPlayKey((k) => k + 1)}>
            {stepIdx === -1 && plan ? <Play size={14} /> : <RotateCcw size={14} />} Replay
          </button>
        </div>
        <AvatarStage steps={steps} playKey={playKey} onStep={setStepIdx} onDone={() => setStepIdx(-1)} />
        <p className="text-center text-xs text-zinc-600">
          Poses come from real signers in the ISL-CSLTR dataset. Known sentences replay a real signer's motion.
        </p>
      </div>
    </div>
  );
}
