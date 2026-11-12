import { useEffect, useRef, useState, type ReactElement } from "react";
import { useNavigate } from "react-router-dom";
import { Pause, Play, SkipForward, Volume2, VolumeX } from "lucide-react";
import Logo from "../components/Logo";
import { ttsUrl } from "../lib/api";

const HAND_PATHS = [
  "M200 560 L140 505 L100 445 L75 385 L55 330",
  "M200 560 L150 360 L140 265 L135 205 L130 150",
  "M150 360 L200 348 L200 240 L200 174 L200 114",
  "M200 348 L250 360 L258 258 L263 198 L267 144",
  "M250 360 L295 384 L315 306 L327 258 L337 216",
  "M200 560 L295 384",
];

function Hand({ size = 260, glow = false }: { size?: number; glow?: boolean }) {
  return (
    <svg width={size} height={size * 1.5} viewBox="0 0 400 600" aria-hidden className={glow ? "drop-shadow-[0_0_40px_rgba(240,166,74,0.35)]" : ""}>
      <g stroke="#c9ced6" strokeWidth="6" strokeLinecap="round" fill="none">
        {HAND_PATHS.map((d, i) => (
          <path key={i} d={d} className="intro-draw" style={{ animationDelay: `${i * 0.18}s` }} />
        ))}
      </g>
      <g fill="#f0a64a">
        {[
          [130, 150],
          [200, 114],
          [267, 144],
          [337, 216],
          [55, 330],
        ].map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="11" className="intro-pop" style={{ animationDelay: `${1.1 + i * 0.12}s` }} />
        ))}
      </g>
    </svg>
  );
}

type Scene = { eyebrow: string; title: string; body: string; narration: string; visual: () => ReactElement };

const SCENES: Scene[] = [
  {
    eyebrow: "Welcome",
    title: "Welcome to Ishara AI.",
    body: "An accessibility agent for two-way communication in Indian Sign Language and AAC.",
    narration: "Welcome to Ishara AI, an accessibility agent for two-way communication.",
    visual: () => <Hand size={240} glow />,
  },
  {
    eyebrow: "Recognition",
    title: "Continuous ISL recognition.",
    body: "Raise your hands to start and lower them to finish. Ishara AI reads the whole sentence, not single letters.",
    narration: "Ishara AI recognises continuous Indian Sign Language, one whole sentence at a time.",
    visual: () => (
      <div className="flex items-end gap-6">
        <Hand size={150} />
        <div className="mb-10 rounded-2xl border border-white/10 bg-white/[0.04] px-6 py-4 text-left">
          <p className="label">Recognised</p>
          <p className="intro-type mt-1 text-2xl text-zinc-100">Bring water for me</p>
        </div>
      </div>
    ),
  },
  {
    eyebrow: "Conversation",
    title: "A context-aware AI partner.",
    body: "A multimodal assistant remembers the conversation and uses context when a sign is unclear.",
    narration: "A context-aware assistant remembers the conversation and uses it when a sign is unclear.",
    visual: () => (
      <div className="w-[420px] space-y-3 text-left">
        <p className="intro-rise ml-auto w-fit rounded-2xl bg-gradient-to-b from-zinc-200 to-zinc-400 px-4 py-2 text-zinc-950">Bring water for me</p>
        <p className="intro-rise w-fit rounded-2xl border border-white/10 bg-white/[0.05] px-4 py-2 text-zinc-200" style={{ animationDelay: "0.6s" }}>
          I will get some water for you right away.
        </p>
      </div>
    ),
  },
  {
    eyebrow: "Languages",
    title: "Text and speech in your language.",
    body: "Every reply is shown and spoken in Hindi, Kannada, Tamil or Telugu.",
    narration: "Every reply is shown and spoken in Hindi, Kannada, Tamil or Telugu.",
    visual: () => (
      <div className="grid grid-cols-2 gap-3">
        {["नमस्ते", "ನಮಸ್ಕಾರ", "வணக்கம்", "నమస్కారం"].map((w, i) => (
          <p key={w} className="intro-rise rounded-2xl border border-white/10 bg-white/[0.04] px-6 py-4 text-3xl text-zinc-100" style={{ animationDelay: `${i * 0.25}s` }}>
            {w}
          </p>
        ))}
      </div>
    ),
  },
  {
    eyebrow: "Your way",
    title: "Hands, or eyes and face.",
    body: "Next, choose how you want to communicate. You can choose with your eyes alone.",
    narration: "Next, choose how you want to talk. You can choose with your eyes alone.",
    visual: () => (
      <div className="flex gap-4">
        <p className="intro-rise rounded-2xl border border-white/10 bg-white/[0.04] px-6 py-5 text-xl text-zinc-200">Sign language</p>
        <p className="intro-rise rounded-2xl border border-amber-400/30 bg-amber-400/10 px-6 py-5 text-xl text-amber-200" style={{ animationDelay: "0.4s" }}>
          Eyes and face (AAC)
        </p>
      </div>
    ),
  },
];

const SCENE_MS = 6000;

export default function Intro() {
  const nav = useNavigate();
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [muted, setMuted] = useState(false);
  const [t, setT] = useState(0); // ms into current scene
  const audio = useRef<HTMLAudioElement | null>(null);

  // narration per scene
  useEffect(() => {
    audio.current?.pause();
    if (muted) return;
    const a = new Audio(ttsUrl(SCENES[i].narration, "en"));
    audio.current = a;
    a.play().catch(() => undefined); // autoplay may be blocked; captions carry the content
    return () => a.pause();
  }, [i, muted]);

  useEffect(() => {
    if (!playing) {
      audio.current?.pause();
      return;
    }
    const step = 100;
    const h = window.setInterval(() => {
      setT((prev) => {
        if (prev + step < SCENE_MS) return prev + step;
        setI((cur) => {
          if (cur + 1 >= SCENES.length) {
            window.clearInterval(h);
            nav("/setup");
            return cur;
          }
          return cur + 1;
        });
        return 0;
      });
    }, step);
    return () => window.clearInterval(h);
  }, [playing, nav]);

  const scene = SCENES[i];
  const total = ((i * SCENE_MS + t) / (SCENES.length * SCENE_MS)) * 100;

  return (
    <div className="flex min-h-full flex-col">
      <header className="flex items-center justify-between px-6 py-5">
        <Logo />
        <button className="btn btn-ghost" onClick={() => nav("/setup")}>
          Skip intro <SkipForward size={16} />
        </button>
      </header>
      <main className="grid flex-1 place-items-center px-6">
        <div key={i} className="intro-scene grid w-full max-w-6xl items-center gap-12 md:grid-cols-[1.1fr_1fr]">
          <div>
            <p className="label mb-4 !text-amber-400/90">{scene.eyebrow}</p>
            <h1 className="chrome-text text-5xl font-semibold leading-[1.05] tracking-tight md:text-7xl">{scene.title}</h1>
            <p className="mt-6 max-w-lg text-lg text-zinc-400">{scene.body}</p>
          </div>
          <div className="grid min-h-[340px] place-items-center">{scene.visual()}</div>
        </div>
      </main>
      <footer className="px-6 pb-8">
        <div className="mx-auto flex max-w-6xl items-center gap-4">
          <button className="btn btn-ghost px-3" onClick={() => setPlaying((p) => !p)} title={playing ? "Pause" : "Play"}>
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <button className="btn btn-ghost px-3" onClick={() => setMuted((m) => !m)} title={muted ? "Unmute" : "Mute"}>
            {muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/10">
            <div className="h-full bg-gradient-to-r from-zinc-400 to-amber-400 transition-[width] duration-100" style={{ width: `${total}%` }} />
          </div>
          <span className="font-mono text-xs text-zinc-500">
            {i + 1}/{SCENES.length}
          </span>
        </div>
        <p className="mx-auto mt-3 max-w-6xl text-center text-sm text-zinc-500">{scene.narration}</p>
      </footer>
    </div>
  );
}
