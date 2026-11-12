import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, FlipHorizontal2, Hand, Loader2, ScanFace } from "lucide-react";
import Logo from "../components/Logo";
import { usePrefs } from "../lib/prefs";
import { useFaceSignals, type FaceSignals } from "../lib/useFaceSignals";

const DWELL_MS = 2000;
const ZONE = 0.22; // |gazeX| beyond this counts as looking at a side

type Choice = "isl" | "aac";

function DwellRing({ progress }: { progress: number }) {
  return (
    <svg width="112" height="112" viewBox="0 0 112 112" className="-rotate-90" aria-hidden>
      <circle cx="56" cy="56" r="48" stroke="rgba(255,255,255,0.08)" strokeWidth="6" fill="none" />
      <circle
        cx="56"
        cy="56"
        r="48"
        stroke="#f0a64a"
        strokeWidth="6"
        fill="none"
        strokeLinecap="round"
        strokeDasharray="302"
        strokeDashoffset={302 * (1 - progress)}
      />
    </svg>
  );
}

export default function Setup() {
  const nav = useNavigate();
  const { save } = usePrefs();
  const [zone, setZone] = useState<Choice | null>(null);
  const [progress, setProgress] = useState(0);
  const [gaze, setGaze] = useState(0);
  const [flip, setFlip] = useState(false);
  const [saving, setSaving] = useState(false);
  const dwell = useRef<{ zone: Choice | null; since: number }>({ zone: null, since: 0 });
  const done = useRef(false);
  const flipRef = useRef(flip);
  flipRef.current = flip;

  const choose = useCallback(
    async (mode: Choice) => {
      if (done.current) return;
      done.current = true;
      setSaving(true);
      try {
        await save({ mode, onboarded: true });
      } finally {
        nav(mode === "aac" ? "/aac" : "/", { replace: true });
      }
    },
    [nav, save],
  );

  const onFrame = useCallback(
    (s: FaceSignals) => {
      if (done.current) return;
      const g = flipRef.current ? -s.gazeX : s.gazeX;
      const z: Choice | null = !s.face ? null : g < -ZONE ? "isl" : g > ZONE ? "aac" : null;
      const now = performance.now();
      if (z !== dwell.current.zone) dwell.current = { zone: z, since: now };
      const p = z ? Math.min(1, (now - dwell.current.since) / DWELL_MS) : 0;
      setGaze(g);
      setZone(z);
      setProgress(p);
      if (z && p >= 1) choose(z);
    },
    [choose],
  );

  const face = useFaceSignals(onFrame);

  useEffect(() => {
    face.start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const card = (id: Choice, title: string, body: string, Icon: typeof Hand) => {
    const active = zone === id;
    return (
      <button
        onClick={() => choose(id)}
        className={`glass relative flex min-h-[340px] flex-col items-center justify-center gap-5 p-10 text-center transition ${
          active ? "!border-amber-400/50 !bg-amber-400/[0.07] scale-[1.02]" : ""
        }`}
      >
        <div className="relative grid place-items-center">
          <DwellRing progress={active ? progress : 0} />
          <Icon className={`absolute ${active ? "text-amber-300" : "text-zinc-300"}`} size={40} />
        </div>
        <h2 className="chrome-text text-3xl font-semibold">{title}</h2>
        <p className="max-w-xs text-zinc-400">{body}</p>
      </button>
    );
  };

  return (
    <div className="flex min-h-full flex-col">
      <header className="flex items-center justify-between px-6 py-5">
        <Logo />
        <span className="label">First-time setup</span>
      </header>
      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col justify-center gap-8 px-6 pb-10">
        <div className="text-center">
          <h1 className="chrome-text text-4xl font-semibold tracking-tight md:text-5xl">How would you like to talk?</h1>
          <p className="mt-3 flex items-center justify-center gap-2 text-zinc-400">
            <Eye size={18} /> Look at an option and hold your gaze for 2 seconds. Turning your head slightly also works.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {card("isl", "Sign language", "Use Indian Sign Language with the camera. Mouse and keyboard work as usual.", Hand)}
          {card("aac", "Eyes and face (AAC)", "No mouse or keyboard needed. Choose phrases with eyebrow raises and speak them aloud.", ScanFace)}
        </div>

        <div className="glass flex flex-wrap items-center gap-5 p-4">
          <div className="relative h-24 w-32 overflow-hidden rounded-xl bg-black">
            <video ref={face.videoRef} playsInline muted className="h-full w-full -scale-x-100 object-cover" />
            {face.status === "loading" && (
              <div className="absolute inset-0 grid place-items-center">
                <Loader2 className="animate-spin text-zinc-400" size={18} />
              </div>
            )}
          </div>
          <div className="min-w-[220px] flex-1">
            <p className="label mb-2">Where you are looking</p>
            <div className="relative h-3 rounded-full bg-white/[0.06]">
              <div className="absolute inset-y-0 left-1/2 w-px bg-white/20" />
              <div
                className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-gradient-to-b from-zinc-100 to-zinc-400 shadow-[0_0_16px_rgba(240,166,74,0.6)] transition-[left] duration-75"
                style={{ left: `${50 + Math.max(-1, Math.min(1, gaze)) * 50}%` }}
              />
            </div>
            <p className="mt-2 text-xs text-zinc-500">
              {face.status === "error"
                ? `Camera unavailable: ${face.error}. You can click an option instead.`
                : face.status === "running"
                  ? "Camera on. Nothing is recorded or uploaded."
                  : "Starting camera…"}
            </p>
          </div>
          <button className="btn btn-ghost" onClick={() => setFlip((f) => !f)} title="Use if the pointer moves the wrong way">
            <FlipHorizontal2 size={16} /> Flip direction
          </button>
          {saving && <Loader2 className="animate-spin text-zinc-400" />}
        </div>
      </main>
    </div>
  );
}
