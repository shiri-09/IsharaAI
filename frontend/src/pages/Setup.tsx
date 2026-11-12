import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Eye, Hand, Loader2, ScanFace } from "lucide-react";
import GazeProvider, { useGaze } from "../components/GazeProvider";
import Logo from "../components/Logo";
import { usePrefs } from "../lib/prefs";

type Choice = "isl" | "aac";

/** The red ring around the icon fills while you look at the card (GazeProvider sets --gaze-p on the card you are looking at). */
function DwellRing() {
  return (
    <svg width="112" height="112" viewBox="0 0 112 112" className="-rotate-90" aria-hidden>
      <circle cx="56" cy="56" r="48" stroke="rgba(255,255,255,0.08)" strokeWidth="6" fill="none" />
      <circle cx="56" cy="56" r="48" stroke="#ef4444" strokeWidth="6" fill="none" strokeLinecap="round" strokeDasharray="302" style={{ strokeDashoffset: "calc(302px * (1 - var(--gaze-p, 0)))" }} />
    </svg>
  );
}

function SetupInner() {
  const nav = useNavigate();
  const { save } = usePrefs();
  const gaze = useGaze();
  const [saving, setSaving] = useState(false);
  const done = useRef(false);

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

  const card = (id: Choice, title: string, body: string, Icon: typeof Hand) => (
    <button
      onClick={() => choose(id)}
      data-gaze
      data-dwell="1400"
      className="glass relative flex min-h-[340px] flex-col items-center justify-center gap-5 p-10 text-center transition hover:border-white/25"
    >
      <div className="relative grid place-items-center">
        <DwellRing />
        <Icon className="absolute text-zinc-300" size={40} />
      </div>
      <h2 className="chrome-text text-3xl font-semibold">{title}</h2>
      <p className="max-w-xs text-zinc-400">{body}</p>
    </button>
  );

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
            <Eye size={18} />
            {gaze?.calibrated
              ? "Look at an option until the red ring fills. You can also click or tap."
              : "Calibrate eye tracking first (the screen that opens), or just click an option."}
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {card("isl", "Sign language", "Use Indian Sign Language with the camera. Mouse and keyboard work as usual.", Hand)}
          {card("aac", "Eyes and face (AAC)", "No mouse or keyboard needed. Calibrated eye tracking, an on-screen keyboard, AI conversation, live speech and the agent, all hands-free.", ScanFace)}
        </div>

        <p className="text-center text-sm text-zinc-500">
          {gaze?.status === "error" ? `Camera problem: ${gaze.error}. You can still click an option.` : "The camera is only used on this device. Nothing is recorded or uploaded."}
        </p>
        {saving && <Loader2 className="mx-auto animate-spin text-zinc-400" />}
      </main>
    </div>
  );
}

export default function Setup() {
  return (
    <GazeProvider>
      <SetupInner />
    </GazeProvider>
  );
}
