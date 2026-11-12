import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Crosshair, Eye, EyeOff, Loader2, Pause, Play, Target } from "lucide-react";
import { autoStore, calStore, centerCalibration, predictFrac, FeatureSmoother, GazeTracker, type AutoCal, type Calibration, type Feat } from "../lib/gaze";
import GazeCalibration from "./GazeCalibration";
import { useFaceSignals, type FaceSignals } from "../lib/useFaceSignals";

/**
 * Eye tracking and face gestures for hands-free (AAC) use.
 *
 *  - Look: a circle follows your gaze. SILVER while tracking, RED when a target is lined up.
 *  - Eyebrows up = YES: selects the thing you are looking at, or presses a visible "Yes" button.
 *  - Mouth open = NO: presses a visible "No" / "Cancel" button, or Back (delete a letter) on the keyboard.
 *  - Or keep looking at a target for a moment and it selects by itself (you can switch this off).
 *
 * Anything with `data-gaze` can be selected. `data-yes`, `data-no` and `data-back` mark the buttons the gestures press.
 * Setup is one dot: look at the centre of the screen for 3 seconds. It keeps learning from every selection you make.
 */

type Ctx = {
  status: "idle" | "loading" | "running" | "error";
  error: string;
  calibrated: boolean;
  errorPx: number | null;
  paused: boolean;
  setPaused: (p: boolean) => void;
  startCalibration: () => void;
  gaze: { x: number; y: number } | null;
};

const GazeCtx = createContext<Ctx | null>(null);
export const useGaze = () => useContext(GazeCtx);

const BROW_RISE = 0.2; // eyebrows must rise this far above their resting level, quickly
const BROW_REARM = 0.09;
const JAW_NO = 0.5;
const JAW_NO_MS = 380;
const DEFAULT_DWELL = 1200;
const GRACE_MS = 320;
const DWELL_LEVELS = [
  { label: "Fast", k: 1 },
  { label: "Normal", k: 1.5 },
  { label: "Slow", k: 2.1 },
];
type SelectMode = "both" | "brow" | "dwell";
const SELECT_MODES: { id: SelectMode; label: string }[] = [
  { id: "both", label: "Brows or look" },
  { id: "brow", label: "Brows only" },
  { id: "dwell", label: "Look only" },
];
const readLS = (k: string, d: string) => {
  try {
    return localStorage.getItem(k) ?? d;
  } catch {
    return d;
  }
};
const writeLS = (k: string, v: string) => {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* ignore */
  }
};

function usable(el: HTMLElement) {
  if ((el as HTMLButtonElement).disabled || el.getAttribute("aria-disabled") === "true") return false;
  const r = el.getBoundingClientRect();
  return r.width > 2 && r.height > 2 && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
}
const firstVisible = (sel: string) => Array.from(document.querySelectorAll<HTMLElement>(sel)).find(usable) ?? null;

function nearestTarget(x: number, y: number, tol: number, keep: HTMLElement | null): HTMLElement | null {
  let best: HTMLElement | null = null;
  let bestD = Infinity;
  document.querySelectorAll<HTMLElement>("[data-gaze]").forEach((el) => {
    if (!usable(el)) return;
    const r = el.getBoundingClientRect();
    const d = Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom));
    const limit = el === keep ? tol * 1.7 : tol; // stick to the current target a little longer: no flicker between neighbours
    if (d <= limit && d < bestD) {
      best = el;
      bestD = d;
    }
  });
  return best;
}

function GazeCircle({ x, y, progress, armed, lost, paused }: { x: number; y: number; progress: number; armed: boolean; lost: boolean; paused: boolean }) {
  const R = 26;
  const ring = 2 * Math.PI * (R + 6);
  return (
    <div className="pointer-events-none fixed left-0 top-0 z-[90]" style={{ transform: `translate(${x}px, ${y}px)`, transition: "transform 80ms linear" }} aria-hidden>
      <svg width="76" height="76" viewBox="-38 -38 76 76" className="-ml-[38px] -mt-[38px]">
        <defs>
          <radialGradient id="gz-silver" cx="50%" cy="50%" r="50%">
            <stop offset="0" stopColor="rgba(235,238,243,0.35)" />
            <stop offset="1" stopColor="rgba(235,238,243,0.05)" />
          </radialGradient>
          <radialGradient id="gz-red" cx="50%" cy="50%" r="50%">
            <stop offset="0" stopColor="rgba(248,113,113,0.5)" />
            <stop offset="1" stopColor="rgba(239,68,68,0.08)" />
          </radialGradient>
        </defs>
        <circle r={R} fill={armed ? "url(#gz-red)" : "url(#gz-silver)"} stroke={paused ? "#71717a" : armed ? "#ef4444" : "#d4d8de"} strokeWidth="3" strokeDasharray={lost || paused ? "4 5" : undefined} opacity={lost || paused ? 0.55 : 1} />
        {armed && progress > 0.02 && <circle r={R + 6} fill="none" stroke="#ef4444" strokeWidth="4" strokeLinecap="round" strokeDasharray={ring} strokeDashoffset={ring * (1 - progress)} transform="rotate(-90)" />}
        <circle r="3.5" fill={armed ? "#fecaca" : "#f4f5f7"} />
      </svg>
    </div>
  );
}

/** One dot at the centre of the screen, about 3 seconds. */
function CenterOverlay({ signals, prev, onDone, onCancel }: { signals: React.MutableRefObject<FaceSignals>; prev: AutoCal | null; onDone: (a: AutoCal) => void; onCancel: () => void }) {
  const [t, setT] = useState(0);
  const [failed, setFailed] = useState(false);
  const frames = useRef<Feat[]>([]);
  const sm = useRef(new FeatureSmoother());
  const last = useRef(0);
  const t0 = useRef(performance.now());
  const [run, setRun] = useState(0);

  useEffect(() => {
    frames.current = [];
    sm.current.reset();
    t0.current = performance.now();
    setFailed(false);
    const h = window.setInterval(() => {
      const el = performance.now() - t0.current;
      setT(el);
      const s = signals.current;
      if (el > 700 && el < 3200 && s.face && s.gf && s.blink < 0.4 && s.frame !== last.current) {
        last.current = s.frame;
        frames.current.push(sm.current.push(s.gf));
      }
      if (el >= 3300) {
        const c = centerCalibration(frames.current, prev);
        if (c) {
          window.clearInterval(h);
          onDone(c);
        } else if (el > 9000) {
          window.clearInterval(h);
          setFailed(true);
        }
      }
    }, 30);
    return () => window.clearInterval(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  const p = Math.min(1, Math.max(0, (t - 700) / 2600));
  return (
    <div className="fixed inset-0 z-[100] grid place-items-center bg-[#050506]">
      <div className="text-center">
        <div className="relative mx-auto grid h-40 w-40 place-items-center">
          <svg viewBox="-60 -60 120 120" className="absolute inset-0 -rotate-90">
            <circle r="46" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="5" />
            <circle r="46" fill="none" stroke="#ef4444" strokeWidth="5" strokeLinecap="round" strokeDasharray={2 * Math.PI * 46} strokeDashoffset={2 * Math.PI * 46 * (1 - p)} />
          </svg>
          <div className={`h-6 w-6 rounded-full ${p > 0 ? "bg-red-400" : "bg-zinc-100"}`} />
        </div>
        {!failed ? (
          <div className="mt-8 space-y-2">
            <p className="chrome-text text-3xl font-semibold">Look at the dot</p>
            <p className="mx-auto max-w-md text-zinc-400">Sit close, a hand's length or a little more from the camera is fine. Keep your head still. That is all the setup there is.</p>
          </div>
        ) : (
          <div className="mt-8 space-y-4">
            <p className="text-2xl text-red-300">I could not see your face</p>
            <p className="text-zinc-400">Move into the camera view and make sure there is light on your face.</p>
            <div className="flex justify-center gap-3">
              <button className="btn btn-chrome px-5 py-2" onClick={() => setRun((x) => x + 1)}>
                Try again
              </button>
              <button className="btn btn-ghost px-5 py-2" onClick={onCancel}>
                Skip
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function GazeProvider({ children }: { children: ReactNode }) {
  const tracker = useRef(new GazeTracker());
  const [cal, setCal] = useState<Calibration | null>(() => calStore.load());
  const [auto, setAuto] = useState<AutoCal | null>(() => autoStore.load());
  const [mode, setMode] = useState<"none" | "center" | "nine">("none");
  const [paused, setPaused] = useState(false);
  const [gaze, setGaze] = useState<{ x: number; y: number } | null>(null);
  const [hover, setHover] = useState<{ p: number; on: boolean }>({ p: 0, on: false });
  const [showCam, setShowCam] = useState(true);
  const [dwellIdx, setDwellIdx] = useState(() => Math.min(2, Math.max(0, Number(readLS("ishara.gaze.dwell", "1")))));
  const [selMode, setSelMode] = useState<SelectMode>(() => readLS("ishara.gaze.select", "both") as SelectMode);
  const [meters, setMeters] = useState({ rise: 0, jaw: 0 });
  const [flash, setFlash] = useState<"yes" | "no" | null>(null);
  const dwellK = DWELL_LEVELS[dwellIdx].k;

  const stateRef = useRef({ cal, mode, paused, dwellK, selMode });
  stateRef.current = { cal, mode, paused, dwellK, selMode };
  const autoRef = useRef(auto);
  autoRef.current = auto;
  const brow = useRef({ base: 0, init: false, armed: true, rise: 0, jaw: 0 });
  const dw = useRef<{ el: HTMLElement | null; since: number; cool: number; lock: HTMLElement | null; jawSince: number; jawFired: boolean; lostAt: number; lostEl: HTMLElement | null; lostSince: number }>({
    el: null, since: 0, cool: 0, lock: null, jawSince: 0, jawFired: false, lostAt: 0, lostEl: null, lostSince: 0,
  });

  useEffect(() => {
    tracker.current.setModel(cal?.model ?? null, cal ? { x: cal.bx ?? 0, y: cal.by ?? 0 } : undefined);
    tracker.current.setAuto(cal ? null : auto);
  }, [cal, auto]);

  const flashIt = (k: "yes" | "no") => {
    setFlash(k);
    window.setTimeout(() => setFlash(null), 800);
  };

  /** Press a target. `fromGaze`: the user was looking at it, so the tracker can learn from where it is. */
  const fire = useCallback((el: HTMLElement, fromGaze: boolean) => {
    el.setAttribute("data-gaze-fired", "1");
    window.setTimeout(() => el.removeAttribute("data-gaze-fired"), 450);
    if (fromGaze) {
      const r = el.getBoundingClientRect();
      const t = { x: (r.left + r.width / 2) / innerWidth, y: (r.top + r.height / 2) / innerHeight };
      const c = stateRef.current.cal;
      if (c) {
        tracker.current.nudge(t, 0.2);
        calStore.save({ ...c, bx: tracker.current.bias.x, by: tracker.current.bias.y });
      } else if (autoRef.current) {
        tracker.current.nudge(t, 0.25);
        autoStore.save(autoRef.current);
      }
    }
    el.click();
  }, []);

  const setHoverEl = (el: HTMLElement | null) => {
    const d = dw.current;
    if (d.el && d.el !== el) {
      d.el.removeAttribute("data-gaze-hover");
      d.el.style.removeProperty("--gaze-p");
    }
    if (el && el !== d.el) el.setAttribute("data-gaze-hover", "1");
    d.el = el;
  };

  const onFrame = useCallback(
    (f: FaceSignals) => {
      const S = stateRef.current;
      const d = dw.current;
      const now = performance.now();
      if (S.mode !== "none") return;

      // eyebrows: compare with their own resting level, and look for a quick rise (looking up lifts them slowly and less)
      const b = brow.current;
      if (!b.init) {
        b.base = f.brow;
        b.init = true;
      }
      if (Math.abs(f.brow - b.base) < 0.08) b.base += (f.brow - b.base) * 0.02;
      b.rise = f.brow - b.base;
      b.jaw = f.jaw;
      if (b.rise < BROW_REARM) b.armed = true;

      // mouth open = NO
      if (f.face && f.jaw > JAW_NO) {
        if (!d.jawSince) d.jawSince = now;
        if (!d.jawFired && now - d.jawSince > JAW_NO_MS && now > d.cool) {
          d.jawFired = true;
          d.cool = now + 1100;
          const target = firstVisible("[data-no]") ?? firstVisible("[data-back]");
          if (target) {
            fire(target, false);
            flashIt("no");
          }
        }
      } else {
        d.jawSince = 0;
        d.jawFired = false;
      }

      const g = S.paused ? null : tracker.current.update(f.gf, f.blink, now);
      if (!g) {
        setGaze(null);
        setHoverEl(null);
        setHover({ p: 0, on: false });
        // eyebrows still answer a visible Yes even without a gaze point
        if (S.selMode !== "dwell" && f.face && b.armed && b.rise > BROW_RISE && now > d.cool && !S.paused) {
          const yes = firstVisible("[data-yes]");
          if (yes) {
            b.armed = false;
            d.cool = now + 900;
            fire(yes, false);
            flashIt("yes");
          }
        }
        return;
      }
      const x = Math.max(0, Math.min(innerWidth, g.x * innerWidth));
      const y = Math.max(0, Math.min(innerHeight, g.y * innerHeight));
      setGaze({ x, y });
      const tol = Math.max(30, Math.min(110, (S.cal?.errorPx ?? 120) * 0.75));
      // scroll the page when the eyes rest at the very top or bottom edge
      if (y > innerHeight * 0.94) window.scrollBy(0, 9 + (y / innerHeight - 0.94) * 260);
      else if (y < innerHeight * 0.05) window.scrollBy(0, -(9 + (0.05 - y / innerHeight) * 260));
      let el = nearestTarget(x, y, tol, d.el);
      if (el && el === d.lock) {
        if (now - d.cool > 2500) d.lock = null;
        else el = null;
      } else if (!el) d.lock = null;
      if (el !== d.el) {
        if (d.el && el === null) {
          d.lostEl = d.el;
          d.lostAt = now;
          d.lostSince = d.since;
        }
        if (el && el === d.lostEl && now - d.lostAt < GRACE_MS) d.since = d.lostSince + (now - d.lostAt);
        else d.since = now;
        setHoverEl(el);
      }

      // eyebrows up = YES: choose what you are looking at, or press a visible Yes
      if (S.selMode !== "dwell" && f.face && b.armed && b.rise > BROW_RISE && now > d.cool) {
        const target = el ?? firstVisible("[data-yes]");
        if (target) {
          b.armed = false;
          d.cool = now + 900;
          d.lock = el;
          const fromGaze = !!el;
          setHoverEl(null);
          setHover({ p: 0, on: false });
          fire(target, fromGaze);
          flashIt("yes");
          return;
        }
      }
      if (!el) {
        setHover({ p: 0, on: false });
        return;
      }
      if (S.selMode === "brow") {
        setHover({ p: 0, on: true }); // lined up: waiting for the eyebrows
        return;
      }
      const need = (Number(el.dataset.dwell) || DEFAULT_DWELL) * S.dwellK;
      const p = Math.min(1, Math.max(0, now - Math.max(d.since, d.cool)) / need);
      el.style.setProperty("--gaze-p", String(p));
      setHover({ p, on: true });
      if (p >= 1) {
        d.cool = now + 900;
        d.lock = el;
        setHoverEl(null);
        setHover({ p: 0, on: false });
        fire(el, true);
      }
    },
    [fire],
  );

  const face = useFaceSignals(onFrame);
  useEffect(() => {
    face.start();
    return () => setHoverEl(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // live meters for the panel
  useEffect(() => {
    const h = window.setInterval(() => setMeters({ rise: brow.current.rise, jaw: brow.current.jaw }), 90);
    return () => window.clearInterval(h);
  }, []);

  // first visit: when the face is in view, run the one-dot setup (no button to press)
  const seen = useRef(0);
  useEffect(() => {
    const h = window.setInterval(() => {
      if (face.status !== "running" || cal || auto || stateRef.current.mode !== "none") return;
      seen.current = face.signals.current.face ? seen.current + 1 : 0;
      if (seen.current >= 4) {
        seen.current = -9999;
        setMode("nine"); // the guided setup, with a quick 3 second alternative inside
      }
    }, 300);
    return () => window.clearInterval(h);
  }, [face.status, face.signals, cal, auto]);

  const tweakGain = (k: "gainX" | "gainY", f: number) => {
    if (!autoRef.current) return;
    const a = { ...autoRef.current, [k]: Math.max(0.3, Math.min(3, autoRef.current[k] * f)) };
    autoStore.save(a);
    setAuto(a);
  };

  const ctx = useMemo<Ctx>(
    () => ({
      status: face.status,
      error: face.error,
      calibrated: !!cal || !!auto,
      errorPx: cal?.errorPx ?? null,
      paused,
      setPaused,
      startCalibration: () => setMode("nine"),
      gaze,
    }),
    [face.status, face.error, cal, auto, paused, gaze],
  );

  const miniBtn = "btn btn-ghost px-2 py-1 text-[11px]";
  return (
    <GazeCtx.Provider value={ctx}>
      {children}
      {gaze && mode === "none" && <GazeCircle x={gaze.x} y={gaze.y} progress={hover.p} armed={hover.on} lost={!face.signals.current.face} paused={paused} />}
      {mode === "center" && (
        <CenterOverlay
          signals={face.signals}
          prev={auto}
          onCancel={() => setMode("none")}
          onDone={(a) => {
            const c = stateRef.current.cal;
            if (c) {
              // already calibrated: Re-center only corrects the offset, it does not throw the calibration away
              const p = predictFrac(c.model, a.ref);
              const nc = { ...c, bx: 0.5 - p.x, by: 0.5 - p.y };
              calStore.save(nc);
              setCal(nc);
            } else {
              autoStore.save(a);
              setAuto(a);
            }
            setMode("none");
          }}
        />
      )}
      {mode === "nine" && (
        <GazeCalibration
          signals={face.signals}
          onCancel={() => setMode("none")}
          onQuick={() => setMode("center")}
          onDone={(c: Calibration) => {
            setCal(c);
            setMode("none");
          }}
        />
      )}

      {flash && (
        <div className={`pointer-events-none fixed bottom-24 left-1/2 z-[95] -translate-x-1/2 rounded-full border px-6 py-2 text-lg font-semibold ${flash === "yes" ? "border-emerald-400/50 bg-emerald-400/15 text-emerald-200" : "border-red-400/50 bg-red-400/15 text-red-200"}`}>
          {flash === "yes" ? "Yes ✓" : "No ✕"}
        </div>
      )}

      {/* camera, gestures and settings */}
      <div className="fixed bottom-3 right-3 z-[80] w-[270px] overflow-hidden rounded-2xl border border-white/10 bg-black/85 shadow-2xl backdrop-blur">
        <div className="relative aspect-[16/10] bg-black" style={{ display: showCam ? "block" : "none" }}>
          <video ref={face.videoRef} playsInline muted className="h-full w-full -scale-x-100 object-cover" />
          {face.status !== "running" && (
            <div className="absolute inset-0 grid place-items-center p-3 text-center text-xs text-zinc-500">
              {face.status === "loading" ? <Loader2 className="animate-spin" /> : face.status === "error" ? `Camera problem: ${face.error}` : "Starting camera…"}
            </div>
          )}
          <span className={`absolute left-2 top-2 rounded-full px-2 py-0.5 font-mono text-[10px] ${face.signals.current.face ? "bg-emerald-500/20 text-emerald-300" : "bg-black/60 text-zinc-500"}`}>{face.signals.current.face ? "FACE" : "NO FACE"}</span>
        </div>
        <div className="space-y-2 p-3">
          <p className="flex items-center gap-2 text-xs text-zinc-400">
            <span className={`h-2 w-2 shrink-0 rounded-full ${paused ? "bg-zinc-500" : cal || auto ? "bg-emerald-400" : "bg-amber-400"}`} />
            {paused ? "Paused. Press play." : cal ? `Calibrated: ${cal.errorPx < 80 ? "very good" : cal.errorPx < 130 ? "good" : cal.errorPx < 210 ? "fair" : "poor"} (about ${Math.round(cal.errorPx)} px)` : auto ? "Quick setup (rough). Calibrate for accuracy." : "Setting up…"}
          </p>
          <div className="space-y-1.5 text-[11px] text-zinc-400">
            <div className="flex items-center gap-2">
              <span className="w-[104px] shrink-0">Eyebrows up = Yes</span>
              <div className="relative h-1.5 flex-1 rounded-full bg-white/10">
                <div className={`h-full rounded-full ${meters.rise > BROW_RISE ? "bg-emerald-400" : "bg-zinc-400"}`} style={{ width: `${Math.max(0, Math.min(100, (meters.rise / (BROW_RISE * 1.5)) * 100))}%` }} />
                <div className="absolute inset-y-[-2px] w-px bg-white/50" style={{ left: `${(1 / 1.5) * 100}%` }} />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-[104px] shrink-0">Mouth open = No</span>
              <div className="relative h-1.5 flex-1 rounded-full bg-white/10">
                <div className={`h-full rounded-full ${meters.jaw > JAW_NO ? "bg-red-400" : "bg-zinc-400"}`} style={{ width: `${Math.max(0, Math.min(100, (meters.jaw / (JAW_NO * 1.5)) * 100))}%` }} />
                <div className="absolute inset-y-[-2px] w-px bg-white/50" style={{ left: `${(1 / 1.5) * 100}%` }} />
              </div>
            </div>
          </div>
          <div className="flex gap-1.5">
            <button className="btn btn-chrome flex-1 px-2 py-1.5 text-xs" onClick={() => setMode("center")} data-gaze data-dwell="1600" title="Look at the centre dot for 3 seconds">
              <Target size={13} /> Re-center
            </button>
            <button className={miniBtn} onClick={() => setPaused((p) => !p)} data-gaze data-dwell="1400" title="Pause or resume eye tracking">
              {paused ? <Play size={13} /> : <Pause size={13} />}
            </button>
            <button className={miniBtn} onClick={() => setShowCam((v) => !v)} title="Show or hide the camera picture">
              {showCam ? <EyeOff size={13} /> : <Eye size={13} />}
            </button>
          </div>
          <div className="flex gap-1.5">
            <button
              className={`${miniBtn} flex-1`}
              onClick={() => {
                const n = SELECT_MODES[(SELECT_MODES.findIndex((m) => m.id === selMode) + 1) % SELECT_MODES.length].id;
                setSelMode(n);
                writeLS("ishara.gaze.select", n);
              }}
              title="How things get selected"
            >
              Select: {SELECT_MODES.find((m) => m.id === selMode)?.label}
            </button>
            <button
              className={miniBtn}
              onClick={() => {
                const n = (dwellIdx + 1) % DWELL_LEVELS.length;
                setDwellIdx(n);
                writeLS("ishara.gaze.dwell", String(n));
              }}
              title="How long you must look before it selects by itself"
            >
              {DWELL_LEVELS[dwellIdx].label}
            </button>
          </div>
          {auto && !cal && (
            <div className="grid grid-cols-2 gap-1.5 text-[11px]">
              <div className="flex items-center gap-1">
                <span className="w-5 text-zinc-500">H</span>
                <button className={`${miniBtn} flex-1`} onClick={() => tweakGain("gainX", 0.88)} title="Dot moves less sideways">−</button>
                <button className={`${miniBtn} flex-1`} onClick={() => tweakGain("gainX", 1.14)} title="Dot moves more sideways">+</button>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-5 text-zinc-500">V</span>
                <button className={`${miniBtn} flex-1`} onClick={() => tweakGain("gainY", 0.88)} title="Dot moves less up and down">−</button>
                <button className={`${miniBtn} flex-1`} onClick={() => tweakGain("gainY", 1.14)} title="Dot moves more up and down">+</button>
              </div>
            </div>
          )}
          <button
            className={cal ? "flex w-full items-center justify-center gap-1.5 text-[11px] text-zinc-500 hover:text-zinc-300" : "btn btn-chrome flex w-full items-center justify-center gap-1.5 px-2 py-1.5 text-xs"}
            onClick={() => setMode("nine")}
            data-gaze
            data-dwell="1800"
          >
            <Crosshair size={12} /> {cal ? "Calibrate again" : "Calibrate for accuracy (25 s)"}
          </button>
        </div>
      </div>
    </GazeCtx.Provider>
  );
}
