import { useEffect, useRef, useState } from "react";
import { Eye } from "lucide-react";
import { calibrate, CAL_POINTS, calStore, FeatureSmoother, type CalPoint, type Calibration, type Pt } from "../lib/gaze";
import type { FaceSignals } from "../lib/useFaceSignals";

type Step = { target: Pt; wiggle?: boolean };
const STEPS: Step[] = [...CAL_POINTS.map((t) => ({ target: t })), { target: { x: 0.5, y: 0.5 }, wiggle: true }];
const fresh = (): CalPoint[] => STEPS.map((st) => ({ target: st.target, frames: [], wiggle: st.wiggle }));

/** The guided calibration: 9 dots, then one dot at the centre while the head turns. About 25 seconds, once. */
export default function GazeCalibration({
  signals,
  onDone,
  onCancel,
  onQuick,
}: {
  signals: React.MutableRefObject<FaceSignals>;
  onDone: (c: Calibration) => void;
  onCancel: () => void;
  onQuick: () => void;
}) {
  const [idx, setIdx] = useState(-1);
  const [phase, setPhase] = useState<"move" | "look">("move");
  const [prog, setProg] = useState(0);
  const [result, setResult] = useState<Calibration | "fail" | null>(null);
  const [seeFace, setSeeFace] = useState(true);
  const data = useRef<CalPoint[]>(fresh());
  const lastFrame = useRef(0);
  const smoother = useRef(new FeatureSmoother());
  const wiggle = idx >= 0 && !!STEPS[idx]?.wiggle;

  useEffect(() => {
    if (idx < 0 || result) return;
    setPhase("move");
    setProg(0);
    smoother.current.reset();
    const need = wiggle ? 100 : 28;
    const settle = wiggle ? 700 : 380;
    const t0 = performance.now();
    let lookAt = 0;
    const h = window.setInterval(() => {
      const el = performance.now() - t0;
      if (el < 750) return;
      if (!lookAt) {
        lookAt = performance.now();
        setPhase("look");
      }
      const s = signals.current;
      setSeeFace(s.face);
      if (!s.face) {
        lookAt += 25; // the clock for this dot waits until the face is back
        return;
      }
      const le = performance.now() - lookAt;
      if (le > settle && s.frame !== lastFrame.current && s.face && s.gf && s.blink < 0.4) {
        lastFrame.current = s.frame;
        data.current[idx].frames.push(smoother.current.push(s.gf));
      }
      const n = data.current[idx].frames.length;
      setProg(wiggle ? Math.min(1, le / 5200) : Math.min(1, n / need));
      const enough = wiggle ? le >= 5200 : n >= need;
      if (enough || le > 7000) {
        window.clearInterval(h);
        if (idx + 1 < STEPS.length) setIdx(idx + 1);
        else {
          const c = calibrate(data.current, innerWidth, innerHeight);
          setResult(c ?? "fail");
          if (c) calStore.save(c);
        }
      }
    }, 25);
    return () => window.clearInterval(h);
  }, [idx, result, signals, wiggle]);

  const dot = idx >= 0 && !result ? STEPS[idx].target : null;
  const q = result && result !== "fail" ? (result.errorPx < 80 ? "Very good" : result.errorPx < 130 ? "Good" : result.errorPx < 210 ? "Fair" : "Poor") : "";
  const C = 2 * Math.PI * 44;
  return (
    <div className="fixed inset-0 z-[100] bg-[#050506]">
      {idx < 0 && (
        <div className="absolute inset-0 grid place-items-center px-6 text-center">
          <div className="max-w-xl space-y-5">
            <Eye className="mx-auto text-zinc-300" size={44} />
            <h2 className="chrome-text text-4xl font-semibold">Set up your eyes</h2>
            <p className="text-lg text-zinc-400">
              A dot appears in 9 places. Look at each one until the ring fills. Then keep looking at the last dot while you slowly turn your head. About 25 seconds, once.
            </p>
            <ul className="mx-auto max-w-md space-y-1 text-left text-sm text-zinc-500">
              <li>• Sit a hand to an arm's length from the camera, face lit from the front.</li>
              <li>• For the first 9 dots, move only your eyes. Keep your head comfortable and still.</li>
              <li>• Keep the browser at the size you will use it.</li>
            </ul>
            <div className="flex flex-wrap justify-center gap-3">
              <button className="btn btn-chrome px-6 py-3 text-lg" onClick={() => setIdx(0)}>
                Start
              </button>
              <button className="btn btn-ghost px-6 py-3 text-lg" onClick={onQuick}>
                Quick start (3 s, less precise)
              </button>
              <button className="btn btn-ghost px-6 py-3 text-lg" onClick={onCancel}>
                Not now
              </button>
            </div>
          </div>
        </div>
      )}
      {dot && (
        <>
          <div className="absolute" style={{ left: `${dot.x * 100}%`, top: `${dot.y * 100}%`, transform: "translate(-50%, -50%)" }}>
            <div className="relative grid h-24 w-24 place-items-center">
              <svg viewBox="-50 -50 100 100" className={`absolute inset-0 -rotate-90 transition-transform duration-300 ${phase === "look" ? "scale-100" : "scale-125"}`}>
                <circle r="44" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="4" />
                <circle r="44" fill="none" stroke="#ef4444" strokeWidth="5" strokeLinecap="round" strokeDasharray={C} strokeDashoffset={C * (1 - (phase === "look" ? prog : 0))} />
              </svg>
              <div className={`h-4 w-4 rounded-full transition-colors ${phase === "look" ? "bg-red-400" : "bg-zinc-100"}`} />
            </div>
          </div>
          {wiggle && phase === "look" && <p className="absolute inset-x-0 top-[18%] text-center text-xl text-zinc-200">Keep looking at the dot. Slowly turn your head left, right, up and down.</p>}
          <div className="absolute inset-x-0 bottom-6 flex flex-col items-center gap-2">
            <div className="flex gap-1.5">
              {STEPS.map((_, i) => (
                <span key={i} className={`h-1.5 w-6 rounded-full ${i < idx ? "bg-red-400" : i === idx ? "bg-zinc-200" : "bg-white/15"}`} />
              ))}
            </div>
            <p className="text-sm text-zinc-500">
              {wiggle ? "Last step" : `Dot ${idx + 1} of ${CAL_POINTS.length}`}
              {!seeFace && <span className="ml-3 text-red-400">I can't see your face. Move into view.</span>}
            </p>
          </div>
        </>
      )}
      {result && (
        <div className="absolute inset-0 grid place-items-center px-6 text-center">
          <div className="max-w-xl space-y-5">
            {result === "fail" ? (
              <h2 className="text-3xl font-semibold text-red-300">I could not see your face for long enough. Try again.</h2>
            ) : (
              <>
                <h2 className="chrome-text text-4xl font-semibold">Eyes set up: {q}</h2>
                <p className="text-lg text-zinc-400">Typical error about {Math.round(result.errorPx)} pixels on this screen.</p>
                {result.errorPx >= 130 && <p className="text-sm text-amber-300/90">Tip: better light on your face, sit a little closer, and look at each dot with your eyes only. Then press Redo.</p>}
              </>
            )}
            <div className="flex justify-center gap-3">
              {result !== "fail" && (
                <button className="btn btn-chrome px-6 py-3 text-lg" onClick={() => onDone(result)}>
                  Done
                </button>
              )}
              <button
                className="btn btn-ghost px-6 py-3 text-lg"
                onClick={() => {
                  data.current = fresh();
                  setResult(null);
                  setIdx(0);
                }}
              >
                Redo
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
