import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import type { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";
import { CaseUpper, Circle, Delete, GraduationCap, Hand, Loader2, Send, Trash2, Waves } from "lucide-react";
import CameraStage from "../components/CameraStage";
import { api } from "../lib/api";
import { hasHands, toFrame } from "../lib/keypoints";
import { classify, letterFeatures, type Template } from "../lib/letters";
import { useHolistic } from "../lib/useHolistic";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const HOLD_FRAMES = 12; // ~0.4 s of the same letter
const WORD_GAP = 24; // ~0.8 s with no hands ends the word
const RECORD_FRAMES = 20;

export default function Spell() {
  const nav = useNavigate();
  const [tab, setTab] = useState<"spell" | "teach">("spell");
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [templates, setTemplates] = useState<Template[]>([]);
  const [word, setWord] = useState("");
  const [words, setWords] = useState<string[]>([]);
  const [guess, setGuess] = useState<{ letter: string; dist: number } | null>(null);
  const [hold, setHold] = useState(0);
  const [teachLetter, setTeachLetter] = useState("A");
  const [phase, setPhase] = useState<"idle" | "countdown" | "recording" | "saving">("idle");
  const [count, setCount] = useState(0);
  const [msg, setMsg] = useState("");

  const st = useRef({ tab, templates, phase, teachLetter });
  st.current = { tab, templates, phase, teachLetter };
  const run = useRef({ last: "", streak: 0, gap: 0, emitted: false });
  const rec = useRef<number[][]>([]);

  const load = useCallback(async () => {
    const [c, t] = await Promise.all([
      api<Record<string, number>>("/api/letters"),
      api<{ letter: string; frame: number[] }[]>("/api/letters/templates"),
    ]);
    setCounts(c);
    setTemplates(t.map((x) => ({ letter: x.letter, feat: letterFeatures(x.frame) })));
  }, []);

  useEffect(() => {
    load().catch(() => undefined);
  }, [load]);

  const saveRecording = useCallback(async () => {
    const frames = rec.current;
    rec.current = [];
    setPhase("saving");
    try {
      await api("/api/letters", { body: { letter: st.current.teachLetter, frames } });
      setMsg(`Saved a sample of ${st.current.teachLetter}.`);
      await load();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setPhase("idle");
    }
  }, [load]);

  const onResult = useCallback(
    (r: HolisticLandmarkerResult) => {
      const s = st.current;
      if (s.tab === "teach") {
        if (s.phase === "recording" && hasHands(r)) {
          rec.current.push(toFrame(r));
          if (rec.current.length >= RECORD_FRAMES) {
            st.current.phase = "saving";
            saveRecording();
          }
        }
        return;
      }
      const R = run.current;
      if (!hasHands(r)) {
        setGuess(null);
        setHold(0);
        R.streak = 0;
        R.gap += 1;
        if (R.gap === WORD_GAP) {
          setWord((w) => {
            if (w) setWords((ws) => [...ws, w]);
            return "";
          });
        }
        R.emitted = false;
        R.last = "";
        return;
      }
      R.gap = 0;
      const g = classify(letterFeatures(toFrame(r)), s.templates);
      setGuess(g);
      if (!g) return;
      if (g.letter === R.last) R.streak += 1;
      else {
        R.last = g.letter;
        R.streak = 1;
        R.emitted = false;
      }
      setHold(Math.min(1, R.streak / HOLD_FRAMES));
      if (R.streak >= HOLD_FRAMES && !R.emitted) {
        R.emitted = true; // a held letter is typed once; change shape (or drop hands) to type the next
        setWord((w) => w + g.letter);
      }
    },
    [saveRecording],
  );

  const cam = useHolistic(onResult);

  function record() {
    if (phase !== "idle") return;
    setMsg("");
    rec.current = [];
    setPhase("countdown");
    let n = 3;
    setCount(n);
    const t = setInterval(() => {
      n -= 1;
      setCount(n);
      if (n === 0) {
        clearInterval(t);
        setPhase("recording");
      }
    }, 600);
  }

  async function clearLetter(l: string) {
    setCounts(await api<Record<string, number>>(`/api/letters/${l}`, { method: "DELETE" }));
    await load();
  }

  const full = [...words, word].filter(Boolean).join(" ");
  const taught = new Set(templates.map((t) => t.letter)).size;

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      <CameraStage
        {...cam}
        onStart={cam.start}
        onStop={cam.stop}
        overlay={
          <>
            {phase === "countdown" && (
              <div className="absolute inset-0 grid place-items-center bg-black/40">
                <span className="chrome-text text-8xl font-semibold">{count}</span>
              </div>
            )}
            {tab === "spell" && guess && (
              <div className="absolute right-3 top-14 grid h-28 w-24 place-items-center rounded-2xl border border-amber-400/50 bg-black/70">
                <span className="text-6xl font-semibold text-amber-200">{guess.letter}</span>
                <div className="absolute inset-x-3 bottom-2 h-1 overflow-hidden rounded-full bg-white/10">
                  <div className="h-full bg-amber-400" style={{ width: `${hold * 100}%` }} />
                </div>
              </div>
            )}
            {phase === "recording" && (
              <div className="absolute bottom-3 left-3 flex items-center gap-2 rounded-full border border-red-400/30 bg-black/70 px-3 py-1 font-mono text-xs text-red-300">
                <Circle size={8} className="animate-pulse fill-red-400" /> HOLD “{teachLetter}” STILL
              </div>
            )}
          </>
        }
        footer={
          <div className="flex flex-wrap items-center gap-2">
            <button className={`btn ${tab === "spell" ? "btn-chrome" : "btn-ghost"}`} onClick={() => setTab("spell")}>
              <CaseUpper size={14} /> Spell
            </button>
            <button className={`btn ${tab === "teach" ? "btn-chrome" : "btn-ghost"}`} onClick={() => setTab("teach")}>
              <GraduationCap size={14} /> Teach letters
            </button>
            <span className="ml-auto text-xs text-zinc-500">{taught} of 26 letters ready</span>
          </div>
        }
      />

      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <Hand className="text-amber-300" />
          <div>
            <h1 className="text-xl font-semibold text-zinc-100">Fingerspelling</h1>
            <p className="text-sm text-zinc-500">For names, colleges, cities and short forms that have no sign.</p>
          </div>
        </div>

        {tab === "spell" ? (
          <div className="glass space-y-4 p-5">
            <p className="text-sm text-zinc-400">
              Hold each letter still until the bar fills. Change shape for the next letter. Lower your hands to finish the word.
            </p>
            <div className="min-h-[88px] rounded-2xl border border-white/10 bg-black/40 px-5 py-4">
              <p className="label mb-1">Spelled so far</p>
              <p className="break-all font-mono text-3xl tracking-[0.2em] text-zinc-100">
                {full || <span className="text-zinc-600">…</span>}
                {word && <span className="animate-pulse text-amber-300">|</span>}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button className="btn btn-ghost" onClick={() => setWord((w) => w.slice(0, -1))} disabled={!word}>
                <Delete size={14} /> Undo letter
              </button>
              <button className="btn btn-ghost" onClick={() => { setWord(""); setWords([]); }} disabled={!full}>
                <Trash2 size={14} /> Clear
              </button>
              <button className="btn btn-chrome ml-auto" onClick={() => nav("/studio", { state: { draft: full } })} disabled={!full}>
                <Send size={14} /> Use in conversation
              </button>
              <button className="btn btn-ghost" onClick={() => nav("/reply", { state: { text: full, spell: true } })} disabled={!full}>
                <Waves size={14} /> Avatar spells it
              </button>
            </div>
            {templates.length === 0 && (
              <p className="text-sm text-amber-300/80">Letters are still loading. If this stays, check the backend is running.</p>
            )}
          </div>
        ) : (
          <div className="glass space-y-4 p-5">
            <p className="text-sm text-zinc-400">
              Pick a letter, press Record, and hold the ISL hand shape still for about a second. Two or three samples per letter, from
              each signer, make spelling much more reliable.
            </p>
            <div className="grid grid-cols-7 gap-2 sm:grid-cols-9">
              {ALPHABET.map((l) => (
                <button
                  key={l}
                  onClick={() => setTeachLetter(l)}
                  className={`relative rounded-xl border py-2 text-lg font-semibold transition ${
                    teachLetter === l ? "border-amber-400/60 bg-amber-400/10 text-amber-100" : counts[l] ? "border-white/20 text-zinc-200" : "border-white/10 text-zinc-500"
                  }`}
                >
                  {l}
                  {counts[l] ? <span className="absolute right-1 top-0.5 font-mono text-[10px] text-emerald-300">{counts[l]}</span> : null}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button className="btn btn-chrome" onClick={record} disabled={cam.status !== "running" || phase !== "idle"}>
                {phase === "saving" ? <Loader2 size={14} className="animate-spin" /> : <Circle size={14} />} Record “{teachLetter}”
              </button>
              {counts[teachLetter] ? (
                <button className="btn btn-ghost" onClick={() => clearLetter(teachLetter)}>
                  <Trash2 size={14} /> Clear {teachLetter}
                </button>
              ) : null}
              {cam.status !== "running" && <span className="text-xs text-zinc-500">Start the camera first.</span>}
            </div>
            {msg && <p className="text-sm text-zinc-400">{msg}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
