import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { BookOpenCheck, CaseUpper, Circle, Hand, Loader2, Play, RotateCcw, Search, Type, X } from "lucide-react";
import AvatarStage from "../components/AvatarStage";
import { api } from "../lib/api";
import { buildSteps, fill, FINGERTIPS, HAND_EDGES, LH0, RH0, type RawPose, type Segment, type Step } from "../lib/avatar";

export type LibSentence = {
  id: string;
  text: string;
  samples: number;
  in_model: boolean;
  reliability: number | null;
  has_video: boolean;
  has_avatar?: boolean;
  active?: boolean;
};

type Filter = "reliable" | "all" | "untrained";

const RELIABLE = 0.8;

function badge(s: LibSentence) {
  if (!s.in_model) return { text: "not trained yet", cls: "border-white/15 text-zinc-500" };
  if (s.reliability == null) return { text: "not tested", cls: "border-white/15 text-zinc-400" };
  const pct = `${Math.round(s.reliability * 100)}%`;
  if (s.reliability >= RELIABLE) return { text: `reliable ${pct}`, cls: "border-emerald-400/40 bg-emerald-400/10 text-emerald-300" };
  if (s.reliability >= 0.4) return { text: `works sometimes ${pct}`, cls: "border-amber-400/40 bg-amber-400/10 text-amber-200" };
  return { text: `hard ${pct}`, cls: "border-red-400/30 text-red-300/80" };
}

/** A real signer's video next to the Ishara avatar performing the same sentence from its keypoints. */
function WatchModal({ s, onClose }: { s: LibSentence; onClose: () => void }) {
  const [steps, setSteps] = useState<Step[] | null>(null);
  const [state, setState] = useState<"loading" | "ok" | "none">("loading");
  const [key, setKey] = useState(0);
  useEffect(() => {
    api<{ segments: Segment[] }>(`/api/sentences/${s.id}/avatar`)
      .then((r) => {
        setSteps(buildSteps(r.segments, 1.6));
        setState("ok");
        setKey((k) => k + 1);
      })
      .catch(() => setState("none"));
  }, [s.id]);
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/80 p-4" onClick={onClose}>
      <div className="glass w-full max-w-5xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 border-b border-white/10 px-5 py-3">
          <p className="flex-1 text-lg text-zinc-100">{s.text}</p>
          <button className="rounded-md p-1 text-zinc-400 hover:text-zinc-100" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="grid gap-px bg-white/5 md:grid-cols-2">
          <div className="flex flex-col bg-black">
            <p className="label px-4 pt-3">Real signer</p>
            {s.has_video ? (
              <video key={s.id} src={`/api/sentences/${s.id}/video`} controls autoPlay loop muted playsInline className="max-h-[60vh] w-full flex-1 bg-black object-contain" />
            ) : (
              <p className="grid flex-1 place-items-center p-8 text-sm text-zinc-500">No video for this sentence.</p>
            )}
          </div>
          <div className="flex flex-col items-center bg-black">
            <div className="flex w-full items-center px-4 pt-3">
              <p className="label">Ishara avatar</p>
              {state === "ok" && (
                <button className="btn btn-ghost ml-auto px-3 py-1 text-xs" onClick={() => setKey((k) => k + 1)}>
                  <RotateCcw size={13} /> Replay
                </button>
              )}
            </div>
            {state === "loading" && <Loader2 className="m-auto animate-spin text-zinc-500" />}
            {state === "none" && <p className="m-auto p-8 text-sm text-zinc-500">No keypoint recording for this sentence.</p>}
            {state === "ok" && (
              <div className="w-full max-w-[420px]">
                <AvatarStage steps={steps} playKey={key} onDone={() => window.setTimeout(() => setKey((k) => k + 1), 900)} />
              </div>
            )}
          </div>
        </div>
        <p className="px-5 py-3 text-xs text-zinc-500">
          Left: a real signer from the ISL-CSLTR dataset (Elakkiya R and Natarajan B, 2021, CC BY 4.0). Right: the avatar replays the hand and body points
          the sign reader learned from, slowed down so you can follow it.
        </p>
      </div>
    </div>
  );
}

/** What the sign reader knows, how reliably, and a real signer showing each sentence. */
function SentencesView() {
  const nav = useNavigate();
  const [items, setItems] = useState<LibSentence[]>([]);
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("reliable");
  const [playing, setPlaying] = useState<LibSentence | null>(null);

  useEffect(() => {
    api<LibSentence[]>("/api/sentences").then((all) => setItems(all.filter((s) => s.active !== false && s.in_model))).catch(() => setItems([]));
  }, []);

  const shown = useMemo(() => {
    const rank = (s: LibSentence) => (!s.in_model ? 4 : s.reliability == null ? 3 : -s.reliability);
    return items
      .filter((s) => (filter === "reliable" ? s.in_model && (s.reliability ?? 0) >= RELIABLE : filter === "untrained" ? !s.in_model : true))
      .filter((s) => s.text.toLowerCase().includes(q.toLowerCase()))
      .sort((a, b) => rank(a) - rank(b) || a.text.localeCompare(b.text));
  }, [items, q, filter]);

  const reliable = items.filter((s) => s.in_model && (s.reliability ?? 0) >= RELIABLE).length;
  const known = items.filter((s) => s.in_model).length;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <BookOpenCheck className="text-amber-300" />
        <div>
          <h1 className="text-xl font-semibold text-zinc-100">Sign library</h1>
          <p className="text-sm text-zinc-500">
            The sign reader knows {known} sentences. {reliable} of them were recognised at least {RELIABLE * 100}% of the time on videos it never saw. Watch
            a real signer and the avatar, then sign it the same way.
          </p>
        </div>
      </div>

      <div className="glass flex flex-wrap items-center gap-3 p-4">
        <div className="relative min-w-[220px] flex-1">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
          <input className="input pl-9" placeholder="Search sentences" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {([
          ["reliable", `Most reliable (${reliable})`],
          ["all", `All known (${known})`],
          ["untrained", "Not trained yet"],
        ] as [Filter, string][]).map(([k, v]) => (
          <button
            key={k}
            onClick={() => setFilter(k)}
            className={`rounded-full border px-3 py-1.5 text-sm ${filter === k ? "border-amber-400/50 bg-amber-400/10 text-amber-100" : "border-white/10 text-zinc-400"}`}
          >
            {v}
          </button>
        ))}
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {shown.map((s) => {
          const b = badge(s);
          return (
            <div key={s.id} className="glass flex flex-col gap-3 p-4">
              <div className="flex items-start gap-2">
                <p className="flex-1 text-lg text-zinc-100">{s.text}</p>
                <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[11px] ${b.cls}`}>{b.text}</span>
              </div>
              <div className="mt-auto flex flex-wrap gap-2">
                {(s.has_video || s.has_avatar) && (
                  <button className="btn btn-chrome px-3 py-1.5 text-xs" onClick={() => setPlaying(s)}>
                    <Play size={13} /> Watch how
                  </button>
                )}
                <button className="btn btn-ghost px-3 py-1.5 text-xs" onClick={() => nav("/collect", { state: { sentenceId: s.id } })}>
                  <Circle size={13} /> Record my takes
                </button>
                {s.samples > 0 && <span className="self-center font-mono text-[11px] text-zinc-500">{s.samples} takes</span>}
              </div>
            </div>
          );
        })}
        {shown.length === 0 && <p className="text-zinc-500">No sentences match.</p>}
      </div>

      {filter === "untrained" && (
        <p className="text-sm text-zinc-500">
          These sentences have no training videos yet, so the sign reader cannot recognise them. Record 10 or more takes of one on Train, then press Train.
        </p>
      )}

      {playing && <WatchModal s={playing} onClose={() => setPlaying(null)} />}
    </div>
  );
}


// ------------------------------------------------------------------ letters A to Z
type LetterInfo = { letter: string; pose: RawPose | null; hands: number; has_photo: boolean; taught: number };

/** The avatar's hands for one letter, drawn large (both hands, fitted to the canvas). */
function HandsCanvas({ pose, w, h }: { pose: RawPose; w: number; h: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const dpr = Math.min(2, devicePixelRatio || 1);
    c.width = w * dpr;
    c.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const p = fill(pose);
    const hands = [LH0, RH0].filter((h0) => p[h0][1] < 1.5); // a hand resting at the side is not part of the letter
    if (!hands.length) return;
    const pts = hands.flatMap((h0) => Array.from({ length: 21 }, (_, k) => p[h0 + k]));
    const xs = pts.map((q) => q[0]);
    const ys = pts.map((q) => q[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const span = Math.max(Math.max(...xs) - Math.min(...xs), (Math.max(...ys) - Math.min(...ys)) * (w / h), 0.3);
    const z = (w * 0.82) / span;
    const X = (q: number[]) => w / 2 + (q[0] - cx) * z;
    const Y = (q: number[]) => h / 2 + (q[1] - cy) * z;
    hands.forEach((h0, i) => {
      ctx.strokeStyle = i === 0 && hands.length === 2 ? "rgba(201,206,214,0.95)" : "rgba(255,226,186,0.95)";
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.beginPath();
      for (const [a, b] of HAND_EDGES) {
        ctx.moveTo(X(p[h0 + a]), Y(p[h0 + a]));
        ctx.lineTo(X(p[h0 + b]), Y(p[h0 + b]));
      }
      ctx.stroke();
      ctx.fillStyle = "#EEF1F5";
      for (let k = 0; k < 21; k++) {
        ctx.beginPath();
        ctx.arc(X(p[h0 + k]), Y(p[h0 + k]), 2.4, 0, 6.283);
        ctx.fill();
      }
      ctx.fillStyle = "#F0A64A";
      for (const k of FINGERTIPS) {
        ctx.beginPath();
        ctx.arc(X(p[h0 + k]), Y(p[h0 + k]), 4.2, 0, 6.283);
        ctx.fill();
      }
    });
  }, [pose, w, h]);
  return <canvas ref={ref} style={{ width: w, height: h }} />;
}

function LetterModal({ L, onClose }: { L: LetterInfo; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/80 p-4" onClick={onClose}>
      <div className="glass w-full max-w-4xl overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 border-b border-white/10 px-5 py-3">
          <span className="chrome-text text-4xl font-semibold">{L.letter}</span>
          <p className="flex-1 text-zinc-400">{L.hands === 2 ? "Two hands" : "One hand"}{L.taught > 0 ? ` · you taught your own version (${L.taught} takes)` : ""}</p>
          <button className="rounded-md p-1 text-zinc-400 hover:text-zinc-100" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="grid gap-px bg-white/5 md:grid-cols-2">
          <div className="bg-black p-3">
            <p className="label mb-2">Real signer</p>
            {L.has_photo ? <img src={`/api/alphabet/${L.letter}/photo`} alt={`ISL letter ${L.letter}`} className="mx-auto max-h-[60vh] rounded-lg object-contain" /> : <p className="p-8 text-sm text-zinc-500">No photo.</p>}
          </div>
          <div className="flex flex-col items-center bg-black p-3">
            <p className="label mb-2 self-start">Ishara avatar (hand points read from that photo)</p>
            {L.pose && <HandsCanvas pose={L.pose} w={420} h={380} />}
          </div>
        </div>
        <p className="px-5 py-3 text-xs text-zinc-500">
          Photos: Hambir, Khandagale, Santosh, Gaikwad et al., Mendeley Data, doi:10.17632/2tj6c7pzfk.1 (CC BY 4.0). Hand points were read from the photo with MediaPipe.
          Want it your way? Teach your own version on <Link to="/spell" className="text-amber-300 underline">Fingerspelling</Link>.
        </p>
      </div>
    </div>
  );
}

function LettersView() {
  const nav = useNavigate();
  const [letters, setLetters] = useState<LetterInfo[]>([]);
  const [open, setOpen] = useState<LetterInfo | null>(null);
  const [word, setWord] = useState("");
  const [err, setErr] = useState("");
  const [tries, setTries] = useState(0);
  useEffect(() => {
    let live = true;
    setErr("");
    api<LetterInfo[]>("/api/alphabet")
      .then((r) => live && setLetters(r))
      .catch((e) => live && setErr(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [tries]);
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <CaseUpper className="text-amber-300" />
        <div className="flex-1">
          <h1 className="text-xl font-semibold text-zinc-100">ISL alphabet, A to Z</h1>
          <p className="text-sm text-zinc-500">All 26 letters. Each card shows a real signer and the avatar's handshape read from the same photo. Tap a card for a larger view.</p>
        </div>
      </div>
      <form
        className="glass flex flex-wrap items-center gap-3 p-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (word.trim()) nav("/reply", { state: { text: word.trim(), spell: true } });
        }}
      >
        <Type size={16} className="text-zinc-500" />
        <input className="input min-w-[220px] flex-1" placeholder="Type a name or word and watch the avatar spell it" value={word} onChange={(e) => setWord(e.target.value)} />
        <button className="btn btn-chrome px-4" disabled={!word.trim()}>
          <Hand size={15} /> Spell it
        </button>
      </form>
      {letters.length === 0 && !err && <p className="flex items-center gap-2 text-zinc-500"><Loader2 size={15} className="animate-spin" /> Loading the alphabet…</p>}
      {err && (
        <p className="flex flex-wrap items-center gap-3 text-amber-300/90">
          Could not load the alphabet ({err}). The backend may need a restart.
          <button className="btn btn-ghost" onClick={() => setTries((n) => n + 1)}>Try again</button>
        </p>
      )}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
        {letters.map((L) => (
          <button key={L.letter} onClick={() => setOpen(L)} className="glass group overflow-hidden text-left transition hover:border-white/25">
            <div className="flex items-center gap-2 px-3 pt-2">
              <span className="chrome-text text-3xl font-semibold leading-none">{L.letter}</span>
              <span className="ml-auto font-mono text-[10px] uppercase tracking-wider text-zinc-500">{L.hands === 2 ? "2 hands" : "1 hand"}</span>
            </div>
            <div className="grid grid-cols-2 gap-px bg-white/5 p-px">
              <div className="aspect-square bg-black">{L.has_photo && <img src={`/api/alphabet/${L.letter}/photo`} alt="" loading="lazy" className="h-full w-full object-cover" />}</div>
              <div className="grid aspect-square place-items-center bg-black">{L.pose && <HandsCanvas pose={L.pose} w={96} h={96} />}</div>
            </div>
          </button>
        ))}
      </div>
      <p className="text-xs text-zinc-600">Left: a real signer. Right: the avatar. Photos CC BY 4.0, Hambir et al., Mendeley Data, doi:10.17632/2tj6c7pzfk.1.</p>
      {open && <LetterModal L={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

export default function SignLibrary() {
  const [tab, setTab] = useState<"sentences" | "letters">("sentences");
  return (
    <div className="space-y-5">
      <div className="flex gap-1 rounded-xl border border-white/10 bg-black/40 p-1">
        {([["sentences", "Sentences"], ["letters", "Letters A to Z"]] as const).map(([k, label]) => (
          <button key={k} onClick={() => setTab(k)} className={`flex-1 rounded-lg px-3 py-2 text-sm transition ${tab === k ? "bg-white/10 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"}`}>
            {label}
          </button>
        ))}
      </div>
      {tab === "sentences" ? <SentencesView /> : <LettersView />}
    </div>
  );
}
