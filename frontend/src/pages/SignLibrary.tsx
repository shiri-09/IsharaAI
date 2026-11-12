import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BookOpenCheck, Circle, Loader2, Play, RotateCcw, Search, X } from "lucide-react";
import AvatarStage from "../components/AvatarStage";
import { api } from "../lib/api";
import { buildSteps, type Segment, type Step } from "../lib/avatar";

export type LibSentence = {
  id: string;
  text: string;
  samples: number;
  in_model: boolean;
  reliability: number | null;
  has_video: boolean;
  has_avatar?: boolean;
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
export default function SignLibrary() {
  const nav = useNavigate();
  const [items, setItems] = useState<LibSentence[]>([]);
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("reliable");
  const [playing, setPlaying] = useState<LibSentence | null>(null);

  useEffect(() => {
    api<LibSentence[]>("/api/sentences").then(setItems).catch(() => setItems([]));
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
