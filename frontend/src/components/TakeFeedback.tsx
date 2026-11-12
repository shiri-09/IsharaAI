import { useEffect, useRef, useState } from "react";
import { Check, GraduationCap, Loader2 } from "lucide-react";
import { api, type Prediction, type Sentence } from "../lib/api";

type Ft = { running: boolean; result?: { status: string; new_takes?: number; message?: string } | null };

/**
 * "Was that right?" for a signed sentence. A correction (or a confirmation) keeps the take as the person's own training sample;
 * "Learn from my takes" then fine-tunes the model on them in about 15 seconds, so it gets better at THEIR signing.
 */
export default function TakeFeedback({ pred }: { pred: Prediction | null }) {
  const [list, setList] = useState<Sentence[]>([]);
  const [saved, setSaved] = useState<{ take: string; text: string } | null>(null);
  const [waiting, setWaiting] = useState(0);
  const [ft, setFt] = useState<Ft | null>(null);
  const [err, setErr] = useState("");
  const seen = useRef("");

  useEffect(() => {
    api<Sentence[]>("/api/sentences")
      .then((s) => setList(s.filter((x) => x.in_model && x.active !== false)))
      .catch(() => undefined);
  }, []);

  // a new prediction resets the card
  useEffect(() => {
    if (pred?.take && pred.take !== seen.current) {
      seen.current = pred.take;
      setSaved(null);
      setErr("");
    }
  }, [pred?.take]);

  useEffect(() => {
    if (!ft?.running) return;
    const t = setInterval(async () => {
      try {
        const s = await api<Ft>("/api/model/train");
        setFt(s);
        if (!s.running) setWaiting(0);
      } catch {
        /* keep polling */
      }
    }, 1500);
    return () => clearInterval(t);
  }, [ft?.running]);

  if (!pred?.take) return null;

  async function pick(id: string, text: string) {
    setErr("");
    try {
      await api("/api/recognize/correct", { body: { take: pred!.take, sentence_id: id } });
      setSaved({ take: pred!.take!, text });
      setWaiting((n) => n + 1);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }
  async function learn() {
    setErr("");
    try {
      await api("/api/model/train?mode=finetune", { method: "POST" });
      setFt({ running: true });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  const here = saved && saved.take === pred.take;
  return (
    <div className="space-y-2 rounded-xl border border-white/10 bg-black/30 p-3 text-sm">
      {!here ? (
        <>
          <p className="flex flex-wrap items-center gap-2 text-zinc-300">
            Heard: <b className="text-zinc-100">{pred.text}</b>
            <button className="btn btn-ghost !px-2 !py-1 text-xs" onClick={() => pick(pred.id, pred.text)}>
              <Check size={12} /> Yes, right
            </button>
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-zinc-500">Not right? I signed:</span>
            {pred.top
              .filter((t) => t.id !== pred.id)
              .map((t) => (
                <button key={t.id} className="btn btn-ghost !px-2.5 !py-1 text-xs" onClick={() => pick(t.id, t.text)}>
                  {t.text}
                </button>
              ))}
            <select
              className="input !w-auto !py-1 text-xs"
              value=""
              onChange={(e) => {
                const s = list.find((x) => x.id === e.target.value);
                if (s) pick(s.id, s.text);
              }}
            >
              <option value="">another sentence…</option>
              {list.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.text}
                </option>
              ))}
            </select>
          </div>
        </>
      ) : (
        <p className="flex items-center gap-2 text-emerald-300">
          <Check size={14} /> Saved your version of “{saved!.text}”.
        </p>
      )}
      {waiting > 0 && (
        <div className="flex flex-wrap items-center gap-2 border-t border-white/10 pt-2">
          <button className="btn btn-chrome !px-3 !py-1.5 text-xs" onClick={learn} disabled={ft?.running}>
            {ft?.running ? <Loader2 size={13} className="animate-spin" /> : <GraduationCap size={13} />}
            {ft?.running ? "Learning your signing…" : `Learn from my ${waiting} take${waiting === 1 ? "" : "s"} (about 15 s)`}
          </button>
          {ft && !ft.running && ft.result?.status === "ok" && <span className="text-xs text-emerald-300">Learned {ft.result.new_takes} takes. Try signing it again.</span>}
          {ft && !ft.running && ft.result && ft.result.status !== "ok" && <span className="text-xs text-zinc-500">{ft.result.message}</span>}
        </div>
      )}
      {err && <p className="text-xs text-red-400/90">{err}</p>}
    </div>
  );
}
