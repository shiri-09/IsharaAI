import { useMemo } from "react";
import { CornerDownLeft, Delete, Eraser, Volume2 } from "lucide-react";

/**
 * On-screen keyboard for eye typing. Every key has `data-gaze`, so looking at it (or raising the eyebrows) types it.
 * Big keys, a row of word suggestions, and Space / Backspace / Clear / Speak / Send.
 */
const ROWS = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"];

const COMMON = [
  "I", "you", "the", "to", "and", "a", "is", "am", "are", "can", "please", "thank", "help", "need", "want", "water", "food", "doctor",
  "professor", "meeting", "tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", "friday", "morning", "afternoon", "remind",
  "book", "order", "lunch", "interpreter", "library", "washroom", "where", "when", "what", "how", "yes", "no", "hello", "sorry", "tell",
  "message", "class", "seminar", "assignment", "submit", "meet", "with", "at", "on", "for", "me", "my", "about", "project", "internship",
];

function suggest(text: string): string[] {
  const m = text.match(/([A-Za-z']*)$/);
  const prefix = (m?.[1] ?? "").toLowerCase();
  if (!prefix) return ["I need", "Please", "Thank you", "Yes", "No"].slice(0, 5);
  const hits = COMMON.filter((w) => w.toLowerCase().startsWith(prefix) && w.toLowerCase() !== prefix);
  return hits.slice(0, 5);
}

const KEY = "flex min-h-[clamp(42px,5.6vh,62px)] items-center justify-center rounded-xl border border-white/15 bg-white/[0.05] px-2 text-xl font-medium text-zinc-100 hover:bg-white/[0.09]";

export default function GazeKeyboard({
  value,
  onChange,
  onSubmit,
  onSpeak,
  submitLabel = "Send",
  busy,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onSpeak?: () => void;
  submitLabel?: string;
  busy?: boolean;
}) {
  const words = useMemo(() => suggest(value), [value]);
  const press = (ch: string) => onChange(value + (value.length === 0 || /[.!?]\s$/.test(value) ? ch.toUpperCase() : ch.toLowerCase()));
  const pickWord = (w: string) => onChange(value.replace(/([A-Za-z']*)$/, "") + w + " ");
  return (
    <div className="space-y-1.5 select-none">
      <div className="grid grid-cols-5 gap-1.5">
        {words.map((w) => (
          <button key={w} type="button" data-gaze data-dwell="1000" onClick={() => pickWord(w)} className={`${KEY} !min-h-[clamp(38px,4.8vh,52px)] !text-base !text-amber-200`}>
            {w}
          </button>
        ))}
      </div>
      {ROWS.map((row, i) => (
        <div key={row} className="flex justify-center gap-1.5" style={{ paddingInline: i === 1 ? 24 : i === 2 ? 64 : 0 }}>
          {row.split("").map((c) => (
            <button key={c} type="button" data-gaze data-dwell="800" onClick={() => press(c)} className={`${KEY} flex-1`}>
              {c}
            </button>
          ))}
        </div>
      ))}
      <div className="grid grid-cols-[1.1fr_3fr_1fr_1fr_1.4fr_1.4fr] gap-1.5">
        <button type="button" data-gaze data-dwell="900" onClick={() => press(",")} className={KEY}>,</button>
        <button type="button" data-gaze data-dwell="800" onClick={() => onChange(value + " ")} className={KEY}>space</button>
        <button type="button" data-gaze data-dwell="900" onClick={() => onChange(value + ".")} className={KEY}>.</button>
        <button type="button" data-gaze data-dwell="900" onClick={() => onChange(value + "?")} className={KEY}>?</button>
        <button type="button" data-gaze data-back data-dwell="900" onClick={() => onChange(value.slice(0, -1))} className={`${KEY} !text-base`}>
          <Delete size={20} /> <span className="ml-1">Back</span>
        </button>
        <button type="button" data-gaze data-dwell="1300" onClick={() => onChange("")} className={`${KEY} !text-base`}>
          <Eraser size={20} /> <span className="ml-1">Clear</span>
        </button>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {onSpeak && (
          <button type="button" data-gaze data-dwell="1300" onClick={onSpeak} disabled={!value.trim()} className={`${KEY} !text-lg disabled:opacity-40`}>
            <Volume2 size={22} /> <span className="ml-2">Speak it</span>
          </button>
        )}
        <button
          type="button"
          data-gaze
          data-dwell="1500"
          onClick={onSubmit}
          disabled={!value.trim() || busy}
          className={`${KEY} !text-lg !border-amber-400/50 !bg-amber-400/10 !text-amber-100 disabled:opacity-40 ${onSpeak ? "" : "col-span-2"}`}
        >
          <CornerDownLeft size={22} /> <span className="ml-2">{submitLabel}</span>
        </button>
      </div>
    </div>
  );
}
