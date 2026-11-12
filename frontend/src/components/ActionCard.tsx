import { CalendarCheck, CircleDashed, Loader2, ShieldCheck, X } from "lucide-react";
import type { AgentAction } from "../lib/api";

const FIELD_LABEL: Record<string, string> = { with: "Who", date: "Date", time: "Time", purpose: "Purpose" };

/** The task agent's current task. Nothing runs until the user presses Confirm (or selects Yes on the AAC board). */
export default function ActionCard({
  action,
  busy,
  onConfirm,
  onCancel,
  highlight,
}: {
  action: AgentAction;
  busy?: boolean;
  onConfirm?: () => void;
  onCancel?: () => void;
  highlight?: "confirm" | "cancel" | null;
}) {
  const ready = action.status === "awaiting_consent";
  const done = action.status === "done";
  return (
    <div
      className={`rounded-2xl border p-4 ${
        done ? "border-emerald-400/30 bg-emerald-400/[0.05]" : ready ? "border-amber-400/40 bg-amber-400/[0.06]" : "border-white/10 bg-white/[0.03]"
      }`}
    >
      <div className="mb-2 flex items-center gap-2">
        {done ? <ShieldCheck size={16} className="text-emerald-300" /> : ready ? <CalendarCheck size={16} className="text-amber-300" /> : <CircleDashed size={16} className="text-zinc-400" />}
        <span className="label !text-zinc-300">
          {done ? "Booked and verified" : ready ? "Waiting for your OK" : "Task agent is collecting details"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        {(["with", "date", "time", "purpose"] as const).map((k) =>
          action.fields[k] || action.missing.includes(k) ? (
            <div key={k} className="flex gap-2">
              <span className="w-16 shrink-0 text-zinc-500">{FIELD_LABEL[k]}</span>
              <span className={action.fields[k] ? "text-zinc-100" : "italic text-amber-300/80"}>{action.fields[k] || "needed"}</span>
            </div>
          ) : null,
        )}
      </div>
      {action.problem && <p className="mt-2 text-sm text-amber-300/90">{action.problem}</p>}
      {done && action.reference && <p className="mt-2 font-mono text-xs text-emerald-300">Reference {action.reference}</p>}
      {ready && (
        <div className="mt-3 flex gap-2">
          <button
            className={`btn btn-chrome flex-1 ${highlight === "confirm" ? "ring-4 ring-amber-400/60" : ""}`}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? <Loader2 size={14} className="animate-spin" /> : <CalendarCheck size={14} />} Confirm booking
          </button>
          <button className={`btn btn-ghost ${highlight === "cancel" ? "ring-4 ring-amber-400/60" : ""}`} onClick={onCancel} disabled={busy}>
            <X size={14} /> Cancel
          </button>
        </div>
      )}
    </div>
  );
}
