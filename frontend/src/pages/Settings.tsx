import { Check, Contrast, Gauge, Languages, SlidersHorizontal, Type, Volume2 } from "lucide-react";
import type { ReactNode } from "react";
import { LANGS } from "../lib/api";
import { usePrefs, type Prefs } from "../lib/prefs";

function Row({ icon, title, hint, children }: { icon: ReactNode; title: string; hint: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-4 border-b border-white/[0.06] py-4 last:border-0">
      <div className="grid h-10 w-10 place-items-center rounded-xl border border-white/10 bg-white/[0.03] text-zinc-300">{icon}</div>
      <div className="min-w-[200px] flex-1">
        <p className="text-zinc-100">{title}</p>
        <p className="text-sm text-zinc-500">{hint}</p>
      </div>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  );
}

function Choice<T extends string | number | boolean>({ value, current, label, onPick }: { value: T; current: T; label: string; onPick: (v: T) => void }) {
  const on = value === current;
  return (
    <button
      onClick={() => onPick(value)}
      className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm transition ${
        on ? "border-amber-400/50 bg-amber-400/10 text-amber-100" : "border-white/10 text-zinc-400 hover:text-zinc-200"
      }`}
    >
      {on && <Check size={14} />} {label}
    </button>
  );
}

/** Adaptive Access: each user's accessibility preferences, saved to their account and applied everywhere. */
export default function Settings() {
  const { prefs, save } = usePrefs();
  if (!prefs) return null;
  const set = <K extends keyof Prefs>(k: K) => (v: Prefs[K]) => save({ [k]: v } as Partial<Prefs>);
  const aac = prefs.mode === "aac";

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <div className="flex items-center gap-3">
        <SlidersHorizontal className="text-zinc-400" />
        <h1 className="text-xl font-semibold text-zinc-100">Accessibility preferences</h1>
      </div>
      <p className="text-sm text-zinc-500">Saved to your account, so they follow you on any device. Changes apply straight away.</p>
      <div className="glass px-5">
        <Row icon={<Languages size={18} />} title="Reply language" hint="Replies are shown and spoken in this language.">
          <Choice value="en" current={prefs.voice} label="English" onPick={set("voice")} />
          {Object.entries(LANGS).map(([k, v]) => (
            <Choice key={k} value={k} current={prefs.voice} label={v} onPick={set("voice")} />
          ))}
        </Row>
        <Row icon={<Volume2 size={18} />} title="Speak replies automatically" hint="Read every reply aloud as soon as it arrives.">
          <Choice value={true} current={prefs.auto_speak} label="On" onPick={set("auto_speak")} />
          <Choice value={false} current={prefs.auto_speak} label="Off" onPick={set("auto_speak")} />
        </Row>
        <Row icon={<Type size={18} />} title="Text size" hint="Bigger text across the whole app.">
          <Choice value="normal" current={prefs.text_size} label="Normal" onPick={set("text_size")} />
          <Choice value="large" current={prefs.text_size} label="Large" onPick={set("text_size")} />
          <Choice value="xlarge" current={prefs.text_size} label="Extra large" onPick={set("text_size")} />
        </Row>
        <Row icon={<Contrast size={18} />} title="High contrast" hint="Brighter text and stronger borders.">
          <Choice value={false} current={prefs.high_contrast} label="Off" onPick={set("high_contrast")} />
          <Choice value={true} current={prefs.high_contrast} label="On" onPick={set("high_contrast")} />
        </Row>
        {aac ? (
          <Row icon={<Gauge size={18} />} title="Scanning speed" hint="How long each phrase stays highlighted on the AAC board.">
            <Choice value={2400} current={prefs.scan_ms} label="Slow" onPick={set("scan_ms")} />
            <Choice value={1600} current={prefs.scan_ms} label="Normal" onPick={set("scan_ms")} />
            <Choice value={1000} current={prefs.scan_ms} label="Fast" onPick={set("scan_ms")} />
          </Row>
        ) : (
          <Row icon={<Gauge size={18} />} title="Avatar signing speed" hint="How fast the avatar signs replies back to you.">
            <Choice value="slow" current={prefs.avatar_speed} label="Slow" onPick={set("avatar_speed")} />
            <Choice value="normal" current={prefs.avatar_speed} label="Normal" onPick={set("avatar_speed")} />
            <Choice value="fast" current={prefs.avatar_speed} label="Fast" onPick={set("avatar_speed")} />
          </Row>
        )}
      </div>
    </div>
  );
}
