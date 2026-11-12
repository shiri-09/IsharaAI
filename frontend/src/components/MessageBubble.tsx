import { useState } from "react";
import { BookOpen, Hand, Keyboard, Loader2, Mic, ScanFace, ShieldCheck, Volume2 } from "lucide-react";
import { LANGS, ttsUrl, type Msg } from "../lib/api";

let current: HTMLAudioElement | null = null;

export function speak(text: string, lang: string): Promise<void> {
  current?.pause();
  const a = new Audio(ttsUrl(text, lang));
  current = a;
  return new Promise((resolve, reject) => {
    a.onended = () => resolve();
    a.onerror = () => reject(new Error("TTS failed"));
    a.play().catch(reject);
  });
}

function SpeakButton({ text, lang }: { text: string; lang: string }) {
  const [busy, setBusy] = useState(false);
  return (
    <button
      className="rounded-md p-1 text-zinc-500 transition hover:bg-white/5 hover:text-zinc-200"
      title="Play audio"
      onClick={async () => {
        setBusy(true);
        try {
          await speak(text, lang);
        } catch {
          /* surfaced as no audio */
        } finally {
          setBusy(false);
        }
      }}
    >
      {busy ? <Loader2 size={14} className="animate-spin" /> : <Volume2 size={14} />}
    </button>
  );
}

const SOURCE_LABEL: Record<string, { icon: typeof Hand; text: string }> = {
  sign: { icon: Hand, text: "Recognised from sign" },
  aac: { icon: ScanFace, text: "Chosen on AAC board" },
  voice: { icon: Mic, text: "Spoken" },
  text: { icon: Keyboard, text: "Typed" },
};

export default function MessageBubble({ m }: { m: Msg }) {
  const langs = Object.keys(m.translations || {}).filter((k) => k in LANGS);
  const [tab, setTab] = useState<string>("en");
  const isUser = m.role === "user";
  const text = tab === "en" ? m.content : m.translations[tab];
  const src = SOURCE_LABEL[m.source] ?? SOURCE_LABEL.text;
  const SrcIcon = src.icon;
  const sources = m.meta?.sources ?? [];

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-gradient-to-b from-zinc-200 to-zinc-400 text-zinc-950"
            : "border border-white/[0.08] bg-white/[0.035] text-zinc-200"
        }`}
      >
        {isUser && (
          <div className="mb-1 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-700">
            <SrcIcon size={11} /> {src.text}
          </div>
        )}
        <div className="flex items-start gap-2">
          <p className="flex-1 whitespace-pre-wrap">{text}</p>
          {!isUser && <SpeakButton text={text} lang={tab} />}
        </div>
        {isUser && m.interpreted && (
          <p className="mt-2 rounded-lg bg-black/10 px-2 py-1 text-xs text-zinc-800" title="The assistant acted on this meaning. Your recognised words above are kept unchanged.">
            Understood from context as: <b>{m.interpreted}</b>
          </p>
        )}
        {!isUser && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-white/[0.06] pt-2">
            {["en", ...langs].map((k) => (
              <button
                key={k}
                onClick={() => setTab(k)}
                className={`rounded-md px-2 py-0.5 text-[11px] transition ${
                  tab === k ? "bg-white/10 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {k === "en" ? "English" : LANGS[k]}
              </button>
            ))}
            {m.meta?.verified && (
              <span className="flex items-center gap-1 rounded-md bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-300">
                <ShieldCheck size={11} /> verified
              </span>
            )}
            {m.provider && <span className="ml-auto font-mono text-[10px] text-zinc-600">{m.provider}</span>}
          </div>
        )}
        {!isUser && sources.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <BookOpen size={11} className="text-zinc-500" />
            {sources.map((s) => (
              <span key={s.title} className="rounded-md border border-white/10 px-1.5 py-0.5 text-[10px] text-zinc-400" title={s.source}>
                {s.title}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
