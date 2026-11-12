import { useEffect, useRef, useState } from "react";
import { Loader2, MessagesSquare, Plus } from "lucide-react";
import ActionCard from "../components/ActionCard";
import GazeKeyboard from "../components/GazeKeyboard";
import { speak } from "../components/MessageBubble";
import { api, type AgentAction, type ChatResp, type Msg } from "../lib/api";
import { usePrefs } from "../lib/prefs";

const QUICK = [
  "Yes", "No", "Thank you", "I need water", "I need the washroom", "Can you repeat that?", "I am in pain", "I need help",
  "I want to meet my professor", "Where is the library?", "Please call my family", "I need a sign language interpreter",
];

/** AI conversation for eye-tracking users: choose a phrase or type with the on-screen keyboard; the AI replies out loud. */
export default function AacChat() {
  const { prefs } = usePrefs();
  const voice = prefs?.voice || "hi";
  const [messages, setMessages] = useState<Msg[]>([]);
  const [conv, setConv] = useState<number | null>(null);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [action, setAction] = useState<AgentAction | null>(null);
  const [error, setError] = useState("");
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  async function say(m: Msg) {
    if (prefs?.auto_speak === false) return;
    const spoken = voice === "en" ? m.content : m.translations?.[voice] || m.content;
    await speak(spoken, voice === "en" || !m.translations?.[voice] ? "en" : voice).catch(() => undefined);
  }

  async function send(t: string) {
    if (!t.trim() || busy) return;
    setBusy(true);
    setError("");
    setText("");
    speak(t, "en").catch(() => undefined);
    try {
      const r = await api<ChatResp>("/api/chat", {
        body: { text: t, conversation_id: conv, context: "campus", source: "aac", languages: [voice].filter((l) => l !== "en") },
      });
      setConv(r.conversation_id);
      setMessages((m) => [...m, r.user_message, r.reply]);
      setAction(r.action);
      await say(r.reply);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function decide(kind: "confirm" | "cancel") {
    if (!action) return;
    setBusy(true);
    try {
      const r = await api<{ action: AgentAction; message: Msg }>(`/api/actions/${action.id}/${kind}`, { body: { languages: [voice].filter((l) => l !== "en") } });
      setAction(r.action.status === "cancelled" ? null : r.action);
      setMessages((m) => [...m, r.message]);
      await say(r.message);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const consent = action?.status === "awaiting_consent";
  return (
    <div className="flex h-[calc(100vh-3.2rem)] min-h-[560px] flex-col gap-2">
      <div className="flex items-center gap-3">
        <MessagesSquare className="text-amber-300" />
        <h1 className="text-xl font-semibold text-zinc-100">AI conversation</h1>
        <span className="text-sm text-zinc-500">Look at a phrase or type with your eyes. The AI answers aloud.</span>
        <button className="btn btn-ghost ml-auto px-3 py-2" data-gaze data-dwell="1500" onClick={() => { setMessages([]); setConv(null); setAction(null); }}>
          <Plus size={16} /> New
        </button>
      </div>

      <div ref={scroller} className="glass min-h-[90px] flex-1 space-y-2 overflow-y-auto p-4">
        {messages.length === 0 && !busy && <p className="text-xl text-zinc-600">Your conversation appears here.</p>}
        {messages.map((m) => (
          <div key={m.id} className={m.role === "user" ? "text-right" : ""}>
            <p className={`inline-block max-w-[85%] rounded-2xl px-4 py-1.5 text-xl ${m.role === "user" ? "bg-white/10 text-zinc-100" : "border border-white/10 text-zinc-200"}`}>{m.content}</p>
            {m.role !== "user" && voice !== "en" && m.translations?.[voice] && <p className="mt-1 text-lg text-zinc-400">{m.translations[voice]}</p>}
          </div>
        ))}
        {busy && (
          <p className="flex items-center gap-2 text-lg text-zinc-400">
            <Loader2 size={18} className="animate-spin" /> Thinking…
          </p>
        )}
      </div>

      {action && <ActionCard action={action} busy={busy} highlight={null} />}
      {consent ? (
        <div className="grid grid-cols-2 gap-4">
          <button data-gaze data-yes data-dwell="1800" disabled={busy} onClick={() => decide("confirm")} className="min-h-[18vh] rounded-2xl border border-emerald-400/40 bg-emerald-400/10 text-3xl font-semibold text-emerald-100">
            Yes, book it
          </button>
          <button data-gaze data-no data-dwell="1800" disabled={busy} onClick={() => decide("cancel")} className="min-h-[18vh] rounded-2xl border border-white/15 bg-white/[0.04] text-3xl font-semibold text-zinc-200">
            No, cancel
          </button>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-1.5 md:grid-cols-4">
            {QUICK.slice(0, 8).map((q) => (
              <button key={q} data-gaze data-dwell="1400" disabled={busy} onClick={() => send(q)} className="min-h-[clamp(40px,5.4vh,56px)] rounded-xl border border-white/10 bg-white/[0.03] px-3 text-left text-base text-zinc-200">
                {q}
              </button>
            ))}
          </div>
          <div className="glass space-y-1.5 p-3">
            <div className="min-h-[46px] rounded-xl border border-white/15 bg-black/50 px-4 py-1.5 text-2xl text-zinc-100">
              {text || <span className="text-zinc-600">Type here with your eyes…</span>}
              <span className="animate-pulse text-amber-300">|</span>
            </div>
            <GazeKeyboard value={text} onChange={setText} busy={busy} submitLabel="Send" onSubmit={() => send(text)} onSpeak={() => speak(text, "en").catch(() => undefined)} />
          </div>
        </>
      )}
      {error && <p className="text-base text-red-400/90">{error}</p>}
    </div>
  );
}
