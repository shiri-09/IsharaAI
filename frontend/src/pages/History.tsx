import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Bot, MessagesSquare, Radio, Sparkles, Trash2, type LucideIcon } from "lucide-react";
import MessageBubble from "../components/MessageBubble";
import { api, type ConversationSummary, type Msg } from "../lib/api";
import { RunView, STATUS, type Run } from "./Agent";

type Mode = "ai" | "live" | "agent";
type Item = ConversationSummary & { mode?: Mode; messages?: number; updated_at?: string };

const MODES: Record<Mode, { label: string; icon: LucideIcon; badge: string }> = {
  ai: { label: "AI conversation", icon: Sparkles, badge: "border-white/15 text-zinc-300" },
  live: { label: "Real-time speech", icon: Radio, badge: "border-sky-300/30 text-sky-200" },
  agent: { label: "Agentic orchestration", icon: Bot, badge: "border-amber-400/40 text-amber-200" },
};

function LiveTurn({ m }: { m: Msg }) {
  const signer = m.role === "signer";
  const local = Object.values(m.translations || {})[0];
  return (
    <div className={`flex ${signer ? "justify-start" : "justify-end"}`}>
      <div className={`max-w-[80%] rounded-2xl border px-4 py-2.5 ${signer ? "border-amber-400/25 bg-amber-400/[0.05]" : "border-white/10 bg-white/[0.04]"}`}>
        <p className="text-[11px] uppercase tracking-wider text-zinc-500">
          {signer ? "Signer" : "Hearing person"} · {m.source} · {new Date(m.created_at).toLocaleTimeString()}
        </p>
        <p className="text-zinc-100">{m.content}</p>
        {!signer && m.interpreted && <p className="text-sm text-zinc-400">English: {m.interpreted}</p>}
        {signer && local && <p className="text-sm text-zinc-400">{local}</p>}
      </div>
    </div>
  );
}

export default function History() {
  const { id } = useParams();
  const nav = useNavigate();
  const [items, setItems] = useState<Item[]>([]);
  const [filter, setFilter] = useState<Mode | "all">("all");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [mode, setMode] = useState<Mode>("ai");
  const [runs, setRuns] = useState<Run[]>([]);

  useEffect(() => {
    api<Item[]>("/api/conversations").then(setItems);
  }, []);

  useEffect(() => {
    if (!id) {
      setMessages([]);
      setRuns([]);
      return;
    }
    api<{ messages: Msg[]; mode?: Mode }>(`/api/conversations/${id}`)
      .then((c) => {
        setMessages(c.messages);
        setMode(c.mode ?? "ai");
        if (c.mode === "agent") api<Run[]>("/api/agent/runs").then((rs) => setRuns(rs.filter((r) => String(r.conversation_id) === id)));
        else setRuns([]);
      })
      .catch(() => setMessages([]));
  }, [id]);

  async function remove(cid: number) {
    await api(`/api/conversations/${cid}`, { method: "DELETE" });
    setItems((x) => x.filter((c) => c.id !== cid));
    if (String(cid) === id) nav("/history");
  }

  const shown = useMemo(() => items.filter((c) => filter === "all" || (c.mode ?? "ai") === filter), [items, filter]);
  const count = (m: Mode) => items.filter((c) => (c.mode ?? "ai") === m).length;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {(["all", "ai", "live", "agent"] as const).map((k) => {
          const Icon = k === "all" ? MessagesSquare : MODES[k].icon;
          return (
            <button
              key={k}
              onClick={() => setFilter(k)}
              className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm ${filter === k ? "border-zinc-300/40 bg-white/10 text-zinc-100" : "border-white/10 text-zinc-500"}`}
            >
              <Icon size={14} /> {k === "all" ? `All (${items.length})` : `${MODES[k].label} (${count(k)})`}
            </button>
          );
        })}
      </div>
      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        <div className="glass p-3">
          {shown.length === 0 && <p className="px-2 py-4 text-sm text-zinc-600">Nothing yet.</p>}
          <div className="max-h-[72vh] space-y-1 overflow-y-auto">
            {shown.map((c) => {
              const m = MODES[(c.mode ?? "ai") as Mode];
              const Icon = m.icon;
              return (
                <div key={c.id} className={`group flex items-center rounded-lg transition ${String(c.id) === id ? "bg-white/[0.08]" : "hover:bg-white/[0.03]"}`}>
                  <Link to={`/history/${c.id}`} className="min-w-0 flex-1 px-3 py-2">
                    <span className="flex items-center gap-2">
                      <Icon size={13} className="shrink-0 text-zinc-500" />
                      <span className="truncate text-sm text-zinc-300">{c.title.replace(/^Live: /, "")}</span>
                    </span>
                    <span className="mt-0.5 flex items-center gap-2 pl-5">
                      <span className={`rounded-full border px-1.5 text-[10px] ${m.badge}`}>{m.label}</span>
                      <span className="font-mono text-[10px] text-zinc-600">
                        {new Date(c.updated_at ?? c.created_at).toLocaleString()}
                        {c.messages != null && ` · ${c.messages} msg`}
                      </span>
                    </span>
                  </Link>
                  <button className="mr-2 rounded p-1 text-zinc-600 opacity-0 transition hover:text-red-400 group-hover:opacity-100" onClick={() => remove(c.id)} title="Delete">
                    <Trash2 size={14} />
                  </button>
                </div>
              );
            })}
          </div>
        </div>
        <div className="glass min-h-[420px] space-y-4 p-5">
          {!id ? (
            <div className="grid h-full min-h-[380px] place-items-center text-zinc-600">
              <MessagesSquare />
            </div>
          ) : mode === "agent" && runs.length ? (
            runs.map((r) => (
              <div key={r.id} className="space-y-2">
                <p className="font-mono text-[11px] text-zinc-600">
                  Run #{r.id} · {new Date(r.created_at).toLocaleString()} · <span className={STATUS[r.status].cls.split(" ").pop()}>{STATUS[r.status].text}</span>
                </p>
                <RunView run={r} />
              </div>
            ))
          ) : mode === "live" ? (
            messages.map((m) => <LiveTurn key={m.id} m={m} />)
          ) : (
            messages.map((m) => <MessageBubble key={m.id} m={m} />)
          )}
        </div>
      </div>
    </div>
  );
}
