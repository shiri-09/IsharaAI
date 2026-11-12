import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { MessagesSquare, Trash2 } from "lucide-react";
import MessageBubble from "../components/MessageBubble";
import { api, type ConversationSummary, type Msg } from "../lib/api";

export default function History() {
  const { id } = useParams();
  const nav = useNavigate();
  const [items, setItems] = useState<ConversationSummary[]>([]);
  const [messages, setMessages] = useState<Msg[]>([]);

  useEffect(() => {
    api<ConversationSummary[]>("/api/conversations").then(setItems);
  }, []);

  useEffect(() => {
    if (!id) return setMessages([]);
    api<{ messages: Msg[] }>(`/api/conversations/${id}`)
      .then((c) => setMessages(c.messages))
      .catch(() => setMessages([]));
  }, [id]);

  async function remove(cid: number) {
    await api(`/api/conversations/${cid}`, { method: "DELETE" });
    setItems((x) => x.filter((c) => c.id !== cid));
    if (String(cid) === id) nav("/history");
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <div className="glass p-3">
        <p className="label px-2 pb-2 pt-1">Conversations</p>
        {items.length === 0 && <p className="px-2 py-4 text-sm text-zinc-600">Nothing yet.</p>}
        <div className="space-y-1">
          {items.map((c) => (
            <div
              key={c.id}
              className={`group flex items-center rounded-lg transition ${
                String(c.id) === id ? "bg-white/[0.08]" : "hover:bg-white/[0.03]"
              }`}
            >
              <Link to={`/history/${c.id}`} className="flex-1 truncate px-3 py-2 text-sm text-zinc-300">
                {c.title}
                <span className="block font-mono text-[10px] text-zinc-600">{new Date(c.created_at).toLocaleString()}</span>
              </Link>
              <button
                className="mr-2 rounded p-1 text-zinc-600 opacity-0 transition hover:text-red-400 group-hover:opacity-100"
                onClick={() => remove(c.id)}
                title="Delete"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>
      <div className="glass min-h-[420px] space-y-4 p-5">
        {!id ? (
          <div className="grid h-full min-h-[380px] place-items-center text-zinc-600">
            <MessagesSquare />
          </div>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} m={m} />)
        )}
      </div>
    </div>
  );
}
