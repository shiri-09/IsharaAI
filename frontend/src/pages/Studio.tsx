import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import type { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";
import { Hand, Loader2, Plus, Send, Sparkles } from "lucide-react";
import CameraStage from "../components/CameraStage";
import MessageBubble, { speak } from "../components/MessageBubble";
import { api, LANGS, wsUrl, type Msg, type Prediction } from "../lib/api";
import { toFrame } from "../lib/keypoints";
import { useHolistic } from "../lib/useHolistic";

type ChatResp = { conversation_id: number; user_message: Msg; reply: Msg };

/** Grab a small frame of the webcam as a canvas. */
function grabFrame(video: HTMLVideoElement | null, w = 256): HTMLCanvasElement | null {
  if (!video || !video.videoWidth) return null;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = Math.round((video.videoHeight / video.videoWidth) * w);
  c.getContext("2d")?.drawImage(video, 0, 0, c.width, c.height);
  return c;
}

/** Beginning / middle / end of the signing, side by side, as one JPEG data URL for the vision model. */
function filmstrip(frames: HTMLCanvasElement[]): string | null {
  if (!frames.length) return null;
  const pick = frames.length <= 3 ? frames : [frames[0], frames[Math.floor(frames.length / 2)], frames[frames.length - 1]];
  const out = document.createElement("canvas");
  out.width = pick.reduce((a, f) => a + f.width, 0);
  out.height = Math.max(...pick.map((f) => f.height));
  const ctx = out.getContext("2d")!;
  let x = 0;
  for (const f of pick) {
    ctx.drawImage(f, x, 0);
    x += f.width;
  }
  return out.toDataURL("image/jpeg", 0.75);
}

export default function Studio() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [langs, setLangs] = useState<string[]>(Object.keys(LANGS));
  const [voice, setVoice] = useState<string>("hi");
  const [autoSend, setAutoSend] = useState(true);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [signing, setSigning] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const scroller = useRef<HTMLDivElement>(null);

  const state = useRef({ conversationId, langs, voice, autoSend, sending });
  state.current = { conversationId, langs, voice, autoSend, sending };

  useEffect(() => {
    api<{ ready: boolean }>("/api/model")
      .then((m) => setModelReady(m.ready))
      .catch(() => setModelReady(false));
  }, []);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const snapshot = useRef<string | null>(null);
  const frames = useRef<HTMLCanvasElement[]>([]);
  const grabTimer = useRef<number | undefined>(undefined);

  const send = useCallback(async (msg: string, source: "sign" | "text", candidates: Prediction["top"] = []) => {
    const s = state.current;
    if (!msg.trim() || s.sending) return;
    setSending(true);
    setError("");
    try {
      const r = await api<ChatResp>("/api/chat", {
        body: {
          text: msg,
          conversation_id: s.conversationId,
          source,
          languages: s.langs,
          candidates: candidates.map((c) => ({ text: c.text, p: c.p })),
          image: source === "sign" ? snapshot.current : null,
        },
      });
      setConversationId(r.conversation_id);
      setMessages((prev) => [...prev, r.user_message, r.reply]);
      if (s.voice !== "off") {
        const spoken = s.voice === "en" ? r.reply.content : r.reply.translations[s.voice];
        if (spoken) speak(spoken, s.voice).catch(() => undefined);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }, []);

  const onResult = useCallback((r: HolisticLandmarkerResult) => {
    const sock = ws.current;
    if (sock?.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ type: "frame", kp: toFrame(r) }));
  }, []);

  const cam = useHolistic(onResult);

  // open the recognition socket while the camera runs
  useEffect(() => {
    if (cam.status !== "running") return;
    const sock = new WebSocket(wsUrl("/ws/recognize"));
    ws.current = sock;
    sock.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "state") {
        setSigning(m.signing);
        // sample webcam frames while signing; they become a filmstrip for the multimodal model
        window.clearInterval(grabTimer.current);
        if (m.signing) {
          frames.current = [];
          grabTimer.current = window.setInterval(() => {
            const f = grabFrame(cam.videoRef.current);
            if (f && frames.current.length < 12) frames.current.push(f);
          }, 350);
        } else {
          snapshot.current = filmstrip(frames.current);
        }
      } else if (m.type === "prediction") {
        setPrediction(m);
        if (m.accepted && state.current.autoSend) send(m.text, "sign", m.top);
      } else if (m.type === "error") setError(m.message);
    };
    sock.onclose = (ev) => {
      if (ev.code === 4401) setError("Session expired, please log in again.");
    };
    return () => {
      window.clearInterval(grabTimer.current);
      sock.close();
      ws.current = null;
      setSigning(false);
    };
  }, [cam.status, cam.videoRef, send]);

  function submit(e: FormEvent) {
    e.preventDefault();
    const t = text;
    setText("");
    send(t, "text");
  }

  function newChat() {
    setMessages([]);
    setConversationId(null);
    setPrediction(null);
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
      <div className="space-y-4">
        <CameraStage
          {...cam}
          onStart={cam.start}
          onStop={cam.stop}
          overlay={
            <div className="absolute inset-x-3 bottom-3">
              <div
                className={`flex items-center gap-3 rounded-xl border px-4 py-3 backdrop-blur transition ${
                  signing ? "border-zinc-300/40 bg-zinc-200/10" : "border-white/10 bg-black/60"
                }`}
              >
                <Hand size={16} className={signing ? "animate-pulse text-zinc-100" : "text-zinc-500"} />
                <span className="text-sm text-zinc-300">
                  {signing ? "Reading your signs… lower your hands to finish" : "Raise your hands to start signing"}
                </span>
              </div>
            </div>
          }
          footer={
            <div className="space-y-3">
              {modelReady === false && (
                <p className="text-sm text-amber-300/80">
                  No trained model yet — record samples on <b>Collect &amp; Train</b>. You can still type below.
                </p>
              )}
              {prediction ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="label">Last recognised</span>
                    <span className="font-mono text-xs text-zinc-500">{(prediction.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <p className={`text-lg ${prediction.accepted ? "text-zinc-100" : "text-zinc-500"}`}>{prediction.text}</p>
                  <div className="h-1 overflow-hidden rounded-full bg-white/5">
                    <div
                      className="h-full bg-gradient-to-r from-zinc-500 to-zinc-100"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    />
                  </div>
                  {(!prediction.accepted || !autoSend) && (
                    <div className="flex flex-wrap gap-2 pt-1">
                      {prediction.top.map((t) => (
                        <button key={t.id} className="btn btn-ghost px-3 py-1.5 text-xs" onClick={() => send(t.text, "sign")}>
                          {t.text} <span className="font-mono text-zinc-600">{(t.p * 100).toFixed(0)}%</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-zinc-600">Recognised sentences appear here.</p>
              )}
              <label className="flex items-center gap-2 text-xs text-zinc-500">
                <input type="checkbox" checked={autoSend} onChange={(e) => setAutoSend(e.target.checked)} />
                Auto-send confident recognitions to the conversation
              </label>
            </div>
          }
        />
      </div>

      <div className="glass flex h-[calc(100vh-8.5rem)] min-h-[520px] flex-col">
        <div className="flex flex-wrap items-center gap-3 border-b border-white/[0.06] px-5 py-4">
          <Sparkles size={16} className="text-zinc-400" />
          <h2 className="font-medium text-zinc-100">Conversation</h2>
          <button className="btn btn-ghost ml-auto px-3 py-1.5 text-xs" onClick={newChat}>
            <Plus size={14} /> New
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2 border-b border-white/[0.06] px-5 py-3">
          <span className="label mr-1">Translate</span>
          {Object.entries(LANGS).map(([k, v]) => (
            <button
              key={k}
              onClick={() => setLangs((l) => (l.includes(k) ? l.filter((x) => x !== k) : [...l, k]))}
              className={`rounded-full border px-2.5 py-0.5 text-xs transition ${
                langs.includes(k) ? "border-zinc-300/40 bg-white/10 text-zinc-100" : "border-white/10 text-zinc-500"
              }`}
            >
              {v}
            </button>
          ))}
          <span className="label ml-auto mr-1">Voice</span>
          <select
            value={voice}
            onChange={(e) => setVoice(e.target.value)}
            className="rounded-lg border border-white/10 bg-black/60 px-2 py-1 text-xs text-zinc-300"
          >
            <option value="off">Off</option>
            <option value="en">English</option>
            {Object.entries(LANGS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </div>

        <div ref={scroller} className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
          {messages.length === 0 && !sending && (
            <div className="grid h-full place-items-center text-center">
              <div>
                <p className="chrome-text text-2xl font-semibold">Start signing</p>
                <p className="mt-2 text-sm text-zinc-600">or type a message to test the dialogue.</p>
              </div>
            </div>
          )}
          {messages.map((m) => (
            <MessageBubble key={m.id} m={m} />
          ))}
          {sending && (
            <div className="flex items-center gap-2 text-sm text-zinc-500">
              <Loader2 size={14} className="animate-spin" /> Thinking and translating…
            </div>
          )}
        </div>

        {error && <p className="px-5 pb-2 text-sm text-red-400/90">{error}</p>}
        <form onSubmit={submit} className="flex gap-2 border-t border-white/[0.06] p-4">
          <input className="input" placeholder="Type a message…" value={text} onChange={(e) => setText(e.target.value)} />
          <button className="btn btn-chrome px-4" disabled={sending || !text.trim()}>
            <Send size={16} />
          </button>
        </form>
      </div>
    </div>
  );
}
