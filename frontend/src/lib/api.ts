export type User = { id: number; email: string; name: string };
export type Msg = {
  id: number;
  role: "user" | "assistant" | "signer" | "speaker";
  /** For signed turns: exactly what the recogniser produced. Never rewritten. */
  content: string;
  /** What the agent understood, when it differs from `content` (e.g. chosen from context). */
  interpreted?: string;
  meta?: {
    candidates?: { text: string; p: number }[];
    intent?: string;
    sources?: { title: string; source: string }[];
    action_id?: number | null;
    verified?: boolean;
  };
  source: string;
  translations: Record<string, string>;
  provider: string;
  created_at: string;
};
export type AgentAction = {
  id: number;
  kind: string;
  status: "draft" | "awaiting_consent" | "done" | "cancelled";
  fields: Record<string, string>;
  missing: string[];
  problem: string;
  summary: string;
  reference: string | null;
  result: string;
  conversation_id: number;
  created_at: string;
};
export type ChatResp = {
  conversation_id: number;
  context: string;
  user_message: Msg;
  reply: Msg;
  intent?: string;
  action: AgentAction | null;
};
export const CONTEXTS: Record<string, string> = {
  general: "General",
  campus: "Campus",
  classroom: "Classroom",
  office: "Office desk",
  hospital: "Hospital",
  home: "Home",
};
export type ConversationSummary = { id: number; title: string; context?: string; created_at: string };
export type Sentence = { id: string; text: string; samples: number; in_model?: boolean; reliability?: number | null; has_video?: boolean };
export type Prediction = {
  text: string;
  id: string;
  confidence: number;
  top: { id: string; text: string; p: number }[];
  accepted: boolean;
};

export const LANGS: Record<string, string> = { hi: "हिन्दी", kn: "ಕನ್ನಡ", ta: "தமிழ்", te: "తెలుగు" };

const TOKEN_KEY = "ishara.token";

export const tokenStore = {
  get(): string | null {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set(t: string) {
    try {
      localStorage.setItem(TOKEN_KEY, t);
    } catch {
      /* storage unavailable: session-only login */
    }
  },
  clear() {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* ignore */
    }
  },
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  const token = tokenStore.get();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, {
    method: opts.method ?? (opts.body !== undefined ? "POST" : "GET"),
    headers,
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* not json */
    }
    if (res.status === 401) tokenStore.clear();
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${path}?token=${encodeURIComponent(tokenStore.get() ?? "")}`;
}

export function ttsUrl(text: string, lang: string): string {
  return `/api/tts?text=${encodeURIComponent(text)}&lang=${lang}`;
}
