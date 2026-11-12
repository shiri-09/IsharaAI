import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import { api, tokenStore, type User } from "./api";

type AuthConfig = { provider: "supabase" | "local"; supabase_url: string | null; supabase_anon_key: string | null };

type AuthCtx = {
  user: User | null;
  loading: boolean;
  provider: "supabase" | "local";
  login: (email: string, password: string) => Promise<void>;
  /** Resolves to a notice string when the user must confirm their email first. */
  register: (name: string, email: string, password: string) => Promise<string | void>;
  logout: () => void;
};

const Ctx = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [provider, setProvider] = useState<"supabase" | "local">("local");
  const sb = useRef<SupabaseClient | null>(null);

  const loadMe = useCallback(async () => {
    try {
      setUser(await api<User>("/api/auth/me"));
    } catch {
      tokenStore.clear();
      setUser(null);
    }
  }, []);

  useEffect(() => {
    let unsub: (() => void) | undefined;
    (async () => {
      let cfg: AuthConfig = { provider: "local", supabase_url: null, supabase_anon_key: null };
      try {
        cfg = await api<AuthConfig>("/api/auth/config");
      } catch {
        /* backend unreachable: stay on local */
      }
      setProvider(cfg.provider);
      if (cfg.provider === "supabase" && cfg.supabase_url && cfg.supabase_anon_key) {
        const client = createClient(cfg.supabase_url, cfg.supabase_anon_key);
        sb.current = client;
        const { data } = await client.auth.getSession();
        if (data.session) {
          tokenStore.set(data.session.access_token);
          await loadMe();
        }
        // keep our bearer token in sync with Supabase's auto-refresh
        const { data: sub } = client.auth.onAuthStateChange((_evt, session) => {
          if (session) tokenStore.set(session.access_token);
          else {
            tokenStore.clear();
            setUser(null);
          }
        });
        unsub = () => sub.subscription.unsubscribe();
      } else if (tokenStore.get()) {
        await loadMe();
      }
      setLoading(false);
    })();
    return () => unsub?.();
  }, [loadMe]);

  const login = useCallback(
    async (email: string, password: string) => {
      if (sb.current) {
        const { data, error } = await sb.current.auth.signInWithPassword({ email, password });
        if (error) throw new Error(error.message);
        tokenStore.set(data.session.access_token);
        await loadMe();
        return;
      }
      const r = await api<{ token: string; user: User }>("/api/auth/login", { body: { email, password } });
      tokenStore.set(r.token);
      setUser(r.user);
    },
    [loadMe],
  );

  const register = useCallback(
    async (name: string, email: string, password: string) => {
      if (sb.current) {
        const { data, error } = await sb.current.auth.signUp({ email, password, options: { data: { name } } });
        if (error) throw new Error(error.message);
        if (!data.session) return "Check your inbox to confirm your email, then sign in.";
        tokenStore.set(data.session.access_token);
        await loadMe();
        return;
      }
      const r = await api<{ token: string; user: User }>("/api/auth/register", { body: { name, email, password } });
      tokenStore.set(r.token);
      setUser(r.user);
    },
    [loadMe],
  );

  const logout = useCallback(() => {
    sb.current?.auth.signOut();
    tokenStore.clear();
    try {
      sessionStorage.removeItem("ishara.intro.seen"); // the intro plays again after the next login
    } catch {
      /* ignore */
    }
    setUser(null);
  }, []);

  return <Ctx.Provider value={{ user, loading, provider, login, register, logout }}>{children}</Ctx.Provider>;
}

export function useAuth() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useAuth outside AuthProvider");
  return c;
}
