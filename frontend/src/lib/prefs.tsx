import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "./api";

export type Prefs = { mode: "" | "isl" | "aac"; onboarded: boolean; voice: string };

type PrefsCtx = { prefs: Prefs | null; save: (p: Partial<Prefs>) => Promise<Prefs> };

const Ctx = createContext<PrefsCtx | null>(null);

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [prefs, setPrefs] = useState<Prefs | null>(null);

  useEffect(() => {
    api<Prefs>("/api/prefs")
      .then(setPrefs)
      .catch(() => setPrefs({ mode: "", onboarded: false, voice: "hi" }));
  }, []);

  const save = useCallback(async (p: Partial<Prefs>) => {
    const next = await api<Prefs>("/api/prefs", { method: "PUT", body: p });
    setPrefs(next);
    return next;
  }, []);

  return <Ctx.Provider value={{ prefs, save }}>{children}</Ctx.Provider>;
}

export function usePrefs() {
  const c = useContext(Ctx);
  if (!c) throw new Error("usePrefs outside PrefsProvider");
  return c;
}
