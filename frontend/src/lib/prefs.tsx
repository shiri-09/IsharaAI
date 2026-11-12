import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "./api";

export type Prefs = {
  mode: "" | "isl" | "aac";
  onboarded: boolean;
  voice: string;
  text_size: "normal" | "large" | "xlarge";
  scan_ms: number;
  avatar_speed: "slow" | "normal" | "fast";
  auto_speak: boolean;
  high_contrast: boolean;
};

export const DEFAULT_PREFS: Prefs = {
  mode: "",
  onboarded: false,
  voice: "hi",
  text_size: "normal",
  scan_ms: 1600,
  avatar_speed: "normal",
  auto_speak: true,
  high_contrast: false,
};

type PrefsCtx = { prefs: Prefs | null; save: (p: Partial<Prefs>) => Promise<Prefs> };

const Ctx = createContext<PrefsCtx | null>(null);

export function PrefsProvider({ children }: { children: ReactNode }) {
  const [prefs, setPrefs] = useState<Prefs | null>(null);

  useEffect(() => {
    api<Prefs>("/api/prefs")
      .then(setPrefs)
      .catch(() => setPrefs(DEFAULT_PREFS));
  }, []);

  // Adaptive access: apply text size and contrast to the whole app
  useEffect(() => {
    const root = document.documentElement;
    root.style.fontSize = prefs?.text_size === "xlarge" ? "20px" : prefs?.text_size === "large" ? "18px" : "";
    root.classList.toggle("hc", !!prefs?.high_contrast);
  }, [prefs?.text_size, prefs?.high_contrast]);

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
