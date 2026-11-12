import { useCallback, useEffect, useRef, useState } from "react";

export const LOCALES: Record<string, { label: string; code: string }> = {
  en: { label: "English", code: "en-IN" },
  hi: { label: "हिन्दी", code: "hi-IN" },
  kn: { label: "ಕನ್ನಡ", code: "kn-IN" },
  ta: { label: "தமிழ்", code: "ta-IN" },
  te: { label: "తెలుగు", code: "te-IN" },
};

// Web Speech API (Chrome / Edge). Typed loosely because it is not in TypeScript's DOM types everywhere.
/* eslint-disable @typescript-eslint/no-explicit-any */
type Rec = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  start(): void;
  stop(): void;
  onresult: ((e: any) => void) | null;
  onend: (() => void) | null;
  onerror: ((e: any) => void) | null;
};
const SR: (new () => Rec) | undefined = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
export const speechSupported = !!SR;

/** One-shot speech recognition: interim text while speaking, `onFinal` with the final sentence. */
export function useSpeech(onFinal: (text: string) => void) {
  const rec = useRef<Rec | null>(null);
  const cb = useRef(onFinal);
  cb.current = onFinal;
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState("");

  useEffect(() => () => rec.current?.stop(), []);

  const toggle = useCallback(
    (lang: string) => {
      if (!SR) {
        setError("Speech recognition needs Chrome or Edge. You can type instead.");
        return;
      }
      if (listening) {
        rec.current?.stop();
        return;
      }
      const r = new SR();
      r.lang = LOCALES[lang]?.code ?? "en-IN";
      r.interimResults = true;
      r.continuous = false;
      r.onresult = (e: any) => {
        const t = Array.from(e.results as ArrayLike<any>).map((x: any) => x[0].transcript).join(" ");
        setInterim(t);
        if (e.results[e.results.length - 1].isFinal) {
          setInterim("");
          cb.current(t);
        }
      };
      r.onend = () => setListening(false);
      r.onerror = (e: any) => {
        setError(e.error === "not-allowed" ? "Microphone permission was denied." : `Speech recognition: ${e.error}`);
        setListening(false);
      };
      rec.current = r;
      setError("");
      setListening(true);
      r.start();
    },
    [listening],
  );

  return { listening, interim, error, toggle };
}
