import { useCallback, useEffect, useRef, useState } from "react";
import type { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";
import { wsUrl, type Prediction } from "./api";
import { toFrame } from "./keypoints";
import { useHolistic } from "./useHolistic";

/** Webcam + MediaPipe + the live recognition socket. Calls `onPrediction` for every recognised sentence. */
export function useSignReader(onPrediction: (p: Prediction) => void) {
  const ws = useRef<WebSocket | null>(null);
  const cb = useRef(onPrediction);
  cb.current = onPrediction;
  const [signing, setSigning] = useState(false);
  const [error, setError] = useState("");

  const onResult = useCallback((r: HolisticLandmarkerResult) => {
    const sock = ws.current;
    if (sock?.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ type: "frame", kp: toFrame(r) }));
  }, []);
  const cam = useHolistic(onResult);

  useEffect(() => {
    if (cam.status !== "running") return;
    const sock = new WebSocket(wsUrl("/ws/recognize"));
    ws.current = sock;
    sock.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "state") setSigning(m.signing);
      else if (m.type === "prediction") cb.current(m);
      else if (m.type === "error") setError(m.message);
    };
    sock.onclose = (ev) => {
      if (ev.code === 4401) setError("Session expired, please log in again.");
    };
    return () => {
      sock.close();
      ws.current = null;
      setSigning(false);
    };
  }, [cam.status]);

  return { cam, signing, error };
}
