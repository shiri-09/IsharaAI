import { useCallback, useEffect, useRef, useState } from "react";
import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerResult,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task";

let landmarkerPromise: Promise<HolisticLandmarker> | null = null;

function getLandmarker(): Promise<HolisticLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
      const make = (delegate: "GPU" | "CPU") =>
        HolisticLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate },
          runningMode: "VIDEO",
        });
      try {
        return await make("GPU");
      } catch {
        return await make("CPU");
      }
    })();
    landmarkerPromise.catch(() => (landmarkerPromise = null));
  }
  return landmarkerPromise;
}

type Status = "idle" | "loading" | "running" | "error";

function drawPoints(
  ctx: CanvasRenderingContext2D,
  pts: NormalizedLandmark[] | undefined,
  conns: { start: number; end: number }[],
  w: number,
  h: number,
  color: string,
  radius: number,
) {
  if (!pts?.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const c of conns) {
    const a = pts[c.start];
    const b = pts[c.end];
    if (!a || !b) continue;
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
  }
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Webcam + in-browser MediaPipe Holistic. `onResult` is called for every processed frame. */
export function useHolistic(onResult: (r: HolisticLandmarkerResult) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cb = useRef(onResult);
  cb.current = onResult;
  const raf = useRef<number>(0);
  const stream = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string>("");
  const [fps, setFps] = useState(0);

  const stop = useCallback(() => {
    cancelAnimationFrame(raf.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    canvasRef.current?.getContext("2d")?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setStatus("idle");
    setFps(0);
  }, []);

  const start = useCallback(async () => {
    setError("");
    setStatus("loading");
    try {
      const [lm, media] = await Promise.all([
        getLandmarker(),
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false }),
      ]);
      stream.current = media;
      const video = videoRef.current!;
      video.srcObject = media;
      await video.play();
      const canvas = canvasRef.current!;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d")!;
      setStatus("running");

      let lastVideoTime = -1;
      let frames = 0;
      let fpsT = performance.now();
      const loop = () => {
        if (video.currentTime !== lastVideoTime && video.readyState >= 2) {
          lastVideoTime = video.currentTime;
          const r = lm.detectForVideo(video, performance.now());
          const { width: w, height: h } = canvas;
          ctx.clearRect(0, 0, w, h);
          drawPoints(ctx, r.poseLandmarks[0], HolisticLandmarker.POSE_CONNECTIONS, w, h, "rgba(200,206,216,0.35)", 2);
          drawPoints(ctx, r.leftHandLandmarks[0], HolisticLandmarker.HAND_CONNECTIONS, w, h, "rgba(238,241,245,0.9)", 2.5);
          drawPoints(ctx, r.rightHandLandmarks[0], HolisticLandmarker.HAND_CONNECTIONS, w, h, "rgba(238,241,245,0.9)", 2.5);
          cb.current(r);
          frames++;
          const now = performance.now();
          if (now - fpsT > 1000) {
            setFps(Math.round((frames * 1000) / (now - fpsT)));
            frames = 0;
            fpsT = now;
          }
        }
        raf.current = requestAnimationFrame(loop);
      };
      raf.current = requestAnimationFrame(loop);
    } catch (e) {
      stop();
      setStatus("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [stop]);

  useEffect(() => stop, [stop]);

  return { videoRef, canvasRef, status, error, fps, start, stop };
}
