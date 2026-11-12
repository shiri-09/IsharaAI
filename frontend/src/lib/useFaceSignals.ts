import { useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver, type Category } from "@mediapipe/tasks-vision";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

let faceLmPromise: Promise<FaceLandmarker> | null = null;

function getFaceLandmarker(): Promise<FaceLandmarker> {
  if (!faceLmPromise) {
    faceLmPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
      const make = (delegate: "GPU" | "CPU") =>
        FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
        });
      try {
        return await make("GPU");
      } catch {
        return await make("CPU");
      }
    })();
    faceLmPromise.catch(() => (faceLmPromise = null));
  }
  return faceLmPromise;
}

export type FaceSignals = {
  face: boolean;
  /** Where the user is looking on screen: -1 = left, +1 = right (eyes + head turn, smoothed). */
  gazeX: number;
  /** Eyebrow raise 0..1 (blendshape browInnerUp). */
  brow: number;
  /** Mouth open 0..1 (blendshape jawOpen). */
  jaw: number;
};

const score = (cats: Category[], name: string) => cats.find((c) => c.categoryName === name)?.score ?? 0;

/**
 * Hands-free input for AAC users: webcam + MediaPipe Face Landmarker blendshapes.
 * Read `signals.current` every frame (no re-render); `onFrame` fires per processed frame.
 */
export function useFaceSignals(onFrame?: (s: FaceSignals) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const signals = useRef<FaceSignals>({ face: false, gazeX: 0, brow: 0, jaw: 0 });
  const cb = useRef(onFrame);
  cb.current = onFrame;
  const raf = useRef(0);
  const stream = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "running" | "error">("idle");
  const [error, setError] = useState("");

  const stop = useCallback(() => {
    cancelAnimationFrame(raf.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }, []);

  const start = useCallback(async () => {
    setError("");
    setStatus("loading");
    try {
      const [lm, media] = await Promise.all([
        getFaceLandmarker(),
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false }),
      ]);
      stream.current = media;
      const video = videoRef.current!;
      video.srcObject = media;
      await video.play();
      setStatus("running");
      let last = -1;
      const loop = () => {
        if (video.readyState >= 2 && video.currentTime !== last) {
          last = video.currentTime;
          const r = lm.detectForVideo(video, performance.now());
          const cats = r.faceBlendshapes?.[0]?.categories ?? [];
          const pts = r.faceLandmarks?.[0];
          const s = signals.current;
          if (cats.length && pts) {
            // Blendshapes follow ARKit naming (subject's own left/right). Looking at the screen's left
            // = the subject's left: left eye looks out, right eye looks in.
            const eyeLeftness =
              (score(cats, "eyeLookOutLeft") + score(cats, "eyeLookInRight") - score(cats, "eyeLookOutRight") -
                score(cats, "eyeLookInLeft")) /
              2;
            // Head turn: in the (unmirrored) camera image the nose moves right when the head turns to the subject's left.
            const width = Math.max(1e-3, pts[454].x - pts[234].x);
            const headLeftness = ((pts[1].x - (pts[234].x + pts[454].x) / 2) / width) * 4;
            const raw = -Math.max(-1, Math.min(1, eyeLeftness * 1.6 + headLeftness));
            s.gazeX = s.gazeX * 0.7 + raw * 0.3;
            s.brow = s.brow * 0.5 + score(cats, "browInnerUp") * 0.5;
            s.jaw = s.jaw * 0.5 + score(cats, "jawOpen") * 0.5;
            s.face = true;
          } else {
            s.face = false;
          }
          cb.current?.(s);
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
  return { videoRef, signals, status, error, start, stop };
}
