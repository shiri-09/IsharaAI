// Appearance-based gaze network: MobileGaze (MobileOne S0, trained on Gaze360) from github.com/yakhyo/gaze-estimation
// (MIT licence), run in the browser with onnxruntime-web. It looks at the face image itself, not just landmark positions,
// and outputs where the person is looking as yaw and pitch angles. These angles join the landmark features of the
// calibrated tracker (lib/gaze.ts): the per-person calibration still maps them to the screen.
//
// Inference runs asynchronously on the latest face crop whenever the previous one has finished, so it never blocks the
// camera loop; the newest angles are read with latest().

import type { Pt } from "./gaze";

type Ort = typeof import("onnxruntime-web");
type Session = import("onnxruntime-web").InferenceSession;

const SIZE = 448;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ort: Ort | null = null;
let session: Session | null = null;
let loading: Promise<void> | null = null;
let failed = false;
let busy = false;
let lastStart = 0;
let last: { yaw: number; pitch: number; at: number } | null = null;
let canvas: HTMLCanvasElement | null = null;

function load(): Promise<void> {
  if (!loading) {
    loading = (async () => {
      try {
        ort = await import("onnxruntime-web");
        ort.env.wasm.wasmPaths = location.origin + "/ort/";
        ort.env.wasm.numThreads = 1; // no cross-origin isolation needed
        ort.env.wasm.proxy = true; // run in a web worker: the page and the camera loop never wait for it
        session = await ort.InferenceSession.create("/models/gaze/mobileone_s0_gaze.onnx", { executionProviders: ["wasm"] });
      } catch (e) {
        failed = true;
        console.warn("gaze network unavailable, landmarks only:", e);
      }
    })();
  }
  return loading;
}

const softmaxExpect = (logits: Float32Array | number[]): number => {
  let m = -Infinity;
  for (const v of logits) m = Math.max(m, v);
  let sum = 0;
  let acc = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - m);
    sum += e;
    acc += e * i;
  }
  return (acc / sum) * 4 - 180; // 90 bins of 4 degrees, centred on 0
};

/** Start an inference on this frame if the network is free. `pts` are MediaPipe face landmarks (normalised). */
export function maybeRun(video: HTMLVideoElement, pts: Pt[]): void {
  if (failed || busy || performance.now() - lastStart < 200) return; // at most 5 a second: plenty for a smoothed gaze
  if (!session) {
    load();
    return;
  }
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) return;
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of pts) {
    if (p.x < x0) x0 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.x > x1) x1 = p.x;
    if (p.y > y1) y1 = p.y;
  }
  // a square face crop with some margin, like the face detector the network was trained with
  const cx = ((x0 + x1) / 2) * vw;
  const cy = ((y0 + y1) / 2) * vh;
  const side = Math.max((x1 - x0) * vw, (y1 - y0) * vh) * 1.15;
  canvas ??= Object.assign(document.createElement("canvas"), { width: SIZE, height: SIZE });
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return;
  ctx.drawImage(video, cx - side / 2, cy - side / 2, side, side, 0, 0, SIZE, SIZE);
  const px = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE;
  const input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    input[i] = (px[i * 4] / 255 - MEAN[0]) / STD[0];
    input[n + i] = (px[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    input[2 * n + i] = (px[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  busy = true;
  lastStart = performance.now();
  const s = session;
  const feeds = { [s.inputNames[0]]: new ort!.Tensor("float32", input, [1, 3, SIZE, SIZE]) };
  s.run(feeds)
    .then((out) => {
      const yaw = softmaxExpect(out[s.outputNames[0]].data as Float32Array);
      const pitch = softmaxExpect(out[s.outputNames[1]].data as Float32Array);
      // light smoothing: the network is a little noisy frame to frame
      last = last && performance.now() - last.at < 500 ? { yaw: last.yaw * 0.5 + yaw * 0.5, pitch: last.pitch * 0.5 + pitch * 0.5, at: performance.now() } : { yaw, pitch, at: performance.now() };
    })
    .catch(() => undefined)
    .finally(() => {
      busy = false;
    });
}

/** Newest gaze angles in degrees, or zeros when the network is not running (the calibration then ignores them). */
export function latest(): [number, number] {
  if (!last) return [0, 0]; // never ran: the calibration gives this column no weight
  // keep the last value through short gaps (a blink, a slow frame): a sudden drop to 0 would throw the estimate
  return [last.yaw, last.pitch];
}

export const gazeNetReady = () => !!session && !failed;
