// Mirror of backend/app/ml/features.py — keep the layout identical.
// pose (33) | left hand (21) | right hand (21) | face subset (20), each point = x, y, z
import type { HolisticLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";

export const FACE_IDX = [61, 291, 0, 17, 13, 14, 78, 308, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 1, 152];
const N_POSE = 33;
const N_HAND = 21;
export const FRAME_DIM = (N_POSE + 2 * N_HAND + FACE_IDX.length) * 3; // 285

function push(out: number[], pts: NormalizedLandmark[] | undefined, n: number, pick?: number[]) {
  if (!pts || pts.length === 0) {
    for (let i = 0; i < n * 3; i++) out.push(0);
    return;
  }
  const idx = pick ?? [...Array(n).keys()];
  for (const i of idx) {
    const p = pts[i];
    out.push(p?.x ?? 0, p?.y ?? 0, p?.z ?? 0);
  }
}

export function toFrame(r: HolisticLandmarkerResult): number[] {
  const out: number[] = [];
  push(out, r.poseLandmarks[0], N_POSE);
  push(out, r.leftHandLandmarks[0], N_HAND);
  push(out, r.rightHandLandmarks[0], N_HAND);
  push(out, r.faceLandmarks[0], FACE_IDX.length, FACE_IDX);
  return out;
}

export function hasHands(r: HolisticLandmarkerResult): boolean {
  return (r.leftHandLandmarks[0]?.length ?? 0) > 0 || (r.rightHandLandmarks[0]?.length ?? 0) > 0;
}
