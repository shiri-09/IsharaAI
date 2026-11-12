// Pose maths for the signing avatar. Poses come from the backend as 95 points (same layout as the recogniser:
// pose 33 | left hand 21 | right hand 21 | face 20), body-normalised: origin between the shoulders, shoulder
// width = 1, y pointing down. A missing point is null and is filled from a neutral rest pose.

export type Pt = [number, number];
export type RawPose = (Pt | null)[];
export type Pose = Pt[];

export type Segment =
  | { type: "sign"; label: string; pose: RawPose }
  | { type: "letter"; label: string; pose: RawPose | null; word: string }
  | { type: "clip"; label: string; frames: RawPose[] };

export const LH0 = 33;
export const RH0 = 54;
export const FACE0 = 75;
const N = 95;

// Canonical relaxed hand, relative to the wrist, fingers pointing down (wrist, thumb 1-4, index 5-8 ... pinky 17-20)
const HAND_DOWN: Pt[] = [
  [0, 0], [0.06, 0.05], [0.1, 0.11], [0.12, 0.17], [0.13, 0.22],
  [0.05, 0.16], [0.05, 0.26], [0.05, 0.32], [0.05, 0.37],
  [0, 0.17], [0, 0.28], [0, 0.35], [0, 0.4],
  [-0.05, 0.16], [-0.05, 0.26], [-0.05, 0.32], [-0.05, 0.36],
  [-0.09, 0.14], [-0.1, 0.22], [-0.1, 0.27], [-0.1, 0.31],
];
// Fingerspelling stance: right hand raised beside the chest, palm out
const HAND_UP: Pt[] = HAND_DOWN.map(([x, y]) => [x, -y]);

function restPose(): Pose {
  const p: Pose = Array.from({ length: N }, () => [0, 0] as Pt);
  const set = (i: number, x: number, y: number) => (p[i] = [x, y]);
  // head (pose indices 0-10). Image is not mirrored: the person's LEFT side is on the viewer's RIGHT.
  set(0, 0, -0.78); set(1, 0.07, -0.88); set(2, 0.12, -0.88); set(3, 0.16, -0.88);
  set(4, -0.07, -0.88); set(5, -0.12, -0.88); set(6, -0.16, -0.88);
  set(7, 0.27, -0.84); set(8, -0.27, -0.84); set(9, 0.06, -0.66); set(10, -0.06, -0.66);
  // shoulders, elbows, wrists, hand markers
  set(11, 0.5, 0); set(12, -0.5, 0); set(13, 0.62, 0.92); set(14, -0.62, 0.92);
  set(15, 0.6, 1.78); set(16, -0.6, 1.78);
  set(17, 0.62, 1.95); set(18, -0.62, 1.95); set(19, 0.6, 1.98); set(20, -0.6, 1.98); set(21, 0.56, 1.9); set(22, -0.56, 1.9);
  // hips, legs (rarely visible; kept for the torso)
  set(23, 0.33, 2.15); set(24, -0.33, 2.15); set(25, 0.35, 3.1); set(26, -0.35, 3.1);
  set(27, 0.35, 4); set(28, -0.35, 4); set(29, 0.35, 4.1); set(30, -0.35, 4.1); set(31, 0.4, 4.1); set(32, -0.4, 4.1);
  // hands hanging at the wrists
  HAND_DOWN.forEach(([x, y], k) => {
    p[LH0 + k] = [0.6 + x, 1.78 + y];
    p[RH0 + k] = [-0.6 - x, 1.78 + y];
  });
  // face subset: lip corners, upper/lower lip, inner lips, brows, nose, chin
  const face: Pt[] = [
    [0.07, -0.66], [-0.07, -0.66], [0, -0.69], [0, -0.62], [0, -0.67], [0, -0.65], [0.05, -0.66], [-0.05, -0.66],
    [0.2, -0.97], [0.16, -0.99], [0.12, -1.0], [0.08, -0.99], [0.04, -0.97],
    [-0.04, -0.97], [-0.08, -0.99], [-0.12, -1.0], [-0.16, -0.99], [-0.2, -0.97],
    [0, -0.76], [0, -0.55],
  ];
  face.forEach((pt, k) => (p[FACE0 + k] = pt));
  return p;
}

export const REST: Pose = restPose();

const visible = (raw: RawPose, from: number, n: number) => raw.slice(from, from + n).filter(Boolean).length;

/** Fill missing points: body from the rest pose (shifted to stay attached), hands as relaxed hands at the wrists. */
export function fill(raw: RawPose): Pose {
  const out: Pose = REST.map((p, i) => (raw[i] ? (raw[i] as Pt) : p));
  // head: if the face/nose is missing, keep the rest head
  for (const [h0, wrist] of [
    [LH0, 15],
    [RH0, 16],
  ] as const) {
    if (visible(raw, h0, 21) < 10) {
      const [wx, wy] = out[wrist];
      const sign = h0 === LH0 ? 1 : -1;
      HAND_DOWN.forEach(([x, y], k) => (out[h0 + k] = [wx + sign * x, wy + y]));
    }
  }
  return out;
}

/** A neutral fingerspelling stance for letters nobody has taught yet: right hand raised beside the chest. */
export function spellStance(): Pose {
  const p = REST.map((q) => [...q] as Pt);
  p[14] = [-0.75, 0.55];
  p[16] = [-0.45, 0.1];
  HAND_UP.forEach(([x, y], k) => (p[RH0 + k] = [-0.45 - x, 0.1 + y]));
  return p;
}

export const lerpPose = (a: Pose, b: Pose, t: number): Pose => a.map(([x, y], i) => [x + (b[i][0] - x) * t, y + (b[i][1] - y) * t]);
export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export type Step = { pose: Pose; move: number; hold: number; label: string; letter?: string; part?: number };

/** Turn the backend's segments into timed key poses. `k` scales all durations (avatar speed preference). */
export function buildSteps(segments: Segment[], k = 1): Step[] {
  const steps: Step[] = [];
  for (const s of segments) {
    if (s.type === "clip") {
      const frames = s.frames.map(fill);
      frames.forEach((f, i) => steps.push({ pose: f, move: i === 0 ? 380 * k : 66 * k, hold: 0, label: s.label }));
      steps[steps.length - 1].hold = 350 * k;
    } else if (s.type === "sign") {
      steps.push({ pose: fill(s.pose), move: 380 * k, hold: 560 * k, label: s.label });
    } else {
      steps.push({ pose: s.pose ? fill(s.pose) : spellStance(), move: 260 * k, hold: 520 * k, label: s.word, letter: s.label });
    }
  }
  steps.push({ pose: REST, move: 500 * k, hold: 0, label: "" });
  return steps;
}

export const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
export const FINGERTIPS = [4, 8, 12, 16, 20];
