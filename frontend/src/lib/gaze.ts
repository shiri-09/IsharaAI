// Calibrated eye-gaze estimation from MediaPipe Face Landmarker (478 landmarks, with irises).
//
// How it works:
//  1. Per frame we turn the landmarks into a small feature vector: where each iris sits inside its eye (in the eye's own
//     axes, so head roll does not matter), plus head yaw, pitch, roll, position and distance.
//  2. During calibration the user looks at 9 dots. For every dot we collect many frames.
//  3. A ridge regression maps features to the on-screen position (as a fraction of the viewport, so it survives a resize).
//  4. At run time the prediction is smoothed with a speed-adaptive filter (steady when still, quick when moving) and
//     blinks are ignored.
// Accuracy is measured honestly on 4 check dots the model was not fitted on (they then join the final fit).

export type Pt = { x: number; y: number };

export type Feat = number[]; // base features, see baseFeatures()

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);

function eyeFeatures(p: Pt[], c0: number, c1: number, iris: number): [number, number] {
  const a = p[c0];
  const b = p[c1];
  const len = Math.max(1e-6, dist(a, b));
  const ux = (b.x - a.x) / len;
  const uy = (b.y - a.y) / len;
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const rx = p[iris].x - mx;
  const ry = p[iris].y - my;
  return [(rx * ux + ry * uy) / len, (-rx * uy + ry * ux) / len]; // along the eye, across the eye
}

/** Eye direction and eyelid state from MediaPipe's own blendshapes (trained estimates, less sensitive to head pose than raw
 *  landmarks). The eyelids matter for vertical gaze: looking down lowers them, looking up widens the eyes. */
export type EyeBs = { h: number; v: number; blink: number; squint: number; wide: number };
export function eyeBlendshapes(score: (name: string) => number): EyeBs {
  // positive h = looking toward the subject's right, positive v = looking up
  const h = (score("eyeLookOutRight") + score("eyeLookInLeft") - score("eyeLookInRight") - score("eyeLookOutLeft")) / 2;
  const v = (score("eyeLookUpLeft") + score("eyeLookUpRight") - score("eyeLookDownLeft") - score("eyeLookDownRight")) / 2;
  const avg = (a: string) => (score(a + "Left") + score(a + "Right")) / 2;
  return { h, v, blink: avg("eyeBlink"), squint: avg("eyeSquint"), wide: avg("eyeWide") };
}

/** Head pose from MediaPipe's 4x4 facial transformation matrix (column-major): yaw, pitch, roll in radians, translation in cm. */
export function headPose(m: ArrayLike<number> | undefined): number[] {
  if (!m || m.length < 16) return [0, 0, 0, 0, 0, 0];
  const sy = Math.hypot(m[0], m[1]);
  return [Math.atan2(-m[2], sy), Math.atan2(m[6], m[10]), Math.atan2(m[1], m[0]), m[12], m[13], m[14]];
}

/**
 * Features for one frame, or null if the iris landmarks are missing.
 *  0 iris x (both eyes)   1 iris y          2 nose yaw   3 nose pitch   4 roll   5 face cx   6 face cy   7 face width
 *  8 blendshape h         9 blendshape v    10..12 head yaw, pitch, roll (matrix)   13..15 head tx, ty, tz
 *  16 right iris x        17 right iris y   18 left iris x   19 left iris y
 *  20 eye opening (lid gap / eye width)     21 blink   22 squint   23 wide
 */
export function baseFeatures(p: Pt[] | undefined, bs?: EyeBs, hp?: number[]): Feat | null {
  if (!p || p.length < 478) return null;
  const [rx, ry] = eyeFeatures(p, 33, 133, 468); // subject's right eye (image left)
  const [lx, ly] = eyeFeatures(p, 362, 263, 473); // subject's left eye (image right)
  const faceL = p[234];
  const faceR = p[454];
  const width = Math.max(1e-6, dist(faceL, faceR));
  const mid = { x: (faceL.x + faceR.x) / 2, y: (p[10].y + p[152].y) / 2 };
  const yaw = (p[1].x - mid.x) / width;
  const pitch = (p[1].y - mid.y) / Math.max(1e-6, p[152].y - p[10].y);
  const roll = Math.atan2(faceR.y - faceL.y, faceR.x - faceL.x);
  const open = (dist(p[159], p[145]) / Math.max(1e-6, dist(p[33], p[133])) + dist(p[386], p[374]) / Math.max(1e-6, dist(p[362], p[263]))) / 2;
  const h = hp ?? [0, 0, 0, 0, 0, 0];
  return [
    (rx + lx) / 2, (ry + ly) / 2, yaw, pitch, roll, mid.x, mid.y, width,
    bs?.h ?? 0, bs?.v ?? 0, h[0], h[1], h[2], h[3], h[4], h[5],
    rx, ry, lx, ly, open, bs?.blink ?? 0, bs?.squint ?? 0, bs?.wide ?? 0,
  ];
}

/**
 * One model per screen axis, each with only the features that carry that axis. Fewer terms per model means far less
 * over-fitting with a handful of calibration dots. Vertical gaze is the hard one for webcams, so the vertical model also
 * gets the eyelids (they follow the eyes up and down).
 * Head terms let the estimate survive a head that moves after calibration; the head-turn step teaches them.
 */
function designX(f: Feat): number[] {
  const g = (i: number) => f[i] ?? 0;
  const ix = g(0), bh = g(8);
  return [1, ix, g(16), g(18), bh, g(2), g(10), g(4), g(12), g(5), g(7), g(13), ix * ix, bh * bh, ix * bh];
}
const PEN_X = [0, 1, 1, 1, 1, 1.5, 1.5, 2, 2, 2, 2, 2, 6, 6, 6];
function designY(f: Feat): number[] {
  const g = (i: number) => f[i] ?? 0;
  const iy = g(1), bv = g(9), op = g(20);
  return [1, iy, g(17), g(19), bv, g(3), g(11), op, g(21), g(22), g(23), g(6), g(15), g(14), iy * iy, bv * bv, op * op, iy * bv];
}
const PEN_Y = [0, 1, 1, 1, 1, 1.5, 1.5, 1, 1.5, 1.5, 1.5, 2, 2, 2, 6, 6, 6, 6];
const QUAD_X = 12; // index where the quadratic terms start
const QUAD_Y = 14;

function solve(A: number[][], B: number[]): number[] {
  const n = B.length;
  const M = A.map((r, i) => [...r, B[i]]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
    [M[i], M[piv]] = [M[piv], M[i]];
    const d = M[i][i] || 1e-12;
    for (let c = i; c <= n; c++) M[i][c] /= d;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const k = M[r][i];
      for (let c = i; c <= n; c++) M[r][c] -= k * M[i][c];
    }
  }
  return M.map((r) => r[n]);
}

export type AxisModel = { axis: "x" | "y"; mean: number[]; std: number[]; w: number[]; lambda: number; quad: boolean };
export type Model = { x: AxisModel; y: AxisModel };
type Sample = { f: Feat; x: number; y: number; w: number };

function fitAxis(samples: Sample[], axis: "x" | "y", lambda: number, quad: boolean): AxisModel {
  const des = axis === "x" ? designX : designY;
  const pen = axis === "x" ? PEN_X : PEN_Y;
  const q0 = axis === "x" ? QUAD_X : QUAD_Y;
  const D = samples.map((s) => des(s.f));
  const n = D[0].length;
  const W = samples.reduce((a, s) => a + s.w, 0);
  const mean = Array(n).fill(0);
  const std = Array(n).fill(1);
  for (let j = 1; j < n; j++) {
    mean[j] = D.reduce((a, r, k) => a + r[j] * samples[k].w, 0) / W;
    const v = D.reduce((a, r, k) => a + samples[k].w * (r[j] - mean[j]) ** 2, 0) / W;
    std[j] = Math.sqrt(v) > 1e-9 ? Math.sqrt(v) : 1; // a column that never varies gets weight 0 through the ridge
  }
  const XtX = Array.from({ length: n }, () => Array(n).fill(0));
  const Xty = Array(n).fill(0);
  D.forEach((row, k) => {
    const r = row.map((v, j) => (j === 0 ? 1 : (v - mean[j]) / std[j]));
    const w = samples[k].w;
    const t = axis === "x" ? samples[k].x : samples[k].y;
    for (let i = 0; i < n; i++) {
      Xty[i] += w * r[i] * t;
      for (let j = 0; j < n; j++) XtX[i][j] += w * r[i] * r[j];
    }
  });
  for (let i = 1; i < n; i++) XtX[i][i] += lambda * (i >= q0 && !quad ? 1e4 : pen[i]) * W + 1e-9;
  return { axis, mean, std, w: solve(XtX, Xty), lambda, quad };
}

function predictAxis(m: AxisModel, f: Feat): number {
  const d = m.axis === "x" ? designX(f) : designY(f);
  let v = 0;
  for (let j = 0; j < d.length; j++) v += (j === 0 ? 1 : (d[j] - m.mean[j]) / m.std[j]) * m.w[j];
  return v;
}

export function predictFrac(m: Model, f: Feat): Pt {
  return { x: predictAxis(m.x, f), y: predictAxis(m.y, f) };
}

/** 9 calibration targets as viewport fractions (kept away from the very edge). */
export const CAL_POINTS: Pt[] = [
  { x: 0.5, y: 0.5 },
  { x: 0.08, y: 0.1 },
  { x: 0.92, y: 0.1 },
  { x: 0.08, y: 0.9 },
  { x: 0.92, y: 0.9 },
  { x: 0.5, y: 0.1 },
  { x: 0.5, y: 0.9 },
  { x: 0.08, y: 0.5 },
  { x: 0.92, y: 0.5 },
];
/** 4 check dots the model is NOT fitted on at first: their error is the honest accuracy, then they join the fit. */
export const CHECK_POINTS: Pt[] = [
  { x: 0.29, y: 0.3 },
  { x: 0.71, y: 0.3 },
  { x: 0.71, y: 0.7 },
  { x: 0.29, y: 0.7 },
];

export type CalPoint = { target: Pt; frames: Feat[]; wiggle?: boolean; check?: boolean };
export type Calibration = {
  model: Model;
  errorPx: number;
  errX: number;
  errY: number;
  w: number;
  h: number;
  at: number;
  perDot?: number[];
  v: 3;
  bx?: number;
  by?: number;
};

const median = (v: number[]) => {
  const s = [...v].sort((p, q) => p - q);
  return s[Math.floor(s.length / 2)];
};

/** Drop frames that disagree with the rest of the dot (a glance away, a blink the detector missed, a landmark glitch). */
export function robustFrames(frames: Feat[], keep = 40): Feat[] {
  if (frames.length < 6) return frames;
  const idx = [0, 1, 8, 9]; // iris x,y and blendshape h,v
  const med = idx.map((j) => median(frames.map((f) => f[j] ?? 0)));
  const mad = idx.map((j, k) => median(frames.map((f) => Math.abs((f[j] ?? 0) - med[k]))) * 1.4826 + 1e-4);
  const scored = frames.map((f) => ({ f, d: Math.max(...idx.map((j, k) => Math.abs((f[j] ?? 0) - med[k]) / mad[k])) }));
  const good = scored.filter((s) => s.d <= 3).map((s) => s.f);
  const use = good.length >= 6 ? good : frames;
  if (use.length <= keep) return use;
  const step = use.length / keep;
  return Array.from({ length: keep }, (_, i) => use[Math.floor(i * step)]);
}

function samplesOf(points: CalPoint[], wiggleWeight = 2): Sample[] {
  return points.flatMap((p) => {
    const fr = robustFrames(p.frames);
    const total = p.wiggle ? wiggleWeight : 1; // every dot counts equally however many frames it got
    return fr.map((f) => ({ f, x: p.target.x, y: p.target.y, w: total / fr.length }));
  });
}

const LAMBDAS = [0.0005, 0.002, 0.006, 0.02, 0.06, 0.2];
const dotPred = (m: AxisModel, p: CalPoint) => median(robustFrames(p.frames).map((f) => predictAxis(m, f)));

/** Pick the ridge strength and model shape per axis by how well it predicts dots it was not fitted on. */
function choose(axis: "x" | "y", train: (exclude: CalPoint | null) => Sample[], held: CalPoint[], scale: number, lopo: boolean) {
  let best = { err: Infinity, lambda: 0.02, quad: false };
  for (const quad of [false, true]) {
    for (const lambda of LAMBDAS) {
      const errs: number[] = [];
      if (lopo) {
        for (const p of held) {
          const m = fitAxis(train(p), axis, lambda, quad);
          errs.push(Math.abs(dotPred(m, p) - (axis === "x" ? p.target.x : p.target.y)) * scale);
        }
      } else {
        const m = fitAxis(train(null), axis, lambda, quad);
        for (const p of held) errs.push(Math.abs(dotPred(m, p) - (axis === "x" ? p.target.x : p.target.y)) * scale);
      }
      const e = errs.reduce((a, b) => a + b, 0) / errs.length; // mean: one badly wrong dot should count
      if (e < best.err - 0.5) best = { err: e, lambda, quad };
    }
  }
  return best;
}

/**
 * Fit from the dots. The model shape and ridge strength are chosen per axis on the check dots (or leave-one-dot-out when
 * there are none), the reported error is measured on dots the model never saw, and the final model then uses every dot.
 */
export function calibrate(points: CalPoint[], w: number, h: number): Calibration | null {
  const usable = points.filter((p) => p.frames.length >= 8);
  const fitPts = usable.filter((p) => !p.check);
  const checks = usable.filter((p) => p.check);
  const dots = fitPts.filter((p) => !p.wiggle);
  if (dots.length < 7) return null;
  const useChecks = checks.length >= 3;
  const train = (ex: CalPoint | null) => samplesOf(fitPts.filter((q) => q !== ex));
  const held = useChecks ? checks : dots;
  const cx = choose("x", train, held, w, !useChecks);
  const cy = choose("y", train, held, h, !useChecks);
  // honest error on the held dots with the chosen settings
  const perDot: number[] = [];
  if (useChecks) {
    const mx = fitAxis(train(null), "x", cx.lambda, cx.quad);
    const my = fitAxis(train(null), "y", cy.lambda, cy.quad);
    for (const p of checks) perDot.push(Math.hypot((dotPred(mx, p) - p.target.x) * w, (dotPred(my, p) - p.target.y) * h));
  } else {
    for (const p of dots) {
      const mx = fitAxis(train(p), "x", cx.lambda, cx.quad);
      const my = fitAxis(train(p), "y", cy.lambda, cy.quad);
      perDot.push(Math.hypot((dotPred(mx, p) - p.target.x) * w, (dotPred(my, p) - p.target.y) * h));
    }
  }
  const all = samplesOf(usable);
  const model = { x: fitAxis(all, "x", cx.lambda, cx.quad), y: fitAxis(all, "y", cy.lambda, cy.quad) };
  const errorPx = perDot.reduce((a, b) => a + b, 0) / perDot.length;
  return { model, errorPx, errX: cx.err, errY: cy.err, w, h, at: Date.now(), perDot, v: 3 };
}

const KEY = "ishara.gaze.calibration";
export const calStore = {
  load(): Calibration | null {
    try {
      const raw = localStorage.getItem(KEY);
      const c = raw ? (JSON.parse(raw) as Calibration) : null;
      return c && c.v === 3 ? c : null; // an older, differently shaped calibration is discarded
    } catch {
      return null;
    }
  },
  save(c: Calibration) {
    try {
      localStorage.setItem(KEY, JSON.stringify(c));
    } catch {
      /* storage unavailable: calibration lasts this session only */
    }
  },
  clear() {
    try {
      localStorage.removeItem(KEY);
    } catch {
      /* ignore */
    }
  },
};

/** Low-pass for the raw landmark features: iris landmarks jitter by a pixel or so, and the regression amplifies it. */
export class FeatureSmoother {
  private f: Feat | null = null;
  reset() {
    this.f = null;
  }
  push(x: Feat, alpha = 0.35): Feat {
    if (!this.f) this.f = [...x];
    else this.f = this.f.map((v, i) => v + (x[i] - v) * alpha);
    return this.f;
  }
}

/** One Euro filter: very steady when the eyes rest, quick when they really move. Units: seconds, viewport fractions. */
class OneEuro {
  private xPrev = 0;
  private dxPrev = 0;
  private init = false;
  constructor(private minCut = 0.25, private beta = 14, private dCut = 1.0) {}
  reset() {
    this.init = false;
  }
  private a(cut: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cut);
    return 1 / (1 + tau / dt);
  }
  filter(x: number, dt: number): number {
    if (!this.init) {
      this.init = true;
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dx = (x - this.xPrev) / dt;
    this.dxPrev += this.a(this.dCut, dt) * (dx - this.dxPrev);
    const cut = this.minCut + this.beta * Math.abs(this.dxPrev);
    this.xPrev += this.a(cut, dt) * (x - this.xPrev);
    return this.xPrev;
  }
}

/** Turns per-frame features into a steady on-screen point (viewport fractions). */
export class GazeTracker {
  model: Model | null = null;
  auto: AutoCal | null = null;
  private fs = new FeatureSmoother();
  private ex = new OneEuro();
  private ey = new OneEuro();
  private win: { x: number; y: number }[] = [];
  private out: Pt | null = null;
  private lastGood = 0;
  private lastT = 0;

  /** Learned offset (viewport fractions) added to a calibrated estimate: set by Re-center and refined by every selection. */
  bias: Pt = { x: 0, y: 0 };
  setModel(m: Model | null, bias?: Pt) {
    this.model = m;
    this.bias = bias ? { ...bias } : { x: 0, y: 0 };
    this.reset();
  }
  setAuto(a: AutoCal | null) {
    this.auto = a;
    this.reset();
  }
  /** After a selection: the user was looking at `target` (viewport fractions), so move the estimate a little toward it. */
  nudge(target: Pt, rate = 0.3) {
    if (this.model && this.out) {
      // calibrated: only a small, bounded correction (a wrong selection must not drag the estimate far)
      const ex = target.x - this.out.x;
      const ey = target.y - this.out.y;
      if (Math.hypot(ex, ey) > 0.14) return;
      const c = (v: number) => Math.max(-0.1, Math.min(0.1, v));
      this.bias = { x: c(this.bias.x + rate * ex), y: c(this.bias.y + rate * ey) };
      return;
    }
    if (!this.auto || this.model || !this.out) return;
    const c = (v: number) => Math.max(-0.3, Math.min(0.3, v));
    this.auto.bx = c(this.auto.bx + rate * (target.x - this.out.x));
    this.auto.by = c(this.auto.by + rate * (target.y - this.out.y));
  }
  reset() {
    this.fs.reset();
    this.ex.reset();
    this.ey.reset();
    this.win = [];
    this.out = null;
  }

  /** `blink` is 0..1. Returns null when there is no face or no calibration. */
  update(f: Feat | null, blink: number, now: number): Pt | null {
    if ((!this.model && !this.auto) || !f) {
      if (this.out && now - this.lastGood <= 700) return this.out; // brief tracking loss: hold the dot
      this.reset();
      return null;
    }
    if (blink > 0.4) return this.out; // hold through a blink
    const dt = Math.min(0.1, Math.max(0.008, (now - this.lastT) / 1000 || 0.033));
    this.lastT = now;
    const sf = this.fs.push(f);
    const pr = this.model ? predictFrac(this.model, sf) : autoPredict(this.auto!, sf);
    const raw = this.model ? { x: pr.x + this.bias.x, y: pr.y + this.bias.y } : pr;
    const x = this.ex.filter(Math.max(-0.05, Math.min(1.05, raw.x)), dt);
    const y = this.ey.filter(Math.max(-0.05, Math.min(1.05, raw.y)), dt);
    // fixation lock: while the eyes rest (small spread over ~0.4 s) show the centre of the cluster, so the dot stops trembling
    this.win.push({ x, y });
    if (this.win.length > 12) this.win.shift();
    let ox = x;
    let oy = y;
    if (this.win.length >= 6) {
      const xs = this.win.map((p) => p.x);
      const ys = this.win.map((p) => p.y);
      const spread = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
      if (spread < 0.045) {
        const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
        const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
        const k = spread < 0.025 ? 0.92 : 0.6;
        ox = x + (cx - x) * k;
        oy = y + (cy - y) * k;
      }
    }
    // a big jump is a real eye movement: forget the old cluster so the dot follows without lag
    if (this.out && Math.hypot(ox - this.out.x, oy - this.out.y) > 0.12) this.win = [{ x: ox, y: oy }];
    this.out = { x: ox, y: oy };
    this.lastGood = now;
    return this.out;
  }
}

// ------------------------------------------------------------------ zero-effort calibration
// Instead of a long calibration, the tracker starts from a typical mapping (for someone sitting close to the camera, about a
// hand to an arm's length away) and needs only ONE dot: look at the centre of the screen for 3 seconds. After that it keeps
// learning: every time you select something, the tracker moves its estimate a little toward where that thing actually is.

export type AutoCal = { ref: Feat; gainX: number; gainY: number; bx: number; by: number; at: number };

/** Screen position change per unit of eye or head movement. Signs: the camera image is not mirrored, so looking to the
 *  screen's right moves the iris and the nose toward the image's left (negative), looking down increases y. */
const AUTO = { ix: -4.2, yaw: -1.6, iy: 7, pitch: 2.2 };

export function autoPredict(a: AutoCal, f: Feat): Pt {
  const k = a.ref[7] / Math.max(1e-6, f[7]); // closer to the camera -> bigger eye movements for the same screen: scale them back
  const dix = (f[0] - a.ref[0]) * k;
  const diy = (f[1] - a.ref[1]) * k;
  const dyaw = (f[2] - a.ref[2]) * k;
  const dpitch = (f[3] - a.ref[3]) * k;
  return {
    x: 0.5 + a.bx + a.gainX * (AUTO.ix * dix + AUTO.yaw * dyaw),
    y: 0.5 + a.by + a.gainY * (AUTO.iy * diy + AUTO.pitch * dpitch),
  };
}

/** One dot: the user looked at the centre of the screen. `frames` are smoothed feature vectors. */
export function centerCalibration(frames: Feat[], prev?: AutoCal | null): AutoCal | null {
  if (frames.length < 18) return null;
  const ref = frames[0].map((_, j) => median(frames.map((f) => f[j])));
  return { ref, gainX: prev?.gainX ?? 1, gainY: prev?.gainY ?? 1, bx: 0, by: 0, at: Date.now() };
}

const AUTO_KEY = "ishara.gaze.auto";
export const autoStore = {
  load(): AutoCal | null {
    try {
      const raw = localStorage.getItem(AUTO_KEY);
      return raw ? (JSON.parse(raw) as AutoCal) : null;
    } catch {
      return null;
    }
  },
  save(a: AutoCal) {
    try {
      localStorage.setItem(AUTO_KEY, JSON.stringify(a));
    } catch {
      /* ignore */
    }
  },
  clear() {
    try {
      localStorage.removeItem(AUTO_KEY);
    } catch {
      /* ignore */
    }
  },
};
