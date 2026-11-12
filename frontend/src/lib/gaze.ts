// Calibrated eye-gaze estimation from MediaPipe Face Landmarker (478 landmarks, with irises).
//
// How it works:
//  1. Per frame we turn the landmarks into a small feature vector: where each iris sits inside its eye (in the eye's own
//     axes, so head roll does not matter), plus head yaw, pitch, roll, position and distance.
//  2. During calibration the user looks at 9 dots. For every dot we collect many frames.
//  3. A ridge regression maps features to the on-screen position (as a fraction of the viewport, so it survives a resize).
//  4. At run time the prediction is smoothed with a speed-adaptive filter (steady when still, quick when moving) and
//     blinks are ignored.
// Accuracy is measured honestly with leave-one-dot-out: the model is fitted without a dot and asked where it is.

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

/** Eye direction from MediaPipe's own blendshapes (trained gaze estimates, much less sensitive to head pose than raw landmarks). */
export type EyeBs = { h: number; v: number };
export function eyeBlendshapes(score: (name: string) => number): EyeBs {
  // positive h = looking toward the subject's right, positive v = looking up
  const h = (score("eyeLookOutRight") + score("eyeLookInLeft") - score("eyeLookInRight") - score("eyeLookOutLeft")) / 2;
  const v = (score("eyeLookUpLeft") + score("eyeLookUpRight") - score("eyeLookDownLeft") - score("eyeLookDownRight")) / 2;
  return { h, v };
}

/** Head pose from MediaPipe's 4x4 facial transformation matrix (column-major): yaw, pitch, roll in radians, translation in cm. */
export function headPose(m: ArrayLike<number> | undefined): number[] {
  if (!m || m.length < 16) return [0, 0, 0, 0, 0, 0];
  const sy = Math.hypot(m[0], m[1]);
  return [Math.atan2(-m[2], sy), Math.atan2(m[6], m[10]), Math.atan2(m[1], m[0]), m[12], m[13], m[14]];
}

/** Features for one frame, or null if the iris landmarks are missing. Index 0..7 are the original features. */
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
  const h = hp ?? [0, 0, 0, 0, 0, 0];
  return [(rx + lx) / 2, (ry + ly) / 2, yaw, pitch, roll, mid.x, mid.y, width, bs?.h ?? 0, bs?.v ?? 0, h[0], h[1], h[2], h[3], h[4], h[5]];
}

/**
 * Design vector: bias, eye direction (iris geometry and blendshapes), 3D head pose and position, and a few quadratic terms.
 * Head terms are what lets the estimate survive a head that moves after calibration; the calibration asks the person to keep
 * looking at one dot while turning the head, so these coefficients are learned from data rather than guessed.
 */
function design(f: Feat): number[] {
  const g = (i: number) => f[i] ?? 0;
  const ix = g(0), iy = g(1), bh = g(8), bv = g(9);
  return [1, ix, iy, bh, bv, g(2), g(3), g(10), g(11), g(4), g(12), g(5), g(6), g(7), g(15), ix * ix, iy * iy, ix * iy, bh * bh, bv * bv, bh * bv];
}
// ridge strength per term: light on the main terms, heavier on the quadratic ones and on position terms
const PENALTY = [0, 1, 1, 1, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6];

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

export type Model = { mean: number[]; std: number[]; wx: number[]; wy: number[] };
type Sample = { f: Feat; x: number; y: number; w: number };

function fit(samples: Sample[], lambda: number): Model {
  const D = samples.map((s) => design(s.f));
  const n = D[0].length;
  const W = samples.reduce((a, s) => a + s.w, 0);
  const mean = Array(n).fill(0);
  const std = Array(n).fill(1);
  for (let j = 1; j < n; j++) {
    mean[j] = D.reduce((a, r, k) => a + r[j] * samples[k].w, 0) / W;
    const v = D.reduce((a, r, k) => a + samples[k].w * (r[j] - mean[j]) ** 2, 0) / W;
    std[j] = Math.sqrt(v) > 1e-9 ? Math.sqrt(v) : 1; // a column that never varies (no head matrix) gets weight 0 through the ridge
  }
  const Z = D.map((r) => r.map((v, j) => (j === 0 ? 1 : (v - mean[j]) / std[j])));
  const XtX = Array.from({ length: n }, () => Array(n).fill(0));
  const Xtx = Array(n).fill(0);
  const Xty = Array(n).fill(0);
  Z.forEach((r, k) => {
    const w = samples[k].w;
    for (let i = 0; i < n; i++) {
      Xtx[i] += w * r[i] * samples[k].x;
      Xty[i] += w * r[i] * samples[k].y;
      for (let j = 0; j < n; j++) XtX[i][j] += w * r[i] * r[j];
    }
  });
  for (let i = 1; i < n; i++) XtX[i][i] += lambda * PENALTY[i] * W + 1e-9;
  return { mean, std, wx: solve(XtX, Xtx), wy: solve(XtX.map((r) => [...r]), Xty) };
}

export function predictFrac(m: Model, f: Feat): Pt {
  const z = design(f).map((v, j) => (j === 0 ? 1 : (v - m.mean[j]) / m.std[j]));
  let x = 0;
  let y = 0;
  for (let j = 0; j < z.length; j++) {
    x += z[j] * m.wx[j];
    y += z[j] * m.wy[j];
  }
  return { x, y };
}

/** 9 calibration targets as viewport fractions (kept away from the very edge). The 10th, at the centre, is the head-turn dot. */
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

export type CalPoint = { target: Pt; frames: Feat[]; wiggle?: boolean };
export type Calibration = { model: Model; errorPx: number; w: number; h: number; at: number; perDot?: number[]; v: 2; bx?: number; by?: number };

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

/** Fit from per-dot samples and report leave-one-dot-out error in pixels (median over the 9 dots, and each dot). */
export function calibrate(points: CalPoint[], w: number, h: number): Calibration | null {
  const usable = points.filter((p) => p.frames.length >= 8);
  const dots = usable.filter((p) => !p.wiggle);
  if (dots.length < 7) return null;
  const model = fit(samplesOf(usable), 0.03);
  const errs: number[] = [];
  dots.forEach((p) => {
    const rest = samplesOf(usable.filter((q) => q !== p));
    const m = fit(rest, 0.03);
    const fr = robustFrames(p.frames);
    const preds = fr.map((f) => predictFrac(m, f));
    const px = median(preds.map((q) => q.x));
    const py = median(preds.map((q) => q.y));
    errs.push(Math.hypot((px - p.target.x) * w, (py - p.target.y) * h));
  });
  const sorted = [...errs].sort((a, b) => a - b);
  return { model, errorPx: sorted[Math.floor(sorted.length / 2)], w, h, at: Date.now(), perDot: errs, v: 2 };
}

const KEY = "ishara.gaze.calibration";
export const calStore = {
  load(): Calibration | null {
    try {
      const raw = localStorage.getItem(KEY);
      const c = raw ? (JSON.parse(raw) as Calibration) : null;
      return c && c.v === 2 ? c : null; // an older, differently shaped calibration is discarded
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
