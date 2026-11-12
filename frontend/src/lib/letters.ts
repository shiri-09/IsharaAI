// Few-shot fingerspelling: features for one frame + a k-nearest-neighbour classifier over taught letters.
// ISL letters are mostly two-handed, so the feature combines both hands' shapes and where the hands are.

const N_POSE = 33;
const LH = N_POSE;
const RH = N_POSE + 21;

function hand(frame: number[], start: number): number[] | null {
  const pts: [number, number, number][] = [];
  for (let k = 0; k < 21; k++) {
    const i = (start + k) * 3;
    pts.push([frame[i], frame[i + 1], frame[i + 2]]);
  }
  if (pts.every(([x, y]) => x === 0 && y === 0)) return null;
  const [wx, wy, wz] = pts[0];
  const size = Math.hypot(pts[9][0] - wx, pts[9][1] - wy) || 1e-3;
  return pts.flatMap(([x, y, z]) => [(x - wx) / size, (y - wy) / size, (z - wz) / size]);
}

export type LetterFeat = { lh: number[] | null; rh: number[] | null; rel: number[] | null };

export function letterFeatures(frame: number[]): LetterFeat {
  const lh = hand(frame, LH);
  const rh = hand(frame, RH);
  const ls = [frame[11 * 3], frame[11 * 3 + 1]];
  const rs = [frame[12 * 3], frame[12 * 3 + 1]];
  const sw = Math.hypot(ls[0] - rs[0], ls[1] - rs[1]) || 0.3;
  let rel: number[] | null = null;
  if (lh && rh) {
    const a = LH * 3;
    const b = RH * 3;
    rel = [(frame[a] - frame[b]) / sw, (frame[a + 1] - frame[b + 1]) / sw];
  }
  return { lh, rh, rel };
}

const sq = (a: number[], b: number[]) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);

export function distance(a: LetterFeat, b: LetterFeat): number {
  // a different number of hands is a different letter
  if (!!a.lh !== !!b.lh || !!a.rh !== !!b.rh) return Infinity;
  let d = 0;
  let n = 0;
  if (a.lh && b.lh) {
    d += sq(a.lh, b.lh);
    n += 1;
  }
  if (a.rh && b.rh) {
    d += sq(a.rh, b.rh);
    n += 1;
  }
  if (a.rel && b.rel) d += 4 * sq(a.rel, b.rel);
  return n ? Math.sqrt(d / n) : Infinity;
}

export type Template = { letter: string; feat: LetterFeat };

export function classify(f: LetterFeat, templates: Template[], k = 3): { letter: string; dist: number } | null {
  if (!f.lh && !f.rh) return null;
  const scored = templates.map((t) => ({ letter: t.letter, d: distance(f, t.feat) })).filter((s) => Number.isFinite(s.d));
  if (!scored.length) return null;
  scored.sort((a, b) => a.d - b.d);
  const top = scored.slice(0, k);
  const votes = new Map<string, number>();
  for (const s of top) votes.set(s.letter, (votes.get(s.letter) ?? 0) + 1 / (1e-3 + s.d));
  const [letter] = [...votes.entries()].sort((a, b) => b[1] - a[1])[0];
  return { letter, dist: scored.find((s) => s.letter === letter)!.d };
}
