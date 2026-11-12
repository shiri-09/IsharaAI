import { useEffect, useRef } from "react";
import { ease, FACE0, FINGERTIPS, HAND_EDGES, LH0, lerpPose, REST, RH0, type Pose, type Step } from "../lib/avatar";

type Props = {
  steps: Step[] | null;
  /** Bump to (re)play the current steps. */
  playKey: number;
  onStep?: (i: number) => void;
  onDone?: () => void;
};

const W = 520;
const H = 560;

export function draw(ctx: CanvasRenderingContext2D, p: Pose, letter?: string) {
  const S = 150; // pixels per shoulder width
  const ox = W / 2;
  const oy = H * 0.36;
  const X = (i: number) => ox + p[i][0] * S;
  const Y = (i: number) => oy + p[i][1] * S;
  ctx.clearRect(0, 0, W, H);

  // soft floor glow
  const g = ctx.createRadialGradient(ox, H * 0.55, 20, ox, H * 0.55, 300);
  g.addColorStop(0, "rgba(240,166,74,0.10)");
  g.addColorStop(1, "rgba(240,166,74,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  // torso
  const torso = ctx.createLinearGradient(0, Y(11), 0, Y(23));
  torso.addColorStop(0, "rgba(201,206,214,0.22)");
  torso.addColorStop(1, "rgba(201,206,214,0.04)");
  ctx.fillStyle = torso;
  ctx.strokeStyle = "rgba(201,206,214,0.55)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(X(12), Y(12));
  ctx.lineTo(X(11), Y(11));
  ctx.lineTo(X(23), Y(23));
  ctx.lineTo(X(24), Y(24));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // neck + head
  const hx = (X(0) + (X(7) + X(8)) / 2) / 2;
  const hy = Y(0) - 0.08 * S;
  ctx.beginPath();
  ctx.moveTo(ox, oy);
  ctx.lineTo(hx, hy + 0.3 * S);
  ctx.stroke();
  const head = ctx.createRadialGradient(hx - 10, hy - 12, 6, hx, hy, 0.34 * S);
  head.addColorStop(0, "rgba(238,241,245,0.30)");
  head.addColorStop(1, "rgba(201,206,214,0.08)");
  ctx.fillStyle = head;
  ctx.beginPath();
  ctx.ellipse(hx, hy, 0.29 * S, 0.34 * S, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(238,241,245,0.6)";
  ctx.stroke();

  // face: eyes, brows, mouth (non-manual signals)
  ctx.fillStyle = "#F0A64A";
  for (const i of [2, 5]) {
    ctx.beginPath();
    ctx.arc(X(i), Y(i), 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = "rgba(238,241,245,0.85)";
  ctx.lineWidth = 2.2;
  for (const [a, b] of [
    [FACE0 + 8, FACE0 + 12],
    [FACE0 + 13, FACE0 + 17],
  ]) {
    ctx.beginPath();
    for (let i = a; i <= b; i++) (i === a ? ctx.moveTo : ctx.lineTo).call(ctx, X(i), Y(i));
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(X(FACE0), Y(FACE0));
  ctx.quadraticCurveTo(X(FACE0 + 2), Y(FACE0 + 2), X(FACE0 + 1), Y(FACE0 + 1));
  ctx.quadraticCurveTo(X(FACE0 + 3), Y(FACE0 + 3), X(FACE0), Y(FACE0));
  ctx.stroke();

  // arms
  ctx.strokeStyle = "rgba(220,224,232,0.9)";
  ctx.lineCap = "round";
  ctx.lineWidth = 9;
  for (const [a, b, c] of [
    [11, 13, 15],
    [12, 14, 16],
  ]) {
    ctx.beginPath();
    ctx.moveTo(X(a), Y(a));
    ctx.lineTo(X(b), Y(b));
    ctx.lineTo(X(c), Y(c));
    ctx.stroke();
  }

  // hands
  for (const h0 of [LH0, RH0]) {
    ctx.strokeStyle = "rgba(238,241,245,0.95)";
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    for (const [a, b] of HAND_EDGES) {
      ctx.moveTo(X(h0 + a), Y(h0 + a));
      ctx.lineTo(X(h0 + b), Y(h0 + b));
    }
    ctx.stroke();
    ctx.fillStyle = "#EEF1F5";
    for (let k = 0; k < 21; k++) {
      ctx.beginPath();
      ctx.arc(X(h0 + k), Y(h0 + k), 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
    for (const k of FINGERTIPS) {
      ctx.fillStyle = "rgba(240,166,74,0.35)";
      ctx.beginPath();
      ctx.arc(X(h0 + k), Y(h0 + k), 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#F0A64A";
      ctx.beginPath();
      ctx.arc(X(h0 + k), Y(h0 + k), 3.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // close-up of the hands while finger-spelling, so the handshape is easy to read
  if (letter) {
    const pts: [number, number][] = [];
    for (const h0 of [LH0, RH0]) {
      if (p[h0][1] > 1.5) continue; // a hand resting at the side is not part of the letter
      for (let k = 0; k < 21; k++) pts.push(p[h0 + k]);
    }
    if (pts.length) {
      const xs = pts.map((q) => q[0]);
      const ys = pts.map((q) => q[1]);
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
      const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 0.35);
      const box = 190;
      const bx = W - box - 24;
      const by = H - box - 24;
      const z = (box - 36) / span;
      ctx.save();
      ctx.fillStyle = "rgba(9,10,13,0.85)";
      ctx.strokeStyle = "rgba(201,206,214,0.35)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(bx, by, box, box, 16);
      ctx.fill();
      ctx.stroke();
      ctx.clip();
      const ZX = (i: number) => bx + box / 2 + (p[i][0] - cx) * z;
      const ZY = (i: number) => by + box / 2 + (p[i][1] - cy) * z;
      for (const [h0, col] of [
        [LH0, "rgba(201,206,214,0.95)"],
        [RH0, "rgba(255,214,160,0.95)"],
      ] as const) {
        if (p[h0][1] > 1.5) continue;
        ctx.strokeStyle = col;
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.beginPath();
        for (const [a, b] of HAND_EDGES) {
          ctx.moveTo(ZX(h0 + a), ZY(h0 + a));
          ctx.lineTo(ZX(h0 + b), ZY(h0 + b));
        }
        ctx.stroke();
        for (const k of FINGERTIPS) {
          ctx.fillStyle = "#F0A64A";
          ctx.beginPath();
          ctx.arc(ZX(h0 + k), ZY(h0 + k), 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
      ctx.fillStyle = "rgba(161,161,170,0.9)";
      ctx.font = "500 11px Inter, Arial, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("CLOSE-UP", bx + 12, by + 20);
    }
  }

  // letter card while finger-spelling
  if (letter) {
    ctx.fillStyle = "rgba(9,10,13,0.75)";
    ctx.strokeStyle = "rgba(240,166,74,0.8)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(W - 128, 24, 96, 112, 18);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#FFC27A";
    ctx.font = "600 64px Inter, Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(letter, W - 80, 82);
  }
}

/** Plays the avatar's key poses with eased transitions. */
export default function AvatarStage({ steps, playKey, onStep, onDone }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cb = useRef({ onStep, onDone });
  cb.current = { onStep, onDone };

  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    draw(ctx, REST);
    if (!steps || !steps.length) return;
    let raf = 0;
    let i = 0;
    let from: Pose = REST;
    let t0 = performance.now();
    cb.current.onStep?.(0);
    const tick = (now: number) => {
      const st = steps[i];
      const el = now - t0;
      if (el < st.move) {
        draw(ctx, lerpPose(from, st.pose, ease(Math.min(1, el / Math.max(1, st.move)))), st.letter);
      } else if (el < st.move + st.hold) {
        draw(ctx, st.pose, st.letter);
      } else {
        from = st.pose;
        i += 1;
        t0 = now;
        if (i >= steps.length) {
          draw(ctx, REST);
          cb.current.onDone?.();
          return;
        }
        cb.current.onStep?.(i);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [steps, playKey]);

  return <canvas ref={ref} width={W} height={H} className="h-auto w-full max-w-[520px]" aria-label="Signing avatar" />;
}
