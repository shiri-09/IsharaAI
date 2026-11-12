import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Maximize2, Minimize2, SkipForward } from "lucide-react";
import { usePrefs } from "../lib/prefs";

/**
 * Cinematic introduction (about 45 seconds, silent, black and silver, made for full screen).
 * One particle field is the whole picture: dust gathers into "Hello.", then two hands, a waveform, a keyboard being typed with
 * an eye, the agent's four steps, and finally the title. Serif film-title type, letterbox bars, slow camera push, grain.
 * F = full screen, Esc = skip, any key on the last card = begin.
 */

const SESSION_KEY = "ishara.intro.seen";
export const introSeen = () => {
  try {
    return sessionStorage.getItem(SESSION_KEY) === "1";
  } catch {
    return true;
  }
};

type P = { x: number; y: number; u?: number; b?: number; hot?: boolean };

const HAND = [
  "200 560 140 505 100 445 75 385 55 330",
  "200 560 150 360 140 265 135 205 130 150",
  "150 360 200 348 200 240 200 174 200 114",
  "200 348 250 360 258 258 263 198 267 144",
  "250 360 295 384 315 306 327 258 337 216",
  "200 560 295 384",
];
const SERIF = '"Cormorant Garamond", Georgia, "Times New Roman", serif';

// ------------------------------------------------------------------ shape builders
function alongPoly(poly: P[], step: number): P[] {
  const out: P[] = [];
  for (let i = 0; i < poly.length - 1; i++) {
    const a = poly[i];
    const b = poly[i + 1];
    const n = Math.max(1, Math.round(Math.hypot(b.x - a.x, b.y - a.y) / step));
    for (let k = 0; k < n; k++) out.push({ x: a.x + ((b.x - a.x) * k) / n, y: a.y + ((b.y - a.y) * k) / n });
  }
  out.push(poly[poly.length - 1]);
  return out;
}

function textPoints(text: string, widthPx: number, font: string, step: number): P[] {
  const c = document.createElement("canvas");
  const x = c.getContext("2d")!;
  const fs = 240;
  const f = font.replace("SIZE", `${fs}px`);
  x.font = f;
  const w = Math.ceil(x.measureText(text).width) + 30;
  c.width = w;
  c.height = Math.ceil(fs * 1.5);
  x.font = f;
  x.fillStyle = "#fff";
  x.fillText(text, 15, fs * 1.1);
  const d = x.getImageData(0, 0, c.width, c.height).data;
  const pts: P[] = [];
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (let yy = 0; yy < c.height; yy += step)
    for (let xx = 0; xx < c.width; xx += step)
      if (d[(yy * c.width + xx) * 4 + 3] > 140) {
        pts.push({ x: xx, y: yy });
        x0 = Math.min(x0, xx); x1 = Math.max(x1, xx); y0 = Math.min(y0, yy); y1 = Math.max(y1, yy);
      }
  const s = widthPx / Math.max(1, x1 - x0);
  return pts.map((p) => ({ x: (p.x - (x0 + x1) / 2) * s, y: (p.y - (y0 + y1) / 2) * s }));
}

type Shape = { pts: P[]; dyn?: (i: number, t: number, p: P) => { x: number; y: number }; };
type Built = { shapes: Record<string, Shape>; keys: Record<string, { x: number; y: number }>; keyW: number; nodes: { x: number; y: number }[]; nodeR: number };

function build(W: number, H: number): Built {
  const cx = W / 2;
  const cy = H / 2;
  const shapes: Record<string, Shape> = {};

  // "Hello."
  shapes.hello = { pts: textPoints("Hello.", Math.min(W * 0.46, H * 1.05), `italic 500 SIZE ${SERIF}`, 3).map((p) => ({ x: cx + p.x, y: cy - H * 0.04 + p.y })) };

  // two hands built from the same landmark lines the reader sees
  const hands: P[] = [];
  const s = (H * 0.6) / 450;
  const mk = (ox: number, mirror: number, rot: number) => {
    for (const line of HAND) {
      const n = line.split(" ").map(Number);
      const poly: P[] = [];
      for (let i = 0; i < n.length; i += 2) {
        const hx = (n[i] - 200) * mirror;
        const hy = n[i + 1] - 337;
        poly.push({ x: ox + (hx * Math.cos(rot) - hy * Math.sin(rot)) * s, y: cy + (hx * Math.sin(rot) + hy * Math.cos(rot)) * s });
      }
      alongPoly(poly, 5).forEach((p) => hands.push(p));
      poly.forEach((p) => { for (let k = 0; k < 5; k++) hands.push({ x: p.x + (Math.random() - 0.5) * 6, y: p.y + (Math.random() - 0.5) * 6, hot: true }); });
    }
  };
  mk(W * 0.33, -1, 0.14);
  mk(W * 0.67, 1, -0.14);
  shapes.hands = { pts: hands };

  // waveform
  const bars = 63;
  const wave: P[] = [];
  const perBar = 24;
  for (let b = 0; b < bars; b++)
    for (let k = 0; k < perBar; k++) {
      const u = (k / (perBar - 1)) * 2 - 1;
      wave.push({ x: cx + ((b - (bars - 1) / 2) / (bars - 1)) * W * 0.62, y: cy + u * H * 0.05, u, b });
    }
  shapes.wave = {
    pts: wave,
    dyn: (_i, t, p) => {
      const b = p.b ?? 0;
      const env = Math.exp(-(((b - 31) / 22) ** 2));
      const amp = H * 0.2 * env * (0.25 + 0.75 * Math.abs(Math.sin(t * 2.6 + b * 0.5) * Math.sin(t * 1.1 + b * 0.19 + 1)));
      return { x: p.x, y: cy + (p.u ?? 0) * amp };
    },
  };

  // keyboard
  const rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"];
  const kw = Math.min(W * 0.052, H * 0.085);
  const gap = kw * 0.2;
  const keys: Record<string, { x: number; y: number }> = {};
  const kb: P[] = [];
  rows.forEach((row, r) => {
    const rowW = row.length * kw + (row.length - 1) * gap;
    row.split("").forEach((ch, i) => {
      const x0 = cx - rowW / 2 + i * (kw + gap);
      const y0 = H * 0.5 + r * (kw + gap);
      keys[ch] = { x: x0 + kw / 2, y: y0 + kw / 2 };
      const rect = [{ x: x0, y: y0 }, { x: x0 + kw, y: y0 }, { x: x0 + kw, y: y0 + kw }, { x: x0, y: y0 + kw }, { x: x0, y: y0 }];
      alongPoly(rect, 5).forEach((p) => kb.push(p));
    });
  });
  shapes.keys = { pts: kb };

  // the agent's four steps
  const nodes = [0.2, 0.4, 0.6, 0.8].map((f) => ({ x: W * f, y: cy + H * 0.02 }));
  const nodeR = H * 0.07;
  const net: P[] = [];
  nodes.forEach((n, i) => {
    for (let k = 0; k < 110; k++) {
      const a = (k / 110) * Math.PI * 2;
      net.push({ x: n.x + Math.cos(a) * nodeR, y: n.y + Math.sin(a) * nodeR, hot: true });
    }
    if (i < 3) alongPoly([{ x: n.x + nodeR, y: n.y }, { x: nodes[i + 1].x - nodeR, y: n.y }], 6).forEach((p) => net.push(p));
  });
  shapes.net = { pts: net };

  // title
  shapes.title = { pts: textPoints("ISHARA AI", Math.min(W * 0.66, H * 1.5), `600 SIZE ${SERIF}`, 3).map((p) => ({ x: cx + p.x, y: cy - H * 0.1 + p.y })) };
  return { shapes, keys, keyW: kw, nodes, nodeR };
}

// ------------------------------------------------------------------ particle engine
class Engine {
  N = 1700;
  W = 0;
  H = 0;
  dpr = 1;
  px = new Float32Array(this.N);
  py = new Float32Array(this.N);
  vx = new Float32Array(this.N);
  vy = new Float32Array(this.N);
  tx = new Float32Array(this.N);
  ty = new Float32Array(this.N);
  al = new Float32Array(this.N);
  alT = new Float32Array(this.N);
  sz = new Float32Array(this.N);
  go = new Float32Array(this.N); // time each particle starts moving
  par: P[] = [];
  dyn: Shape["dyn"] = undefined;
  bokeh: { x: number; y: number; r: number; v: number; a: number }[] = [];
  sprite: HTMLCanvasElement;

  constructor() {
    const s = document.createElement("canvas");
    s.width = s.height = 64;
    const x = s.getContext("2d")!;
    const g = x.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.18, "rgba(235,238,244,0.85)");
    g.addColorStop(0.5, "rgba(190,196,206,0.16)");
    g.addColorStop(1, "rgba(190,196,206,0)");
    x.fillStyle = g;
    x.fillRect(0, 0, 64, 64);
    this.sprite = s;
  }

  resize(W: number, H: number, dpr: number) {
    const first = this.W === 0;
    this.W = W;
    this.H = H;
    this.dpr = dpr;
    if (first) {
      for (let i = 0; i < this.N; i++) {
        this.px[i] = Math.random() * W;
        this.py[i] = Math.random() * H;
        this.al[i] = 0.12 + Math.random() * 0.3;
        this.alT[i] = this.al[i];
        this.tx[i] = this.px[i];
        this.ty[i] = this.py[i];
        this.sz[i] = 4 + Math.random() * 5;
      }
      this.bokeh = Array.from({ length: 22 }, () => ({ x: Math.random() * W, y: Math.random() * H, r: 30 + Math.random() * 90, v: 3 + Math.random() * 10, a: 0.025 + Math.random() * 0.05 }));
    }
  }

  setShape(sh: Shape, now: number) {
    const pts = sh.pts;
    this.dyn = sh.dyn;
    const M = pts.length;
    // choose N points (random subset when there are more, repeat with jitter when fewer)
    for (let i = 0; i < this.N; i++) {
      const p = M >= this.N ? pts[(Math.random() * M) | 0] : pts[i % M];
      this.par[i] = p;
      const jit = M >= this.N ? 0.4 : 1.6;
      this.tx[i] = p.x + (Math.random() - 0.5) * jit;
      this.ty[i] = p.y + (Math.random() - 0.5) * jit;
      this.alT[i] = p.hot ? 1 : 0.62 + Math.random() * 0.3;
      this.sz[i] = p.hot ? 7 + Math.random() * 3 : 4 + Math.random() * 3;
      this.go[i] = now + Math.random() * 0.3;
    }
  }

  step(dt: number, t: number) {
    const K = 40;
    const D = 8.6;
    const breathe = 1 + 0.008 * Math.sin(t * 1.3);
    const cx = this.W / 2;
    const cy = this.H / 2;
    for (let i = 0; i < this.N; i++) {
      if (t < this.go[i]) continue;
      let tx = this.tx[i];
      let ty = this.ty[i];
      if (this.dyn) {
        const d = this.dyn(i, t, this.par[i]);
        tx = d.x;
        ty = d.y;
      }
      tx = cx + (tx - cx) * breathe + Math.sin(t * 0.9 + i * 1.7) * 0.9;
      ty = cy + (ty - cy) * breathe + Math.cos(t * 0.8 + i * 2.3) * 0.9;
      this.vx[i] += (tx - this.px[i]) * K * dt;
      this.vy[i] += (ty - this.py[i]) * K * dt;
      const damp = Math.exp(-D * dt);
      this.vx[i] *= damp;
      this.vy[i] *= damp;
      this.px[i] += this.vx[i] * dt;
      this.py[i] += this.vy[i] * dt;
      this.al[i] += (this.alT[i] - this.al[i]) * Math.min(1, dt * 5);
    }
    for (const b of this.bokeh) {
      b.y -= b.v * dt;
      if (b.y < -b.r) {
        b.y = this.H + b.r;
        b.x = Math.random() * this.W;
      }
    }
  }

  draw(ctx: CanvasRenderingContext2D) {
    const d = this.dpr;
    ctx.globalCompositeOperation = "lighter";
    for (const b of this.bokeh) {
      ctx.globalAlpha = b.a;
      ctx.drawImage(this.sprite, (b.x - b.r) * d, (b.y - b.r) * d, b.r * 2 * d, b.r * 2 * d);
    }
    for (let i = 0; i < this.N; i++) {
      const s = this.sz[i] * 2.6 * d;
      ctx.globalAlpha = this.al[i];
      ctx.drawImage(this.sprite, this.px[i] * d - s / 2, this.py[i] * d - s / 2, s, s);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }
}

function gazeCircle(ctx: CanvasRenderingContext2D, x: number, y: number, prog: number, d: number) {
  const red = prog > 0.02;
  ctx.save();
  ctx.lineWidth = 3 * d;
  ctx.fillStyle = red ? "rgba(248,113,113,0.22)" : "rgba(235,238,243,0.12)";
  ctx.strokeStyle = red ? "#ef4444" : "#d4d8de";
  ctx.beginPath();
  ctx.arc(x * d, y * d, 26 * d, 0, 7);
  ctx.fill();
  ctx.stroke();
  if (red) {
    ctx.lineWidth = 4 * d;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.arc(x * d, y * d, 32 * d, -Math.PI / 2, -Math.PI / 2 + prog * Math.PI * 2);
    ctx.stroke();
  }
  ctx.fillStyle = red ? "#fecaca" : "#f4f5f7";
  ctx.beginPath();
  ctx.arc(x * d, y * d, 3.5 * d, 0, 7);
  ctx.fill();
  ctx.restore();
}

// ------------------------------------------------------------------ scenes (text and timing)
const Line = ({ children, d = 0, className = "" }: { children: React.ReactNode; d?: number; className?: string }) => (
  <span className={`cin-mask ${className}`}>
    <span style={{ animationDelay: `${d}s` }}>{children}</span>
  </span>
);
const Mono = ({ children, d = 0 }: { children: React.ReactNode; d?: number }) => (
  <span className="cin-fade block font-mono text-[11px] uppercase tracking-[0.4em] text-zinc-400 md:text-sm" style={{ animationDelay: `${d}s` }}>
    {children}
  </span>
);

type Scene = { name: string; ms: number; view: (typed: string) => React.ReactNode };

const SCENES: Scene[] = [
  { name: "hello", ms: 4000, view: () => (
    <div className="absolute inset-x-0 bottom-[17vh] space-y-2 text-center">
      <p className="cin-serif text-xl italic text-zinc-300 md:text-3xl"><Line d={0.9}>It takes one second to say.</Line></p>
      <p className="cin-serif text-xl italic text-zinc-500 md:text-3xl"><Line d={1.8}>For some people, it takes an interpreter.</Line></p>
    </div>
  ) },
  { name: "hands", ms: 5000, view: () => (
    <div className="absolute inset-x-0 bottom-[14vh] space-y-3 text-center">
      <p className="cin-serif cin-silver text-4xl font-medium md:text-6xl"><Line d={0.3}>It reads sign language.</Line></p>
      <Mono d={0.9}>Whole sentences · straight from a webcam</Mono>
    </div>
  ) },
  { name: "wave", ms: 4800, view: () => (
    <>
      <div className="absolute inset-x-0 bottom-[14vh] space-y-3 text-center">
        <p className="cin-serif cin-silver text-4xl font-medium md:text-6xl"><Line d={0.4}>You speak. It signs back.</Line></p>
        <Mono d={0.9}>And answers in the language you know</Mono>
      </div>
      {[["नमस्ते", "left-[7vw] top-[18vh]", 0.5], ["ನಮಸ್ಕಾರ", "right-[7vw] top-[18vh]", 0.8], ["வணக்கம்", "left-[7vw] bottom-[30vh]", 1.1], ["నమస్కారం", "right-[7vw] bottom-[30vh]", 1.4]].map(([w, pos, d]) => (
        <span key={w as string} className={`cin-fade absolute ${pos} text-3xl text-zinc-400 md:text-5xl`} style={{ animationDelay: `${d}s` }}>{w}</span>
      ))}
    </>
  ) },
  { name: "keys", ms: 5600, view: (typed) => (
    <>
      <div className="absolute inset-x-0 top-[14vh] space-y-3 text-center">
        <p className="cin-serif cin-silver text-4xl font-medium md:text-6xl"><Line d={0.2}>Eyes that type.</Line></p>
        <Mono d={0.9}>Eye tracking · on-screen keyboard</Mono>
      </div>
      <div className="absolute inset-x-0 top-[34vh] text-center font-mono text-4xl tracking-[0.5em] text-white md:text-6xl">{typed}<span className="cin-caret">|</span></div>
    </>
  ) },
  { name: "net", ms: 5200, view: () => (
    <div className="absolute inset-x-0 top-[16vh] space-y-3 text-center">
      <p className="cin-serif cin-silver text-4xl font-medium md:text-6xl"><Line d={0.2}>An agent that asks first,</Line></p>
      <p className="cin-serif cin-silver text-4xl font-medium italic md:text-6xl"><Line d={0.45}>then acts.</Line></p>
    </div>
  ) },
  { name: "title", ms: 12000, view: () => (
    <div className="absolute inset-x-0 bottom-[10vh] space-y-3 text-center">
      <p className="cin-serif cin-fade text-2xl italic text-zinc-200 md:text-4xl" style={{ animationDelay: "1.3s" }}>Sign. Speak. Understand.</p>
      <Mono d={1.9}>Team Quad Force</Mono>
      <p className="cin-fade text-base text-zinc-400 md:text-xl" style={{ animationDelay: "2.3s" }}>Vijay M · Sriraksha · Atharv S · Amruta K Halemani</p>
      <Mono d={2.7}>CodeSpark 2026 · AMC Engineering College</Mono>
      <p className="cin-blink pt-3 font-mono text-xs uppercase tracking-[0.35em] text-zinc-200">Press any key to begin</p>
    </div>
  ) },
];

export default function Cinema() {
  const nav = useNavigate();
  const { prefs } = usePrefs();
  const [i, setI] = useState(0);
  const [leaving, setLeaving] = useState(false);
  const [typed, setTyped] = useState("");
  const [full, setFull] = useState(!!document.fullscreenElement);
  const [hint, setHint] = useState(true);
  const [ready, setReady] = useState(false);
  const done = useRef(false);
  const cv = useRef<HTMLCanvasElement>(null);
  const eng = useRef<Engine | null>(null);
  const built = useRef<Built | null>(null);
  const sceneT0 = useRef(0);
  const sceneIdx = useRef(0);
  sceneIdx.current = i;

  const finish = useCallback(() => {
    if (done.current) return;
    done.current = true;
    try {
      sessionStorage.setItem(SESSION_KEY, "1");
    } catch {
      /* ignore */
    }
    if (document.fullscreenElement) document.exitFullscreen().catch(() => undefined);
    nav(prefs?.onboarded ? "/" : "/setup", { replace: true });
  }, [nav, prefs?.onboarded]);

  // canvas, engine and shapes
  useEffect(() => {
    const c = cv.current!;
    const ctx = c.getContext("2d")!;
    const e = new Engine();
    eng.current = e;
    let raf = 0;
    let last = performance.now();
    let alive = true;
    const setup = async () => {
      try {
        await Promise.race([document.fonts.load(`600 220px ${SERIF}`), new Promise((r) => setTimeout(r, 1500))]);
        await Promise.race([document.fonts.load(`italic 500 220px ${SERIF}`), new Promise((r) => setTimeout(r, 800))]);
      } catch {
        /* fall back to the system serif */
      }
      if (!alive) return;
      const W = innerWidth;
      const H = innerHeight;
      const dpr = Math.min(2, devicePixelRatio || 1);
      c.width = W * dpr;
      c.height = H * dpr;
      e.resize(W, H, dpr);
      built.current = build(W, H);
      e.setShape(built.current.shapes[SCENES[sceneIdx.current].name], performance.now() / 1000);
      sceneT0.current = performance.now() / 1000;
      setReady(true);
    };
    setup();
    const frame = (nowMs: number) => {
      const now = nowMs / 1000;
      const dt = Math.min(0.05, (nowMs - last) / 1000);
      last = nowMs;
      const W = e.W;
      const H = e.H;
      if (W && built.current) {
        e.step(dt, now);
        ctx.clearRect(0, 0, c.width, c.height);
        e.draw(ctx);
        // overlays that belong to a scene
        const name = SCENES[sceneIdx.current].name;
        const st = now - sceneT0.current;
        const d = e.dpr;
        const B = built.current;
        if (name === "keys") {
          ctx.fillStyle = `rgba(226,230,237,${Math.min(0.7, Math.max(0, st - 0.3) * 0.8)})`;
          ctx.font = `${Math.round(B.keyW * 0.42 * d)}px Inter, sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          for (const [ch, p] of Object.entries(B.keys)) ctx.fillText(ch, p.x * d, p.y * d + d);
          // the eye: glide to each letter of HELLO, then dwell (ring fills and turns red)
          const word = "HELLO";
          const k = Math.floor((st - 0.6) / 0.65);
          const start = { x: W / 2, y: H * 0.42 };
          if (st < 0.6) gazeCircle(ctx, start.x, start.y, 0, d);
          else {
            const idx = Math.min(word.length - 1, Math.max(0, k));
            const prev = idx === 0 ? start : B.keys[word[idx - 1]];
            const cur = B.keys[word[idx]];
            const local = Math.min(1, Math.max(0, (st - 0.6 - idx * 0.65) / 0.65));
            const move = Math.min(1, local / 0.3);
            const e1 = 1 - Math.pow(1 - move, 3);
            const prog = local < 0.3 ? 0 : Math.min(1, (local - 0.3) / 0.62);
            gazeCircle(ctx, prev.x + (cur.x - prev.x) * e1, prev.y + (cur.y - prev.y) * e1, prog, d);
          }
        } else if (name === "net") {
          const labels = ["Plan", "Ask", "Do", "Verify"];
          ctx.textAlign = "center";
          ctx.textBaseline = "alphabetic";
          B.nodes.forEach((n, j) => {
            const on = st > 0.7 + j * 0.75;
            ctx.fillStyle = on ? "#f4f5f7" : "rgba(160,166,176,0.7)";
            ctx.font = `${Math.round(H * 0.034 * d)}px ${SERIF}`;
            ctx.fillText(labels[j], n.x * d, (n.y + B.nodeR + H * 0.06) * d);
            if (on) {
              ctx.strokeStyle = "#34d399";
              ctx.lineWidth = 3.5 * d;
              ctx.lineCap = "round";
              ctx.beginPath();
              ctx.moveTo((n.x - 10) * d, n.y * d);
              ctx.lineTo((n.x - 2) * d, (n.y + 9) * d);
              ctx.lineTo((n.x + 13) * d, (n.y - 10) * d);
              ctx.stroke();
            }
          });
          const seg = Math.min(2.999, Math.max(0, (st - 0.7) / 0.75));
          if (st > 0.7 && seg < 3) {
            const j = Math.floor(seg);
            const f = seg - j;
            const a = B.nodes[j];
            const b = B.nodes[j + 1];
            const x = a.x + B.nodeR + (b.x - a.x - 2 * B.nodeR) * f;
            const g = ctx.createRadialGradient(x * d, a.y * d, 0, x * d, a.y * d, 22 * d);
            g.addColorStop(0, "rgba(255,255,255,0.95)");
            g.addColorStop(1, "rgba(255,255,255,0)");
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(x * d, a.y * d, 22 * d, 0, 7);
            ctx.fill();
          }
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    const onResize = () => {
      if (!built.current) return;
      const W = innerWidth;
      const H = innerHeight;
      const dpr = Math.min(2, devicePixelRatio || 1);
      c.width = W * dpr;
      c.height = H * dpr;
      e.W = W;
      e.H = H;
      e.dpr = dpr;
      built.current = build(W, H);
      e.setShape(built.current.shapes[SCENES[sceneIdx.current].name], performance.now() / 1000);
    };
    addEventListener("resize", onResize);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      removeEventListener("resize", onResize);
    };
  }, []);

  // advance scenes
  useEffect(() => {
    if (!ready) return;
    const now = performance.now() / 1000;
    sceneT0.current = now;
    setLeaving(false);
    if (built.current && eng.current) eng.current.setShape(built.current.shapes[SCENES[i].name], now);
    setTyped("");
    const timers: number[] = [];
    if (SCENES[i].name === "keys") "HELLO".split("").forEach((_, k) => timers.push(window.setTimeout(() => setTyped("HELLO".slice(0, k + 1)), (0.6 + (k + 1) * 0.65) * 1000)));
    const last = i === SCENES.length - 1;
    timers.push(window.setTimeout(() => setLeaving(true), SCENES[i].ms - 450));
    timers.push(window.setTimeout(() => (last ? finish() : setI(i + 1)), SCENES[i].ms));
    return () => timers.forEach(clearTimeout);
  }, [i, ready, finish]);

  useEffect(() => {
    const h = setTimeout(() => setHint(false), 5000);
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "f") {
        document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen().catch(() => undefined);
      } else if (e.key === "Escape" && !document.fullscreenElement) finish();
      else if (e.key === "ArrowRight") setI((x) => Math.min(SCENES.length - 1, x + 1));
      else if (e.key !== "Escape" && sceneIdx.current === SCENES.length - 1) finish();
    };
    const onFs = () => setFull(!!document.fullscreenElement);
    addEventListener("keydown", onKey);
    document.addEventListener("fullscreenchange", onFs);
    return () => {
      clearTimeout(h);
      removeEventListener("keydown", onKey);
      document.removeEventListener("fullscreenchange", onFs);
    };
  }, [finish]);

  const last = i === SCENES.length - 1;
  return (
    <div className="fixed inset-0 z-[200] cursor-none overflow-hidden bg-black" onClick={() => last && finish()}>
      <canvas ref={cv} className="cin-canvas absolute inset-0 h-full w-full" />
      <div className="cin-camera absolute inset-0" key={`cam${i}`}>
        {ready && (
          <div className={`absolute inset-0 ${leaving && !last ? "cin-out" : ""}`} key={i}>
            {SCENES[i].view(typed)}
          </div>
        )}
      </div>
      {ready && <div key={`fl${i}`} className="cin-streak pointer-events-none absolute left-0 right-0 top-1/2 h-px bg-gradient-to-r from-transparent via-white to-transparent" />}
      {/* film framing */}
      <div className="cin-bar-top pointer-events-none absolute inset-x-0 top-0 h-[7vh] bg-black" />
      <div className="cin-bar-bottom pointer-events-none absolute inset-x-0 bottom-0 h-[7vh] bg-black" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_40%,rgba(0,0,0,0.8)_100%)]" />
      <div className="cin-grain pointer-events-none absolute inset-0 opacity-[0.06] mix-blend-screen" />
      <div className="absolute inset-x-0 bottom-[7vh] h-px bg-white/10">
        <div className="h-full bg-gradient-to-r from-zinc-600 to-white transition-[width] duration-500" style={{ width: `${((i + 1) / SCENES.length) * 100}%` }} />
      </div>
      <div className={`absolute bottom-[calc(7vh+14px)] right-6 flex cursor-default items-center gap-2 transition-opacity duration-500 ${hint ? "opacity-100" : "opacity-0 hover:opacity-100"}`} onClick={(e) => e.stopPropagation()}>
        <button className="flex items-center gap-2 rounded-full border border-white/15 bg-black/60 px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-zinc-300 hover:text-white" onClick={() => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen().catch(() => undefined))}>
          {full ? <Minimize2 size={13} /> : <Maximize2 size={13} />} {full ? "Exit full screen" : "Full screen (F)"}
        </button>
        <button className="flex items-center gap-2 rounded-full border border-white/15 bg-black/60 px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-zinc-300 hover:text-white" onClick={finish}>
          <SkipForward size={13} /> Skip (Esc)
        </button>
      </div>
    </div>
  );
}
