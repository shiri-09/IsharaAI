import type { ReactNode, RefObject } from "react";
import { Camera, CameraOff, Loader2 } from "lucide-react";

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  status: "idle" | "loading" | "running" | "error";
  error: string;
  fps: number;
  onStart: () => void;
  onStop: () => void;
  overlay?: ReactNode;
  footer?: ReactNode;
};

export default function CameraStage({ videoRef, canvasRef, status, error, fps, onStart, onStop, overlay, footer }: Props) {
  return (
    <div className="glass overflow-hidden">
      <div className="relative aspect-[4/3] w-full bg-black">
        <video ref={videoRef} playsInline muted className="absolute inset-0 h-full w-full -scale-x-100 object-cover" />
        <canvas ref={canvasRef} className="absolute inset-0 h-full w-full -scale-x-100 object-cover" />
        <div className="pointer-events-none absolute inset-0 rounded-none ring-1 ring-inset ring-white/5" />

        {status !== "running" && (
          <div className="absolute inset-0 grid place-items-center">
            <div className="flex flex-col items-center gap-4 text-center">
              {status === "loading" ? (
                <>
                  <Loader2 className="animate-spin text-zinc-400" />
                  <p className="text-sm text-zinc-500">Loading camera and MediaPipe Holistic…</p>
                </>
              ) : (
                <>
                  <div className="grid h-16 w-16 place-items-center rounded-2xl border border-white/10 bg-white/[0.03]">
                    <Camera className="text-zinc-400" />
                  </div>
                  <button className="btn btn-chrome" onClick={onStart}>
                    Start camera
                  </button>
                  {error && <p className="max-w-xs text-xs text-red-400/80">{error}</p>}
                </>
              )}
            </div>
          </div>
        )}

        {status === "running" && (
          <>
            <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-3 py-1 font-mono text-[11px] text-zinc-400 backdrop-blur">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> LIVE · {fps} fps
            </div>
            <button
              className="absolute right-3 top-3 rounded-full border border-white/10 bg-black/60 p-2 text-zinc-400 backdrop-blur hover:text-zinc-100"
              onClick={onStop}
              title="Stop camera"
            >
              <CameraOff size={16} />
            </button>
            {overlay}
          </>
        )}
      </div>
      {footer && <div className="border-t border-white/[0.06] p-4">{footer}</div>}
    </div>
  );
}
