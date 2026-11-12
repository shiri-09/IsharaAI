import { useId } from "react";

export default function Logo({ size = 32 }: { size?: number }) {
  const gid = `chrome-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  return (
    <div className="flex items-center gap-3">
      <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#ffffff" />
            <stop offset="0.5" stopColor="#c9ced6" />
            <stop offset="1" stopColor="#6b7280" />
          </linearGradient>
        </defs>
        <rect x="1" y="1" width="30" height="30" rx="9" fill="#0d0f12" stroke={`url(#${gid})`} strokeOpacity="0.5" />
        <path
          d="M11 22V12.5a1.5 1.5 0 0 1 3 0V17m0-6.5a1.5 1.5 0 0 1 3 0V17m0-5a1.5 1.5 0 0 1 3 0V18c0 3-2 5-5 5h-1c-2 0-3-1-4-2.5L8.7 17.6a1.4 1.4 0 0 1 2.3-1.6"
          fill="none"
          stroke={`url(#${gid})`}
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <span className="chrome-text text-lg font-semibold tracking-tight">Ishara AI</span>
    </div>
  );
}
