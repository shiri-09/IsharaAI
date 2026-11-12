import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import {
  BookOpenCheck,
  Bot,
  CaseUpper,
  ChevronsLeft,
  ChevronsRight,
  Database,
  History as HistoryIcon,
  LogOut,
  Menu,
  PlayCircle,
  Radio,
  Repeat,
  ScanFace,
  ScanLine,
  Settings as SettingsIcon,
  Waves,
  X,
  type LucideIcon,
} from "lucide-react";
import Logo from "./Logo";
import GazeProvider from "./GazeProvider";
import { useAuth } from "../lib/auth";
import { usePrefs } from "../lib/prefs";

type Item = { to: string; label: string; icon: LucideIcon; hint?: string };
type Group = { title: string; items: Item[] };

/**
 * Two completely separate experiences share one account:
 *  - ISL users get the Sign Studio (camera, mouse, keyboard) and its tools.
 *  - AAC users get only the eyes-and-face board; no sign-language screens exist for them at all.
 */
const ISL_NAV: Group[] = [
  {
    title: "Modes",
    items: [
      { to: "/studio", label: "AI conversation", icon: ScanLine, hint: "Sign or upload, chat with the AI" },
      { to: "/live", label: "Real-time speech", icon: Radio, hint: "Live signer and speaker interpreter" },
      { to: "/agent", label: "Agentic orchestration", icon: Bot, hint: "Plan, confirm, do, verify" },
    ],
  },
  {
    title: "Tools",
    items: [
      { to: "/reply", label: "Reply in sign", icon: Waves, hint: "Speech or text to sign avatar" },
      { to: "/spell", label: "Fingerspelling", icon: CaseUpper, hint: "Names and short words" },
    ],
  },
  {
    title: "Learn",
    items: [
      { to: "/library", label: "Sign library", icon: BookOpenCheck, hint: "What the reader knows" },
      { to: "/collect", label: "Train", icon: Database, hint: "Record and retrain" },
    ],
  },
  {
    title: "You",
    items: [
      { to: "/history", label: "History", icon: HistoryIcon },
      { to: "/settings", label: "Settings", icon: SettingsIcon },
    ],
  },
];
const AAC_NAV: Group[] = [
  {
    title: "Modes (eyes only)",
    items: [
      { to: "/aac", label: "AI conversation", icon: ScanFace, hint: "Phrases and eye typing" },
      { to: "/aac/live", label: "Real-time speech", icon: Radio, hint: "Speak, then listen" },
      { to: "/aac/agent", label: "Agentic orchestration", icon: Bot, hint: "Plan, confirm, do, verify" },
    ],
  },
  {
    title: "You",
    items: [
      { to: "/history", label: "Conversation log", icon: HistoryIcon },
      { to: "/settings", label: "Settings", icon: SettingsIcon },
    ],
  },
];

const KEY = "ishara.sidebar.collapsed";

function readCollapsed() {
  try {
    return localStorage.getItem(KEY) === "1";
  } catch {
    return false;
  }
}

export default function Shell() {
  const { user, logout } = useAuth();
  const { prefs } = usePrefs();
  const loc = useLocation();
  const aac = prefs?.mode === "aac";
  const nav = aac ? AAC_NAV : ISL_NAV;
  const [collapsed, setCollapsed] = useState(readCollapsed);
  const [open, setOpen] = useState(false); // mobile drawer

  useEffect(() => {
    try {
      localStorage.setItem(KEY, collapsed ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [collapsed]);
  useEffect(() => setOpen(false), [loc.pathname]);

  const mini = collapsed && !open;
  const initial = (user?.name || "?").trim().charAt(0).toUpperCase();

  const sidebar = (
    <div className="flex h-full flex-col">
      <div className={`flex items-center gap-3 px-4 pb-4 pt-5 ${mini ? "justify-center px-2" : ""}`}>
        {mini ? (
          <Link to={aac ? "/aac" : "/studio"} title="Ishara AI">
            <Logo compact />
          </Link>
        ) : (
          <>
            <Logo />
            <button className="ml-auto rounded-md p-1 text-zinc-400 hover:text-zinc-100 lg:hidden" onClick={() => setOpen(false)} aria-label="Close menu">
              <X size={18} />
            </button>
          </>
        )}
      </div>

      <div className={`px-4 pb-3 ${mini ? "px-2 text-center" : ""}`}>
        <span
          className={`inline-block rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.18em] ${
            aac ? "border-amber-400/40 text-amber-300" : "border-white/15 text-zinc-400"
          }`}
        >
          {mini ? (aac ? "AAC" : "ISL") : aac ? "AAC mode" : "ISL mode"}
        </span>
      </div>

      <nav className="flex-1 space-y-5 overflow-y-auto px-3 py-2">
        {nav.map((g) => (
          <div key={g.title} className="space-y-1">
            {!mini && <p className="px-3 pb-1 font-mono text-[10px] uppercase tracking-[0.2em] text-zinc-600">{g.title}</p>}
            {mini && <div className="mx-auto mb-2 h-px w-6 bg-white/10" />}
            {g.items.map(({ to, label, icon: Icon, hint }) => (
              <NavLink
                key={to}
                to={to}
                {...(aac ? { "data-gaze": "", "data-dwell": "1600" } : {})}
                end={to === "/aac"}
                title={mini ? label : undefined}
                className={({ isActive }) =>
                  `group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${mini ? "justify-center" : ""} ${
                    isActive
                      ? "bg-gradient-to-r from-white/[0.10] to-white/[0.03] text-zinc-50 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
                      : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-100"
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    {isActive && <span className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-zinc-200" />}
                    <Icon size={18} className={isActive ? "text-zinc-100" : "text-zinc-500 group-hover:text-zinc-300"} />
                    {!mini && (
                      <span className="min-w-0 flex-1">
                        <span className="block truncate">{label}</span>
                        {hint && <span className="block truncate text-[11px] text-zinc-600">{hint}</span>}
                      </span>
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </div>
        ))}
      </nav>

      <div className="space-y-1 border-t border-white/[0.06] p-3">
        <Link
          to="/setup"
          title="Change how you communicate (ISL or AAC)"
          className={`flex items-center gap-3 rounded-xl px-3 py-2 text-sm text-zinc-400 transition hover:bg-white/[0.04] hover:text-zinc-100 ${mini ? "justify-center" : ""}`}
        >
          <Repeat size={17} /> {!mini && "Change mode"}
        </Link>
        {!aac && (
          <Link
            to="/intro"
            title="Replay intro"
            className={`flex items-center gap-3 rounded-xl px-3 py-2 text-sm text-zinc-400 transition hover:bg-white/[0.04] hover:text-zinc-100 ${mini ? "justify-center" : ""}`}
          >
            <PlayCircle size={17} /> {!mini && "Replay intro"}
          </Link>
        )}
        <div className={`mt-2 flex items-center gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-2 ${mini ? "flex-col" : ""}`}>
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-gradient-to-br from-zinc-200 to-zinc-500 text-sm font-semibold text-black">
            {initial}
          </span>
          {!mini && (
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm text-zinc-200">{user?.name}</span>
              <span className="block truncate text-[11px] text-zinc-600">{user?.email}</span>
            </span>
          )}
          <button className="rounded-lg p-2 text-zinc-500 transition hover:bg-white/[0.06] hover:text-zinc-100" onClick={logout} title="Log out">
            <LogOut size={16} />
          </button>
        </div>
        <button
          className="mt-1 hidden w-full items-center justify-center gap-2 rounded-lg py-1.5 text-xs text-zinc-600 transition hover:text-zinc-300 lg:flex"
          onClick={() => setCollapsed((c) => !c)}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronsRight size={15} /> : <><ChevronsLeft size={15} /> Collapse</>}
        </button>
      </div>
    </div>
  );

  return (
    <div className="flex min-h-full">
      {/* desktop sidebar */}
      <aside
        className={`sticky top-0 hidden h-screen shrink-0 border-r border-white/[0.06] bg-black/50 backdrop-blur-xl transition-[width] duration-200 lg:block ${
          collapsed ? "w-[76px]" : "w-[248px]"
        }`}
      >
        {sidebar}
      </aside>

      {/* mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/70" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-[260px] border-r border-white/10 bg-[#0a0a0b]">{sidebar}</aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-white/[0.06] bg-black/60 px-4 py-3 backdrop-blur-xl lg:hidden">
          <button className="rounded-lg p-2 text-zinc-300 hover:bg-white/[0.06]" onClick={() => setOpen(true)} aria-label="Open menu">
            <Menu size={20} />
          </button>
          <Logo />
        </header>
        <main className={`mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6 lg:px-8 ${aac ? "lg:pr-[290px]" : ""}`}>
          {aac ? (
            <GazeProvider>
              <Outlet />
            </GazeProvider>
          ) : (
            <Outlet />
          )}
        </main>
      </div>
    </div>
  );
}
