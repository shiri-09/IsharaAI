import { Link, NavLink, Outlet } from "react-router-dom";
import { Database, History as HistoryIcon, LogOut, PlayCircle, Repeat, ScanFace, ScanLine } from "lucide-react";
import Logo from "./Logo";
import { useAuth } from "../lib/auth";
import { usePrefs } from "../lib/prefs";

/**
 * Two completely separate experiences share one account:
 *  - ISL users get the Sign Studio (camera, mouse, keyboard) and its tools.
 *  - AAC users get only the eyes-and-face board; no sign-language screens exist for them at all.
 */
const ISL_NAV = [
  { to: "/studio", label: "Sign Studio", icon: ScanLine },
  { to: "/collect", label: "Collect & Train", icon: Database },
  { to: "/history", label: "History", icon: HistoryIcon },
];
const AAC_NAV = [
  { to: "/aac", label: "AAC board", icon: ScanFace },
  { to: "/history", label: "Conversation log", icon: HistoryIcon },
];

export default function Shell() {
  const { user, logout } = useAuth();
  const { prefs } = usePrefs();
  const aac = prefs?.mode === "aac";
  const nav = aac ? AAC_NAV : ISL_NAV;
  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-20 border-b border-white/[0.06] bg-black/40 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6">
          <Logo />
          <span
            className={`hidden rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.18em] sm:inline ${
              aac ? "border-amber-400/40 text-amber-300" : "border-white/15 text-zinc-400"
            }`}
          >
            {aac ? "AAC mode" : "ISL mode"}
          </span>
          <nav className="flex flex-1 items-center gap-1 overflow-x-auto">
            {nav.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `flex items-center gap-2 whitespace-nowrap rounded-lg px-3 py-2 text-sm transition ${
                    isActive ? "bg-white/[0.07] text-zinc-100" : "text-zinc-500 hover:text-zinc-200"
                  }`
                }
              >
                <Icon size={16} />
                <span className="hidden sm:inline">{label}</span>
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <Link to="/setup" className="btn btn-ghost px-3" title="Change how you communicate (ISL or AAC)">
              <Repeat size={16} />
              <span className="hidden lg:inline">Change mode</span>
            </Link>
            {!aac && (
              <Link to="/welcome" className="btn btn-ghost px-3" title="Replay intro">
                <PlayCircle size={16} />
              </Link>
            )}
            <span className="hidden text-sm text-zinc-400 md:inline">{user?.name}</span>
            <button className="btn btn-ghost px-3" onClick={logout} title="Log out">
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        <Outlet />
      </main>
    </div>
  );
}
