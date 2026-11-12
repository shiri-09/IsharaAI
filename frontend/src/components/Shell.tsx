import { NavLink, Outlet } from "react-router-dom";
import { Database, History as HistoryIcon, LogOut, ScanLine } from "lucide-react";
import Logo from "./Logo";
import { useAuth } from "../lib/auth";

const NAV = [
  { to: "/", label: "Studio", icon: ScanLine },
  { to: "/collect", label: "Collect & Train", icon: Database },
  { to: "/history", label: "History", icon: HistoryIcon },
];

export default function Shell() {
  const { user, logout } = useAuth();
  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-20 border-b border-white/[0.06] bg-black/40 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3 sm:px-6">
          <Logo />
          <nav className="flex flex-1 items-center gap-1 overflow-x-auto">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
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
          <div className="flex items-center gap-3">
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
