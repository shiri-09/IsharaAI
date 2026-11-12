import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Outlet, Route, Routes, useLocation } from "react-router-dom";
import "./index.css";
import { AuthProvider, useAuth } from "./lib/auth";
import { PrefsProvider, usePrefs } from "./lib/prefs";
import Shell from "./components/Shell";
import Login from "./pages/Login";
import Cinema, { introSeen } from "./pages/Cinema";
import Setup from "./pages/Setup";
import Studio from "./pages/Studio";
import Aac from "./pages/Aac";
import Collect from "./pages/Collect";
import History from "./pages/History";
import SignBack from "./pages/SignBack";
import Spell from "./pages/Spell";
import Settings from "./pages/Settings";
import SignLibrary from "./pages/SignLibrary";
import Agent from "./pages/Agent";
import Live from "./pages/Live";
import AacChat from "./pages/AacChat";
import AacLive from "./pages/AacLive";

function Loading() {
  return <div className="grid h-full place-items-center text-zinc-500">Loading…</div>;
}

/** Signed-in area: loads preferences and sends first-time users through intro + setup. */
function Protected() {
  const { user, loading } = useAuth();
  if (loading) return <Loading />;
  if (!user) return <Navigate to="/login" replace />;
  return (
    <PrefsProvider>
      <Onboarding />
    </PrefsProvider>
  );
}

function Onboarding() {
  const { prefs } = usePrefs();
  const loc = useLocation();
  if (!prefs) return <Loading />;
  const inSetup = loc.pathname === "/welcome" || loc.pathname === "/setup";
  if (!prefs.onboarded && !inSetup) return <Navigate to="/welcome" replace />;
  // the cinematic intro plays once after every login
  if (prefs.onboarded && !introSeen() && !inSetup && loc.pathname !== "/intro") return <Navigate to="/intro" replace />;
  return <Outlet />;
}

/** Keeps each user inside their own interface: ISL screens for ISL users, the AAC board for AAC users. */
function ModeGate({ mode }: { mode: "isl" | "aac" }) {
  const { prefs } = usePrefs();
  if (!prefs) return <Loading />;
  const current = prefs.mode === "aac" ? "aac" : "isl";
  if (current !== mode) return <Navigate to={current === "aac" ? "/aac" : "/studio"} replace />;
  return <Outlet />;
}

function Home() {
  const { prefs } = usePrefs();
  return <Navigate to={prefs?.mode === "aac" ? "/aac" : "/studio"} replace />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<Protected />}>
            <Route path="/welcome" element={<Cinema />} />
            <Route path="/intro" element={<Cinema />} />
            <Route path="/setup" element={<Setup />} />
            <Route element={<Shell />}>
              <Route path="/" element={<Home />} />
              <Route element={<ModeGate mode="isl" />}>
                <Route path="/studio" element={<Studio />} />
                <Route path="/library" element={<SignLibrary />} />
                <Route path="/live" element={<Live />} />
                <Route path="/agent" element={<Agent />} />
                <Route path="/reply" element={<SignBack />} />
                <Route path="/spell" element={<Spell />} />
                <Route path="/collect" element={<Collect />} />
              </Route>
              <Route element={<ModeGate mode="aac" />}>
                <Route path="/aac" element={<AacChat />} />
                <Route path="/aac/live" element={<AacLive />} />
                <Route path="/aac/agent" element={<Agent aac />} />
              </Route>
              <Route path="/settings" element={<Settings />} />
              <Route path="/history" element={<History />} />
              <Route path="/history/:id" element={<History />} />
            </Route>
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  </StrictMode>,
);
