import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Navigate, Outlet, Route, Routes, useLocation } from "react-router-dom";
import "./index.css";
import { AuthProvider, useAuth } from "./lib/auth";
import { PrefsProvider, usePrefs } from "./lib/prefs";
import Shell from "./components/Shell";
import Login from "./pages/Login";
import Intro from "./pages/Intro";
import Setup from "./pages/Setup";
import Studio from "./pages/Studio";
import Aac from "./pages/Aac";
import Collect from "./pages/Collect";
import History from "./pages/History";

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
            <Route path="/welcome" element={<Intro />} />
            <Route path="/setup" element={<Setup />} />
            <Route element={<Shell />}>
              <Route path="/" element={<Home />} />
              <Route element={<ModeGate mode="isl" />}>
                <Route path="/studio" element={<Studio />} />
                <Route path="/collect" element={<Collect />} />
              </Route>
              <Route element={<ModeGate mode="aac" />}>
                <Route path="/aac" element={<Aac />} />
              </Route>
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
