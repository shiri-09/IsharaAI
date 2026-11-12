import { useState, type FormEvent } from "react";
import { Navigate } from "react-router-dom";
import { ArrowRight, Loader2 } from "lucide-react";
import Logo from "../components/Logo";
import { useAuth } from "../lib/auth";

export default function Login() {
  const { user, login, register, provider } = useAuth();
  const [notice, setNotice] = useState("");
  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setNotice("");
    setBusy(true);
    try {
      if (mode === "login") await login(email, password);
      else {
        const msg = await register(name, email, password);
        if (msg) {
          setNotice(msg);
          setMode("login");
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-full lg:grid-cols-2">
      <section className="hidden flex-col justify-between border-r border-white/[0.06] p-12 lg:flex">
        <Logo size={36} />
        <div>
          <p className="label mb-4">Continuous Indian Sign Language</p>
          <h1 className="chrome-text text-5xl font-semibold leading-[1.05] tracking-tight">
            Sign a sentence.
            <br />
            Be heard in every language.
          </h1>
          <p className="mt-6 max-w-md text-zinc-500">
            Ishara AI reads full ISL sentences from your webcam, holds a conversation with an AI partner, and speaks the
            reply in Hindi, Kannada, Tamil and Telugu.
          </p>
        </div>
        <p className="font-mono text-xs text-zinc-600">MediaPipe Holistic · Transformer · Gemini / Ollama</p>
      </section>

      <section className="grid place-items-center p-6">
        <form onSubmit={submit} className="glass w-full max-w-sm space-y-5 p-8">
          <div className="lg:hidden">
            <Logo />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-zinc-100">{mode === "login" ? "Welcome back" : "Create account"}</h2>
            <p className="mt-1 text-sm text-zinc-500">
              {mode === "login" ? "Sign in to continue your conversations." : "Your history stays on this server."}
              <span className="mt-2 block font-mono text-[10px] uppercase tracking-wider text-zinc-600">
                {provider === "supabase" ? "Secured by Supabase Auth" : "Local auth mode"}
              </span>
            </p>
          </div>
          {mode === "register" && (
            <input className="input" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} required />
          )}
          <input
            className="input"
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            className="input"
            type="password"
            placeholder="Password (min 6)"
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          {error && <p className="text-sm text-red-400/90">{error}</p>}
          {notice && <p className="text-sm text-emerald-300/90">{notice}</p>}
          <button className="btn btn-chrome w-full" disabled={busy}>
            {busy ? <Loader2 size={16} className="animate-spin" /> : <ArrowRight size={16} />}
            {mode === "login" ? "Sign in" : "Create account"}
          </button>
          <button
            type="button"
            className="w-full text-center text-sm text-zinc-500 hover:text-zinc-300"
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError("");
            }}
          >
            {mode === "login" ? "New here? Create an account" : "Have an account? Sign in"}
          </button>
        </form>
      </section>
    </div>
  );
}
