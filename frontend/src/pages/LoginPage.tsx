import { LogIn, ShieldCheck } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Navigate, useLocation } from "react-router-dom";

import { Button, Field, Input, Notice } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { useAuth } from "@/hooks/useAuth";
import { api, errorMessage } from "@/services/api";

export function LoginPage() {
  const { user, login, loading } = useAuth();
  const location = useLocation();
  const setup = useApi(() => api.setupStatus(), []);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const from = (location.state as { from?: string } | null)?.from ?? "/";

  if (!loading && user) return <Navigate to={from} replace />;

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(email.trim(), password);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-bg px-4">
      <div className="w-full max-w-sm">
        <div className="mb-6 flex items-center gap-3">
          <span className="rounded-lg bg-accent p-2 text-white">
            <ShieldCheck className="h-5 w-5" aria-hidden />
          </span>
          <div>
            <h1 className="text-lg font-semibold text-ink">BotShield AI</h1>
            <p className="text-xs text-ink-2">Social bot &amp; fake-follower intelligence</p>
          </div>
        </div>
        <form onSubmit={onSubmit} className="space-y-4 rounded-xl border border-border bg-surface p-6 shadow-sm" aria-labelledby="login-title">
          <h2 id="login-title" className="text-base font-semibold text-ink">Sign in</h2>
          {setup.data && !setup.data.initialized && (
            <Notice tone="warning">
              This installation has no users yet. Create the first administrator on the server: <code className="font-mono">python -m app.cli create-admin</code>
            </Notice>
          )}
          <Field label="Email">
            <Input type="email" autoComplete="username" required value={email} onChange={(e) => setEmail(e.target.value)} autoFocus />
          </Field>
          <Field label="Password">
            <Input type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
          </Field>
          {error && (
            <p role="alert" className="text-sm text-status-critical">
              {error}
            </p>
          )}
          <Button type="submit" icon={LogIn} loading={busy} className="w-full">
            Sign in
          </Button>
          <p className="text-[11px] text-ink-3">Forgot your password? Ask an administrator to reset it from Settings → Users.</p>
        </form>
      </div>
    </main>
  );
}
