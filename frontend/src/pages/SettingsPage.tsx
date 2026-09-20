import { Moon, RefreshCw, Sun } from "lucide-react";

import { Badge, Button, Card, ErrorState, Notice, PageHeader, Table, Td, Th } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { useHealth } from "@/hooks/useHealth";
import { useTheme } from "@/hooks/useTheme";
import { API_BASE, api } from "@/services/api";

export function SettingsPage() {
  const { theme, setTheme } = useTheme();
  const health = useHealth();
  const adapters = useApi(() => api.adapters(), []);
  const features = useApi(() => api.features(), []);

  return (
    <div className="space-y-6">
      <PageHeader title="Settings" description="Client preferences and a read-only view of backend configuration reported by the API. Secrets are never exposed; backend settings are configured through environment variables (.env)." />

      <Card title="Appearance">
        <div className="flex gap-2">
          <Button variant={theme === "light" ? "primary" : "secondary"} icon={Sun} onClick={() => setTheme("light")}>Light</Button>
          <Button variant={theme === "dark" ? "primary" : "secondary"} icon={Moon} onClick={() => setTheme("dark")}>Dark</Button>
        </div>
      </Card>

      <Card title="Backend connection" actions={<Button size="sm" variant="secondary" icon={RefreshCw} onClick={health.reload} loading={health.loading}>Re-check</Button>}>
        {health.error && <ErrorState title="Backend unavailable" message={health.error} />}
        {health.data && (
          <dl className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["API base", API_BASE || "Vite dev proxy → http://localhost:8000"],
              ["App", `${health.data.app} v${health.data.version}`],
              ["Environment", health.data.environment],
              ["Python", health.data.python],
              ["Active model", health.data.active_model ? `${health.data.active_model.name}${health.data.active_model.is_demo ? " (demo)" : ""}` : "none"],
              ["Datasets registered", String(health.data.datasets_available)],
              ["Demo mode", health.data.demo_mode_enabled ? "enabled" : "disabled"],
              ["Feature version", `${health.data.feature_version} (${health.data.n_features} features)`],
            ].map(([k, v]) => <div key={k}><dt className="text-[11px] text-ink-3">{k}</dt><dd className="truncate font-medium text-ink" title={v}>{v}</dd></div>)}
          </dl>
        )}
      </Card>

      <Card title="Backend environment variables" subtitle="Set in .env (see .env.example). Values are not read back by the UI.">
        <Table>
          <thead><tr><Th>Variable</Th><Th>Purpose</Th><Th>Default</Th></tr></thead>
          <tbody>
            {[
              ["BOTSHIELD_CORS_ORIGINS", "Allowed frontend origins (comma-separated)", "http://localhost:5173"],
              ["BOTSHIELD_MAX_UPLOAD_MB", "CSV upload size limit", "50"],
              ["BOTSHIELD_MAX_BATCH_ROWS", "Maximum rows per batch prediction", "50000"],
              ["BOTSHIELD_RATE_LIMIT_PER_MINUTE", "Per-client rate limit on predict/upload/train", "120"],
              ["BOTSHIELD_DEMO_MODE_ENABLED", "Allow generating the synthetic demo dataset", "true"],
              ["BOTSHIELD_MAX_TWEETS_PER_USER", "Tweets used per account for text features during Cresci import", "100"],
              ["BOTSHIELD_DATABASE_URL", "SQLAlchemy URL (SQLite by default)", "sqlite:///backend/data/botshield.db"],
              ["BOTSHIELD_X_BEARER_TOKEN", "App-only bearer token that enables the live X API v2 adapter (Analyze → Fetch a live account)", "unset"],
              ["BOTSHIELD_X_MAX_TWEETS", "Recent tweets fetched per account (5-100)", "100"],
              ["BOTSHIELD_X_CACHE_TTL_SECONDS", "Cache for fetched X accounts (rate-limit friendly)", "600"],
            ].map(([k, p, d]) => <tr key={k}><Td className="font-mono text-xs">{k}</Td><Td className="text-xs text-ink-2">{p}</Td><Td className="font-mono text-xs text-ink-2">{d}</Td></tr>)}
          </tbody>
        </Table>
      </Card>

      <Card title="Integrations (adapters)">
        {adapters.error && <ErrorState message={adapters.error} onRetry={adapters.reload} />}
        <ul className="space-y-2">
          {adapters.data?.map((a) => (
            <li key={a.name} className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border px-3 py-2 text-sm">
              <span><span className="font-medium text-ink">{a.name}</span> <span className="text-xs text-ink-3">({a.kind})</span><div className="text-xs text-ink-2">{a.description}</div></span>
              <span className="flex gap-1">{a.is_sample && <Badge tone="demo">sample</Badge>}<Badge tone={a.configured ? "good" : "neutral"} dot={a.configured ? "var(--status-good)" : "var(--text-3)"}>{a.configured ? "configured" : "not configured"}</Badge></span>
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Feature schema" subtitle={features.data ? `${features.data.n_features} features · version ${features.data.feature_version}` : undefined}>
        {features.data && <div className="flex flex-wrap gap-1.5">{features.data.order.map((f) => <Badge key={f}>{f}</Badge>)}</div>}
      </Card>

      <Notice>Risk Score is an application-level representation of the model probability (round(100 × P(bot))) and is NOT a definitive statement that an account is malicious.</Notice>
    </div>
  );
}
