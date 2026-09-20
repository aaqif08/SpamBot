import { ExternalLink } from "lucide-react";

import { Badge, Card, PageHeader, Table, Td, Th } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { API_BASE, api } from "@/services/api";

const ENDPOINTS: { method: string; path: string; purpose: string; role?: string }[] = [
  { method: "GET", path: "/api/v1/health", purpose: "Liveness (public)" },
  { method: "GET", path: "/api/v1/health/ready", purpose: "Readiness: database, migrations, storage (503 when not ready)" },
  { method: "GET", path: "/api/v1/auth/setup-status", purpose: "Whether the first admin has been created (public)" },
  { method: "POST", path: "/api/v1/auth/login", purpose: "Email + password → access token (JSON) + httpOnly refresh cookie" },
  { method: "POST", path: "/api/v1/auth/refresh", purpose: "Rotate the refresh cookie and issue a new access token" },
  { method: "POST", path: "/api/v1/auth/logout", purpose: "Revoke the refresh session" },
  { method: "GET", path: "/api/v1/auth/me", purpose: "Current user" },
  { method: "POST", path: "/api/v1/auth/change-password", purpose: "Change own password (revokes other sessions)" },
  { method: "POST", path: "/api/v1/auth/password-reset/request | confirm", purpose: "Reset flow (token delivered out-of-band by the operator)" },
  { method: "GET/POST", path: "/api/v1/users", purpose: "List / create users in the organization", role: "ADMIN" },
  { method: "PATCH", path: "/api/v1/users/{id}", purpose: "Change role, status or name", role: "ADMIN" },
  { method: "POST", path: "/api/v1/users/{id}/reset-password", purpose: "Set a user's password", role: "ADMIN" },
  { method: "GET/PATCH", path: "/api/v1/organization", purpose: "Organization profile", role: "ADMIN for PATCH" },
  { method: "GET", path: "/api/v1/dashboard", purpose: "Cards + charts computed from the organization's records" },
  { method: "POST", path: "/api/v1/analyses", purpose: "Analyze one account (SHAP + LIME) and store the prediction", role: "ANALYST" },
  { method: "GET", path: "/api/v1/analyses", purpose: "Filterable, sortable, paginated prediction history" },
  { method: "GET/DELETE", path: "/api/v1/analyses/{id}", purpose: "Full stored analysis / delete it" },
  { method: "GET", path: "/api/v1/analyses/{id}/explanations/{shap|lime}", purpose: "Local explanation (computed on demand and persisted)" },
  { method: "POST", path: "/api/v1/analyses/retention/purge", purpose: "Delete predictions older than N days", role: "ADMIN" },
  { method: "POST", path: "/api/v1/batches", purpose: "multipart CSV or dataset_id → 202 + background job", role: "ANALYST" },
  { method: "GET", path: "/api/v1/batches · /batches/{id} · /batches/{id}/download", purpose: "List, summary, predictions.csv" },
  { method: "GET/POST", path: "/api/v1/datasets", purpose: "List / upload a versioned CSV (dataset_id → new version)", role: "ANALYST for POST" },
  { method: "GET/DELETE", path: "/api/v1/datasets/{id}", purpose: "Profile with column mapping and versions / delete" },
  { method: "POST", path: "/api/v1/datasets/{id}/evaluate", purpose: "Evaluate a model on a labelled dataset", role: "ANALYST" },
  { method: "GET", path: "/api/v1/datasets/benchmarks", purpose: "Cresci-15/17 availability and paper-reported statistics", role: "ADMIN" },
  { method: "POST", path: "/api/v1/datasets/benchmarks/{kind}/import", purpose: "Import a public benchmark (job)", role: "ADMIN" },
  { method: "GET", path: "/api/v1/models", purpose: "Models, production model id, supported algorithms" },
  { method: "POST", path: "/api/v1/models/train", purpose: "Submit a training job → {job_id, model_id}", role: "ANALYST" },
  { method: "GET", path: "/api/v1/models/{id} · /evaluation · /explanation", purpose: "Model card, stored evaluation runs + feature importance, global SHAP" },
  { method: "POST", path: "/api/v1/models/{id}/activate | deprecate", purpose: "Lifecycle (checksums verified before activation)", role: "ADMIN" },
  { method: "DELETE", path: "/api/v1/models/{id}", purpose: "Delete a non-production model and its artefacts", role: "ADMIN" },
  { method: "GET", path: "/api/v1/jobs · /jobs/{id}", purpose: "Background job status, progress, log and result" },
  { method: "GET", path: "/api/v1/providers", purpose: "Data providers and whether each is configured" },
  { method: "POST", path: "/api/v1/providers/{name}/fetch", purpose: "Fetch an account from a configured provider (409 when not configured)", role: "ANALYST" },
  { method: "GET", path: "/api/v1/audit", purpose: "Audit log", role: "ADMIN" },
  { method: "GET", path: "/api/v1/research · /features · /runtime", purpose: "Citation and paper-reported results (public), feature catalogue, runtime info" },
];

const EXAMPLE = `{
  "account": {
    "account_id": "example_user",
    "verified": false,
    "friends_count": 120,
    "followers_count": 15,
    "listed_count": 0,
    "favorites_count": 50,
    "statuses_count": 500,
    "hashtag_count": 25,
    "mentions_count": 40,
    "retweet_count": 300,
    "reply_count": 2,
    "url_count": 20,
    "description": "…",
    "tweets": ["…"]
  },
  "model_id": null,
  "source": "manual",
  "explain": true
}`;

export function ApiDocsPage() {
  const base = API_BASE || window.location.origin.replace(/:\d+$/, ":8000");
  const docs = useApi(() => api.runtime(), []);
  return (
    <div className="space-y-6">
      <PageHeader
        title="API Documentation"
        description="Versioned REST API under /api/v1 with OpenAPI 3 documentation. Requests carry a Bearer access token (30 min) obtained from /auth/login; a rotating httpOnly refresh cookie renews it. Errors are {detail, code} with an X-Request-ID header for support."
        actions={
          <>
            <a href={`${base}/docs`} target="_blank" rel="noreferrer" className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-medium text-white hover:bg-accent-2">Swagger UI <ExternalLink className="h-4 w-4" /></a>
            <a href={`${base}/redoc`} target="_blank" rel="noreferrer" className="inline-flex h-10 items-center gap-2 rounded-lg border border-border-strong px-4 text-sm font-medium text-ink hover:bg-surface-2">ReDoc <ExternalLink className="h-4 w-4" /></a>
          </>
        }
      />
      {docs.data && docs.data.environment === "production" ? (
        <Card title="Live documentation"><p className="text-sm text-ink-2">Interactive Swagger UI is disabled in production unless <code className="font-mono">BOTSHIELD_DOCS_ENABLED=true</code>. The endpoint reference below is authoritative; see also <code className="font-mono">docs/api.md</code>.</p></Card>
      ) : (
        <Card title="Live documentation" subtitle="Rendered by the backend (Swagger UI). Use “Authorize” with a Bearer token from /auth/login. If the frame is blank, open it in a new tab.">
          <iframe title="Swagger UI" src={`${base}/docs`} className="h-[640px] w-full rounded-lg border border-border bg-white" />
        </Card>
      )}
      <Card title="Endpoints" padded={false}>
        <Table className="rounded-none border-0">
          <thead><tr><Th>Method</Th><Th>Path</Th><Th>Purpose</Th><Th>Minimum role</Th></tr></thead>
          <tbody>
            {ENDPOINTS.map((e) => <tr key={e.method + e.path}><Td><Badge tone={e.method === "GET" ? "human" : e.method === "DELETE" ? "critical" : "bot"}>{e.method}</Badge></Td><Td className="font-mono text-xs">{e.path}</Td><Td className="text-xs text-ink-2">{e.purpose}</Td><Td className="text-xs text-ink-3">{e.role ?? "any signed-in user"}</Td></tr>)}
          </tbody>
        </Table>
      </Card>
      <Card title="Example — POST /api/v1/analyses" subtitle="Derived features are computed server-side; only raw counts, flags and text are needed.">
        <pre className="scrollbar-thin overflow-auto rounded-lg bg-surface-2 p-4 text-xs text-ink">{EXAMPLE}</pre>
        <p className="mt-2 text-xs text-ink-2">Response (201): prediction_id, prediction, bot_probability, risk_score (= round(100 × P(bot)) — a model output, not a verified fact), model, features, top_features, shap_explanation, lime_explanation, explanation_status, interpretation. Requires header <code className="font-mono">Authorization: Bearer &lt;access_token&gt;</code>. Full schema in <a className="underline" href={`${base}/redoc`} target="_blank" rel="noreferrer">ReDoc</a> and docs/api.md.</p>
      </Card>
    </div>
  );
}
