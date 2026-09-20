import { ExternalLink } from "lucide-react";

import { Badge, Card, PageHeader, Table, Td, Th } from "@/components/ui";
import { API_BASE } from "@/services/api";

const ENDPOINTS: { method: string; path: string; purpose: string }[] = [
  { method: "GET", path: "/api/health", purpose: "Liveness, active model, library versions" },
  { method: "GET", path: "/api/dashboard", purpose: "Cards + chart data from SQLite and the active model" },
  { method: "GET", path: "/api/models", purpose: "Registry listing, supported algorithms, paper-reported results (separately keyed)" },
  { method: "POST", path: "/api/models/{id}/activate", purpose: "Set the active model" },
  { method: "DELETE", path: "/api/models/{id}", purpose: "Delete a model and its artefacts" },
  { method: "POST", path: "/api/predict", purpose: "Single account prediction with SHAP + LIME" },
  { method: "POST", path: "/api/predict/batch", purpose: "multipart CSV or dataset_id → batch summary" },
  { method: "GET", path: "/api/predict/batch/{batch_id}", purpose: "Stored batch summary" },
  { method: "GET", path: "/api/predict/batch/{batch_id}/download", purpose: "predictions.csv" },
  { method: "GET", path: "/api/predict/batches", purpose: "List batches" },
  { method: "GET", path: "/api/sample-accounts", purpose: "DEMO sample accounts" },
  { method: "GET", path: "/api/datasets", purpose: "List datasets" },
  { method: "POST", path: "/api/datasets/upload", purpose: "Upload + inspect a CSV" },
  { method: "GET", path: "/api/datasets/{id}", purpose: "Inspection summary" },
  { method: "DELETE", path: "/api/datasets/{id}", purpose: "Delete an uploaded/demo dataset" },
  { method: "POST", path: "/api/datasets/{id}/evaluate", purpose: "Evaluate a model on a labelled dataset" },
  { method: "POST", path: "/api/datasets/demo", purpose: "Generate the synthetic demo dataset" },
  { method: "GET", path: "/api/datasets/cresci/status", purpose: "Cresci-15/17 local availability + paper stats" },
  { method: "POST", path: "/api/datasets/cresci/{kind}/import", purpose: "Featurise local Cresci files (job)" },
  { method: "GET", path: "/api/datasets/jobs/{job_id}", purpose: "Import job status" },
  { method: "POST", path: "/api/train", purpose: "Submit training job → {job_id}" },
  { method: "GET", path: "/api/train/status/{job_id}", purpose: "Poll training stage/progress/result" },
  { method: "GET", path: "/api/train/runs", purpose: "Training run history" },
  { method: "GET", path: "/api/evaluation/{model_id}", purpose: "Metrics, CM, ROC/PR, CV folds ('active' allowed)" },
  { method: "GET", path: "/api/explain/global/{model_id}", purpose: "SHAP importance + beeswarm sample" },
  { method: "GET", path: "/api/explain/shap/{prediction_id}", purpose: "Local SHAP (computed on demand for batch rows)" },
  { method: "GET", path: "/api/explain/lime/{prediction_id}", purpose: "Local LIME" },
  { method: "GET", path: "/api/history", purpose: "Filterable prediction history" },
  { method: "GET", path: "/api/history/{id}", purpose: "Full stored analysis" },
  { method: "GET", path: "/api/adapters", purpose: "Adapter status (sample · x_api = live X API v2, configured when BOTSHIELD_X_BEARER_TOKEN is set)" },
  { method: "POST", path: "/api/adapters/{name}/fetch", purpose: "Fetch account via adapter ({identifier}: @username for x_api); 401/403/404/429 passed through" },
  { method: "GET", path: "/api/features", purpose: "Feature groups and descriptions" },
  { method: "GET", path: "/api/research", purpose: "Citation, paper-reported results, pipeline" },
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
    "description": "Best deals every day #followback",
    "tweets": ["GET FOLLOWERS FAST http://bit.ly/x #followback", "..."]
  },
  "explain": true
}`;

export function ApiDocsPage() {
  const base = API_BASE || window.location.origin.replace(/:\d+$/, ":8000");
  return (
    <div className="space-y-6">
      <PageHeader
        title="API Documentation"
        description="The backend generates OpenAPI/Swagger documentation automatically. All responses are typed Pydantic schemas; errors are {detail, code}."
        actions={
          <>
            <a href={`${base}/docs`} target="_blank" rel="noreferrer" className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-medium text-white hover:bg-accent-2">Swagger UI <ExternalLink className="h-4 w-4" /></a>
            <a href={`${base}/redoc`} target="_blank" rel="noreferrer" className="inline-flex h-10 items-center gap-2 rounded-lg border border-border-strong px-4 text-sm font-medium text-ink hover:bg-surface-2">ReDoc <ExternalLink className="h-4 w-4" /></a>
          </>
        }
      />
      <Card title="Live documentation" subtitle="Rendered by the backend (Swagger UI). If the frame is blank, open it in a new tab using the button above.">
        <iframe title="Swagger UI" src={`${base}/docs`} className="h-[640px] w-full rounded-lg border border-border bg-white" />
      </Card>
      <Card title="Endpoints" padded={false}>
        <Table className="rounded-none border-0">
          <thead><tr><Th>Method</Th><Th>Path</Th><Th>Purpose</Th></tr></thead>
          <tbody>
            {ENDPOINTS.map((e) => <tr key={e.method + e.path}><Td><Badge tone={e.method === "GET" ? "human" : e.method === "DELETE" ? "critical" : "bot"}>{e.method}</Badge></Td><Td className="font-mono text-xs">{e.path}</Td><Td className="text-xs text-ink-2">{e.purpose}</Td></tr>)}
          </tbody>
        </Table>
      </Card>
      <Card title="Example — POST /api/predict" subtitle="Derived features are computed server-side; only raw counts, flags and text are needed.">
        <pre className="scrollbar-thin overflow-auto rounded-lg bg-surface-2 p-4 text-xs text-ink">{EXAMPLE}</pre>
        <p className="mt-2 text-xs text-ink-2">Response: prediction, bot/human probability, risk_score (= round(100 × P(bot))), model, features, top_features, shap_explanation, lime_explanation, interpretation. Full schema in <a className="underline" href={`${base}/redoc`} target="_blank" rel="noreferrer">ReDoc</a> and docs/api.md.</p>
      </Card>
    </div>
  );
}
