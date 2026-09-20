import { ArrowDown, ArrowRight, Boxes, Braces, Database, FileSpreadsheet, FlaskConical, Lightbulb, MonitorSmartphone, Plug, Server, Workflow, type LucideIcon } from "lucide-react";
import { Fragment, type ReactNode } from "react";

import { Badge, Card, PageHeader, Table, Td, Th } from "@/components/ui";
import { useHealth } from "@/hooks/useHealth";

function Node({ icon: Icon, title, sub, tone = "neutral" }: { icon: LucideIcon; title: string; sub?: ReactNode; tone?: "neutral" | "accent" }) {
  return (
    <div className={`flex w-full items-start gap-3 rounded-xl border px-4 py-3 ${tone === "accent" ? "border-accent/50 bg-accent-soft/50" : "border-border bg-surface"}`}>
      <span className="rounded-lg bg-surface-2 p-2 text-ink-2"><Icon className="h-4 w-4" aria-hidden /></span>
      <div className="min-w-0">
        <div className="text-sm font-semibold text-ink">{title}</div>
        {sub && <div className="text-xs text-ink-2">{sub}</div>}
      </div>
    </div>
  );
}

function Flow({ nodes }: { nodes: { icon: LucideIcon; title: string; sub?: ReactNode; tone?: "neutral" | "accent" }[] }) {
  return (
    <div className="flex flex-col items-center gap-1">
      {nodes.map((n, i) => (
        <Fragment key={n.title}>
          <Node {...n} />
          {i < nodes.length - 1 && <ArrowDown className="h-4 w-4 text-ink-3" aria-hidden />}
        </Fragment>
      ))}
    </div>
  );
}

export function ArchitecturePage() {
  const health = useHealth();
  return (
    <div className="space-y-6">
      <PageHeader title="System Architecture" description="Two independent packages in one repository: a React/TypeScript frontend that only consumes typed JSON, and a FastAPI backend whose HTTP layer wraps a framework-independent ML package." />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Inference architecture" subtitle="Single and batch prediction path">
          <Flow
            nodes={[
              { icon: MonitorSmartphone, title: "Frontend (React + TypeScript + Vite + Tailwind)", sub: "pages · hooks · typed API client · Recharts" },
              { icon: Server, title: "FastAPI backend", sub: "app/api routers → Pydantic validation → services · CORS · rate limit · error handlers" },
              { icon: Workflow, title: "Prediction service", sub: "app/services/prediction_service.py — orchestration, persistence, interpretation text" },
              { icon: Braces, title: "Feature engineering", sub: "ml/features.py FeatureExtractor → 31-feature vector (FEATURE_GROUPS single source of truth)", tone: "accent" },
              { icon: Boxes, title: "ML model", sub: "sklearn Pipeline(imputer → min-max scaler → classifier) loaded from the model registry" },
              { icon: Lightbulb, title: "SHAP / LIME", sub: "ml/explain.py — TreeExplainer/KernelExplainer + LimeTabularExplainer on the same pipeline" },
              { icon: Database, title: "SQLite · results", sub: "predictions (features + SHAP + LIME JSON), batches, exports/predictions.csv" },
            ]}
          />
        </Card>
        <Card title="Training architecture" subtitle="Asynchronous job: POST /api/train → GET /api/train/status/{job_id}">
          <Flow
            nodes={[
              { icon: FileSpreadsheet, title: "Dataset upload / Cresci import / demo generator", sub: "validated CSV → data/uploads · chunked Cresci users.csv + tweets.csv aggregation" },
              { icon: Workflow, title: "Preprocessing", sub: "label coercion · two text paths (feature path / sentiment path) · shuffling + stratified split" },
              { icon: Braces, title: "Feature extraction", sub: "same FeatureExtractor as inference → identical vectors in training and serving", tone: "accent" },
              { icon: FlaskConical, title: "Model training", sub: "optional SHAP feature selection · randomised search with stratified k-fold CV · nine classifiers" },
              { icon: Lightbulb, title: "Evaluation + SHAP analysis", sub: "hold-out metrics, confusion matrix, ROC/PR curves, CV folds, global mean |SHAP| + beeswarm sample" },
              { icon: Boxes, title: "Model registry", sub: "models/<id>/pipeline.joblib · scaler.joblib · feature_metadata.json · metrics.json · shap_global.json · background.npy · lime_sample.npy · registry.json" },
            ]}
          />
        </Card>
      </div>

      <Card title="Social-network adapters" subtitle="An adapter converts platform data into the account schema. The X API v2 adapter is bundled and activates with a bearer token; a sample adapter serves hand-written demo accounts.">
        <div className="flex flex-wrap items-center gap-2">
          {["Social Network API Adapter", "Account data (AccountInput)", "Feature extractor", "Prediction + explanation"].map((s, i, arr) => (
            <Fragment key={s}>
              <span className="rounded-lg border border-border bg-surface-2 px-3 py-2 text-xs font-medium text-ink"><Plug className="mr-1 inline h-3.5 w-3.5" />{s}</span>
              {i < arr.length - 1 && <ArrowRight className="h-4 w-4 text-ink-3" />}
            </Fragment>
          ))}
        </div>
        <p className="mt-3 text-xs text-ink-2">Contract: <code className="font-mono">SocialNetworkAdapter.fetch_account(identifier) → AccountInput</code> (app/services/adapter_service.py). Bundled implementations: <code className="font-mono">sample</code> (hand-written DEMO accounts) and <code className="font-mono">x_api</code> (app/services/x_api_adapter.py — X API v2 <code className="font-mono">/2/users/by/username</code> + <code className="font-mono">/2/users/:id/tweets</code>, active once <code className="font-mono">BOTSHIELD_X_BEARER_TOKEN</code> is set). Until a token is configured the UI reports the X adapter as "not configured" and never fabricates live data.</p>
      </Card>

      <Card title="Components" padded={false}>
        <Table className="rounded-none border-0">
          <thead><tr><Th>Layer</Th><Th>Location</Th><Th>Responsibility</Th></tr></thead>
          <tbody>
            {[
              ["Frontend pages", "frontend/src/pages", "Dashboard, Analyze, Batch, Datasets, Models, Training, Evaluation, Explainability, History, Research, Architecture, API Docs, Settings"],
              ["API client / types", "frontend/src/services/api.ts · src/types/api.ts", "Typed fetch client mirroring the Pydantic schemas; no ML logic in React"],
              ["HTTP layer", "backend/app/api", "Routers, validation, HTTP status codes, rate limiting, upload security"],
              ["Services", "backend/app/services", "Prediction, dataset, training job manager, dashboard aggregation, adapters"],
              ["ML package", "backend/ml", "features · preprocessing · sentiment · train · evaluation · explain · predict · model_registry · datasets · demo_data · paper_results"],
              ["Persistence", "backend/data/botshield.db (SQLite) · backend/models", "predictions, datasets, models, training_runs, batches · joblib artefacts"],
              ["Scripts", "scripts/", "train_model.py · evaluate_model.py · seed_demo.py (reproducible CLI pipeline)"],
              ["Docs", "docs/", "paper-analysis.md · architecture.md · methodology.md · api.md"],
            ].map(([l, p, r]) => <tr key={l}><Td className="font-medium">{l}</Td><Td className="font-mono text-xs text-ink-2">{p}</Td><Td className="text-xs text-ink-2">{r}</Td></tr>)}
          </tbody>
        </Table>
      </Card>

      <Card title="Runtime" subtitle="Reported by GET /api/health">
        {health.data ? (
          <div className="flex flex-wrap gap-2">
            <Badge>Python {health.data.python}</Badge>
            {Object.entries(health.data.libraries).map(([k, v]) => <Badge key={k}>{k} {v}</Badge>)}
            <Badge tone="accent">feature version {health.data.feature_version} · {health.data.n_features} features</Badge>
          </div>
        ) : (
          <p className="text-xs text-ink-3">Backend unavailable.</p>
        )}
      </Card>
    </div>
  );
}
