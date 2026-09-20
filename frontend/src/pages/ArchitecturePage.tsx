import { ArrowDown, ArrowRight, Boxes, Braces, Database, FileSpreadsheet, FlaskConical, Lightbulb, MonitorSmartphone, Plug, Server, Workflow, type LucideIcon } from "lucide-react";
import { Fragment, type ReactNode } from "react";

import { Badge, Card, PageHeader, Table, Td, Th } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api } from "@/services/api";

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
  const runtime = useApi(() => api.runtime(), []);
  return (
    <div className="space-y-6">
      <PageHeader title="System Architecture" description="A React/TypeScript frontend that only consumes typed JSON, a FastAPI backend (authentication, RBAC, multi-tenancy, jobs, storage) and a framework-independent ML package. PostgreSQL in production, SQLite for local development; artefacts in local or S3 storage; background jobs in a thread pool or Celery workers." />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Inference architecture" subtitle="Single and batch prediction path">
          <Flow
            nodes={[
              { icon: MonitorSmartphone, title: "Frontend (React + TypeScript + Vite + Tailwind)", sub: "pages · hooks · typed API client · Recharts" },
              { icon: Server, title: "FastAPI backend (/api/v1)", sub: "JWT access + rotating refresh cookie · RBAC (ADMIN/ANALYST/VIEWER) · organization scoping · request ids · rate limits · security headers" },
              { icon: Workflow, title: "Prediction service", sub: "app/services/prediction_service.py — resolves the production model, persists predictions + explanations, writes audit entries" },
              { icon: Braces, title: "Feature engineering", sub: "ml/features.py FeatureExtractor → 31-feature vector (FEATURE_GROUPS single source of truth)", tone: "accent" },
              { icon: Boxes, title: "ML model", sub: "sklearn Pipeline(imputer → min-max scaler → classifier) loaded from storage after SHA-256 checksum verification" },
              { icon: Lightbulb, title: "SHAP / LIME", sub: "ml/explain.py — TreeExplainer/KernelExplainer + LimeTabularExplainer on the same pipeline" },
              { icon: Database, title: "Database · storage", sub: "predictions, explanations, batches, jobs, audit (PostgreSQL / SQLite) · predictions.csv in local or S3 storage" },
            ]}
          />
        </Card>
        <Card title="Training architecture" subtitle="Asynchronous job: POST /api/v1/models/train → GET /api/v1/jobs/{id}">
          <Flow
            nodes={[
              { icon: FileSpreadsheet, title: "Dataset upload / public benchmark import", sub: "validated, checksummed, versioned CSV in storage · Cresci-15/17 user-level import (admin)" },
              { icon: Workflow, title: "Preprocessing", sub: "label coercion · two text paths (feature path / sentiment path) · shuffling + stratified split" },
              { icon: Braces, title: "Feature extraction", sub: "same FeatureExtractor as inference → identical vectors in training and serving", tone: "accent" },
              { icon: FlaskConical, title: "Model training", sub: "optional SHAP feature selection · randomised search with stratified k-fold CV · nine classifiers" },
              { icon: Lightbulb, title: "Evaluation + SHAP analysis", sub: "hold-out metrics, confusion matrix, ROC/PR curves, CV folds, global mean |SHAP| + beeswarm sample — stored as evaluation runs" },
              { icon: Boxes, title: "Model lifecycle", sub: "TRAINING → READY → PRODUCTION → DEPRECATED · artefacts org/<org>/models/<id>/ (pipeline.joblib, feature_metadata.json, metrics.json, shap_global.json, background.npy, lime_sample.npy, checksums.json)" },
            ]}
          />
        </Card>
      </div>

      <Card title="Social-network adapters" subtitle="A provider converts platform data into the account schema. Manual entry and CSV are always available; the X API v2 provider activates only when the server holds a bearer token.">
        <div className="flex flex-wrap items-center gap-2">
          {["Social Network API Adapter", "Account data (AccountInput)", "Feature extractor", "Prediction + explanation"].map((s, i, arr) => (
            <Fragment key={s}>
              <span className="rounded-lg border border-border bg-surface-2 px-3 py-2 text-xs font-medium text-ink"><Plug className="mr-1 inline h-3.5 w-3.5" />{s}</span>
              {i < arr.length - 1 && <ArrowRight className="h-4 w-4 text-ink-3" />}
            </Fragment>
          ))}
        </div>
        <p className="mt-3 text-xs text-ink-2">Contract: <code className="font-mono">Provider.fetch_account(identifier) → AccountInput</code> (app/services/providers.py). Bundled: <code className="font-mono">manual</code>, <code className="font-mono">csv</code> and <code className="font-mono">x_api</code> (X API v2 <code className="font-mono">/2/users/by/username</code> + <code className="font-mono">/2/users/:id/tweets</code>, active once <code className="font-mono">BOTSHIELD_X_BEARER_TOKEN</code> is set on the server). Until a token is configured the API answers 409 “External data integration is not configured” and never fabricates live data.</p>
      </Card>

      <Card title="Components" padded={false}>
        <Table className="rounded-none border-0">
          <thead><tr><Th>Layer</Th><Th>Location</Th><Th>Responsibility</Th></tr></thead>
          <tbody>
            {[
              ["Frontend pages", "frontend/src/pages", "Dashboard, Analyze, Batch, Datasets, Models, Training, Evaluation, Explainability, History, Research, Architecture, API Docs, Settings"],
              ["API client / types", "frontend/src/services/api.ts · src/types/api.ts", "Typed fetch client mirroring the Pydantic schemas; no ML logic in React"],
              ["HTTP layer", "backend/app/api/v1", "Routers, validation, RBAC dependencies, rate limiting, upload security"],
              ["Services", "backend/app/services", "auth, prediction, dataset, model lifecycle, jobs + handlers, dashboard, audit, providers"],
              ["ML package", "backend/ml", "features · preprocessing · sentiment · train · evaluation · explain · predict · model_registry · datasets · paper_results"],
              ["Persistence", "backend/app/db · alembic/", "SQLAlchemy models + Alembic migrations (PostgreSQL / SQLite) · storage abstraction (local / S3) for artefacts"],
              ["Scripts", "scripts/ · python -m app.cli", "migrate · create-admin · check-config · train_model.py · report_results.py · check_no_demo_data.py · verify_production_readiness.py"],
              ["Docs", "docs/", "architecture.md · methodology.md · api.md · deployment.md · production-audit.md · production-completion-report.md"],
            ].map(([l, p, r]) => <tr key={l}><Td className="font-medium">{l}</Td><Td className="font-mono text-xs text-ink-2">{p}</Td><Td className="text-xs text-ink-2">{r}</Td></tr>)}
          </tbody>
        </Table>
      </Card>

      <Card title="Runtime" subtitle="Reported by GET /api/v1/runtime">
        {runtime.data ? (
          <div className="flex flex-wrap gap-2">
            <Badge>Python {runtime.data.python}</Badge>
            {Object.entries(runtime.data.libraries).map(([k, v]) => <Badge key={k}>{k} {v}</Badge>)}
            <Badge tone="accent">feature version {runtime.data.feature_version} · {runtime.data.n_features} features</Badge>
            <Badge>{runtime.data.environment} · jobs: {runtime.data.job_backend} · storage: {runtime.data.storage_backend}</Badge>
          </div>
        ) : (
          <p className="text-xs text-ink-3">Backend unavailable.</p>
        )}
      </Card>
    </div>
  );
}
