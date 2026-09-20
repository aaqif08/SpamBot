import clsx from "clsx";
import { Check, Circle, FlaskConical, Loader2, Play, XCircle } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { ConfusionMatrixView, ImportanceChart, RocChart } from "@/components/charts/basic";
import { Badge, Button, Card, DemoBanner, EmptyState, ErrorState, Field, Input, Notice, PageHeader, Select, SourceTag, Table, Td, Th, Toggle } from "@/components/ui";
import { useAction, useApi } from "@/hooks/useApi";
import { useHealth } from "@/hooks/useHealth";
import { useJobPolling } from "@/hooks/usePolling";
import { api } from "@/services/api";
import type { TrainRequest } from "@/types/api";
import { algorithmLabel, dateTime, num, seconds } from "@/utils/format";

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  preprocessing: "Preprocessing",
  feature_engineering: "Feature Engineering",
  training: "Training",
  cross_validation: "Cross Validation",
  shap_analysis: "SHAP Analysis",
  saving: "Saving Model",
  completed: "Completed",
};

function StageTracker({ stages, current, status }: { stages: string[]; current: string; status: string }) {
  const idx = stages.indexOf(current);
  return (
    <ol className="grid gap-1 sm:grid-cols-4 lg:grid-cols-8">
      {stages.map((s, i) => {
        const done = status === "completed" || i < idx;
        const active = i === idx && status !== "completed" && status !== "failed";
        const failed = status === "failed" && i === idx;
        return (
          <li key={s} className={clsx("flex items-center gap-1.5 rounded-md border px-2 py-1.5 text-[11px]", active ? "border-accent bg-accent-soft/60 text-ink" : done ? "border-border text-ink" : "border-border text-ink-3")}>
            {failed ? <XCircle className="h-3.5 w-3.5 text-status-critical" /> : done ? <Check className="h-3.5 w-3.5 text-status-good" /> : active ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Circle className="h-3.5 w-3.5" />}
            {STAGE_LABELS[s] ?? s}
          </li>
        );
      })}
    </ol>
  );
}

export function TrainingPage() {
  const health = useHealth();
  const datasets = useApi(() => api.datasets(), []);
  const models = useApi(() => api.models(), []);
  const stages = useApi(() => api.trainStages(), []);
  const runs = useApi(() => api.trainRuns(), []);
  const [jobId, setJobId] = useState<string | null>(null);
  const poll = useJobPolling(jobId, "train", 1000);
  const [form, setForm] = useState<TrainRequest>({
    dataset_id: "",
    algorithm: "lightgbm",
    test_size: 0.25,
    cv_folds: 5,
    hyperparameter_search: true,
    search_iterations: 6,
    feature_selection: false,
    feature_selection_top_k: 20,
    seed: 42,
    activate: true,
    notes: "",
  });
  const submit = useAction(useCallback((body: TrainRequest) => api.train(body), []));
  const demo = useAction(useCallback(() => api.createDemoDataset(600, 7), []));

  const labelled = datasets.data?.filter((d) => d.has_label) ?? [];
  useEffect(() => {
    if (!form.dataset_id && labelled.length) setForm((f) => ({ ...f, dataset_id: labelled[0].id }));
  }, [labelled, form.dataset_id]);

  const reloadRuns = runs.reload;
  const reloadModels = models.reload;
  const reloadHealth = health.reload;
  useEffect(() => {
    if (poll.done) {
      reloadRuns();
      reloadModels();
      reloadHealth();
    }
  }, [poll.done, reloadRuns, reloadModels, reloadHealth]);

  const set = <K extends keyof TrainRequest>(k: K, v: TrainRequest[K]) => setForm((f) => ({ ...f, [k]: v }));
  const onSubmit = async () => {
    const r = await submit.run(form);
    if (r) setJobId(r.job_id);
  };
  const result = poll.status?.result ?? null;
  const selectedDataset = datasets.data?.find((d) => d.id === form.dataset_id);

  return (
    <div className="space-y-6">
      <PageHeader title="Model Training" description="Training runs in a backend job: stratified split → optional SHAP feature selection → randomised hyperparameter search with stratified k-fold CV → hold-out evaluation → SHAP analysis → model registry." />

      {datasets.data && labelled.length === 0 && (
        <Notice tone="warning">
          No labelled dataset registered. Run <code className="font-mono">python scripts/fetch_datasets.py</code> and import Cresci-15 / Cresci-17 on the Datasets page, or upload a labelled CSV.
          {health.data?.demo_mode_enabled && (
            <Button size="sm" variant="secondary" className="ml-3" icon={FlaskConical} loading={demo.loading} onClick={async () => { const r = await demo.run(); if (r) { datasets.reload(); set("dataset_id", r.id); } }}>Generate demo dataset</Button>
          )}
        </Notice>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr_1.3fr]">
        <Card title="Configuration">
          <div className="space-y-3">
            <Field label="Dataset (labelled)">
              <Select value={form.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
                {labelled.length === 0 && <option value="">No labelled dataset</option>}
                {labelled.map((d) => <option key={d.id} value={d.id}>{d.name} · {d.n_rows.toLocaleString()} rows{d.is_demo ? " (DEMO)" : ""}</option>)}
              </Select>
            </Field>
            {selectedDataset?.is_demo && <DemoBanner compact text="Models trained on this dataset are flagged as demo models everywhere in the app." />}
            <Field label="Classifier">
              <Select value={form.algorithm} onChange={(e) => set("algorithm", e.target.value)}>
                {models.data?.supported_algorithms.map((a) => <option key={a.key} value={a.key} disabled={!a.available}>{a.display_name}{a.available ? "" : " (unavailable)"}</option>)}
              </Select>
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Test size" hint="Paper: 75/25 (also mentions 70/30)"><Input type="number" step={0.05} min={0.05} max={0.5} value={form.test_size} onChange={(e) => set("test_size", Number(e.target.value))} /></Field>
              <Field label="CV folds" hint="Paper: 5-fold"><Input type="number" min={2} max={10} value={form.cv_folds} onChange={(e) => set("cv_folds", Number(e.target.value))} /></Field>
              <Field label="Search iterations" hint="Randomised search candidates"><Input type="number" min={1} max={30} value={form.search_iterations} onChange={(e) => set("search_iterations", Number(e.target.value))} /></Field>
              <Field label="Random seed"><Input type="number" min={0} value={form.seed} onChange={(e) => set("seed", Number(e.target.value))} /></Field>
            </div>
            <Toggle label="Hyperparameter optimisation (CV-scored randomised search)" checked={form.hyperparameter_search} onChange={(v) => set("hyperparameter_search", v)} />
            <Toggle label="SHAP-based feature selection (paper §III-C)" hint="Fit a screening model, rank by mean |SHAP|, keep top-k" checked={form.feature_selection} onChange={(v) => set("feature_selection", v)} />
            {form.feature_selection && <Field label="Keep top-k features"><Input type="number" min={5} max={31} value={form.feature_selection_top_k} onChange={(e) => set("feature_selection_top_k", Number(e.target.value))} /></Field>}
            <Toggle label="Activate model after training" checked={form.activate} onChange={(v) => set("activate", v)} />
            <Field label="Notes"><Input value={form.notes} onChange={(e) => set("notes", e.target.value)} placeholder="optional" /></Field>
            <Button icon={Play} onClick={onSubmit} loading={submit.loading} disabled={!form.dataset_id || (poll.status ? !poll.done : false)}>Train model</Button>
            {submit.error && <ErrorState title="Could not start training" message={submit.error} />}
          </div>
        </Card>

        <Card title="Training progress" subtitle={jobId ? `Job ${jobId}` : "Submit a job to see progress"}>
          {!jobId && <EmptyState icon={FlaskConical} title="No job running" />}
          {jobId && (
            <div className="space-y-4">
              <StageTracker stages={stages.data ?? Object.keys(STAGE_LABELS)} current={poll.status?.stage ?? "queued"} status={poll.status?.status ?? "queued"} />
              <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
                <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${Math.round((poll.status?.progress ?? 0) * 100)}%` }} />
              </div>
              <p className="text-xs text-ink-2">{poll.status?.message ?? "Waiting…"}</p>
              {poll.error && <ErrorState message={poll.error} />}
              {poll.status?.status === "failed" && <ErrorState title="Training failed" message={poll.status.error ?? "Unknown error"} />}
              {poll.status?.log && poll.status.log.length > 0 && (
                <pre className="scrollbar-thin max-h-40 overflow-auto rounded-lg bg-surface-2 p-3 text-[11px] text-ink-2">{poll.status.log.join("\n")}</pre>
              )}
            </div>
          )}
        </Card>
      </div>

      {result && (
        <div className="space-y-4 animate-fade-in">
          {result.is_demo && <DemoBanner compact />}
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold text-ink">Result — {result.model.name}</h2>
            <Badge tone="neutral">{result.model_id}</Badge>
            <SourceTag kind="ours" />
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
            {(["accuracy", "precision", "recall", "f1", "roc_auc"] as const).map((k) => (
              <div key={k} className="rounded-lg border border-border bg-surface p-3">
                <div className="text-[11px] uppercase text-ink-3">{k.replace("_", "-")} (hold-out)</div>
                <div className="text-lg font-semibold tabular-nums text-ink">{num(result.metrics.holdout.metrics[k], 3)}</div>
                <div className="text-[11px] text-ink-3">CV mean {num(result.metrics.cross_validation.summary[k].mean, 3)} ± {num(result.metrics.cross_validation.summary[k].std, 3)}</div>
              </div>
            ))}
          </div>
          <div className="grid gap-4 lg:grid-cols-3">
            <Card title="Confusion matrix (hold-out)"><ConfusionMatrixView cm={result.metrics.holdout.confusion_matrix} /></Card>
            <Card title="ROC curve (hold-out)"><RocChart roc={result.metrics.holdout.roc_curve} height={220} /></Card>
            <Card title="Best hyperparameters" subtitle={`training ${seconds(result.metrics.training_seconds)}`}>
              {Object.keys(result.metrics.best_params).length ? (
                <dl className="space-y-1 text-xs">
                  {Object.entries(result.metrics.best_params).map(([k, v]) => <div key={k} className="flex justify-between border-b border-border py-1"><dt className="text-ink-2">{k}</dt><dd className="font-mono text-ink">{String(v)}</dd></div>)}
                </dl>
              ) : <p className="text-xs text-ink-3">Defaults (no search).</p>}
            </Card>
          </div>
          <Card title="SHAP feature importance (top 20)" subtitle={`${result.shap_global.explainer} · scale ${result.shap_global.output_scale}`}>
            <ImportanceChart rows={result.shap_global.importance} outputScale={result.shap_global.output_scale} />
          </Card>
        </div>
      )}

      <Card title="Training runs" padded={false}>
        {runs.error && <div className="p-4"><ErrorState message={runs.error} onRetry={runs.reload} /></div>}
        {runs.data && runs.data.length === 0 && <div className="p-4"><EmptyState title="No training runs yet" /></div>}
        {runs.data && runs.data.length > 0 && (
          <Table className="rounded-none border-0">
            <thead>
              <tr><Th>Model</Th><Th>Dataset</Th><Th>Status</Th><Th align="right">Accuracy</Th><Th align="right">F1</Th><Th align="right">AUC</Th><Th>Started</Th><Th>Completed</Th></tr>
            </thead>
            <tbody>
              {runs.data.map((r) => (
                <tr key={r.id}>
                  <Td>{algorithmLabel(r.model)}{r.is_demo && <Badge tone="demo" className="ml-2">demo</Badge>}</Td>
                  <Td className="text-xs">{r.dataset}</Td>
                  <Td><Badge tone={r.status === "completed" ? "good" : r.status === "failed" ? "critical" : "neutral"}>{r.status}{r.status === "running" ? ` · ${STAGE_LABELS[r.stage] ?? r.stage}` : ""}</Badge>{r.error && <div className="max-w-xs truncate text-[11px] text-status-critical" title={r.error}>{r.error}</div>}</Td>
                  <Td align="right" mono>{num(r.metrics?.holdout.accuracy, 3)}</Td>
                  <Td align="right" mono>{num(r.metrics?.holdout.f1, 3)}</Td>
                  <Td align="right" mono>{num(r.metrics?.holdout.roc_auc, 3)}</Td>
                  <Td className="text-xs text-ink-2">{dateTime(r.created_at)}</Td>
                  <Td className="text-xs text-ink-2">{dateTime(r.completed_at)}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}
