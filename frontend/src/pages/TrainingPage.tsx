import { FlaskConical, Play } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { ConfusionMatrixView, CvFoldsChart, ImportanceChart, RocChart } from "@/components/charts/basic";
import { Button, Card, EmptyState, ErrorState, Field, Input, Notice, PageHeader, Select, SourceTag, StatTile, StatusBadge, Table, Td, Textarea, Th, Toggle } from "@/components/ui";
import { useAction, useApi } from "@/hooks/useApi";
import { useAuth } from "@/hooks/useAuth";
import { useJobPolling } from "@/hooks/usePolling";
import { api } from "@/services/api";
import type { TrainRequest, TrainingJobResult } from "@/types/api";
import { algorithmLabel, dateTime, num, pct, seconds } from "@/utils/format";

const DEFAULTS: Omit<TrainRequest, "dataset_id" | "algorithm"> = { dataset_version_id: null, test_size: 0.2, cv_folds: 5, hyperparameter_search: true, search_iterations: 20, feature_selection: false, feature_selection_top_k: 20, seed: 42, activate: false, notes: "" };

export function TrainingPage() {
  const { hasRole } = useAuth();
  const isAdmin = hasRole("ADMIN");
  const navigate = useNavigate();
  const datasets = useApi(() => api.datasets(), []);
  const models = useApi(() => api.models(), []);
  const jobs = useApi(() => api.jobs("TRAINING"), []);
  const [form, setForm] = useState<TrainRequest>({ dataset_id: "", algorithm: "lightgbm", ...DEFAULTS });
  const [jobId, setJobId] = useState<string | null>(null);
  const poll = useJobPolling(jobId, 1500);
  const submit = useAction(useCallback((body: TrainRequest) => api.train(body), []));
  const set = <K extends keyof TrainRequest>(k: K, v: TrainRequest[K]) => setForm((f) => ({ ...f, [k]: v }));

  const labelled = datasets.data?.filter((d) => d.has_label && d.status === "VALIDATED") ?? [];
  const algos = models.data?.supported_algorithms ?? [];
  useEffect(() => { if (!form.dataset_id && labelled.length) set("dataset_id", labelled[0].id); }, [labelled, form.dataset_id]);
  useEffect(() => { if (algos.length && !algos.some((a) => a.key === form.algorithm && a.available)) { const first = algos.find((a) => a.available); if (first) set("algorithm", first.key); } }, [algos, form.algorithm]);
  const reloadJobs = jobs.reload; const reloadModels = models.reload;
  useEffect(() => { if (poll.done) { reloadJobs(); reloadModels(); } }, [poll.done, reloadJobs, reloadModels]);

  const onSubmit = async () => {
    const r = await submit.run(form);
    if (r) { setJobId(r.job_id); jobs.reload(); }
  };
  const result = poll.status?.status === "COMPLETED" ? (poll.status.result as unknown as TrainingJobResult | null) : null;

  return (
    <div className="space-y-6">
      <PageHeader title="Training" description="Train a classifier on a labelled dataset using the reference pipeline: 31 features → median imputation → Min-Max scaling → stratified hold-out split → 5-fold cross-validation → randomized hyper-parameter search (F1)." />
      <div className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
        <Card title="New training run">
          {datasets.data && labelled.length === 0 ? (
            <EmptyState icon={FlaskConical} title="No labelled dataset available" description="Upload a validated CSV with a bot/human label column first." action={<Link to="/datasets"><Button>Upload a dataset</Button></Link>} />
          ) : (
            <div className="space-y-3">
              <Field label="Dataset"><Select value={form.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>{labelled.map((d) => <option key={d.id} value={d.id}>{d.name} · {d.n_rows.toLocaleString()} rows{d.class_distribution ? ` · ${d.class_distribution.BOT ?? 0} bot / ${d.class_distribution.HUMAN ?? 0} human` : ""}</option>)}</Select></Field>
              <Field label="Algorithm"><Select value={form.algorithm} onChange={(e) => set("algorithm", e.target.value)}>{algos.map((a) => <option key={a.key} value={a.key} disabled={!a.available}>{a.display_name}{a.available ? "" : ` — ${a.unavailable_reason}`}</option>)}</Select></Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Test size"><Input type="number" min={0.1} max={0.5} step={0.05} value={form.test_size} onChange={(e) => set("test_size", Number(e.target.value))} /></Field>
                <Field label="CV folds"><Input type="number" min={2} max={10} value={form.cv_folds} onChange={(e) => set("cv_folds", Number(e.target.value))} /></Field>
                <Field label="Seed"><Input type="number" value={form.seed} onChange={(e) => set("seed", Number(e.target.value))} /></Field>
                <Field label="Search iterations"><Input type="number" min={1} max={200} value={form.search_iterations} disabled={!form.hyperparameter_search} onChange={(e) => set("search_iterations", Number(e.target.value))} /></Field>
              </div>
              <Toggle label="Randomized hyper-parameter search" checked={form.hyperparameter_search} onChange={(v) => set("hyperparameter_search", v)} hint="Scored by F1 inside the cross-validation loop." />
              <Toggle label="Feature selection" checked={form.feature_selection} onChange={(v) => set("feature_selection", v)} hint="Keep only the top-k features by mutual information (computed on the training split)." />
              {form.feature_selection && <Field label="Top-k features"><Input type="number" min={5} max={31} value={form.feature_selection_top_k} onChange={(e) => set("feature_selection_top_k", Number(e.target.value))} /></Field>}
              {isAdmin && <Toggle label="Promote to production when finished" checked={form.activate} onChange={(v) => set("activate", v)} />}
              <Field label="Notes"><Textarea rows={2} value={form.notes} onChange={(e) => set("notes", e.target.value)} maxLength={2000} /></Field>
              <Button icon={Play} onClick={onSubmit} loading={submit.loading} disabled={!form.dataset_id || !form.algorithm}>Start training</Button>
              {submit.error && <ErrorState title="Could not start training" message={submit.error} />}
            </div>
          )}
        </Card>

        <div className="space-y-4">
          {jobId && poll.status && (
            <Card title="Current run" actions={<StatusBadge status={poll.status.status} />}>
              <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${Math.round(poll.status.progress * 100)}%` }} /></div>
              <p className="mt-2 text-sm text-ink-2" role="status">{poll.status.stage} — {poll.status.message}</p>
              {poll.status.log.length > 0 && <pre className="mt-2 max-h-40 overflow-auto rounded-lg bg-surface-2 p-2 text-[11px] text-ink-2">{poll.status.log.join("\n")}</pre>}
              {poll.status.status === "FAILED" && <ErrorState title="Training failed" message={poll.status.error ?? "Unknown error"} />}
              {poll.error && <ErrorState message={poll.error} />}
            </Card>
          )}
          {result && (
            <>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <StatTile label="Model" value={`${result.model.name} v${result.model.version}`} sub={algorithmLabel(result.model.algorithm)} />
                <StatTile label="Hold-out accuracy" value={pct(result.metrics.holdout.metrics.accuracy)} sub={`F1 ${num(result.metrics.holdout.metrics.f1)}`} />
                <StatTile label="CV F1" value={`${num(result.metrics.cross_validation.summary.f1.mean)} ± ${num(result.metrics.cross_validation.summary.f1.std)}`} sub={`${result.metrics.cross_validation.n_folds} folds`} />
                <StatTile label="Training time" value={seconds(result.metrics.training_seconds)} sub={result.activated ? "promoted to production" : "status READY"} />
              </div>
              {!result.activated && isAdmin && <Notice>Model is READY. Promote it on the <Link to="/models" className="underline">Models</Link> page to use it for analyses.</Notice>}
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Confusion matrix (hold-out)" actions={<SourceTag kind="ours" />}><ConfusionMatrixView cm={result.metrics.holdout.confusion_matrix} /></Card>
                <Card title="ROC (hold-out)"><RocChart roc={result.metrics.holdout.roc_curve} /></Card>
                <Card title="Cross-validation folds"><CvFoldsChart folds={result.metrics.cross_validation.folds} /></Card>
                <Card title="Feature importance (mean |SHAP|)" subtitle={`${result.explainer} · ${result.output_scale}`}>{result.feature_importance.length ? <ImportanceChart rows={result.feature_importance} /> : <EmptyState title="SHAP unavailable for this model" />}</Card>
              </div>
              <div className="flex gap-2"><Button variant="secondary" onClick={() => navigate(`/evaluation?model=${result.model_id}`)}>Open evaluation</Button><Button variant="ghost" onClick={() => navigate(`/explainability?model=${result.model_id}`)}>Global explanation</Button></div>
            </>
          )}
          <Card title="Training history" padded={false}>
            {jobs.error && <div className="p-4"><ErrorState message={jobs.error} onRetry={jobs.reload} /></div>}
            {jobs.data && jobs.data.length === 0 && <div className="p-4"><EmptyState title="No training runs yet" /></div>}
            {jobs.data && jobs.data.length > 0 && (
              <Table className="rounded-none border-0" compact>
                <thead><tr><Th>Started</Th><Th>Status</Th><Th>Stage</Th><Th>Model</Th><Th></Th></tr></thead>
                <tbody>{jobs.data.map((j) => <tr key={j.id}><Td className="text-xs">{dateTime(j.created_at)}</Td><Td><StatusBadge status={j.status} /></Td><Td className="text-xs">{j.status === "FAILED" ? j.error : j.message}</Td><Td className="text-xs">{(j.result as { model?: { name?: string; version?: number } } | null)?.model?.name ? `${(j.result as { model: { name: string; version: number } }).model.name} v${(j.result as { model: { version: number } }).model.version}` : j.target_id ? j.target_id.slice(0, 8) : "—"}</Td><Td>{(j.status === "PROCESSING" || j.status === "QUEUED") && j.id !== jobId && <Button size="sm" variant="ghost" onClick={() => setJobId(j.id)}>Follow</Button>}</Td></tr>)}</tbody>
              </Table>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
