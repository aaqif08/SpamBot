import { BarChart3 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { ClassDistributionChart, ConfusionMatrixView, CvFoldsChart, ImportanceChart, PrChart, ProbabilityHistogram, RocChart } from "@/components/charts/basic";
import { Badge, Card, EmptyState, ErrorState, Field, PageHeader, Select, Skeleton, SourceTag, StatTile, StatusBadge, Table, Td, Th } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api } from "@/services/api";
import type { Evaluation, EvaluationRunPublic, MetricSet } from "@/types/api";
import { algorithmLabel, dateTime, int, num, seconds } from "@/utils/format";

function MetricTiles({ m }: { m: MetricSet }) {
  return <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">{(["accuracy", "precision", "recall", "f1", "roc_auc"] as const).map((k) => <div key={k} className="rounded-lg border border-border p-3"><div className="text-[11px] uppercase text-ink-3">{k.replace("_", "-")}</div><div className="text-lg font-semibold tabular-nums text-ink">{num(m[k], 4)}</div></div>)}</div>;
}

function EvaluationBlock({ e }: { e: Evaluation }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Confusion matrix"><ConfusionMatrixView cm={e.confusion_matrix} /><p className="mt-2 text-xs text-ink-3">FPR {num(e.confusion_matrix.false_positive_rate)} · FNR {num(e.confusion_matrix.false_negative_rate)} · {int(e.n_positive)} bots / {int(e.n_negative)} humans</p></Card>
      <Card title="ROC curve"><RocChart roc={e.roc_curve} /></Card>
      <Card title="Precision–recall curve"><PrChart pr={e.pr_curve} /></Card>
      <Card title="Estimated bot probability distribution"><ProbabilityHistogram bins={e.probability_histogram} /></Card>
    </div>
  );
}

export function EvaluationPage() {
  const [params, setParams] = useSearchParams();
  const models = useApi(() => api.models(), []);
  const [modelId, setModelId] = useState(params.get("model") ?? "");
  useEffect(() => { if (!modelId && models.data) { const id = models.data.production_model_id ?? models.data.models.find((m) => m.status !== "TRAINING" && m.status !== "FAILED")?.id ?? ""; if (id) setModelId(id); } }, [models.data, modelId]);
  useEffect(() => { if (modelId && params.get("model") !== modelId) setParams({ model: modelId }, { replace: true }); }, [modelId, params, setParams]);
  const ev = useApi(() => api.modelEvaluation(modelId), [modelId], !!modelId);
  const [runId, setRunId] = useState<string>("");

  const runs = ev.data?.evaluations ?? [];
  const holdout = runs.find((r) => r.kind === "holdout");
  const cv = runs.find((r) => r.kind === "cross_validation");
  const datasetRuns = runs.filter((r) => r.kind === "dataset");
  const selectedRun: EvaluationRunPublic | undefined = runId ? runs.find((r) => r.id === runId) : holdout;
  const train = holdout?.details;

  return (
    <div className="space-y-6">
      <PageHeader title="Evaluation" description="Stored evaluation runs for each model: the stratified hold-out split, cross-validation folds and any later evaluations on labelled datasets. Nothing here is estimated — every figure was computed on real rows and persisted." actions={<Field label="Model" className="min-w-[18rem]"><Select value={modelId} onChange={(e) => { setModelId(e.target.value); setRunId(""); }}>{!models.data?.models.length && <option value="">No models</option>}{models.data?.models.filter((m) => m.status !== "TRAINING").map((m) => <option key={m.id} value={m.id}>{m.name} v{m.version} · {m.status.toLowerCase()}</option>)}</Select></Field>} />

      {models.data && models.data.models.length === 0 && <Card><EmptyState icon={BarChart3} title="No models to evaluate" description="Train a model first; its hold-out and cross-validation results will be stored here." action={<Link to="/training" className="text-sm underline">Go to Training</Link>} /></Card>}
      {ev.error && <ErrorState message={ev.error} onRetry={ev.reload} />}
      {ev.loading && !ev.data && <div className="grid gap-4 sm:grid-cols-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)}</div>}

      {ev.data && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <StatTile label="Model" value={`${ev.data.model.name} v${ev.data.model.version}`} sub={<span className="flex items-center gap-1">{algorithmLabel(ev.data.model.algorithm)} <StatusBadge status={ev.data.model.status} /></span>} />
            <StatTile label="Dataset" value={ev.data.model.dataset_name || "—"} sub={train?.split ? `${int(train.split.train_size)} train / ${int(train.split.test_size_n)} test (stratified)` : undefined} />
            <StatTile label="Features" value={ev.data.feature_metadata.n_features} sub={ev.data.feature_metadata.dropped_constant_features?.length ? `${ev.data.feature_metadata.dropped_constant_features.length} constant features dropped` : ev.data.feature_metadata.feature_selection.enabled ? `top-${ev.data.feature_metadata.feature_selection.top_k} selected` : "all 31 features"} />
            <StatTile label="Training" value={seconds(ev.data.model.training_seconds)} sub={train?.hyperparameter_search ? "with randomized search" : "default hyper-parameters"} />
          </div>

          {runs.length === 0 && <Card><EmptyState title="No evaluation runs stored for this model" /></Card>}

          {runs.length > 0 && (
            <Card title="Evaluation runs" padded={false} actions={<SourceTag kind="ours" />}>
              <Table className="rounded-none border-0" compact>
                <thead><tr><Th>Kind</Th><Th>Dataset</Th><Th align="right">Samples</Th><Th align="right">Acc</Th><Th align="right">Prec</Th><Th align="right">Rec</Th><Th align="right">F1</Th><Th align="right">AUC</Th><Th>Date</Th></tr></thead>
                <tbody>{runs.map((r) => <tr key={r.id} className={`cursor-pointer hover:bg-surface-2 ${selectedRun?.id === r.id ? "bg-accent-soft" : ""}`} onClick={() => setRunId(r.id)}><Td><Badge tone={r.kind === "holdout" ? "good" : "neutral"}>{r.kind.replace("_", " ")}</Badge></Td><Td className="text-xs">{r.details.dataset_name ?? ev.data?.model.dataset_name ?? "—"}</Td><Td align="right" mono>{int(r.n_samples)}</Td><Td align="right" mono>{num(r.metrics.accuracy)}</Td><Td align="right" mono>{num(r.metrics.precision)}</Td><Td align="right" mono>{num(r.metrics.recall)}</Td><Td align="right" mono>{num(r.metrics.f1)}</Td><Td align="right" mono>{num(r.metrics.roc_auc)}</Td><Td className="text-xs">{dateTime(r.created_at)}</Td></tr>)}</tbody>
              </Table>
            </Card>
          )}

          {selectedRun && (
            <>
              <Card title={selectedRun.kind === "holdout" ? "Hold-out test split" : selectedRun.kind === "cross_validation" ? "Cross-validation (mean over folds)" : `Dataset evaluation · ${selectedRun.details.dataset_name ?? ""}`} subtitle={`${int(selectedRun.n_samples)} accounts · ${dateTime(selectedRun.created_at)}`} actions={<SourceTag kind="ours" />}>
                <MetricTiles m={selectedRun.metrics} />
              </Card>
              {selectedRun.kind === "holdout" && selectedRun.details.holdout && <EvaluationBlock e={selectedRun.details.holdout} />}
              {selectedRun.kind === "dataset" && selectedRun.details.holdout && <EvaluationBlock e={selectedRun.details.holdout} />}
              {selectedRun.kind === "cross_validation" && selectedRun.details.cross_validation && <Card title="Per-fold metrics"><CvFoldsChart folds={selectedRun.details.cross_validation.folds} /></Card>}
            </>
          )}

          {train && (
            <div className="grid gap-4 lg:grid-cols-2">
              {cv?.details.cross_validation && <Card title="Cross-validation folds" subtitle={`${cv.details.cross_validation.n_folds}-fold stratified · mean fit ${seconds(cv.details.cross_validation.fit_time_mean)}`}><CvFoldsChart folds={cv.details.cross_validation.folds} /><Table compact className="mt-3"><thead><tr><Th>Metric</Th><Th align="right">Mean</Th><Th align="right">Std</Th></tr></thead><tbody>{Object.entries(cv.details.cross_validation.summary).map(([k, v]) => <tr key={k}><Td>{k}</Td><Td align="right" mono>{num(v.mean, 4)}</Td><Td align="right" mono>{num(v.std, 4)}</Td></tr>)}</tbody></Table></Card>}
              {train.class_distribution && <Card title="Class distribution of the split"><ClassDistributionChart rows={[{ name: "train", human: train.class_distribution.train.human, bot: train.class_distribution.train.bot }, { name: "test", human: train.class_distribution.test.human, bot: train.class_distribution.test.bot }]} />{train.train && <p className="mt-2 text-xs text-ink-3">Training-set F1 {num(train.train.f1)} vs hold-out F1 {num(selectedRun?.metrics.f1 ?? holdout?.metrics.f1)} — a large gap indicates over-fitting.</p>}</Card>}
              {train.best_params && Object.keys(train.best_params).length > 0 && <Card title="Selected hyper-parameters"><Table compact><tbody>{Object.entries(train.best_params).map(([k, v]) => <tr key={k}><Td mono>{k}</Td><Td mono>{String(v)}</Td></tr>)}</tbody></Table></Card>}
              <Card title="Feature importance (mean |SHAP|)" subtitle="Computed on a subset of training rows after fitting">{ev.data.feature_importance.length ? <ImportanceChart rows={ev.data.feature_importance} /> : <EmptyState title="No SHAP analysis stored" />}</Card>
            </div>
          )}
          {datasetRuns.length === 0 && <p className="text-xs text-ink-3">Evaluate this model on another labelled dataset from the <Link to="/datasets" className="underline">Datasets</Link> page to add a run here.</p>}
        </>
      )}
    </div>
  );
}
