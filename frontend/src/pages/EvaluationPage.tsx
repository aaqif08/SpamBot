import { BarChart3 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { ClassDistributionChart, ConfusionMatrixView, CvFoldsChart, PrChart, ProbabilityHistogram, RocChart } from "@/components/charts/basic";
import { Badge, Button, Card, DemoBanner, EmptyState, ErrorState, Notice, PageHeader, Select, Skeleton, SourceTag, Table, Td, Th } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api } from "@/services/api";
import { dateTime, featureLabel, num, seconds } from "@/utils/format";

export function EvaluationPage() {
  const models = useApi(() => api.models(), []);
  const [modelId, setModelId] = useState<string>("active");
  const ev = useApi(() => api.evaluation(modelId), [modelId], true);

  const header = (
    <PageHeader
      title="Model Evaluation"
      description="Hold-out metrics, confusion matrix, ROC and precision–recall curves and cross-validation folds for a trained model — all measured by this implementation."
      actions={
        <Select value={modelId} onChange={(e) => setModelId(e.target.value)} className="w-72">
          <option value="active">Active model</option>
          {models.data?.models.map((m) => <option key={m.id} value={m.id}>{m.name} · {m.dataset_name}{m.is_demo ? " (demo)" : ""}</option>)}
        </Select>
      }
    />
  );

  if (ev.loading && !ev.data) return <div className="space-y-4">{header}<Skeleton className="h-72" /></div>;
  if (ev.error || !ev.data) {
    return (
      <div className="space-y-4">
        {header}
        {ev.status === 404 ? (
          <EmptyState icon={BarChart3} title="No trained model available" description="Run the training pipeline to populate evaluation results." action={<Link to="/training"><Button>Open training</Button></Link>} />
        ) : (
          <ErrorState message={ev.error ?? ""} onRetry={ev.reload} />
        )}
      </div>
    );
  }

  const d = ev.data;
  const m = d.metrics;
  const hold = m.holdout;
  const cv = m.cross_validation;
  const metricKeys = ["accuracy", "precision", "recall", "f1", "roc_auc"] as const;

  return (
    <div className="space-y-6">
      {header}
      {d.is_demo && <DemoBanner text="This model was trained and evaluated on synthetic demo data; the metrics below are not research results." />}

      <Card title="Experiment setup" actions={<SourceTag kind="ours" />}>
        <dl className="grid gap-3 text-sm sm:grid-cols-3 lg:grid-cols-6">
          {[
            ["Dataset", d.dataset.name],
            ["Model", `${d.model.name} (${d.model.algorithm})`],
            ["Feature version", d.feature_metadata.feature_version],
            ["Number of features", String(d.feature_metadata.n_features)],
            ["Train size", m.split.train_size.toLocaleString()],
            ["Test size", `${m.split.test_size_n.toLocaleString()} (${Math.round(m.split.test_size * 100)}%)`],
            ["CV folds", `${m.split.cv_folds} (stratified)`],
            ["Seed", String(m.split.seed)],
            ["Hyperparameter search", m.hyperparameter_search ? "randomised, F1-scored" : "off"],
            ["Training time", seconds(m.training_seconds)],
            ["Trained", dateTime(d.model.trained_at)],
            ["Feature selection", d.feature_metadata.feature_selection.enabled ? `SHAP top-${d.feature_metadata.feature_selection.top_k}` : `off (${d.feature_metadata.n_features} of 31 used)`],
          ].map(([k, v]) => (
            <div key={k}><dt className="text-[11px] text-ink-3">{k}</dt><dd className="truncate font-medium text-ink" title={v}>{v}</dd></div>
          ))}
        </dl>
      </Card>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {metricKeys.map((k) => (
          <div key={k} className="rounded-xl border border-border bg-surface p-4">
            <div className="text-[11px] uppercase tracking-wide text-ink-3">{k.replace("_", "-")}</div>
            <div className="text-2xl font-semibold tabular-nums text-ink">{num(hold.metrics[k], 3)}</div>
            <div className="text-[11px] text-ink-3">hold-out · CV {num(cv.summary[k].mean, 3)} ± {num(cv.summary[k].std, 3)} · train {num(m.train[k], 3)}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Confusion matrix (hold-out)"><ConfusionMatrixView cm={hold.confusion_matrix} /></Card>
        <Card title="ROC curve"><RocChart roc={hold.roc_curve} /></Card>
        <Card title="Precision–recall curve"><PrChart pr={hold.pr_curve} /></Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={`${cv.n_folds}-fold cross-validation (training split)`} subtitle={`mean fit time ${seconds(cv.fit_time_mean)} per fold`}>
          <CvFoldsChart folds={cv.folds} />
          <Table className="mt-3">
            <thead><tr><Th>Fold</Th>{metricKeys.map((k) => <Th key={k} align="right">{k.replace("_", "-")}</Th>)}</tr></thead>
            <tbody>
              {cv.folds.map((f, i) => <tr key={i}><Td>Fold {i + 1}</Td>{metricKeys.map((k) => <Td key={k} align="right" mono>{num(f[k], 3)}</Td>)}</tr>)}
              <tr className="bg-surface-2"><Td className="font-medium">Mean ± std</Td>{metricKeys.map((k) => <Td key={k} align="right" mono>{num(cv.summary[k].mean, 3)} ± {num(cv.summary[k].std, 3)}</Td>)}</tr>
            </tbody>
          </Table>
        </Card>
        <div className="space-y-4">
          <Card title="Predicted probability distribution (hold-out)"><ProbabilityHistogram bins={hold.probability_histogram} height={200} /></Card>
          <Card title="Class distribution (stratified split)">
            <ClassDistributionChart rows={[{ name: "Train", ...m.class_distribution.train }, { name: "Test", ...m.class_distribution.test }]} height={130} />
          </Card>
        </div>
      </div>

      {d.feature_metadata.dropped_constant_features && d.feature_metadata.dropped_constant_features.length > 0 && (
        <Notice tone="warning">
          {d.feature_metadata.dropped_constant_features.length} of the 31 paper features were constant in the training data and were dropped:{" "}
          {d.feature_metadata.dropped_constant_features.map(featureLabel).join(", ")}. For Cresci this happens when only the user-level files (users.csv) are available; the tweet files (tweets.csv) are distributed by the dataset authors on request. Add them next to users.csv and re-import to enable the tweet-derived features.
        </Notice>
      )}
      <Card title="Feature set used by this model" subtitle={`${d.feature_metadata.scaling} · ${d.feature_metadata.imputation}`}>
        <div className="flex flex-wrap gap-1.5">
          {d.feature_metadata.feature_names.map((f) => <Badge key={f}>{featureLabel(f)}</Badge>)}
        </div>
        {d.feature_metadata.feature_selection.enabled && d.feature_metadata.feature_selection.dropped?.length ? (
          <p className="mt-3 text-xs text-ink-2">Dropped by SHAP selection: {d.feature_metadata.feature_selection.dropped.map(featureLabel).join(", ")}</p>
        ) : null}
        {Object.keys(m.best_params).length > 0 && (
          <p className="mt-3 text-xs text-ink-2">Best hyperparameters: <code className="font-mono">{JSON.stringify(m.best_params)}</code></p>
        )}
      </Card>
    </div>
  );
}
