import { Download, FileUp, Layers, Play } from "lucide-react";
import { useCallback, useState } from "react";
import { Link } from "react-router-dom";

import { ConfusionMatrixView, ProbabilityHistogram, ProportionBar, RocChart } from "@/components/charts/basic";
import { Badge, Button, Card, DemoBanner, EmptyState, ErrorState, Field, Notice, PageHeader, PredictionBadge, RiskBadge, Select, StatTile, Table, Td, Th, Toggle } from "@/components/ui";
import { useAction, useApi } from "@/hooks/useApi";
import { useHealth } from "@/hooks/useHealth";
import { api } from "@/services/api";
import type { BatchSummary } from "@/types/api";
import { dateTime, int, num, pct } from "@/utils/format";

export function BatchAnalysisPage() {
  const health = useHealth();
  const datasets = useApi(() => api.datasets(), []);
  const models = useApi(() => api.models(), []);
  const batches = useApi(() => api.batches(), []);
  const [file, setFile] = useState<File | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [modelId, setModelId] = useState("");
  const [evaluate, setEvaluate] = useState(true);
  const [summary, setSummary] = useState<BatchSummary | null>(null);

  const run = useAction(
    useCallback(async () => {
      if (file) return api.predictBatchFile(file, { model_id: modelId || null, evaluate_if_labelled: evaluate });
      if (datasetId) return api.predictBatchDataset(datasetId, { model_id: modelId || null, evaluate_if_labelled: evaluate });
      throw new Error("Choose a CSV file or a registered dataset");
    }, [file, datasetId, modelId, evaluate]),
  );
  const load = useAction(useCallback((id: string) => api.batch(id), []));

  const onRun = async () => {
    const r = await run.run();
    if (r) {
      setSummary(r);
      batches.reload();
    }
  };
  const noModel = health.data ? !health.data.model_available : false;

  return (
    <div className="space-y-6">
      <PageHeader title="Batch Analysis" description="Upload a CSV of accounts (one per row) or pick a registered dataset. The backend featurises every row, predicts with the selected model and produces a downloadable predictions.csv." />
      {noModel && <Notice tone="warning">No trained model available. Train a model first.</Notice>}

      <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
        <Card title="Input" subtitle="CSV columns can be raw counts (followers_count, friends_count, statuses_count, …) or pre-computed features; Cresci users.csv column names are recognised.">
          <div className="space-y-3">
            <Field label="CSV file">
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-border-strong px-4 py-6 text-sm text-ink-2 hover:bg-surface-2">
                <FileUp className="h-4 w-4" aria-hidden />
                {file ? `${file.name} (${(file.size / 1024).toFixed(0)} KB)` : "Choose a .csv file"}
                <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => { setFile(e.target.files?.[0] ?? null); if (e.target.files?.[0]) setDatasetId(""); }} />
              </label>
            </Field>
            <div className="text-center text-xs text-ink-3">— or —</div>
            <Field label="Registered dataset">
              <Select value={datasetId} onChange={(e) => { setDatasetId(e.target.value); if (e.target.value) setFile(null); }}>
                <option value="">Select a dataset…</option>
                {datasets.data?.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} · {d.n_rows.toLocaleString()} rows{d.has_label ? " · labelled" : ""}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Model">
              <Select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                <option value="">Active model</option>
                {models.data?.models.map((m) => (
                  <option key={m.id} value={m.id}>{m.name} · {m.dataset_name}{m.is_demo ? " (demo)" : ""}</option>
                ))}
              </Select>
            </Field>
            <Toggle label="Evaluate against labels when a label column exists" checked={evaluate} onChange={setEvaluate} />
            <Button icon={Play} onClick={onRun} loading={run.loading} disabled={noModel || (!file && !datasetId)}>
              Run batch prediction
            </Button>
            {run.error && <ErrorState title="Batch prediction failed" message={run.error} />}
          </div>
        </Card>

        <Card title="Previous batches" subtitle="Stored batch runs (click to reopen)">
          {batches.error && <ErrorState message={batches.error} onRetry={batches.reload} />}
          {batches.data && batches.data.length === 0 && <EmptyState icon={Layers} title="No batches yet" />}
          {batches.data && batches.data.length > 0 && (
            <Table>
              <thead>
                <tr>
                  <Th>Name</Th>
                  <Th>Model</Th>
                  <Th align="right">Rows</Th>
                  <Th align="right">Bots</Th>
                  <Th align="right">Avg P(bot)</Th>
                  <Th>Date</Th>
                </tr>
              </thead>
              <tbody>
                {batches.data.map((b) => (
                  <tr key={b.batch_id} className="cursor-pointer hover:bg-surface-2" onClick={async () => { const r = await load.run(b.batch_id); if (r) setSummary(r); }}>
                    <Td>{b.name}{b.is_demo && <Badge tone="demo" className="ml-2">demo</Badge>}</Td>
                    <Td>{b.model_name}</Td>
                    <Td align="right" mono>{int(b.total_accounts)}</Td>
                    <Td align="right" mono>{int(b.predicted_bots)}</Td>
                    <Td align="right" mono>{pct(b.average_bot_probability)}</Td>
                    <Td className="text-xs text-ink-2">{dateTime(b.created_at)}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card>
      </div>

      {summary && (
        <div className="space-y-4 animate-fade-in">
          {summary.is_demo && <DemoBanner compact />}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-ink">Results — {summary.name}</h2>
            <a href={api.batchDownloadUrl(summary.batch_id)} download="predictions.csv">
              <Button icon={Download} variant="secondary">Download predictions.csv</Button>
            </a>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
            <StatTile label="Total accounts" value={int(summary.total_accounts)} />
            <StatTile label="Predicted bots" value={int(summary.predicted_bots)} accent="var(--bot)" />
            <StatTile label="Predicted humans" value={int(summary.predicted_humans)} accent="var(--human)" />
            <StatTile label="Average bot probability" value={pct(summary.average_bot_probability)} />
            <StatTile label="High-risk (score ≥ 60)" value={int(summary.high_risk_accounts)} accent="var(--status-serious)" />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Distribution" subtitle="Predicted class share and risk bands">
              <ProportionBar bots={summary.predicted_bots} humans={summary.predicted_humans} />
              <div className="mt-4 flex flex-wrap gap-2">
                {Object.entries(summary.risk_band_distribution).map(([band, n]) => (
                  <span key={band} className="flex items-center gap-1 text-xs text-ink-2">
                    <RiskBadge band={band} /> <span className="font-mono text-ink">{n}</span>
                  </span>
                ))}
              </div>
            </Card>
            <Card title="Bot probability histogram">
              <ProbabilityHistogram bins={summary.probability_histogram} />
            </Card>
          </div>
          {summary.evaluation && (
            <Card title="Evaluation against provided labels" subtitle={`${summary.evaluation.n_samples.toLocaleString()} labelled rows · model ${summary.model.name}`} actions={<Badge tone="accent">Reproduced by this implementation</Badge>}>
              <div className="grid gap-4 lg:grid-cols-3">
                <div className="grid grid-cols-2 gap-2">
                  {(["accuracy", "precision", "recall", "f1", "roc_auc"] as const).map((k) => (
                    <div key={k} className="rounded-lg border border-border p-3">
                      <div className="text-[11px] uppercase text-ink-3">{k.replace("_", "-")}</div>
                      <div className="text-lg font-semibold tabular-nums text-ink">{num(summary.evaluation?.metrics[k], 3)}</div>
                    </div>
                  ))}
                </div>
                <ConfusionMatrixView cm={summary.evaluation.confusion_matrix} />
                <RocChart roc={summary.evaluation.roc_curve} height={200} />
              </div>
            </Card>
          )}
          <Card title="Preview (first 25 rows)" subtitle="Full output incl. the top-10 SHAP-ranked features is in predictions.csv">
            <Table>
              <thead>
                <tr>
                  <Th>Account</Th>
                  {"label" in (summary.preview[0] ?? {}) && <Th>Label</Th>}
                  <Th>Prediction</Th>
                  <Th align="right">P(bot)</Th>
                  <Th align="right">P(human)</Th>
                  <Th>Risk</Th>
                </tr>
              </thead>
              <tbody>
                {summary.preview.map((r, i) => (
                  <tr key={i}>
                    <Td className="font-mono text-xs">{String(r.account_id)}</Td>
                    {"label" in r && <Td>{String(r.label ?? "")}</Td>}
                    <Td><PredictionBadge label={r.prediction as "BOT" | "HUMAN"} /></Td>
                    <Td align="right" mono>{num(Number(r.bot_probability), 3)}</Td>
                    <Td align="right" mono>{num(Number(r.human_probability), 3)}</Td>
                    <Td><RiskBadge band={String(r.risk_band)} score={Number(r.risk_score)} /></Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <p className="mt-2 text-[11px] text-ink-3">
              Rows from this batch are stored in <Link to="/history" className="underline">History</Link>; open one to compute its SHAP/LIME explanation on demand.
            </p>
          </Card>
        </div>
      )}
    </div>
  );
}
