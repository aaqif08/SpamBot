import { Download, FileUp, Layers, Play, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ConfusionMatrixView, ProbabilityHistogram, ProportionBar, RocChart } from "@/components/charts/basic";
import { Button, Card, EmptyState, ErrorState, Field, PageHeader, RiskBadge, Select, StatTile, StatusBadge, Table, Td, Th } from "@/components/ui";
import { ConfirmDialog } from "@/components/ui/Dialog";
import { useAction, useApi } from "@/hooks/useApi";
import { useAuth } from "@/hooks/useAuth";
import { useJobPolling } from "@/hooks/usePolling";
import { api, errorMessage } from "@/services/api";
import type { BatchResponse } from "@/types/api";
import { dateTime, int, num, pct } from "@/utils/format";

export function BatchAnalysisPage() {
  const { hasRole } = useAuth();
  const canRun = hasRole("ADMIN", "ANALYST");
  const datasets = useApi(() => api.datasets(), []);
  const models = useApi(() => api.models(), []);
  const batches = useApi(() => api.batches(), []);
  const [file, setFile] = useState<File | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [modelId, setModelId] = useState("");
  const [selected, setSelected] = useState<BatchResponse | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const poll = useJobPolling(selected && (selected.status === "QUEUED" || selected.status === "PROCESSING") ? selected.job_id : null, 1500);

  const run = useAction(useCallback(async () => {
    if (file) return api.createBatchFile(file, { model_id: modelId || null });
    if (datasetId) return api.createBatchDataset(datasetId, { model_id: modelId || null });
    throw new Error("Choose a CSV file or a registered dataset");
  }, [file, datasetId, modelId]));
  const load = useAction(useCallback((id: string) => api.batch(id), []));
  const del = useAction(useCallback((id: string) => api.deleteBatch(id), []));

  const reloadBatches = batches.reload;
  useEffect(() => {
    if (poll.done && selected) {
      api.batch(selected.id).then(setSelected).catch(() => undefined);
      reloadBatches();
    }
  }, [poll.done, selected, reloadBatches]);

  const onRun = async () => {
    const r = await run.run();
    if (r) { setSelected(r); setFile(null); batches.reload(); }
  };
  const onDownload = async (b: BatchResponse) => {
    try {
      const blob = await api.downloadBatch(b.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = `predictions-${b.id.slice(0, 8)}.csv`; a.click();
      URL.revokeObjectURL(url);
    } catch (e) { alert(errorMessage(e)); }
  };
  const noModel = models.data !== null && !models.data.production_model_id;

  return (
    <div className="space-y-6">
      <PageHeader title="Batch Analysis" description="Score many accounts at once. Upload a CSV or pick a registered dataset; the job runs in the background and results are stored and downloadable." />
      <ConfirmDialog open={!!confirmDelete} title="Delete this batch?" description="The input file, output file and its stored predictions will be removed." confirmLabel="Delete" destructive loading={del.loading} onCancel={() => setConfirmDelete(null)} onConfirm={async () => { if (confirmDelete) { await del.run(confirmDelete); setConfirmDelete(null); if (selected?.id === confirmDelete) setSelected(null); batches.reload(); } }} />

      <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
        <Card title="New batch" subtitle="CSV columns may be raw counts (followers_count, friends_count, statuses_count, …) or pre-computed features; a label column enables evaluation.">
          {!canRun ? <EmptyState title="Read-only access" description="Your role can view batch results but cannot start new batches." /> : noModel ? (
            <EmptyState icon={Layers} title="No production model configured" description="Train and activate a model before running batches." action={<Link to="/training"><Button>Go to Training</Button></Link>} />
          ) : (
            <div className="space-y-3">
              <Field label="CSV file">
                <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-border-strong px-4 py-6 text-sm text-ink-2 hover:bg-surface-2 focus-within:ring-2 focus-within:ring-accent">
                  <FileUp className="h-4 w-4" aria-hidden />{file ? `${file.name} (${(file.size / 1024).toFixed(0)} KB)` : "Choose a .csv file"}
                  <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => { setFile(e.target.files?.[0] ?? null); if (e.target.files?.[0]) setDatasetId(""); }} />
                </label>
              </Field>
              <div className="text-center text-xs text-ink-3">— or —</div>
              <Field label="Registered dataset">
                <Select value={datasetId} onChange={(e) => { setDatasetId(e.target.value); if (e.target.value) setFile(null); }}>
                  <option value="">Select a dataset…</option>
                  {datasets.data?.filter((d) => d.status === "VALIDATED").map((d) => <option key={d.id} value={d.id}>{d.name} · {d.n_rows.toLocaleString()} rows{d.has_label ? " · labelled" : ""}</option>)}
                </Select>
              </Field>
              <Field label="Model">
                <Select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                  <option value="">Production model</option>
                  {models.data?.models.filter((m) => m.status !== "TRAINING" && m.status !== "FAILED").map((m) => <option key={m.id} value={m.id}>{m.name} v{m.version} · {m.status.toLowerCase()}</option>)}
                </Select>
              </Field>
              <Button icon={Play} onClick={onRun} loading={run.loading} disabled={!file && !datasetId}>Start batch</Button>
              {run.error && <ErrorState title="Could not start the batch" message={run.error} />}
            </div>
          )}
        </Card>

        <Card title="Batches" subtitle="Background jobs for this organization" padded={false}>
          {batches.error && <div className="p-4"><ErrorState message={batches.error} onRetry={batches.reload} /></div>}
          {batches.data && batches.data.length === 0 && <div className="p-4"><EmptyState icon={Layers} title="No batches yet" description="Start a batch to score a CSV of accounts." /></div>}
          {batches.data && batches.data.length > 0 && (
            <Table className="rounded-none border-0">
              <thead><tr><Th>Name</Th><Th>Status</Th><Th>Model</Th><Th align="right">Rows</Th><Th align="right">Bots</Th><Th>Created</Th><Th></Th></tr></thead>
              <tbody>
                {batches.data.map((b) => (
                  <tr key={b.id} className="cursor-pointer hover:bg-surface-2" onClick={async () => { const r = await load.run(b.id); if (r) setSelected(r); }}>
                    <Td>{b.name}</Td><Td><StatusBadge status={b.status} /></Td><Td className="text-xs">{b.model ? `${b.model.name} v${b.model.version}` : "—"}</Td>
                    <Td align="right" mono>{int(b.processed_rows)}/{int(b.total_rows)}</Td><Td align="right" mono>{int(b.n_bots)}</Td><Td className="text-xs text-ink-2">{dateTime(b.created_at)}</Td>
                    <Td>{canRun && b.status !== "PROCESSING" && b.status !== "QUEUED" && <Button size="sm" variant="ghost" icon={Trash2} aria-label="Delete batch" onClick={(e) => { e.stopPropagation(); setConfirmDelete(b.id); }} />}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card>
      </div>

      {selected && (
        <div className="space-y-4 animate-fade-in">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2"><h2 className="text-lg font-semibold text-ink">{selected.name}</h2><StatusBadge status={selected.status} /></div>
            {selected.has_output && <Button icon={Download} variant="secondary" onClick={() => onDownload(selected)}>Download predictions.csv</Button>}
          </div>
          {(selected.status === "QUEUED" || selected.status === "PROCESSING") && (
            <Card>
              <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2"><div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${Math.round((poll.status?.progress ?? 0) * 100)}%` }} /></div>
              <p className="mt-2 text-xs text-ink-2" role="status">{poll.status?.message ?? "Queued…"}</p>
            </Card>
          )}
          {selected.status === "FAILED" && <ErrorState title="Batch failed" message={selected.error ?? "Unknown error"} />}
          {selected.status === "COMPLETED" && (
            <>
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
                <StatTile label="Processed" value={`${int(selected.processed_rows)} / ${int(selected.total_rows)}`} sub={selected.failed_rows ? `${selected.failed_rows} failed rows` : "0 failed rows"} />
                <StatTile label="Classified as bot" value={int(selected.n_bots)} accent="var(--bot)" />
                <StatTile label="Classified as human" value={int(selected.n_humans)} accent="var(--human)" />
                <StatTile label="Average bot probability" value={pct(selected.avg_bot_probability)} />
                <StatTile label="High-risk (score ≥ 60)" value={int(selected.high_risk)} accent="var(--status-serious)" />
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                <Card title="Distribution"><ProportionBar bots={selected.n_bots} humans={selected.n_humans} /><div className="mt-4 flex flex-wrap gap-2">{Object.entries(selected.summary.risk_band_distribution ?? {}).map(([band, n]) => <span key={band} className="flex items-center gap-1 text-xs text-ink-2"><RiskBadge band={band} /> <span className="font-mono text-ink">{n}</span></span>)}</div></Card>
                <Card title="Bot probability histogram">{selected.summary.probability_histogram?.length ? <ProbabilityHistogram bins={selected.summary.probability_histogram} /> : <EmptyState title="No data" />}</Card>
              </div>
              {selected.summary.evaluation && (
                <Card title="Evaluation against the provided labels" subtitle={`${int(selected.summary.evaluation.n_samples)} labelled rows`}>
                  <div className="grid gap-4 lg:grid-cols-3">
                    <div className="grid grid-cols-2 gap-2">{(["accuracy", "precision", "recall", "f1", "roc_auc"] as const).map((k) => <div key={k} className="rounded-lg border border-border p-3"><div className="text-[11px] uppercase text-ink-3">{k.replace("_", "-")}</div><div className="text-lg font-semibold tabular-nums text-ink">{num(selected.summary.evaluation?.metrics[k], 3)}</div></div>)}</div>
                    <ConfusionMatrixView cm={selected.summary.evaluation.confusion_matrix} />
                    <RocChart roc={selected.summary.evaluation.roc_curve} height={200} />
                  </div>
                </Card>
              )}
              <p className="text-[11px] text-ink-3">Every scored row is stored in <Link to={`/history?batch_id=${selected.id}`} className="underline">History</Link>; open a row to compute its SHAP/LIME explanation.</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
