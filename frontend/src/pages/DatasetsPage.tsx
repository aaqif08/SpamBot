import { Database, FileUp, FlaskConical, Import, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { ClassDistributionChart } from "@/components/charts/basic";
import { Badge, Button, Card, DemoBanner, EmptyState, ErrorState, Notice, PageHeader, Skeleton, SourceTag, Spinner, Table, Td, Th } from "@/components/ui";
import { useAction, useApi } from "@/hooks/useApi";
import { useHealth } from "@/hooks/useHealth";
import { useJobPolling } from "@/hooks/usePolling";
import { api } from "@/services/api";
import type { DatasetDetail, DatasetInfo } from "@/types/api";
import { dateTime, featureLabel, int, num, pct } from "@/utils/format";

function DatasetInspector({ id, onDeleted }: { id: string; onDeleted: () => void }) {
  const detail = useApi(() => api.dataset(id), [id]);
  const evaluate = useAction(useCallback(() => api.evaluateDataset(id), [id]));
  const del = useAction(useCallback(() => api.deleteDataset(id), [id]));
  if (detail.loading && !detail.data) return <Spinner label="Inspecting dataset…" />;
  if (detail.error || !detail.data) return <ErrorState message={detail.error ?? "Not found"} onRetry={detail.reload} />;
  const d: DatasetDetail = detail.data;
  const s = d.summary;
  const fa = s.feature_availability;
  return (
    <div className="space-y-4 animate-fade-in">
      {d.is_demo && <DemoBanner compact />}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold text-ink">{d.name}</h3>
          <p className="text-xs text-ink-3">kind: {d.kind} · uploaded {dateTime(d.created_at)}{d.original_filename ? ` · ${d.original_filename}` : ""}</p>
        </div>
        <div className="flex gap-2">
          {d.has_label && (
            <Button size="sm" variant="secondary" icon={FlaskConical} loading={evaluate.loading} onClick={() => evaluate.run()}>
              Evaluate active model on this dataset
            </Button>
          )}
          {(d.kind === "upload" || d.kind === "demo") && (
            <Button size="sm" variant="danger" icon={Trash2} loading={del.loading} onClick={async () => { if (window.confirm("Delete this dataset?")) { await del.run(); onDeleted(); } }}>
              Delete
            </Button>
          )}
        </div>
      </div>
      {s.warnings.map((w) => (
        <Notice key={w} tone="warning">{w}</Notice>
      ))}
      <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {[
          ["Records", int(s.n_rows)],
          ["Columns", int(s.n_columns)],
          ["Missing values", int(s.total_missing)],
          ["Duplicate rows", int(s.duplicate_rows)],
          ["Label column", s.label_column ?? "none (inference only)"],
          ["Feature coverage", pct(fa.coverage, 0)],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg border border-border p-3">
            <div className="text-[11px] text-ink-3">{k}</div>
            <div className="truncate text-sm font-semibold text-ink">{v}</div>
          </div>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Class distribution" subtitle={s.class_distribution ? "From the detected label column" : "No labels — inference only"}>
          {s.class_distribution ? <ClassDistributionChart rows={[{ name: "All rows", human: s.class_distribution.HUMAN, bot: s.class_distribution.BOT }]} height={110} /> : <EmptyState title="Unlabelled dataset" description="Batch prediction works; evaluation and training require a label column (label / is_bot / class / account_type)." />}
        </Card>
        <Card title="Feature availability" subtitle="Which of the paper's 31 features this CSV provides directly, can derive, or lacks">
          <div className="space-y-2 text-xs">
            <div><Badge tone="good">direct {fa.direct.length}</Badge> <span className="text-ink-2">{fa.direct.map(featureLabel).join(", ") || "—"}</span></div>
            <div><Badge tone="accent">derivable {fa.derivable.length}</Badge> <span className="text-ink-2">{fa.derivable.map(featureLabel).join(", ") || "—"}</span></div>
            <div><Badge tone="critical">missing {fa.missing.length}</Badge> <span className="text-ink-2">{fa.missing.map(featureLabel).join(", ") || "—"}</span></div>
            <p className="text-[11px] text-ink-3">Missing features are imputed with the training median by the model pipeline (preprocessing status: median imputation → min-max scaling).</p>
          </div>
        </Card>
      </div>
      {evaluate.error && <ErrorState title="Evaluation failed" message={evaluate.error} />}
      {evaluate.result && (
        <Card title={`Evaluation of ${evaluate.result.model.name} on this dataset`} subtitle={`${evaluate.result.n_rows.toLocaleString()} labelled rows`} actions={<SourceTag kind="ours" />}>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
            {(["accuracy", "precision", "recall", "f1", "roc_auc"] as const).map((k) => (
              <div key={k} className="rounded-lg border border-border p-3">
                <div className="text-[11px] uppercase text-ink-3">{k.replace("_", "-")}</div>
                <div className="text-lg font-semibold tabular-nums text-ink">{num(evaluate.result?.evaluation.metrics[k], 3)}</div>
              </div>
            ))}
          </div>
          {evaluate.result.is_demo && <p className="mt-2 text-[11px] text-ink-3">Demo data/model — not a research result.</p>}
        </Card>
      )}
      <Card title="Columns" padded={false}>
        <Table className="rounded-none border-0">
          <thead>
            <tr><Th>Column</Th><Th>Canonical name</Th><Th>dtype</Th><Th align="right">Missing</Th><Th align="right">Unique</Th></tr>
          </thead>
          <tbody>
            {s.columns.map((c) => (
              <tr key={c.name}><Td className="font-mono text-xs">{c.name}</Td><Td className="font-mono text-xs text-ink-2">{c.canonical}</Td><Td className="text-xs">{c.dtype}</Td><Td align="right" mono>{c.missing}</Td><Td align="right" mono>{c.unique}</Td></tr>
            ))}
          </tbody>
        </Table>
      </Card>
      <Card title="Preview (first 10 records)" padded={false}>
        <div className="scrollbar-thin overflow-x-auto">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr>{Object.keys(s.preview[0] ?? {}).map((k) => <Th key={k}>{k}</Th>)}</tr>
            </thead>
            <tbody>
              {s.preview.map((row, i) => (
                <tr key={i}>{Object.values(row).map((v, j) => <Td key={j} className="max-w-[220px] truncate font-mono text-[11px]">{v === null ? "" : String(v)}</Td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function CresciPanel({ onImported }: { onImported: () => void }) {
  const status = useApi(() => api.cresciStatus(), []);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobKind, setJobKind] = useState<string>("");
  const poll = useJobPolling(jobId, "dataset");
  const start = useAction(useCallback((kind: string) => api.importCresci(kind), []));
  const reloadStatus = status.reload;
  useEffect(() => {
    if (poll.done && poll.status?.status === "completed" && jobId) {
      setJobId(null);
      reloadStatus();
      onImported();
    }
  }, [poll.done, poll.status, jobId, reloadStatus, onImported]);
  if (status.loading && !status.data) return <Skeleton className="h-40" />;
  if (status.error || !status.data) return <ErrorState message={status.error ?? ""} onRetry={status.reload} />;
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {status.data.map((c) => (
        <Card key={c.kind} title={c.kind.toUpperCase()} subtitle={c.paper_reported.citation} actions={c.imported ? <Badge tone="good">imported</Badge> : c.available ? <Badge tone="accent">files found</Badge> : <Badge tone="neutral">not installed</Badge>}>
          <div className="space-y-3 text-xs">
            <div>
              <div className="mb-1 flex items-center gap-2 font-semibold text-ink"><SourceTag kind="paper" /></div>
              <Table compact>
                <thead><tr><Th>Subset</Th><Th>Type</Th><Th align="right">Accounts</Th><Th align="right">Tweets</Th></tr></thead>
                <tbody>
                  {c.paper_reported.subsets.map((s) => (
                    <tr key={s.name}><Td className="text-xs">{s.name}</Td><Td className="text-xs">{s.type}</Td><Td align="right" mono>{int(s.accounts)}</Td><Td align="right" mono>{int(s.tweets)}</Td></tr>
                  ))}
                </tbody>
              </Table>
              <p className="mt-1 text-[11px] text-ink-3">Statistics as reported in the base paper (Tables 2–3). Measured statistics appear only after import.</p>
            </div>
            <div>
              <div className="font-semibold text-ink">Local files</div>
              {c.subsets_found.length ? (
                <ul className="mt-1 list-disc pl-4 text-ink-2">
                  {c.subsets_found.map((s) => <li key={s.folder}>{s.folder} → {s.category} (label {s.label}){s.has_tweets ? "" : " — tweets.csv missing"}</li>)}
                </ul>
              ) : (
                <p className="mt-1 text-ink-2">Place the dataset folders at <code className="font-mono">{c.install_path_hint}</code>. Expected subset folders: {c.expected_subsets.join(", ")}. The datasets are obtained from the original authors (Cresci et al.) — see README.</p>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" icon={Import} disabled={!c.available} loading={start.loading && jobKind === c.kind} onClick={async () => { setJobKind(c.kind); const r = await start.run(c.kind); if (r) setJobId(r.job_id); }}>
                {c.imported ? "Re-import" : "Import & featurise"}
              </Button>
              {jobId && jobKind === c.kind && poll.status && <span className="text-ink-2">{poll.status.status}: {poll.status.message}</span>}
              {jobId && jobKind === c.kind && poll.status?.status === "failed" && <span className="text-status-critical">{poll.status.error}</span>}
            </div>
            {start.error && jobKind === c.kind && <p className="text-status-critical">{start.error}</p>}
          </div>
        </Card>
      ))}
    </div>
  );
}

export function DatasetsPage() {
  const health = useHealth();
  const demoEnabled = health.data?.demo_mode_enabled ?? false;
  const list = useApi(() => api.datasets(), []);
  const [selected, setSelected] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const upload = useAction(useCallback((f: File) => api.uploadDataset(f), []));
  const demo = useAction(useCallback(() => api.createDemoDataset(600, 7), []));

  const onUpload = async () => {
    if (!file) return;
    const r = await upload.run(file);
    if (r) {
      setFile(null);
      list.reload();
      setSelected(r.id);
    }
  };
  const onDemo = async () => {
    const r = await demo.run();
    if (r) {
      list.reload();
      setSelected(r.id);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Datasets" description="Upload CSV datasets, inspect columns and feature availability, and import the Cresci-15 / Cresci-17 benchmark files (python scripts/fetch_datasets.py downloads the public user-level mirror)." />

      <div className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
        <div className="space-y-4">
          <Card title="Upload CSV" subtitle="Labelled (label / is_bot / class) or unlabelled. Max size set by BOTSHIELD_MAX_UPLOAD_MB.">
            <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-border-strong px-4 py-6 text-sm text-ink-2 hover:bg-surface-2">
              <FileUp className="h-4 w-4" aria-hidden />
              {file ? file.name : "Choose a .csv file"}
              <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
            <div className="mt-3 flex gap-2">
              <Button icon={FileUp} onClick={onUpload} loading={upload.loading} disabled={!file}>Upload & inspect</Button>
              {demoEnabled && <Button variant="secondary" icon={FlaskConical} onClick={onDemo} loading={demo.loading}>Generate demo dataset</Button>}
            </div>
            {upload.error && <div className="mt-3"><ErrorState title="Upload failed" message={upload.error} /></div>}
            {demo.error && <div className="mt-3"><ErrorState title="Demo generation failed" message={demo.error} /></div>}
            <p className="mt-3 text-[11px] text-ink-3">
              {demoEnabled
                ? "Demo dataset = 600 synthetic accounts (DEMO DATA — NOT REAL SOCIAL MEDIA DATA) with the same 31-feature schema."
                : "Production mode: the synthetic demo generator is disabled (BOTSHIELD_DEMO_MODE_ENABLED=false). Use the real datasets below."}
            </p>
          </Card>
          <Card title="Registered datasets" padded={false}>
            {list.error && <div className="p-4"><ErrorState message={list.error} onRetry={list.reload} /></div>}
            {list.data && list.data.length === 0 && <div className="p-4"><EmptyState icon={Database} title="No datasets yet" description="Upload a CSV, generate the demo dataset or import Cresci files." /></div>}
            <ul className="divide-y divide-border">
              {list.data?.map((d: DatasetInfo) => (
                <li key={d.id}>
                  <button onClick={() => setSelected(d.id)} className={`flex w-full flex-col gap-1 px-4 py-3 text-left hover:bg-surface-2 ${selected === d.id ? "bg-accent-soft/60" : ""}`}>
                    <span className="flex flex-wrap items-center gap-2 text-sm font-medium text-ink">
                      {d.name}
                      {d.is_demo && <Badge tone="demo">demo</Badge>}
                      {d.has_label ? <Badge tone="good">labelled</Badge> : <Badge tone="neutral">unlabelled</Badge>}
                    </span>
                    <span className="text-[11px] text-ink-3">{d.kind} · {d.n_rows.toLocaleString()} rows × {d.n_columns} cols · coverage {pct(d.feature_coverage, 0)} · {dateTime(d.created_at)}</span>
                  </button>
                </li>
              ))}
            </ul>
          </Card>
        </div>
        <Card title="Inspector" subtitle="Schema validation, missing values, duplicates, class distribution, feature availability">
          {selected ? <DatasetInspector id={selected} onDeleted={() => { setSelected(null); list.reload(); }} /> : <EmptyState icon={Database} title="Select a dataset" description="Choose a dataset on the left to inspect it." />}
        </Card>
      </div>

      <div>
        <h2 className="mb-3 text-base font-semibold text-ink">Cresci benchmark datasets</h2>
        <CresciPanel onImported={list.reload} />
      </div>
    </div>
  );
}
