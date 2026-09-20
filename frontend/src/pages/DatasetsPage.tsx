import { Database, Download, FileUp, RefreshCw, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ClassDistributionChart } from "@/components/charts/basic";
import { Badge, Button, Card, EmptyState, ErrorState, Field, Input, Notice, PageHeader, Select, StatTile, StatusBadge, Table, Tabs, Td, Textarea, Th } from "@/components/ui";
import { ConfirmDialog } from "@/components/ui/Dialog";
import { useAction, useApi } from "@/hooks/useApi";
import { useAuth } from "@/hooks/useAuth";
import { useJobPolling } from "@/hooks/usePolling";
import { api } from "@/services/api";
import type { DatasetPublic, DatasetSummary } from "@/types/api";
import { dateTime, int, num, pct } from "@/utils/format";

function SummaryView({ s }: { s: DatasetSummary }) {
  const [tab, setTab] = useState<"overview" | "columns" | "preview">("overview");
  return (
    <div className="space-y-3">
      <Tabs tabs={[{ key: "overview", label: "Overview" }, { key: "columns", label: `Columns (${s.n_columns})` }, { key: "preview", label: "Preview" }]} value={tab} onChange={setTab} />
      {tab === "overview" && (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <StatTile label="Rows" value={int(s.n_rows)} sub={`${int(s.duplicate_rows)} duplicate rows`} />
            <StatTile label="Missing cells" value={int(s.total_missing)} />
            <StatTile label="Feature coverage" value={pct(s.feature_availability.coverage, 0)} sub={`${s.feature_availability.direct.length} direct · ${s.feature_availability.derivable.length} derivable · ${s.feature_availability.missing.length} missing`} />
            <StatTile label="Label column" value={s.label_column ?? "none"} sub={s.label_column ? `${int(s.label_valid_rows)} valid labels` : "unlabelled — usable for batch scoring only"} />
          </div>
          {s.class_distribution && <ClassDistributionChart rows={[{ name: "dataset", human: s.class_distribution.HUMAN, bot: s.class_distribution.BOT }]} />}
          {s.warnings.length > 0 && <Notice tone="warning"><ul className="list-disc pl-4">{s.warnings.map((w) => <li key={w}>{w}</li>)}</ul></Notice>}
          {s.feature_availability.missing.length > 0 && <p className="text-xs text-ink-3">Features not computable from these columns (imputed with training medians at inference): {s.feature_availability.missing.join(", ")}</p>}
        </div>
      )}
      {tab === "columns" && (
        <Table compact>
          <thead><tr><Th>Column</Th><Th>Type</Th><Th>Maps to</Th><Th align="right">Missing</Th><Th align="right">Unique</Th><Th align="right">Min</Th><Th align="right">Median</Th><Th align="right">Max</Th></tr></thead>
          <tbody>{s.columns.map((c) => { const n = s.numeric_summary[c.name]; return <tr key={c.name}><Td mono>{c.name}</Td><Td className="text-xs">{c.dtype}</Td><Td className="text-xs">{c.canonical ? <Badge tone="good">{c.canonical}</Badge> : <span className="text-ink-3">—</span>}</Td><Td align="right" mono>{int(c.missing)}</Td><Td align="right" mono>{int(c.unique)}</Td><Td align="right" mono>{n ? num(n.min, 2) : ""}</Td><Td align="right" mono>{n ? num(n.median, 2) : ""}</Td><Td align="right" mono>{n ? num(n.max, 2) : ""}</Td></tr>; })}</tbody>
        </Table>
      )}
      {tab === "preview" && (s.preview.length ? (
        <div className="overflow-auto"><Table compact><thead><tr>{Object.keys(s.preview[0]).map((k) => <Th key={k}>{k}</Th>)}</tr></thead><tbody>{s.preview.map((row, i) => <tr key={i}>{Object.values(row).map((v, j) => <Td key={j} mono className="max-w-[16rem] truncate">{v === null ? "" : String(v)}</Td>)}</tr>)}</tbody></Table></div>
      ) : <EmptyState title="No preview stored" />)}
    </div>
  );
}

export function DatasetsPage() {
  const { hasRole } = useAuth();
  const canEdit = hasRole("ADMIN", "ANALYST");
  const isAdmin = hasRole("ADMIN");
  const list = useApi(() => api.datasets(), []);
  const models = useApi(() => api.models(), []);
  const benchmarks = useApi(() => api.benchmarks(), [], isAdmin);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const detail = useApi(() => api.dataset(selectedId as string), [selectedId], !!selectedId);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [versionOf, setVersionOf] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<DatasetPublic | null>(null);
  const [evalModel, setEvalModel] = useState("");
  const [importJob, setImportJob] = useState<string | null>(null);
  const importPoll = useJobPolling(importJob, 1500);

  const upload = useAction(useCallback(() => {
    if (!file) throw new Error("Choose a CSV file");
    return api.uploadDataset(file, { name: name || undefined, description: description || undefined, dataset_id: versionOf || undefined });
  }, [file, name, description, versionOf]));
  const del = useAction(useCallback((id: string) => api.deleteDataset(id), []));
  const evaluate = useAction(useCallback((id: string, model_id: string) => api.evaluateDataset(id, model_id || null), []));
  const importBench = useAction(useCallback((kind: string) => api.importBenchmark(kind), []));
  const combine = useAction(useCallback(() => api.combinedBenchmark(), []));

  const reload = list.reload;
  useEffect(() => { if (importPoll.done) { reload(); setImportJob(null); } }, [importPoll.done, reload]);

  const onUpload = async () => {
    const ds = await upload.run();
    if (ds) { setFile(null); setName(""); setDescription(""); setVersionOf(""); list.reload(); setSelectedId(ds.id); }
  };

  return (
    <div className="space-y-6">
      <PageHeader title="Datasets" description="Upload labelled account CSVs to train models, or unlabelled CSVs to score in batch. Every upload is versioned, checksummed and validated before it can be used." />
      <ConfirmDialog open={!!confirmDelete} title={`Delete “${confirmDelete?.name}”?`} description={confirmDelete?.models_trained ? `${confirmDelete.models_trained} model(s) were trained on this dataset; they keep working but lose the dataset link.` : "All versions and stored files will be removed."} confirmLabel="Delete" destructive loading={del.loading} onCancel={() => setConfirmDelete(null)} onConfirm={async () => { if (confirmDelete) { await del.run(confirmDelete.id); if (selectedId === confirmDelete.id) setSelectedId(null); setConfirmDelete(null); list.reload(); } }} />

      <div className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
        <div className="space-y-4">
          {canEdit && (
            <Card title="Upload a dataset" subtitle="CSV with per-account columns (followers_count, friends_count, statuses_count, favourites_count, listed_count, verified, default_profile, description, created_at, …) and optionally a label column (bot/human, 0/1).">
              <div className="space-y-3">
                <Field label="CSV file">
                  <label className="flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-border-strong px-4 py-6 text-sm text-ink-2 hover:bg-surface-2 focus-within:ring-2 focus-within:ring-accent">
                    <FileUp className="h-4 w-4" aria-hidden />{file ? `${file.name} (${(file.size / 1024).toFixed(0)} KB)` : "Choose a .csv file"}
                    <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
                  </label>
                </Field>
                <Field label="New version of"><Select value={versionOf} onChange={(e) => setVersionOf(e.target.value)}><option value="">— create a new dataset —</option>{list.data?.map((d) => <option key={d.id} value={d.id}>{d.name} (v{d.current_version?.version ?? d.n_versions})</option>)}</Select></Field>
                {!versionOf && <Field label="Name" hint="Defaults to the file name"><Input value={name} onChange={(e) => setName(e.target.value)} maxLength={120} /></Field>}
                {!versionOf && <Field label="Description"><Textarea rows={2} value={description} onChange={(e) => setDescription(e.target.value)} maxLength={2000} /></Field>}
                <Button icon={FileUp} onClick={onUpload} loading={upload.loading} disabled={!file}>Upload &amp; validate</Button>
                {upload.error && <ErrorState title="Upload rejected" message={upload.error} />}
              </div>
            </Card>
          )}

          {isAdmin && (
            <Card title="Public benchmark import" subtitle="Cresci-2015 / Cresci-2017 user-level CSVs (Cresci et al.). Files are fetched from their public mirror or read from the server's data folder; imported as regular versioned datasets.">
              {benchmarks.error && <ErrorState message={benchmarks.error} onRetry={benchmarks.reload} />}
              {benchmarks.data && (
                <ul className="space-y-2">
                  {benchmarks.data.map((b) => (
                    <li key={b.kind} className="flex items-center justify-between gap-2 rounded-lg border border-border p-3 text-sm">
                      <div>
                        <div className="font-medium text-ink">{b.kind}</div>
                        <div className="text-xs text-ink-3">{b.available ? `${b.subsets_found.length} subset(s) present locally` : "not present locally — will be downloaded"} · {b.paper_reported.citation}</div>
                      </div>
                      <Button size="sm" variant="secondary" icon={Download} loading={importBench.loading || !!importJob} onClick={async () => { const j = await importBench.run(b.kind); if (j) setImportJob(j.id); }}>Import</Button>
                    </li>
                  ))}
                  <li className="flex items-center justify-between gap-2 rounded-lg border border-border p-3 text-sm">
                    <div><div className="font-medium text-ink">Combined Cresci-15 + Cresci-17</div><div className="text-xs text-ink-3">Requires both imports first; deduplicated by account id.</div></div>
                    <Button size="sm" variant="secondary" loading={combine.loading} onClick={async () => { const d = await combine.run(); if (d) { list.reload(); setSelectedId(d.id); } }}>Build</Button>
                  </li>
                </ul>
              )}
              {importJob && <p className="mt-2 text-xs text-ink-2" role="status"><RefreshCw className="mr-1 inline h-3 w-3 animate-spin" aria-hidden />{importPoll.status?.message ?? "Import queued…"} {importPoll.status ? `(${Math.round(importPoll.status.progress * 100)}%)` : ""}</p>}
              {importPoll.status?.status === "FAILED" && <ErrorState title="Import failed" message={importPoll.status.error ?? "Unknown error"} />}
              {importBench.error && <ErrorState message={importBench.error} />}
              {combine.error && <ErrorState message={combine.error} />}
            </Card>
          )}

          <Card title="Registered datasets" padded={false}>
            {list.error && <div className="p-4"><ErrorState message={list.error} onRetry={list.reload} /></div>}
            {list.data && list.data.length === 0 && <div className="p-4"><EmptyState icon={Database} title="No datasets yet" description={canEdit ? "Upload a CSV to get started." : "An analyst or admin needs to upload a dataset."} /></div>}
            {list.data && list.data.length > 0 && (
              <Table className="rounded-none border-0" compact>
                <thead><tr><Th>Name</Th><Th>Status</Th><Th align="right">Rows</Th><Th>Label</Th><Th align="right">Models</Th><Th></Th></tr></thead>
                <tbody>{list.data.map((d) => (
                  <tr key={d.id} className={`cursor-pointer hover:bg-surface-2 ${selectedId === d.id ? "bg-accent-soft" : ""}`} onClick={() => setSelectedId(d.id)}>
                    <Td><div className="font-medium text-ink">{d.name}</div><div className="text-[11px] text-ink-3">{d.kind} · v{d.current_version?.version ?? d.n_versions} · {dateTime(d.updated_at)}</div></Td>
                    <Td><StatusBadge status={d.status} /></Td><Td align="right" mono>{int(d.n_rows)}</Td><Td>{d.has_label ? <Badge tone="good">labelled</Badge> : <Badge tone="neutral">unlabelled</Badge>}</Td><Td align="right" mono>{d.models_trained}</Td>
                    <Td>{canEdit && <Button size="sm" variant="ghost" icon={Trash2} aria-label={`Delete ${d.name}`} onClick={(e) => { e.stopPropagation(); setConfirmDelete(d); }} />}</Td>
                  </tr>
                ))}</tbody>
              </Table>
            )}
          </Card>
        </div>

        <div className="space-y-4">
          {!selectedId && <Card><EmptyState icon={Database} title="Select a dataset" description="Column mapping, class balance, validation warnings and versions appear here." /></Card>}
          {selectedId && detail.error && <ErrorState message={detail.error} onRetry={detail.reload} />}
          {selectedId && detail.data && (
            <>
              <Card title={detail.data.name} subtitle={detail.data.description || undefined} actions={<StatusBadge status={detail.data.status} />}>
                {detail.data.current_version?.validation_errors?.length ? <Notice tone="warning"><ul className="list-disc pl-4">{detail.data.current_version.validation_errors.map((e) => <li key={e}>{e}</li>)}</ul></Notice> : null}
                {detail.data.summary ? <SummaryView s={detail.data.summary} /> : <EmptyState title="No profile available for this version" />}
              </Card>
              <Card title="Versions" padded={false}>
                <Table className="rounded-none border-0" compact>
                  <thead><tr><Th>v</Th><Th>File</Th><Th align="right">Rows</Th><Th>Status</Th><Th>SHA-256</Th><Th>Uploaded</Th></tr></thead>
                  <tbody>{(detail.data.versions ?? []).map((v) => <tr key={v.id}><Td mono>{v.version}</Td><Td className="text-xs">{v.original_filename} · {(v.size_bytes / 1024).toFixed(0)} KB</Td><Td align="right" mono>{int(v.n_rows)}</Td><Td><StatusBadge status={v.status} /></Td><Td mono className="text-[11px]">{v.checksum_sha256.slice(0, 12)}…</Td><Td className="text-xs">{dateTime(v.created_at)}</Td></tr>)}</tbody>
                </Table>
              </Card>
              {detail.data.has_label && detail.data.status === "VALIDATED" && (
                <Card title="Evaluate a model on this dataset" subtitle="Runs the full pipeline on every labelled row and stores the result as an evaluation run.">
                  {canEdit ? (
                    <div className="flex flex-wrap items-end gap-2">
                      <Field label="Model" className="min-w-[16rem]"><Select value={evalModel} onChange={(e) => setEvalModel(e.target.value)}><option value="">Production model</option>{models.data?.models.filter((m) => m.status === "READY" || m.status === "PRODUCTION").map((m) => <option key={m.id} value={m.id}>{m.name} v{m.version}</option>)}</Select></Field>
                      <Button loading={evaluate.loading} onClick={() => evaluate.run(selectedId, evalModel)}>Evaluate</Button>
                    </div>
                  ) : <p className="text-sm text-ink-2">Analysts and admins can run evaluations.</p>}
                  {evaluate.error && <ErrorState message={evaluate.error} />}
                  {evaluate.result && (
                    <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
                      {(["accuracy", "precision", "recall", "f1", "roc_auc"] as const).map((k) => <div key={k} className="rounded-lg border border-border p-3"><div className="text-[11px] uppercase text-ink-3">{k.replace("_", "-")}</div><div className="text-lg font-semibold tabular-nums text-ink">{num(evaluate.result?.evaluation.metrics[k], 3)}</div></div>)}
                      <p className="col-span-full text-xs text-ink-3">{int(evaluate.result.n_samples)} rows · {evaluate.result.model.name} v{evaluate.result.model.version} · see <Link className="underline" to="/evaluation">Evaluation</Link> for the confusion matrix and curves.</p>
                    </div>
                  )}
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
