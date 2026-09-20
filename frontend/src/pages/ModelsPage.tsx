import { Boxes, CheckCircle2, Trash2 } from "lucide-react";
import { useCallback, useState } from "react";
import { Link } from "react-router-dom";

import { MetricComparisonChart } from "@/components/charts/basic";
import { COLORS } from "@/components/charts/common";
import { Badge, Button, Card, EmptyState, ErrorState, Notice, PageHeader, Select, Skeleton, SourceTag, Table, Td, Th } from "@/components/ui";
import { useAction, useApi } from "@/hooks/useApi";
import { api } from "@/services/api";
import type { MetricSet } from "@/types/api";
import { algorithmLabel, dateTime, num, seconds } from "@/utils/format";

type MetricKey = keyof MetricSet;
const METRICS: { key: MetricKey; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "roc_auc", label: "ROC-AUC" },
];

export function ModelsPage() {
  const { data, loading, error, reload } = useApi(() => api.models(), []);
  const [metric, setMetric] = useState<MetricKey>("f1");
  const [paperDataset, setPaperDataset] = useState<"cresci-15" | "cresci-17">("cresci-15");
  const activate = useAction(useCallback((id: string) => api.activateModel(id), []));
  const remove = useAction(useCallback((id: string) => api.deleteModel(id), []));

  if (loading && !data) return <div className="space-y-4"><PageHeader title="Models" /><Skeleton className="h-64" /></div>;
  if (error || !data) return <div className="space-y-4"><PageHeader title="Models" /><ErrorState message={error ?? ""} onRetry={reload} /></div>;

  const ours = data.models;
  const best = ours.length ? ours.reduce((a, b) => ((b.metrics.holdout[metric] ?? -1) > (a.metrics.holdout[metric] ?? -1) ? b : a)) : null;
  const paperRows = data.paper_reported.datasets[paperDataset].results;
  const chartData = data.supported_algorithms.map((a) => {
    const paper = paperRows.find((r) => r.algorithm === a.key);
    const mine = ours
      .filter((m) => m.algorithm === a.key && !m.is_demo && m.dataset_name.toLowerCase().startsWith(paperDataset))
      .sort((x, y) => (y.metrics.holdout[metric] ?? 0) - (x.metrics.holdout[metric] ?? 0))[0];
    const demo = ours.filter((m) => m.algorithm === a.key && m.is_demo).sort((x, y) => (y.metrics.holdout[metric] ?? 0) - (x.metrics.holdout[metric] ?? 0))[0];
    return { name: a.display_name, paper: paper?.[metric] ?? null, ours: mine?.metrics.holdout[metric] ?? null, demo: demo?.metrics.holdout[metric] ?? null };
  });

  return (
    <div className="space-y-6">
      <PageHeader title="Models" description="All nine classifiers from the base paper are supported. Measured results (this implementation) and paper-reported results are shown side by side but never merged." />

      <Card title="Supported classifiers" subtitle="Availability in this environment">
        <div className="flex flex-wrap gap-2">
          {data.supported_algorithms.map((a) => (
            <Badge key={a.key} tone={a.available ? "good" : "critical"} dot={a.available ? "var(--status-good)" : "var(--status-critical)"}>
              {a.display_name} · {a.family}{a.supports_tree_shap ? " · TreeSHAP" : " · KernelSHAP"}
            </Badge>
          ))}
        </div>
      </Card>

      <Card
        title="Trained models — our experimental results"
        subtitle="Hold-out metrics on each model's own stratified test split; CV means in the Evaluation page"
        actions={<SourceTag kind="ours" />}
      >
        {ours.length === 0 ? (
          <EmptyState icon={Boxes} title="No trained models" description="Run the training pipeline to populate this table." action={<Link to="/training"><Button>Open training</Button></Link>} />
        ) : (
          <>
            {activate.error && <ErrorState message={activate.error} />}
            {remove.error && <ErrorState message={remove.error} />}
            <Table>
              <thead>
                <tr>
                  <Th>Model</Th>
                  <Th>Dataset</Th>
                  {METRICS.map((m) => <Th key={m.key} align="right">{m.label}</Th>)}
                  <Th align="right">Training time</Th>
                  <Th>Trained</Th>
                  <Th></Th>
                </tr>
              </thead>
              <tbody>
                {ours.map((m) => (
                  <tr key={m.id} className={m.is_active ? "bg-accent-soft/40" : ""}>
                    <Td>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium">{m.name}</span>
                        {m.is_active && <Badge tone="accent">active</Badge>}
                        {m.is_demo && <Badge tone="demo">demo</Badge>}
                        {best && best.id === m.id && ours.length > 1 && <Badge tone="neutral">highest {METRICS.find((x) => x.key === metric)?.label} in this experiment</Badge>}
                      </div>
                      <div className="text-[11px] text-ink-3">{m.id}</div>
                    </Td>
                    <Td className="text-xs">{m.dataset_name}</Td>
                    {METRICS.map((k) => <Td key={k.key} align="right" mono>{num(m.metrics.holdout[k.key], 3)}</Td>)}
                    <Td align="right" mono>{seconds(m.training_seconds)}</Td>
                    <Td className="text-xs text-ink-2">{dateTime(m.trained_at)}</Td>
                    <Td>
                      <div className="flex gap-1">
                        {!m.is_active && <Button size="sm" variant="secondary" icon={CheckCircle2} loading={activate.loading} onClick={async () => { await activate.run(m.id); reload(); }}>Activate</Button>}
                        <Button size="sm" variant="ghost" icon={Trash2} aria-label="Delete model" onClick={async () => { if (window.confirm(`Delete model ${m.id}?`)) { await remove.run(m.id); reload(); } }} />
                      </div>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </>
        )}
      </Card>

      <Card
        title="Comparison chart"
        subtitle="Per algorithm: paper-reported value (5-fold CV, 31 features incl. tweet-derived) vs. our best hold-out value on the same Cresci dataset (user-level mirror, tweet-derived features unavailable)"
        actions={
          <div className="flex gap-2">
            <Select value={metric} onChange={(e) => setMetric(e.target.value as MetricKey)} className="h-8 w-36 text-xs">
              {METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
            </Select>
            <Select value={paperDataset} onChange={(e) => setPaperDataset(e.target.value as "cresci-15" | "cresci-17")} className="h-8 w-36 text-xs">
              <option value="cresci-15">Paper: Cresci-15</option>
              <option value="cresci-17">Paper: Cresci-17</option>
            </Select>
          </div>
        }
      >
        <MetricComparisonChart
          data={chartData}
          series={[
            { key: "paper", label: `Reported in base paper (${paperDataset})`, color: COLORS.series[0] },
            { key: "ours", label: `Reproduced by this implementation (${paperDataset}, user-level data)`, color: COLORS.series[1] },
            { key: "demo", label: "This implementation (DEMO data)", color: COLORS.series[2] },
          ]}
          height={300}
        />
        <Notice>
          Paper values come from Tables 5–6 of the base paper (5-fold CV). Values measured here come from a hold-out split on whichever dataset the model was trained on and are directly comparable only when that dataset is the same Cresci dataset.
        </Notice>
      </Card>

      <Card title={`Results reported in base paper — ${paperDataset.toUpperCase()} (${data.paper_reported.datasets[paperDataset].table})`} subtitle={data.paper_reported.datasets[paperDataset].highlight} actions={<SourceTag kind="paper" />}>
        <Table>
          <thead>
            <tr><Th>Classifier</Th>{METRICS.map((m) => <Th key={m.key} align="right">{m.label}</Th>)}</tr>
          </thead>
          <tbody>
            {paperRows.map((r) => (
              <tr key={r.algorithm}>
                <Td>{algorithmLabel(r.algorithm)}</Td>
                {METRICS.map((m) => <Td key={m.key} align="right" mono>{num(r[m.key], 3)}</Td>)}
              </tr>
            ))}
          </tbody>
        </Table>
        <p className="mt-2 text-[11px] text-ink-3">Source: {data.paper_reported.citation.title}, {data.paper_reported.citation.venue}, DOI {data.paper_reported.citation.doi}. Not measured by this implementation.</p>
      </Card>
    </div>
  );
}
