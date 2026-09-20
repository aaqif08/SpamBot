import { ArrowLeft, History as HistoryIcon, Search, Trash2 } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { PredictionResult } from "@/components/PredictionResult";
import { Badge, Button, Card, EmptyState, ErrorState, Field, Input, PageHeader, PredictionBadge, RiskBadge, Select, Skeleton, Spinner, Table, Td, Th, Toggle } from "@/components/ui";
import { useAction, useApi } from "@/hooks/useApi";
import { api } from "@/services/api";
import type { HistoryFilters, LimeLocal, PredictResponse, ShapLocal } from "@/types/api";
import { dateTime, num } from "@/utils/format";

export function HistoryPage() {
  const navigate = useNavigate();
  const models = useApi(() => api.models(), []);
  const [filters, setFilters] = useState<HistoryFilters>({ prediction: "", high_risk: false, model: "", source: "", date_from: "", date_to: "", search: "", page: 1, page_size: 25 });
  const query = useMemo(() => ({ ...filters, prediction: filters.prediction || undefined }), [filters]);
  const { data, loading, error, reload } = useApi(() => api.history(query), [query]);
  const remove = useAction(useCallback((id: string) => api.deleteHistory(id), []));
  const set = <K extends keyof HistoryFilters>(k: K, v: HistoryFilters[K]) => setFilters((f) => ({ ...f, [k]: v, page: k === "page" ? (v as number) : 1 }));
  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;
  const modelNames = Array.from(new Set(models.data?.models.map((m) => m.name) ?? []));

  return (
    <div className="space-y-6">
      <PageHeader title="Prediction History" description="Every prediction is stored in SQLite with its features and explanations. Click a row to open the full analysis." />
      <Card title="Filters" padded>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <Field label="Search account"><div className="relative"><Search className="pointer-events-none absolute left-2.5 top-3 h-4 w-4 text-ink-3" /><Input className="pl-8" value={filters.search ?? ""} onChange={(e) => set("search", e.target.value)} placeholder="identifier…" /></div></Field>
          <Field label="Prediction"><Select value={filters.prediction ?? ""} onChange={(e) => set("prediction", e.target.value as HistoryFilters["prediction"])}><option value="">All</option><option value="BOT">BOT</option><option value="HUMAN">HUMAN</option></Select></Field>
          <Field label="Model"><Select value={filters.model ?? ""} onChange={(e) => set("model", e.target.value)}><option value="">All</option>{modelNames.map((n) => <option key={n} value={n}>{n}</option>)}</Select></Field>
          <Field label="Source"><Select value={filters.source ?? ""} onChange={(e) => set("source", e.target.value)}><option value="">All</option><option value="manual">manual</option><option value="sample">sample</option><option value="batch">batch</option><option value="adapter">adapter</option></Select></Field>
          <Field label="From"><Input type="date" value={filters.date_from ?? ""} onChange={(e) => set("date_from", e.target.value)} /></Field>
          <Field label="To"><Input type="date" value={filters.date_to ?? ""} onChange={(e) => set("date_to", e.target.value ? `${e.target.value}T23:59:59` : "")} /></Field>
        </div>
        <div className="mt-3"><Toggle label="High risk only (risk score ≥ 60)" checked={!!filters.high_risk} onChange={(v) => set("high_risk", v)} /></div>
      </Card>

      <Card title={`Predictions${data ? ` (${data.total.toLocaleString()})` : ""}`} padded={false} actions={data && <span className="text-xs text-ink-2">page {data.page} / {pages}</span>}>
        {loading && !data && <div className="p-4"><Skeleton className="h-40" /></div>}
        {error && <div className="p-4"><ErrorState message={error} onRetry={reload} /></div>}
        {data && data.items.length === 0 && <div className="p-4"><EmptyState icon={HistoryIcon} title="No predictions match" description="Analyze an account or run a batch to populate the history." action={<Link to="/analyze"><Button variant="secondary">Analyze account</Button></Link>} /></div>}
        {data && data.items.length > 0 && (
          <Table className="rounded-none border-0">
            <thead>
              <tr><Th>Date</Th><Th>Account ID</Th><Th>Prediction</Th><Th align="right">Bot probability</Th><Th>Risk score</Th><Th>Model</Th><Th>Source</Th><Th></Th></tr>
            </thead>
            <tbody>
              {data.items.map((h) => (
                <tr key={h.id} className="cursor-pointer hover:bg-surface-2" onClick={() => navigate(`/history/${h.id}`)}>
                  <Td className="text-xs text-ink-2">{dateTime(h.created_at)}</Td>
                  <Td className="font-mono text-xs">{h.account_identifier}{h.is_demo && <Badge tone="demo" className="ml-2">demo</Badge>}</Td>
                  <Td><PredictionBadge label={h.prediction} /></Td>
                  <Td align="right" mono>{num(h.bot_probability, 3)}</Td>
                  <Td><RiskBadge band={h.risk_band} score={h.risk_score} /></Td>
                  <Td className="text-xs">{h.model_name}<div className="text-[10px] text-ink-3">{h.model_version}</div></Td>
                  <Td className="text-xs">{h.source}</Td>
                  <Td><Button size="sm" variant="ghost" icon={Trash2} aria-label="Delete" onClick={async (e) => { e.stopPropagation(); if (window.confirm("Delete this prediction?")) { await remove.run(h.id); reload(); } }} /></Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
        {data && pages > 1 && (
          <div className="flex items-center justify-end gap-2 border-t border-border px-4 py-2">
            <Button size="sm" variant="secondary" disabled={data.page <= 1} onClick={() => set("page", data.page - 1)}>Previous</Button>
            <Button size="sm" variant="secondary" disabled={data.page >= pages} onClick={() => set("page", data.page + 1)}>Next</Button>
          </div>
        )}
      </Card>
    </div>
  );
}

export function HistoryDetailPage() {
  const { id = "" } = useParams();
  const detail = useApi(() => api.historyDetail(id), [id]);
  const needShap = !!detail.data && !detail.data.shap_explanation;
  const needLime = !!detail.data && !detail.data.lime_explanation;
  const shap = useApi(() => api.shapLocal(id), [id], needShap);
  const lime = useApi(() => api.limeLocal(id), [id], needLime);

  const merged: PredictResponse | null = useMemo(() => {
    if (!detail.data) return null;
    const out = { ...detail.data };
    if (needShap && shap.data) out.shap_explanation = shap.data.explanation as ShapLocal;
    if (needLime && lime.data) out.lime_explanation = lime.data.explanation as LimeLocal;
    if (needShap && shap.error) out.explanation_errors = { ...out.explanation_errors, shap: shap.error };
    if (needLime && lime.error) out.explanation_errors = { ...out.explanation_errors, lime: lime.error };
    if (out.shap_explanation && (!out.top_features.length || out.top_features[0]?.impact === null)) {
      out.top_features = out.shap_explanation.contributions.slice(0, 8).map((c) => ({ feature: c.feature, group: c.group, description: c.description, value: c.value, impact: c.shap, direction: c.direction }));
    }
    return out;
  }, [detail.data, shap.data, shap.error, lime.data, lime.error, needShap, needLime]);

  return (
    <div className="space-y-6">
      <PageHeader title="Prediction detail" description={merged ? `${merged.account_identifier} · ${dateTime(merged.created_at)} · source: ${merged.source}` : id} actions={<Link to="/history"><Button variant="secondary" icon={ArrowLeft}>Back to history</Button></Link>} />
      {detail.loading && !detail.data && <Spinner label="Loading prediction…" />}
      {detail.error && <ErrorState message={detail.error} onRetry={detail.reload} />}
      {merged && ((needShap && shap.loading) || (needLime && lime.loading)) && <Spinner label="Computing SHAP / LIME explanation for this stored feature vector…" />}
      {merged && !((needShap && shap.loading) || (needLime && lime.loading)) && <PredictionResult result={merged} />}
    </div>
  );
}
