import { ArrowLeft, History as HistoryIcon, Search, Trash2 } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { PredictionResult } from "@/components/PredictionResult";
import { Badge, Button, Card, EmptyState, ErrorState, Field, Input, PageHeader, PredictionBadge, RiskBadge, Select, Skeleton, Spinner, Table, Td, Th, Toggle } from "@/components/ui";
import { ConfirmDialog } from "@/components/ui/Dialog";
import { useAction, useApi } from "@/hooks/useApi";
import { useAuth } from "@/hooks/useAuth";
import { api } from "@/services/api";
import type { AnalysisResponse, HistoryFilters, LimeLocal, ShapLocal } from "@/types/api";
import { dateTime, num } from "@/utils/format";

const SORTS = [
  { key: "created_at:desc", label: "Newest first" },
  { key: "created_at:asc", label: "Oldest first" },
  { key: "risk_score:desc", label: "Highest risk" },
  { key: "bot_probability:desc", label: "Highest bot probability" },
  { key: "account_identifier:asc", label: "Account A→Z" },
];

export function HistoryPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const { hasRole } = useAuth();
  const canDelete = hasRole("ADMIN", "ANALYST");
  const models = useApi(() => api.models(), []);
  const [filters, setFilters] = useState<HistoryFilters>({ prediction: "", min_risk: undefined, model_id: "", source: "", batch_id: params.get("batch_id") ?? "", date_from: "", date_to: "", search: "", sort: "created_at:desc", page: 1, page_size: 25 });
  const query = useMemo(() => Object.fromEntries(Object.entries(filters).filter(([, v]) => v !== "" && v !== undefined)) as HistoryFilters, [filters]);
  const { data, loading, error, reload } = useApi(() => api.history(query), [query]);
  const remove = useAction(useCallback((id: string) => api.deleteAnalysis(id), []));
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const set = <K extends keyof HistoryFilters>(k: K, v: HistoryFilters[K]) => setFilters((f) => ({ ...f, [k]: v, page: k === "page" ? (v as number) : 1 }));
  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;

  return (
    <div className="space-y-6">
      <PageHeader title="Prediction History" description="Every analysis is stored with its features, model version and explanations. Open a row for the full report." />
      <ConfirmDialog open={!!confirmId} title="Delete this prediction?" description="The stored prediction and its explanations are removed; the action is audited." confirmLabel="Delete" destructive loading={remove.loading} onCancel={() => setConfirmId(null)} onConfirm={async () => { if (confirmId) { await remove.run(confirmId); setConfirmId(null); reload(); } }} />
      <Card title="Filters">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
          <Field label="Search account"><div className="relative"><Search className="pointer-events-none absolute left-2.5 top-3 h-4 w-4 text-ink-3" /><Input className="pl-8" value={filters.search ?? ""} onChange={(e) => set("search", e.target.value)} placeholder="identifier…" /></div></Field>
          <Field label="Classification"><Select value={filters.prediction ?? ""} onChange={(e) => set("prediction", e.target.value as HistoryFilters["prediction"])}><option value="">All</option><option value="BOT">BOT</option><option value="HUMAN">HUMAN</option></Select></Field>
          <Field label="Model"><Select value={filters.model_id ?? ""} onChange={(e) => set("model_id", e.target.value)}><option value="">All</option>{models.data?.models.map((m) => <option key={m.id} value={m.id}>{m.name} v{m.version}</option>)}</Select></Field>
          <Field label="Source"><Select value={filters.source ?? ""} onChange={(e) => set("source", e.target.value)}><option value="">All</option><option value="manual">manual</option><option value="batch">batch</option><option value="x_api">x_api</option></Select></Field>
          <Field label="From"><Input type="date" value={filters.date_from ?? ""} onChange={(e) => set("date_from", e.target.value)} /></Field>
          <Field label="To"><Input type="date" value={(filters.date_to ?? "").slice(0, 10)} onChange={(e) => set("date_to", e.target.value ? `${e.target.value}T23:59:59` : "")} /></Field>
          <Field label="Sort"><Select value={filters.sort ?? ""} onChange={(e) => set("sort", e.target.value)}>{SORTS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}</Select></Field>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-4">
          <Toggle label="High risk only (risk score ≥ 60)" checked={filters.min_risk === 60} onChange={(v) => set("min_risk", v ? 60 : undefined)} />
          {filters.batch_id && <span className="text-xs text-ink-2">Filtered by batch <code className="font-mono">{filters.batch_id.slice(0, 8)}</code> <button className="ml-1 underline" onClick={() => set("batch_id", "")}>clear</button></span>}
        </div>
      </Card>

      <Card title={`Predictions${data ? ` (${data.total.toLocaleString()})` : ""}`} padded={false} actions={data && <span className="text-xs text-ink-2">page {data.page} / {pages}</span>}>
        {loading && !data && <div className="p-4"><Skeleton className="h-40" /></div>}
        {error && <div className="p-4"><ErrorState message={error} onRetry={reload} /></div>}
        {data && data.items.length === 0 && <div className="p-4"><EmptyState icon={HistoryIcon} title={data.total === 0 && !filters.search && !filters.prediction ? "No analyses yet" : "No predictions match these filters"} description="Analyze an account or run a batch to populate the history." action={hasRole("ADMIN", "ANALYST") ? <Link to="/analyze"><Button variant="secondary">Analyze account</Button></Link> : undefined} /></div>}
        {data && data.items.length > 0 && (
          <Table className="rounded-none border-0">
            <thead><tr><Th>Date</Th><Th>Account</Th><Th>Classification</Th><Th align="right">Bot probability</Th><Th>Risk score</Th><Th>Model</Th><Th>Source</Th><Th>Label</Th><Th></Th></tr></thead>
            <tbody>
              {data.items.map((h) => (
                <tr key={h.id} className="cursor-pointer hover:bg-surface-2" onClick={() => navigate(`/history/${h.id}`)}>
                  <Td className="text-xs text-ink-2">{dateTime(h.created_at)}</Td>
                  <Td className="font-mono text-xs">{h.account_identifier}</Td>
                  <Td><PredictionBadge label={h.prediction} /></Td>
                  <Td align="right" mono>{num(h.bot_probability, 3)}</Td>
                  <Td><RiskBadge band={h.risk_band} score={h.risk_score} /></Td>
                  <Td className="text-xs">{h.model_name} <span className="text-ink-3">v{h.model_version}</span></Td>
                  <Td className="text-xs">{h.source}</Td>
                  <Td className="text-xs">{h.label_true ? <Badge tone={h.label_true === h.prediction ? "good" : "critical"}>{h.label_true}</Badge> : <span className="text-ink-3">—</span>}</Td>
                  <Td>{canDelete && <Button size="sm" variant="ghost" icon={Trash2} aria-label="Delete" onClick={(e) => { e.stopPropagation(); setConfirmId(h.id); }} />}</Td>
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
  const detail = useApi(() => api.analysis(id), [id]);
  const needShap = !!detail.data && !detail.data.shap_explanation;
  const needLime = !!detail.data && !detail.data.lime_explanation;
  const shap = useApi(() => api.explanation(id, "shap"), [id], needShap);
  const lime = useApi(() => api.explanation(id, "lime"), [id], needLime);
  const merged = useMemo<AnalysisResponse | null>(() => {
    if (!detail.data) return null;
    const out: AnalysisResponse = { ...detail.data, explanation_errors: { ...detail.data.explanation_errors } };
    if (needShap && shap.data) out.shap_explanation = shap.data.explanation as ShapLocal;
    if (needLime && lime.data) out.lime_explanation = lime.data.explanation as LimeLocal;
    if (needShap && shap.error) out.explanation_errors.shap = shap.error;
    if (needLime && lime.error) out.explanation_errors.lime = lime.error;
    if (out.shap_explanation && (!out.top_features.length || out.top_features[0]?.impact === null)) {
      out.top_features = out.shap_explanation.contributions.slice(0, 8).map((c) => ({ feature: c.feature, group: c.group, description: c.description, value: c.value, impact: c.shap, direction: c.direction }));
    }
    return out;
  }, [detail.data, shap.data, shap.error, lime.data, lime.error, needShap, needLime]);

  return (
    <div className="space-y-4">
      <Link to="/history" className="inline-flex items-center gap-1 text-sm text-ink-2 hover:text-ink"><ArrowLeft className="h-4 w-4" /> Back to history</Link>
      {detail.loading && !detail.data && <Spinner label="Loading analysis…" />}
      {detail.error && <ErrorState message={detail.error} onRetry={detail.reload} />}
      {merged && ((needShap && shap.loading) || (needLime && lime.loading)) && <Spinner label="Computing SHAP / LIME for this account…" />}
      {merged && <PredictionResult result={merged} />}
    </div>
  );
}
