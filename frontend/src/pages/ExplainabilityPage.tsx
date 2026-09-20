import { Lightbulb } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { PredictionResult } from "@/components/PredictionResult";
import { ImportanceChart } from "@/components/charts/basic";
import { COLORS, GROUP_COLORS, LegendRow } from "@/components/charts/common";
import { BeeswarmChart } from "@/components/charts/explain";
import { Badge, Button, Card, EmptyState, ErrorState, PageHeader, Select, Skeleton, SourceTag, Spinner, Table, Tabs, Td, Th } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api } from "@/services/api";
import type { AnalysisResponse, LimeLocal, ShapLocal } from "@/types/api";
import { dateTime, featureLabel, groupLabel, num, pct } from "@/utils/format";

function FeatureDistribution({ points, feature }: { points: { value: number }[]; feature: string }) {
  const bins = useMemo(() => {
    const vals = points.map((p) => p.value);
    if (!vals.length) return [];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const n = 12;
    const w = (hi - lo) / n || 1;
    const counts = Array.from({ length: n }, () => 0);
    vals.forEach((v) => {
      const i = Math.min(n - 1, Math.floor((v - lo) / w));
      counts[i] += 1;
    });
    const max = Math.max(...counts, 1);
    return counts.map((c, i) => ({ start: lo + i * w, end: lo + (i + 1) * w, count: c, h: c / max }));
  }, [points]);
  if (!bins.length) return null;
  return (
    <div>
      <div className="flex h-24 items-end gap-0.5" role="img" aria-label={`Distribution of ${featureLabel(feature)}`}>
        {bins.map((b, i) => (
          <div key={i} className="flex-1 rounded-t-sm" style={{ height: `${Math.max(2, b.h * 100)}%`, background: COLORS.series[0] }} title={`${num(b.start, 2)}–${num(b.end, 2)}: ${b.count}`} />
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-ink-3"><span>{num(bins[0].start, 2)}</span><span>{featureLabel(feature)} (sampled accounts)</span><span>{num(bins[bins.length - 1].end, 2)}</span></div>
    </div>
  );
}

export function ExplainabilityPage() {
  const [params] = useSearchParams();
  const models = useApi(() => api.models(), []);
  const [modelId, setModelId] = useState(params.get("model") ?? "");
  useEffect(() => { if (!modelId && models.data) { const id = models.data.production_model_id ?? models.data.models.find((m) => m.status === "READY")?.id ?? ""; if (id) setModelId(id); } }, [models.data, modelId]);
  const global = useApi(() => api.globalExplanation(modelId), [modelId], !!modelId);
  const history = useApi(() => api.history({ page_size: 50 }), []);
  const [predictionId, setPredictionId] = useState(params.get("analysis") ?? "");
  const local = useApi(() => api.analysis(predictionId), [predictionId], predictionId !== "");
  const [view, setView] = useState<"importance" | "beeswarm">("importance");
  const [feature, setFeature] = useState<string>("");

  const g = global.data?.shap_global;
  const selectedFeature = feature || g?.top_features[0] || "";
  const featurePoints = g?.beeswarm.find((b) => b.feature === selectedFeature)?.points ?? [];
  const groupRows = g ? Object.entries(g.group_importance).sort((a, b) => b[1] - a[1]) : [];
  const groupTotal = groupRows.reduce((a, [, v]) => a + v, 0) || 1;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Explainability"
        description="Global: which features generally influence predictions (SHAP over a subset of training rows). Local: why the model classified one specific account (SHAP waterfall + LIME)."
        actions={
          <Select value={modelId} onChange={(e) => setModelId(e.target.value)} className="w-72" aria-label="Model">
            {!models.data?.models.length && <option value="">No models</option>}
            {models.data?.models.filter((m) => m.status !== "TRAINING" && m.status !== "FAILED").map((m) => <option key={m.id} value={m.id}>{m.name} v{m.version} · {m.status.toLowerCase()}</option>)}
          </Select>
        }
      />

      <section className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold text-ink">Global explanation</h2>
          <span className="text-sm text-ink-2">— Which features generally influence predictions?</span>
        </div>
        {models.data && models.data.models.length === 0 && <EmptyState icon={Lightbulb} title="No trained model available" description="Train a model to compute its global SHAP analysis." action={<Link to="/training"><Button>Open training</Button></Link>} />}
        {global.loading && !global.data && <Skeleton className="h-72" />}
        {global.error && !global.data && (global.status === 404 ? <EmptyState icon={Lightbulb} title="Global SHAP analysis unavailable for this model" description={global.error} /> : <ErrorState message={global.error} onRetry={global.reload} />)}
        {global.data && g && (
          <>
            <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
              <Card
                title={view === "importance" ? "SHAP feature importance" : "SHAP summary (beeswarm)"}
                subtitle={`${g.explainer} · ${g.n_samples} sampled accounts · output scale: ${g.output_scale.replace("_", " ")} · base value ${num(g.base_value, 3)}`}
                actions={<div className="flex items-center gap-2"><Tabs tabs={[{ key: "importance", label: "Bar (mean |SHAP|)" }, { key: "beeswarm", label: "Beeswarm" }]} value={view} onChange={setView} /><SourceTag kind="ours" /></div>}
              >
                {view === "importance" ? <ImportanceChart rows={g.importance.slice(0, 20)} outputScale={g.output_scale} /> : <BeeswarmChart global={g} topN={14} />}
              </Card>
              <div className="space-y-4">
                <Card title="Importance by feature group" subtitle="Sum of mean |SHAP| per feature group">
                  <LegendRow items={groupRows.map(([k]) => ({ label: groupLabel(k), color: GROUP_COLORS[k] ?? COLORS.muted }))} />
                  <div className="mt-3 space-y-2">
                    {groupRows.map(([k, v]) => (
                      <div key={k} className="flex items-center gap-2 text-xs">
                        <span className="w-32 text-ink-2">{groupLabel(k)}</span>
                        <div className="h-3 flex-1 overflow-hidden rounded-sm bg-surface-2"><div className="h-full" style={{ width: `${(v / groupTotal) * 100}%`, background: GROUP_COLORS[k] ?? COLORS.muted }} /></div>
                        <span className="w-12 text-right font-mono text-ink">{pct(v / groupTotal, 0)}</span>
                      </div>
                    ))}
                  </div>
                </Card>
                <Card title="Feature distribution" subtitle="Values of one feature across the explained sample" actions={<Select value={selectedFeature} onChange={(e) => setFeature(e.target.value)} className="h-8 w-44 text-xs">{g.top_features.map((f) => <option key={f} value={f}>{featureLabel(f)}</option>)}</Select>}>
                  <FeatureDistribution points={featurePoints} feature={selectedFeature} />
                  <p className="mt-2 text-[11px] text-ink-3">{global.data.feature_descriptions[selectedFeature]}</p>
                </Card>
              </div>
            </div>
            <Card title="Top features (table view)" padded={false}>
              <Table className="rounded-none border-0">
                <thead><tr><Th>Rank</Th><Th>Feature</Th><Th>Group</Th><Th>Description</Th><Th align="right">mean |SHAP|</Th><Th align="right">mean SHAP</Th></tr></thead>
                <tbody>
                  {g.importance.slice(0, 20).map((r) => (
                    <tr key={r.feature}><Td mono>{r.rank}</Td><Td>{featureLabel(r.feature)}</Td><Td><Badge dot={GROUP_COLORS[r.group]}>{groupLabel(r.group)}</Badge></Td><Td className="text-xs text-ink-2">{global.data?.feature_descriptions[r.feature]}</Td><Td align="right" mono>{num(r.mean_abs_shap, 4)}</Td><Td align="right" mono>{num(r.mean_shap, 4)}</Td></tr>
                  ))}
                </tbody>
              </Table>
            </Card>
          </>
        )}
      </section>

      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-semibold text-ink">Local explanation</h2>
            <span className="text-sm text-ink-2">— Why did the model classify this account this way?</span>
          </div>
          <Select value={predictionId} onChange={(e) => setPredictionId(e.target.value)} className="w-96" aria-label="Select a stored prediction">
            <option value="">Select a stored prediction…</option>
            {history.data?.items.map((h) => <option key={h.id} value={h.id}>{h.account_identifier} · {h.prediction} · p={num(h.bot_probability, 2)} · {h.source} · {dateTime(h.created_at)}</option>)}
          </Select>
        </div>
        {!predictionId && <EmptyState icon={Lightbulb} title="Pick a prediction" description={history.data && history.data.total === 0 ? "No predictions stored yet. Analyze an account first." : "Choose one of the stored predictions above, or analyze a new account."} action={<Link to="/analyze"><Button variant="secondary">Analyze an account</Button></Link>} />}
        {predictionId && local.loading && !local.data && <Spinner label="Loading the stored analysis…" />}
        {predictionId && local.error && <ErrorState message={local.error} onRetry={local.reload} />}
        {predictionId && local.data && <LocalWithLazyExplain id={predictionId} initial={local.data} />}
      </section>
    </div>
  );
}

/** Batch rows are stored without explanations; SHAP/LIME are computed on demand and persisted by the API. */
function LocalWithLazyExplain({ id, initial }: { id: string; initial: AnalysisResponse }) {
  const needShap = !initial.shap_explanation;
  const needLime = !initial.lime_explanation;
  const shap = useApi(() => api.explanation(id, "shap"), [id], needShap);
  const lime = useApi(() => api.explanation(id, "lime"), [id], needLime);
  const merged = useMemo(() => {
    const out: AnalysisResponse = { ...initial, explanation_errors: { ...initial.explanation_errors } };
    if (needShap && shap.data) out.shap_explanation = shap.data.explanation as ShapLocal;
    if (needLime && lime.data) out.lime_explanation = lime.data.explanation as LimeLocal;
    if (needShap && shap.error) out.explanation_errors.shap = shap.error;
    if (needLime && lime.error) out.explanation_errors.lime = lime.error;
    if (out.shap_explanation && (!out.top_features.length || out.top_features[0]?.impact === null)) {
      out.top_features = out.shap_explanation.contributions.slice(0, 8).map((c) => ({ feature: c.feature, group: c.group, description: c.description, value: c.value, impact: c.shap, direction: c.direction }));
    }
    return out;
  }, [initial, shap.data, shap.error, lime.data, lime.error, needShap, needLime]);
  if ((needShap && shap.loading) || (needLime && lime.loading)) return <Spinner label="Computing SHAP / LIME for this account…" />;
  return <PredictionResult result={merged} />;
}
