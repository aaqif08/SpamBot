import { Bot, Boxes, Database, FlaskConical, ScanSearch, ShieldAlert, Users, Workflow } from "lucide-react";
import { Link } from "react-router-dom";

import { ConfusionMatrixView, ImportanceChart, MetricComparisonChart, ProbabilityHistogram, ProportionBar, RocChart, TimelineChart } from "@/components/charts/basic";
import { COLORS } from "@/components/charts/common";
import { Button, Card, EmptyState, ErrorState, PageHeader, RiskBadge, Skeleton, SourceTag, StatTile } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { useAuth } from "@/hooks/useAuth";
import { api } from "@/services/api";
import { dateTime, int, pct } from "@/utils/format";

export function DashboardPage() {
  const { data, loading, error, reload } = useApi(() => api.dashboard(), []);
  const { hasRole } = useAuth();

  if (loading && !data) {
    return (
      <div className="space-y-4">
        <PageHeader title="Dashboard" />
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">{Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-24" />)}</div>
      </div>
    );
  }
  if (error || !data) return <div className="space-y-4"><PageHeader title="Dashboard" /><ErrorState title="Could not load the dashboard" message={error ?? "Unknown error"} onRetry={reload} /></div>;

  const { cards, charts } = data;
  const canAnalyze = hasRole("ADMIN", "ANALYST");

  if (!data.has_model && !data.has_predictions && cards.datasets === 0) {
    return (
      <div className="space-y-6">
        <PageHeader title="Dashboard" description="Live view of analyses, models and jobs for your organization." />
        <Card>
          <div className="mx-auto max-w-xl py-10 text-center">
            <Workflow className="mx-auto h-8 w-8 text-ink-3" aria-hidden />
            <h2 className="mt-3 text-lg font-semibold text-ink">Your intelligence workspace is ready</h2>
            <p className="mt-1 text-sm text-ink-2">No accounts have been analyzed yet. Upload a labelled dataset, train a model, then analyze accounts. Everything shown here comes from your own data.</p>
            <div className="mt-5 flex flex-wrap justify-center gap-2">
              {canAnalyze && <Link to="/datasets"><Button icon={Database}>Upload a dataset</Button></Link>}
              {canAnalyze && <Link to="/training"><Button variant="secondary" icon={FlaskConical}>Train a model</Button></Link>}
              <Link to="/research"><Button variant="ghost">Read the methodology</Button></Link>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Dashboard" description="Every figure is computed from your organization's records and the production model's stored evaluation." actions={canAnalyze && <Link to="/analyze"><Button icon={ScanSearch}>Analyze an account</Button></Link>} />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatTile label="Accounts analyzed" value={int(cards.accounts_analyzed)} icon={Users} sub={cards.latest_analysis_at ? `latest ${dateTime(cards.latest_analysis_at)}` : "no analyses yet"} />
        <StatTile label="Predictions" value={int(cards.predictions_total)} icon={ScanSearch} sub={`${int(cards.batches)} batch run${cards.batches === 1 ? "" : "s"}`} />
        <StatTile label="Classified as bot" value={int(cards.bots_classified)} icon={Bot} accent="var(--bot)" sub={cards.bot_rate !== null ? `bot rate ${pct(cards.bot_rate, 1)}` : undefined} />
        <StatTile label="Classified as human" value={int(cards.humans_classified)} icon={Users} accent="var(--human)" />
        <StatTile label="Average estimated bot probability" value={pct(cards.average_bot_probability, 1)} icon={ShieldAlert} />
        <StatTile label="High-risk (score ≥ 60)" value={int(cards.high_risk)} icon={ShieldAlert} accent="var(--status-serious)" />
        <StatTile label="Production model" value={cards.production_model ? `${cards.production_model.name} v${cards.production_model.version}` : "None"} icon={Boxes} sub={cards.production_model ? cards.production_model.algorithm : "activate a model on the Models page"} />
        <StatTile label="Datasets · models · jobs" value={`${cards.datasets} · ${cards.models} · ${cards.jobs_total}`} icon={Database} sub={cards.jobs_running ? `${cards.jobs_running} job(s) running` : "no jobs running"} />
      </div>

      {!data.has_model && (
        <Card>
          <EmptyState icon={Boxes} title="No production model configured" description="Train a model on a labelled dataset and activate it to enable analyses." action={canAnalyze ? <Link to="/training"><Button>Train a model</Button></Link> : undefined} />
        </Card>
      )}

      {data.has_predictions ? (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Prediction distribution" subtitle="Share of stored predictions by model classification">
              <ProportionBar bots={cards.bots_classified} humans={cards.humans_classified} />
              <div className="mt-4 flex flex-wrap gap-3">
                {charts.risk_distribution.map((r) => (
                  <span key={r.band} className="flex items-center gap-1 text-xs text-ink-2"><RiskBadge band={r.band} /> <span className="font-mono text-ink">{int(r.count)}</span></span>
                ))}
              </div>
            </Card>
            <Card title="Estimated bot probability distribution" subtitle="Last 5,000 predictions">
              <ProbabilityHistogram bins={charts.probability_histogram} />
            </Card>
          </div>
          {charts.timeline.length > 1 && (
            <Card title="Analysis volume (last 30 days)" subtitle="Predictions stored per day by classification">
              <TimelineChart rows={charts.timeline} />
            </Card>
          )}
          {charts.model_usage.length > 0 && (
            <Card title="Model usage" subtitle="Predictions per model version">
              <ul className="divide-y divide-border text-sm">
                {charts.model_usage.map((m) => (
                  <li key={m.model} className="flex justify-between py-1.5"><span className="text-ink">{m.model}</span><span className="font-mono text-ink-2">{int(m.count)}</span></li>
                ))}
              </ul>
            </Card>
          )}
        </>
      ) : (
        data.has_model && (
          <Card>
            <EmptyState icon={ScanSearch} title="No analyses yet" description="Analyze an account or run a batch to populate the prediction charts." action={canAnalyze ? <Link to="/analyze"><Button>Analyze an account</Button></Link> : undefined} />
          </Card>
        )
      )}

      {data.has_model && (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Feature importance (production model)" subtitle="Mean |SHAP| over a subset of training rows" actions={<SourceTag kind="ours" />}>
              {charts.feature_importance.length ? <ImportanceChart rows={charts.feature_importance} /> : <EmptyState title="No SHAP analysis stored for this model" />}
            </Card>
            <Card title="Model comparison (hold-out)" subtitle="Trained model versions in this organization" actions={<SourceTag kind="ours" />}>
              {charts.model_comparison.length ? (
                <MetricComparisonChart
                  data={charts.model_comparison.map((m) => ({ name: `${m.name} v${m.version}`, accuracy: m.accuracy, f1: m.f1, roc_auc: m.roc_auc }))}
                  series={[{ key: "accuracy", label: "Accuracy", color: COLORS.series[0] }, { key: "f1", label: "F1", color: COLORS.series[1] }, { key: "roc_auc", label: "ROC-AUC", color: COLORS.series[2] }]}
                />
              ) : (
                <EmptyState title="No trained models" />
              )}
            </Card>
          </div>
          {charts.production_holdout && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="Confusion matrix (hold-out)" subtitle={`Production model on its stratified test split (${int(charts.production_holdout.n_samples)} accounts)`}>
                {charts.production_holdout.confusion_matrix ? <ConfusionMatrixView cm={charts.production_holdout.confusion_matrix} /> : <EmptyState title="Unavailable" />}
              </Card>
              <Card title="ROC curve (hold-out)">
                {charts.production_holdout.roc_curve ? <RocChart roc={charts.production_holdout.roc_curve} /> : <EmptyState title="Unavailable" />}
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  );
}
