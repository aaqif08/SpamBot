import { Bot, Cpu, Database, ShieldAlert, Users, Wand2 } from "lucide-react";
import { Link } from "react-router-dom";

import { ClassDistributionChart, ConfusionMatrixView, ImportanceChart, MetricComparisonChart, ProbabilityHistogram, ProportionBar, RocChart, TimelineChart } from "@/components/charts/basic";
import { COLORS } from "@/components/charts/common";
import { Button, Card, EmptyState, ErrorState, Notice, PageHeader, Skeleton, SourceTag, StatTile } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api } from "@/services/api";
import { pct } from "@/utils/format";

export function DashboardPage() {
  const { data, loading, error, reload } = useApi(() => api.dashboard(), []);

  if (loading && !data) {
    return (
      <div className="space-y-4">
        <PageHeader title="Dashboard" description="Live overview of predictions, the active model and its evaluation." />
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="space-y-4">
        <PageHeader title="Dashboard" />
        <ErrorState title="Could not load dashboard" message={error ?? "Unknown error"} onRetry={reload} />
      </div>
    );
  }

  const { cards, charts } = data;
  const noModelAction = (
    <div className="flex flex-wrap gap-2">
      <Link to="/training">
        <Button icon={Wand2}>Open training</Button>
      </Link>
      <Link to="/datasets">
        <Button variant="secondary" icon={Database}>Datasets</Button>
      </Link>
    </div>
  );

  return (
    <div className="space-y-6">
      <PageHeader title="Dashboard" description="Every number on this page is read from the SQLite prediction store or from the active model's stored evaluation artefacts." />

      {data.notices.map((n) => (
        <Notice key={n} tone={n.includes("DEMO") ? "warning" : "info"}>
          {n}
        </Notice>
      ))}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <StatTile label="Total accounts analyzed" value={cards.total_accounts_analyzed.toLocaleString()} icon={Users} sub={cards.demo_predictions ? `${cards.demo_predictions.toLocaleString()} from demo data` : undefined} />
        <StatTile label="Bots detected (model prediction)" value={cards.bots_detected.toLocaleString()} icon={Bot} accent="var(--bot)" />
        <StatTile label="Humans detected (model prediction)" value={cards.humans_detected.toLocaleString()} icon={Users} accent="var(--human)" />
        <StatTile label="Average estimated bot probability" value={pct(cards.average_bot_probability, 1)} icon={ShieldAlert} />
        <StatTile label="High-risk accounts (score ≥ 60)" value={cards.high_risk_accounts.toLocaleString()} icon={ShieldAlert} accent="var(--status-serious)" />
        <StatTile label="Current model" value={cards.current_model ?? "None"} icon={Cpu} sub={cards.current_model_algorithm ?? "Run training to activate a model"} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Bot vs human distribution" subtitle="Share of stored predictions by predicted class">
          {data.has_predictions ? <ProportionBar bots={cards.bots_detected} humans={cards.humans_detected} /> : <EmptyState title="No predictions yet" description="Run a single or batch analysis to populate this chart." />}
        </Card>
        <Card title="Bot probability distribution" subtitle="Histogram of estimated bot probability across stored predictions (last 5,000)">
          {charts.probability_histogram.length ? <ProbabilityHistogram bins={charts.probability_histogram} /> : <EmptyState title="No predictions yet" />}
        </Card>
      </div>

      {data.has_predictions && charts.timeline.length > 1 && (
        <Card title="Predictions over time" subtitle="Stored predictions per day by predicted class">
          <TimelineChart rows={charts.timeline} />
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Feature importance (active model)" subtitle="Mean |SHAP| over the training sample" actions={data.has_model && <SourceTag kind="ours" />}>
          {charts.feature_importance.length ? <ImportanceChart rows={charts.feature_importance} /> : <EmptyState title="No trained model available" description="Run the training pipeline to populate evaluation results." action={noModelAction} />}
        </Card>
        <Card title="Model comparison (hold-out F1 / accuracy)" subtitle="Every model in the registry, measured by this implementation" actions={data.has_model && <SourceTag kind="ours" />}>
          {charts.model_comparison.length ? (
            <MetricComparisonChart
              data={charts.model_comparison.map((m) => ({ name: m.name + (m.is_demo ? " (demo)" : ""), accuracy: m.accuracy, f1: m.f1, roc_auc: m.roc_auc }))}
              series={[
                { key: "accuracy", label: "Accuracy", color: COLORS.series[0] },
                { key: "f1", label: "F1", color: COLORS.series[1] },
                { key: "roc_auc", label: "ROC-AUC", color: COLORS.series[2] },
              ]}
            />
          ) : (
            <EmptyState title="No trained model available" description="Run the training pipeline to populate evaluation results." action={noModelAction} />
          )}
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Confusion matrix (hold-out)" subtitle="Active model on its stratified test split">
          {charts.confusion_matrix ? <ConfusionMatrixView cm={charts.confusion_matrix} /> : <EmptyState title="No trained model available" />}
        </Card>
        <Card title="ROC curve (hold-out)" subtitle="Active model">
          {charts.roc_curve ? <RocChart roc={charts.roc_curve} /> : <EmptyState title="No trained model available" />}
        </Card>
        <Card title="Dataset class distribution" subtitle={charts.dataset_class_distribution ? `Training dataset: ${charts.dataset_class_distribution.dataset}${charts.dataset_class_distribution.is_demo ? " (DEMO)" : ""}` : "Training dataset of the active model"}>
          {charts.dataset_class_distribution?.train && charts.dataset_class_distribution.test ? (
            <ClassDistributionChart
              rows={[
                { name: "Train", ...charts.dataset_class_distribution.train },
                { name: "Test", ...charts.dataset_class_distribution.test },
              ]}
            />
          ) : (
            <EmptyState title="No trained model available" />
          )}
        </Card>
      </div>
    </div>
  );
}
