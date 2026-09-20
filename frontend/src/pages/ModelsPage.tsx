import { Boxes, CheckCircle2, FlaskConical, Trash2, XCircle } from "lucide-react";
import { useCallback, useState } from "react";
import { Link } from "react-router-dom";

import { MetricComparisonChart } from "@/components/charts/basic";
import { COLORS } from "@/components/charts/common";
import { Badge, Button, Card, EmptyState, ErrorState, PageHeader, SourceTag, StatusBadge, Table, Td, Th } from "@/components/ui";
import { ConfirmDialog } from "@/components/ui/Dialog";
import { useAction, useApi } from "@/hooks/useApi";
import { useAuth } from "@/hooks/useAuth";
import { api } from "@/services/api";
import type { ModelPublic } from "@/types/api";
import { algorithmLabel, dateTime, num, seconds } from "@/utils/format";

export function ModelsPage() {
  const { hasRole } = useAuth();
  const isAdmin = hasRole("ADMIN");
  const list = useApi(() => api.models(), []);
  const research = useApi(() => api.research(), []);
  const [confirm, setConfirm] = useState<{ kind: "activate" | "deprecate" | "delete"; model: ModelPublic } | null>(null);
  const act = useAction(useCallback(async (kind: "activate" | "deprecate" | "delete", id: string) => {
    if (kind === "activate") return api.activateModel(id);
    if (kind === "deprecate") return api.deprecateModel(id);
    return api.deleteModel(id);
  }, []));

  const models = list.data?.models ?? [];
  const comparison = models.filter((m) => m.test_metrics?.accuracy != null).map((m) => ({ name: `${m.name} v${m.version}`, accuracy: m.test_metrics?.accuracy ?? null, f1: m.test_metrics?.f1 ?? null, roc_auc: m.test_metrics?.roc_auc ?? null }));
  const paper = research.data?.paper_reported;

  return (
    <div className="space-y-6">
      <PageHeader title="Models" description="Model versions trained in your organization. Exactly one model can be in production; analyses use it unless another READY model is chosen explicitly." actions={hasRole("ADMIN", "ANALYST") && <Link to="/training"><Button icon={FlaskConical}>Train a model</Button></Link>} />
      <ConfirmDialog
        open={!!confirm}
        title={confirm?.kind === "activate" ? `Promote ${confirm.model.name} v${confirm.model.version} to production?` : confirm?.kind === "deprecate" ? `Deprecate ${confirm?.model.name} v${confirm?.model.version}?` : `Delete ${confirm?.model.name} v${confirm?.model.version}?`}
        description={confirm?.kind === "activate" ? "Artefact checksums are verified first. The current production model (if any) is set back to READY." : confirm?.kind === "deprecate" ? "Deprecated models cannot be used for new analyses; stored predictions keep referencing them." : "Artefacts are removed from storage. Stored predictions keep the model name but the evaluation history for this model is deleted. Production models cannot be deleted."}
        confirmLabel={confirm?.kind === "activate" ? "Promote" : confirm?.kind === "deprecate" ? "Deprecate" : "Delete"}
        destructive={confirm?.kind !== "activate"}
        loading={act.loading}
        onCancel={() => setConfirm(null)}
        onConfirm={async () => { if (confirm) { await act.run(confirm.kind, confirm.model.id); setConfirm(null); list.reload(); } }}
      />
      {act.error && <ErrorState title="Action failed" message={act.error} />}

      <Card title="Registered models" subtitle="Hold-out metrics come from each model's own stratified test split." padded={false} actions={<SourceTag kind="ours" />}>
        {list.error && <div className="p-4"><ErrorState message={list.error} onRetry={list.reload} /></div>}
        {list.data && models.length === 0 && <div className="p-4"><EmptyState icon={Boxes} title="No models trained yet" description="Train a model on a labelled dataset; it appears here with its metrics and can then be promoted to production." action={hasRole("ADMIN", "ANALYST") ? <Link to="/training"><Button>Go to Training</Button></Link> : undefined} /></div>}
        {models.length > 0 && (
          <Table className="rounded-none border-0">
            <thead><tr><Th>Model</Th><Th>Status</Th><Th>Algorithm</Th><Th>Dataset</Th><Th align="right">Accuracy</Th><Th align="right">F1</Th><Th align="right">ROC-AUC</Th><Th align="right">CV F1</Th><Th align="right">Train time</Th><Th>Trained</Th><Th></Th></tr></thead>
            <tbody>{models.map((m) => (
              <tr key={m.id} className={m.is_production ? "bg-accent-soft/40" : ""}>
                <Td><div className="font-medium text-ink">{m.name} <span className="text-ink-3">v{m.version}</span></div><div className="text-[11px] text-ink-3">{m.n_features} features · {m.feature_version}{m.notes ? ` · ${m.notes}` : ""}</div></Td>
                <Td><StatusBadge status={m.status} /></Td><Td className="text-xs">{algorithmLabel(m.algorithm)}</Td><Td className="text-xs">{m.dataset_name || "—"}</Td>
                <Td align="right" mono>{num(m.test_metrics?.accuracy, 3)}</Td><Td align="right" mono>{num(m.test_metrics?.f1, 3)}</Td><Td align="right" mono>{num(m.test_metrics?.roc_auc, 3)}</Td>
                <Td align="right" mono>{m.validation_metrics?.mean?.f1 != null ? `${num(m.validation_metrics.mean.f1, 3)} ± ${num(m.validation_metrics.std?.f1, 3)}` : "—"}</Td>
                <Td align="right" mono>{seconds(m.training_seconds)}</Td><Td className="text-xs text-ink-2">{dateTime(m.trained_at)}</Td>
                <Td>
                  <div className="flex justify-end gap-1">
                    <Link to={`/evaluation?model=${m.id}`}><Button size="sm" variant="ghost">Details</Button></Link>
                    {isAdmin && m.status === "READY" && <Button size="sm" variant="secondary" icon={CheckCircle2} onClick={() => setConfirm({ kind: "activate", model: m })}>Promote</Button>}
                    {isAdmin && (m.status === "READY" || m.status === "PRODUCTION") && <Button size="sm" variant="ghost" icon={XCircle} onClick={() => setConfirm({ kind: "deprecate", model: m })}>Deprecate</Button>}
                    {isAdmin && m.status !== "PRODUCTION" && m.status !== "TRAINING" && <Button size="sm" variant="ghost" icon={Trash2} aria-label="Delete model" onClick={() => setConfirm({ kind: "delete", model: m })} />}
                  </div>
                </Td>
              </tr>
            ))}</tbody>
          </Table>
        )}
      </Card>

      {comparison.length > 1 && (
        <Card title="Hold-out comparison" subtitle="Your trained versions side by side" actions={<SourceTag kind="ours" />}>
          <MetricComparisonChart data={comparison} series={[{ key: "accuracy", label: "Accuracy", color: COLORS.series[0] }, { key: "f1", label: "F1", color: COLORS.series[1] }, { key: "roc_auc", label: "ROC-AUC", color: COLORS.series[2] }]} />
        </Card>
      )}

      {list.data && (
        <Card title="Supported algorithms" subtitle="Nine classifiers from the reference methodology; availability depends on installed libraries on the server.">
          <div className="flex flex-wrap gap-2">{list.data.supported_algorithms.map((a) => <Badge key={a.key} tone={a.available ? "good" : "neutral"}>{a.display_name}{a.supports_tree_shap ? " · TreeSHAP" : ""}{!a.available ? ` (${a.unavailable_reason})` : ""}</Badge>)}</div>
        </Card>
      )}

      {paper && (
        <Card title="Results reported in the reference paper" subtitle={`${paper.citation.authors[0]} et al., ${paper.citation.venue} ${paper.citation.year} · DOI ${paper.citation.doi}. These numbers are transcribed from the publication and are not measurements of your models.`} actions={<SourceTag kind="paper" />}>
          <div className="grid gap-4 lg:grid-cols-2">
            {Object.entries(paper.datasets).map(([ds, block]) => (
              <div key={ds}>
                <h3 className="mb-1 text-sm font-medium text-ink">{ds} <span className="text-xs text-ink-3">({block.table})</span></h3>
                <Table compact><thead><tr><Th>Algorithm</Th><Th align="right">Acc</Th><Th align="right">Prec</Th><Th align="right">Rec</Th><Th align="right">F1</Th><Th align="right">AUC</Th></tr></thead>
                  <tbody>{block.results.map((r) => <tr key={r.algorithm} className={r.algorithm === block.highlight ? "font-medium" : ""}><Td>{algorithmLabel(r.algorithm)}</Td><Td align="right" mono>{num(r.accuracy, 3)}</Td><Td align="right" mono>{num(r.precision, 3)}</Td><Td align="right" mono>{num(r.recall, 3)}</Td><Td align="right" mono>{num(r.f1, 3)}</Td><Td align="right" mono>{num(r.roc_auc, 3)}</Td></tr>)}</tbody></Table>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
