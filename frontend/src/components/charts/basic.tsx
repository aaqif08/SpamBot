import { useMemo } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { ConfusionMatrix, HistogramBin, ImportanceRow, PrCurve, RocCurve } from "@/types/api";
import { featureLabel, groupLabel, int, num, pct } from "@/utils/format";

import { BAR_MAX, BAR_RADIUS, COLORS, ChartFrame, GROUP_COLORS, HBAR_RADIUS, LegendRow, TooltipBox, axisProps, rampColor } from "./common";

// ---- Proportion bar (BOT vs HUMAN) ------------------------------------------------ //

export function ProportionBar({ bots, humans }: { bots: number; humans: number }) {
  const total = bots + humans;
  const pb = total ? bots / total : 0;
  return (
    <div className="space-y-2">
      <div className="flex h-6 w-full overflow-hidden rounded-md bg-surface-2" role="img" aria-label={`${bots} bots, ${humans} humans`}>
        {bots > 0 && <div style={{ width: `${pb * 100}%`, background: COLORS.bot }} title={`BOT ${int(bots)}`} />}
        {bots > 0 && humans > 0 && <div className="w-0.5 bg-surface" aria-hidden />}
        {humans > 0 && <div style={{ width: `${(1 - pb) * 100}%`, background: COLORS.human }} title={`HUMAN ${int(humans)}`} />}
      </div>
      <div className="flex flex-wrap justify-between gap-2 text-xs text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: COLORS.bot }} aria-hidden />
          BOT <strong className="font-mono text-ink">{int(bots)}</strong> ({pct(pb, 1)})
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: COLORS.human }} aria-hidden />
          HUMAN <strong className="font-mono text-ink">{int(humans)}</strong> ({pct(1 - pb, 1)})
        </span>
      </div>
    </div>
  );
}

// ---- Probability histogram ------------------------------------------------------- //

export function ProbabilityHistogram({ bins, height = 220 }: { bins: HistogramBin[]; height?: number }) {
  const data = bins.map((b) => ({ ...b, label: `${b.bin_start.toFixed(1)}–${b.bin_end.toFixed(1)}`, mid: (b.bin_start + b.bin_end) / 2 }));
  return (
    <ChartFrame
      height={height}
      legend={<LegendRow items={[{ label: "Predicted HUMAN (p < 0.5)", color: COLORS.human }, { label: "Predicted BOT (p ≥ 0.5)", color: COLORS.bot }]} />}
      footer="x-axis: estimated bot probability bins · y-axis: number of accounts"
    >
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }} barCategoryGap={2}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="label" {...axisProps} interval={0} />
          <YAxis {...axisProps} allowDecimals={false} />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={({ payload }) => (payload?.[0] ? <TooltipBox title={`p(bot) ${payload[0].payload.label}`} rows={[{ label: "accounts", value: int(payload[0].payload.count) }]} /> : null)} />
          <Bar isAnimationActive={false} dataKey="count" maxBarSize={BAR_MAX} radius={BAR_RADIUS}>
            {data.map((d) => (
              <Cell key={d.label} fill={d.mid >= 0.5 ? COLORS.bot : COLORS.human} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---- Feature importance (mean |SHAP|) ---------------------------------------------- //

export function ImportanceChart({ rows, height, outputScale }: { rows: ImportanceRow[]; height?: number; outputScale?: string }) {
  const data = rows.map((r) => ({ ...r, label: featureLabel(r.feature) }));
  const groups = Array.from(new Set(rows.map((r) => r.group)));
  const h = height ?? Math.max(160, rows.length * 22 + 30);
  return (
    <ChartFrame
      height={h}
      legend={<LegendRow items={groups.map((g) => ({ label: groupLabel(g), color: GROUP_COLORS[g] ?? COLORS.muted }))} />}
      footer={`mean |SHAP| across the explained sample${outputScale ? ` · scale: ${outputScale.replace("_", " ")}` : ""}`}
    >
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 24, left: 8, bottom: 0 }} barCategoryGap={2}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis type="number" {...axisProps} tickFormatter={(v: number) => v.toFixed(3)} />
          <YAxis type="category" dataKey="label" width={150} {...axisProps} interval={0} />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={({ payload }) => (payload?.[0] ? <TooltipBox title={payload[0].payload.label} rows={[{ label: "mean |SHAP|", value: num(payload[0].payload.mean_abs_shap, 4) }, { label: "mean SHAP", value: num(payload[0].payload.mean_shap, 4) }, { label: "group", value: groupLabel(payload[0].payload.group) }]} /> : null)} />
          <Bar isAnimationActive={false} dataKey="mean_abs_shap" maxBarSize={16} radius={HBAR_RADIUS}>
            {data.map((d) => (
              <Cell key={d.feature} fill={GROUP_COLORS[d.group] ?? COLORS.muted} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---- Grouped metric comparison ------------------------------------------------------ //

export interface ComparisonSeries {
  key: string;
  label: string;
  color: string;
}

export function MetricComparisonChart({ data, series, height = 260, xKey = "name" }: { data: Record<string, string | number | null>[]; series: ComparisonSeries[]; height?: number; xKey?: string }) {
  return (
    <ChartFrame height={height} legend={<LegendRow items={series.map((s) => ({ label: s.label, color: s.color }))} />}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }} barGap={2} barCategoryGap="30%">
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey={xKey} {...axisProps} interval={0} angle={data.length > 6 ? -20 : 0} textAnchor={data.length > 6 ? "end" : "middle"} height={data.length > 6 ? 48 : 30} />
          <YAxis {...axisProps} domain={[0, 1]} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={({ payload, label }) => (payload?.length ? <TooltipBox title={String(label)} rows={payload.map((p) => ({ label: String(p.name), value: typeof p.value === "number" ? num(p.value, 3) : "—", color: p.color }))} /> : null)} />
          {series.map((s) => (
            <Bar isAnimationActive={false} key={s.key} dataKey={s.key} name={s.label} fill={s.color} maxBarSize={BAR_MAX} radius={BAR_RADIUS} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---- Confusion matrix ------------------------------------------------------------------ //

export function ConfusionMatrixView({ cm }: { cm: ConfusionMatrix }) {
  const max = Math.max(1, ...cm.matrix.flat());
  const cells = [
    { r: 0, c: 0, v: cm.tn, name: "True negative" },
    { r: 0, c: 1, v: cm.fp, name: "False positive" },
    { r: 1, c: 0, v: cm.fn, name: "False negative" },
    { r: 1, c: 1, v: cm.tp, name: "True positive" },
  ];
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-xs">
        <div />
        <div className="text-center font-medium text-ink-2">Predicted HUMAN</div>
        <div className="text-center font-medium text-ink-2">Predicted BOT</div>
        {[0, 1].map((r) => (
          <div key={r} className="contents">
            <div className="flex items-center pr-2 font-medium text-ink-2">{r === 0 ? "Actual HUMAN" : "Actual BOT"}</div>
            {[0, 1].map((c) => {
              const cell = cells.find((x) => x.r === r && x.c === c)!;
              const t = cell.v / max;
              const bg = rampColor(0.15 + t * 0.85);
              return (
                <div key={c} className="flex h-20 flex-col items-center justify-center rounded-md" style={{ background: bg, color: t > 0.45 ? "#fff" : "#0b0b0b" }} title={`${cell.name}: ${cell.v}`}>
                  <span className="text-xl font-semibold tabular-nums">{int(cell.v)}</span>
                  <span className="text-[10px] opacity-80">{cell.name}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-4 text-[11px] text-ink-2">
        <span>False-positive rate <strong className="font-mono text-ink">{pct(cm.false_positive_rate, 2)}</strong></span>
        <span>False-negative rate <strong className="font-mono text-ink">{pct(cm.false_negative_rate, 2)}</strong></span>
      </div>
    </div>
  );
}

// ---- ROC / PR curves ------------------------------------------------------------------------- //

export function RocChart({ roc, height = 240 }: { roc: RocCurve; height?: number }) {
  const data = useMemo(() => roc.fpr.map((f, i) => ({ fpr: f, tpr: roc.tpr[i] })), [roc]);
  if (!data.length) return <p className="text-xs text-ink-3">ROC curve unavailable (single class in evaluation set).</p>;
  return (
    <ChartFrame height={height} legend={<LegendRow items={[{ label: `ROC (AUC ${num(roc.auc, 3)})`, color: COLORS.series[0] }, { label: "Chance", color: COLORS.muted }]} />} footer="x: false-positive rate · y: true-positive rate">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
          <CartesianGrid stroke={COLORS.grid} />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v: number) => v.toFixed(1)} />
          <YAxis type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v: number) => v.toFixed(1)} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={COLORS.muted} strokeWidth={1} />
          <Tooltip content={({ payload }) => (payload?.[0] ? <TooltipBox rows={[{ label: "FPR", value: num(payload[0].payload.fpr, 3) }, { label: "TPR", value: num(payload[0].payload.tpr, 3) }]} /> : null)} />
          <Line type="monotone" dataKey="tpr" stroke={COLORS.series[0]} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function PrChart({ pr, height = 240 }: { pr: PrCurve; height?: number }) {
  const data = useMemo(() => pr.recall.map((r, i) => ({ recall: r, precision: pr.precision[i] })), [pr]);
  if (!data.length) return <p className="text-xs text-ink-3">Precision–recall curve unavailable.</p>;
  return (
    <ChartFrame height={height} legend={<LegendRow items={[{ label: `Precision–recall (AP ${num(pr.average_precision, 3)})`, color: COLORS.series[1] }]} />} footer="x: recall · y: precision">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
          <CartesianGrid stroke={COLORS.grid} />
          <XAxis dataKey="recall" type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v: number) => v.toFixed(1)} />
          <YAxis type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v: number) => v.toFixed(1)} />
          <Tooltip content={({ payload }) => (payload?.[0] ? <TooltipBox rows={[{ label: "recall", value: num(payload[0].payload.recall, 3) }, { label: "precision", value: num(payload[0].payload.precision, 3) }]} /> : null)} />
          <Line type="monotone" dataKey="precision" stroke={COLORS.series[1]} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---- Class distribution (stacked, train vs test) ----------------------------------------------- //

export function ClassDistributionChart({ rows, height = 180 }: { rows: { name: string; human: number; bot: number }[]; height?: number }) {
  return (
    <ChartFrame height={height} legend={<LegendRow items={[{ label: "HUMAN", color: COLORS.human }, { label: "BOT", color: COLORS.bot }]} />}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }} barCategoryGap={8}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis type="number" {...axisProps} allowDecimals={false} />
          <YAxis type="category" dataKey="name" width={70} {...axisProps} />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={({ payload, label }) => (payload?.length ? <TooltipBox title={String(label)} rows={payload.map((p) => ({ label: String(p.name), value: int(Number(p.value)), color: p.color }))} /> : null)} />
          <Bar isAnimationActive={false} dataKey="human" name="HUMAN" stackId="a" fill={COLORS.human} maxBarSize={BAR_MAX} stroke="var(--surface)" strokeWidth={2} />
          <Bar isAnimationActive={false} dataKey="bot" name="BOT" stackId="a" fill={COLORS.bot} maxBarSize={BAR_MAX} radius={HBAR_RADIUS} stroke="var(--surface)" strokeWidth={2} />
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---- Timeline --------------------------------------------------------------------------------- //

export function TimelineChart({ rows, height = 200 }: { rows: { date: string; BOT: number; HUMAN: number }[]; height?: number }) {
  return (
    <ChartFrame height={height} legend={<LegendRow items={[{ label: "HUMAN", color: COLORS.human }, { label: "BOT", color: COLORS.bot }]} />} footer="predictions stored per day">
      <ResponsiveContainer>
        <AreaChart data={rows} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="date" {...axisProps} />
          <YAxis {...axisProps} allowDecimals={false} />
          <Tooltip content={({ payload, label }) => (payload?.length ? <TooltipBox title={String(label)} rows={payload.map((p) => ({ label: String(p.name), value: int(Number(p.value)), color: p.color }))} /> : null)} />
          <Area type="monotone" dataKey="HUMAN" stroke={COLORS.human} fill={COLORS.human} fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
          <Area type="monotone" dataKey="BOT" stroke={COLORS.bot} fill={COLORS.bot} fillOpacity={0.1} strokeWidth={2} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---- CV folds ----------------------------------------------------------------------------------- //

export function CvFoldsChart({ folds, height = 220 }: { folds: { accuracy: number; precision: number; recall: number; f1: number; roc_auc: number | null }[]; height?: number }) {
  const data = folds.map((f, i) => ({ name: `Fold ${i + 1}`, ...f }));
  const series = [
    { key: "accuracy", label: "Accuracy", color: COLORS.series[0] },
    { key: "precision", label: "Precision", color: COLORS.series[1] },
    { key: "recall", label: "Recall", color: COLORS.series[2] },
    { key: "f1", label: "F1", color: COLORS.series[3] },
    { key: "roc_auc", label: "ROC-AUC", color: COLORS.series[4] },
  ];
  return <MetricComparisonChart data={data} series={series} height={height} />;
}
