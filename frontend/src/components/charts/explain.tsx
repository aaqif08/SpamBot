import { useMemo } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";

import type { LimeLocal, ShapContribution, ShapGlobal, ShapLocal } from "@/types/api";
import { featureLabel, num, signed } from "@/utils/format";

import { COLORS, ChartFrame, HBAR_RADIUS, LegendRow, TooltipBox, axisProps, rampColor } from "./common";

const scaleLabel = (s: string) => (s === "log_odds" ? "log-odds" : "probability");

// ---- Local SHAP contribution bars --------------------------------------------------- //

export function ShapContributionChart({ shap, topN = 12, height }: { shap: ShapLocal; topN?: number; height?: number }) {
  const rows = shap.contributions.slice(0, topN).map((c) => ({ ...c, label: featureLabel(c.feature) }));
  const h = height ?? Math.max(160, rows.length * 24 + 30);
  const maxAbs = Math.max(0.001, ...rows.map((r) => Math.abs(r.shap)));
  return (
    <ChartFrame
      height={h}
      legend={<LegendRow items={[{ label: "pushes toward BOT (+)", color: COLORS.bot }, { label: "pushes toward HUMAN (−)", color: COLORS.human }]} />}
      footer={`SHAP value = change in ${scaleLabel(shap.output_scale)} of BOT attributable to the feature (${shap.explainer})`}
    >
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 40, left: 8, bottom: 0 }} barCategoryGap={2}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis type="number" domain={[-maxAbs, maxAbs]} {...axisProps} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis type="category" dataKey="label" width={150} {...axisProps} interval={0} />
          <ReferenceLine x={0} stroke="var(--border-strong)" />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={({ payload }) => (payload?.[0] ? <ShapTooltip c={payload[0].payload as ShapContribution} scale={shap.output_scale} /> : null)} />
          <Bar isAnimationActive={false} dataKey="shap" maxBarSize={16} radius={HBAR_RADIUS}>
            {rows.map((r) => (
              <Cell key={r.feature} fill={r.shap >= 0 ? COLORS.bot : COLORS.human} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

function ShapTooltip({ c, scale }: { c: ShapContribution; scale: string }) {
  return (
    <TooltipBox
      title={featureLabel(c.feature)}
      rows={[
        { label: "feature value", value: num(c.value, 3) },
        { label: "scaled (0–1)", value: num(c.value_scaled, 3) },
        { label: `SHAP (${scaleLabel(scale)})`, value: signed(c.shap, 4), color: c.shap >= 0 ? COLORS.bot : COLORS.human },
        { label: "direction", value: c.direction },
      ]}
    />
  );
}

// ---- Waterfall ------------------------------------------------------------------------ //

export function ShapWaterfall({ shap, topN = 10, height }: { shap: ShapLocal; topN?: number; height?: number }) {
  const rows = useMemo(() => {
    const top = shap.contributions.slice(0, topN);
    const rest = shap.contributions.slice(topN);
    const restSum = rest.reduce((a, c) => a + c.shap, 0);
    let cursor = shap.base_value;
    const out: { label: string; offset: number; size: number; delta: number; start: number; end: number }[] = [];
    const push = (label: string, delta: number) => {
      const start = cursor;
      const end = cursor + delta;
      out.push({ label, offset: Math.min(start, end), size: Math.abs(delta), delta, start, end });
      cursor = end;
    };
    top.forEach((c) => push(featureLabel(c.feature), c.shap));
    if (rest.length) push(`${rest.length} other features`, restSum);
    return out;
  }, [shap, topN]);
  const values = [shap.base_value, shap.model_output, ...rows.map((r) => r.start), ...rows.map((r) => r.end)];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = Math.max(0.02, (hi - lo) * 0.1);
  const h = height ?? Math.max(180, rows.length * 24 + 40);
  return (
    <ChartFrame
      height={h}
      legend={<LegendRow items={[{ label: "increase toward BOT", color: COLORS.bot }, { label: "decrease toward HUMAN", color: COLORS.human }, { label: `base value E[f(x)] = ${num(shap.base_value, 3)}`, color: COLORS.muted }]} />}
      footer={`Cumulative path from the base value to the model output f(x) = ${num(shap.model_output, 3)} (${scaleLabel(shap.output_scale)} of BOT)`}
    >
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 24, left: 8, bottom: 0 }} barCategoryGap={2}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis type="number" domain={[lo - pad, hi + pad]} {...axisProps} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis type="category" dataKey="label" width={150} {...axisProps} interval={0} />
          <ReferenceLine x={shap.base_value} stroke={COLORS.muted} strokeWidth={1} />
          <ReferenceLine x={shap.model_output} stroke="var(--accent)" strokeWidth={1} />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={({ payload }) => (payload?.[0] ? <TooltipBox title={payload[0].payload.label} rows={[{ label: "contribution", value: signed(payload[0].payload.delta, 4) }, { label: "from", value: num(payload[0].payload.start, 3) }, { label: "to", value: num(payload[0].payload.end, 3) }]} /> : null)} />
          <Bar dataKey="offset" stackId="w" fill="transparent" isAnimationActive={false} />
          <Bar dataKey="size" stackId="w" maxBarSize={16} radius={HBAR_RADIUS} isAnimationActive={false}>
            {rows.map((r) => (
              <Cell key={r.label} fill={r.delta >= 0 ? COLORS.bot : COLORS.human} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---- Beeswarm ---------------------------------------------------------------------------- //

export function BeeswarmChart({ global, topN = 12, maxPoints = 100 }: { global: ShapGlobal; topN?: number; maxPoints?: number }) {
  const features = global.beeswarm.slice(0, topN);
  const data = useMemo(() => {
    const out: { x: number; y: number; v: number; raw: number; feature: string }[] = [];
    features.forEach((f, fi) => {
      const pts = f.points.slice(0, maxPoints);
      pts.forEach((p, pi) => {
        // deterministic jitter so the layout is stable between renders
        const jitter = ((pi * 7919) % 100) / 100 - 0.5;
        out.push({ x: p.shap, y: fi + jitter * 0.7, v: p.value_scaled, raw: p.value, feature: f.feature });
      });
    });
    return out;
  }, [features, maxPoints]);
  const h = Math.max(200, features.length * 26 + 40);
  return (
    <ChartFrame
      height={h}
      legend={
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-ink-2">
          <span>Feature value (min–max scaled):</span>
          <span className="inline-flex items-center gap-1">
            low <span className="h-2.5 w-16 rounded-sm" style={{ background: `linear-gradient(90deg, ${rampColor(0.1)}, ${rampColor(1)})` }} aria-hidden /> high
          </span>
        </div>
      }
      footer={`One point per account (${Math.min(maxPoints, global.n_samples)} sampled) · x: SHAP value (${scaleLabel(global.output_scale)} of BOT) · features ordered by mean |SHAP|`}
    >
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis type="number" dataKey="x" {...axisProps} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis
            type="number"
            dataKey="y"
            domain={[-0.6, features.length - 0.4]}
            ticks={features.map((_, i) => i)}
            tickFormatter={(i: number) => featureLabel(features[i]?.feature ?? "")}
            width={150}
            {...axisProps}
            reversed
          />
          <ZAxis range={[28, 28]} />
          <ReferenceLine x={0} stroke="var(--border-strong)" />
          <Tooltip cursor={false} content={({ payload }) => (payload?.[0] ? <TooltipBox title={featureLabel(payload[0].payload.feature)} rows={[{ label: "SHAP", value: signed(payload[0].payload.x, 4) }, { label: "value", value: num(payload[0].payload.raw, 3) }, { label: "scaled", value: num(payload[0].payload.v, 3) }]} /> : null)} />
          <Scatter data={data} isAnimationActive={false} shape={(props: { cx?: number; cy?: number; payload?: { v: number } }) => <circle cx={props.cx} cy={props.cy} r={3.5} fill={rampColor(0.1 + 0.9 * (props.payload?.v ?? 0))} stroke="var(--surface)" strokeWidth={1} />} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

// ---- LIME ------------------------------------------------------------------------------------- //

export function LimeChart({ lime, topN = 12, height }: { lime: LimeLocal; topN?: number; height?: number }) {
  const rows = lime.items.slice(0, topN).map((i) => ({ ...i, label: i.rule }));
  const maxAbs = Math.max(0.001, ...rows.map((r) => Math.abs(r.weight)));
  const h = height ?? Math.max(160, rows.length * 28 + 30);
  return (
    <ChartFrame
      height={h}
      legend={<LegendRow items={[{ label: "supports BOT (+)", color: COLORS.bot }, { label: "supports HUMAN (−)", color: COLORS.human }]} />}
      footer={`Local linear surrogate fitted on ${lime.num_samples.toLocaleString()} perturbed samples · rules use min–max scaled values · R² ${num(lime.surrogate_r2, 3)}`}
    >
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 32, left: 8, bottom: 0 }} barCategoryGap={2}>
          <CartesianGrid horizontal={false} stroke={COLORS.grid} />
          <XAxis type="number" domain={[-maxAbs, maxAbs]} {...axisProps} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis type="category" dataKey="label" width={230} {...axisProps} interval={0} tick={{ fontSize: 10 }} />
          <ReferenceLine x={0} stroke="var(--border-strong)" />
          <Tooltip cursor={{ fill: "var(--surface-2)" }} content={({ payload }) => (payload?.[0] ? <TooltipBox title={payload[0].payload.rule} rows={[{ label: "weight", value: signed(payload[0].payload.weight, 4), color: payload[0].payload.weight >= 0 ? COLORS.bot : COLORS.human }, { label: "value", value: num(payload[0].payload.value, 3) }, { label: "scaled", value: num(payload[0].payload.value_scaled, 3) }]} /> : null)} />
          <Bar isAnimationActive={false} dataKey="weight" maxBarSize={16} radius={HBAR_RADIUS}>
            {rows.map((r) => (
              <Cell key={r.rule} fill={r.weight >= 0 ? COLORS.bot : COLORS.human} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartFrame>
  );
}

export function ProbabilityBars({ human, bot }: { human: number; bot: number }) {
  return (
    <div className="space-y-2 text-xs">
      {[
        { label: "HUMAN", v: human, color: COLORS.human },
        { label: "BOT", v: bot, color: COLORS.bot },
      ].map((r) => (
        <div key={r.label} className="flex items-center gap-2">
          <span className="w-14 font-medium text-ink-2">{r.label}</span>
          <div className="h-3 flex-1 overflow-hidden rounded-sm bg-surface-2">
            <div className="h-full rounded-r-sm" style={{ width: `${Math.max(0, Math.min(100, r.v * 100))}%`, background: r.color }} />
          </div>
          <span className="w-14 text-right font-mono tabular-nums text-ink">{(r.v * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
