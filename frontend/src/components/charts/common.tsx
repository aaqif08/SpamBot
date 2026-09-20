import type { ReactNode } from "react";

/** Series colors resolve through CSS variables so light/dark swap in one place. */
export const COLORS = {
  human: "var(--human)",
  bot: "var(--bot)",
  series: ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)"],
  grid: "var(--grid)",
  text: "var(--text-2)",
  muted: "var(--text-3)",
  surface: "var(--surface)",
  accent: "var(--accent)",
};

/** Fixed categorical slot per feature group (never re-assigned on filter). */
export const GROUP_COLORS: Record<string, string> = {
  user_profile: "var(--series-1)",
  content: "var(--series-2)",
  engagement: "var(--series-3)",
  linguistic: "var(--series-4)",
  profile_attributes: "var(--series-5)",
  sentiment: "var(--series-6)",
};

/** Sequential blue ramp (reference palette) for magnitude encodings. */
export const BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

export function rampColor(t: number, ramp: string[] = BLUE_RAMP): string {
  const i = Math.max(0, Math.min(ramp.length - 1, Math.round(t * (ramp.length - 1))));
  return ramp[i];
}

export const BAR_MAX = 24;
export const BAR_RADIUS: [number, number, number, number] = [4, 4, 0, 0];
export const HBAR_RADIUS: [number, number, number, number] = [0, 4, 4, 0];

export interface TooltipRow {
  label: string;
  value: ReactNode;
  color?: string;
}

export function TooltipBox({ title, rows }: { title?: ReactNode; rows: TooltipRow[] }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2 text-xs shadow-lg">
      {title && <div className="mb-1 font-medium text-ink">{title}</div>}
      <div className="space-y-0.5">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center justify-between gap-4 text-ink-2">
            <span className="flex items-center gap-1.5">
              {r.color && <span className="h-2 w-2 rounded-full" style={{ background: r.color }} aria-hidden />}
              {r.label}
            </span>
            <span className="font-mono tabular-nums text-ink">{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function LegendRow({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-2" aria-label="Legend">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: it.color }} aria-hidden />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function ChartFrame({ height = 260, children, legend, footer }: { height?: number; children: ReactNode; legend?: ReactNode; footer?: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      {legend}
      <div style={{ height }} className="w-full">
        {children}
      </div>
      {footer && <div className="text-[11px] text-ink-3">{footer}</div>}
    </div>
  );
}

export const axisProps = { tick: { fontSize: 11 }, stroke: "var(--border)", tickLine: false } as const;
