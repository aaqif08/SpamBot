import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Info, Loader2, RefreshCw, ShieldAlert, XCircle, type LucideIcon } from "lucide-react";
import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes } from "react";

// ---- Card ------------------------------------------------------------------ //

export interface CardProps extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  padded?: boolean;
}

export function Card({ title, subtitle, actions, padded = true, className, children, ...rest }: CardProps) {
  return (
    <section className={clsx("rounded-xl border border-border bg-surface shadow-sm", className)} {...rest}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-border px-5 py-4">
          <div className="min-w-0">
            {title && <h3 className="text-sm font-semibold text-ink">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-ink-2">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx(padded && "p-5")}>{children}</div>
    </section>
  );
}

// ---- Button ----------------------------------------------------------------- //

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: LucideIcon;
}

const variantClass: Record<Variant, string> = {
  primary: "bg-accent text-white hover:bg-accent-2 focus-visible:ring-accent",
  secondary: "border border-border-strong bg-surface text-ink hover:bg-surface-2 focus-visible:ring-accent",
  ghost: "text-ink-2 hover:bg-surface-2 hover:text-ink focus-visible:ring-accent",
  danger: "bg-status-critical text-white hover:opacity-90 focus-visible:ring-status-critical",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", size = "md", loading = false, icon: Icon, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-surface disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-xs" : "h-10 px-4 text-sm",
        variantClass[variant],
        className,
      )}
      {...rest}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : Icon ? <Icon className="h-4 w-4" aria-hidden /> : null}
      {children}
    </button>
  );
});

// ---- Badge ------------------------------------------------------------------ //

export type BadgeTone = "neutral" | "bot" | "human" | "accent" | "good" | "warning" | "serious" | "critical";

const toneClass: Record<BadgeTone, string> = {
  neutral: "bg-surface-2 text-ink-2 border-border",
  bot: "bg-bot-soft text-ink border-transparent",
  human: "bg-human-soft text-ink border-transparent",
  accent: "bg-accent-soft text-ink border-transparent",
  good: "bg-surface-2 text-ink border-border",
  warning: "bg-surface-2 text-ink border-border",
  serious: "bg-surface-2 text-ink border-border",
  critical: "bg-surface-2 text-ink border-border",
};

export function Badge({ tone = "neutral", className, children, dot }: { tone?: BadgeTone; className?: string; children: ReactNode; dot?: string }) {
  return (
    <span className={clsx("inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-4", toneClass[tone], className)}>
      {dot && <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: dot }} aria-hidden />}
      {children}
    </span>
  );
}

export function PredictionBadge({ label }: { label: "BOT" | "HUMAN" }) {
  return (
    <Badge tone={label === "BOT" ? "bot" : "human"} dot={label === "BOT" ? "var(--bot)" : "var(--human)"}>
      {label}
    </Badge>
  );
}

const RISK_COLOR: Record<string, string> = {
  critical: "var(--status-critical)",
  high: "var(--status-serious)",
  medium: "var(--status-warning)",
  low: "var(--status-good)",
  minimal: "var(--status-good)",
};
const RISK_ICON: Record<string, LucideIcon> = {
  critical: ShieldAlert,
  high: AlertTriangle,
  medium: AlertTriangle,
  low: CheckCircle2,
  minimal: CheckCircle2,
};

export function RiskBadge({ band, score }: { band: string; score?: number }) {
  const Icon = RISK_ICON[band] ?? Info;
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-2 px-2 py-0.5 text-[11px] font-medium text-ink">
      <Icon className="h-3.5 w-3.5" style={{ color: RISK_COLOR[band] }} aria-hidden />
      {band}
      {score !== undefined && <span className="tabular-nums text-ink-2">{score}</span>}
    </span>
  );
}

export function riskColor(band: string): string {
  return RISK_COLOR[band] ?? "var(--text-3)";
}

// ---- States ---------------------------------------------------------------------- //

export function Spinner({ label = "Loading…", className }: { label?: string; className?: string }) {
  return (
    <div className={clsx("flex items-center gap-2 text-sm text-ink-2", className)} role="status" aria-live="polite">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
      {label}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded-md bg-surface-2", className)} aria-hidden />;
}

export function EmptyState({ icon: Icon = Info, title, description, action }: { icon?: LucideIcon; title: string; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border-strong px-6 py-10 text-center">
      <Icon className="h-6 w-6 text-ink-3" aria-hidden />
      <p className="text-sm font-medium text-ink">{title}</p>
      {description && <p className="max-w-md text-xs text-ink-2">{description}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function ErrorState({ message, onRetry, title = "Something went wrong" }: { message: string; onRetry?: () => void; title?: string }) {
  return (
    <div role="alert" className="flex flex-col items-start gap-2 rounded-lg border border-status-critical/40 bg-status-critical/5 px-4 py-3 text-sm">
      <div className="flex items-center gap-2 font-medium text-ink">
        <XCircle className="h-4 w-4" style={{ color: "var(--status-critical)" }} aria-hidden />
        {title}
      </div>
      <p className="text-xs text-ink-2">{message}</p>
      {onRetry && (
        <Button variant="secondary" size="sm" icon={RefreshCw} onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function Notice({ tone = "info", children }: { tone?: "info" | "warning"; children: ReactNode }) {
  const Icon = tone === "warning" ? AlertTriangle : Info;
  return (
    <div className={clsx("flex items-start gap-2 rounded-lg border px-3 py-2 text-xs text-ink-2", tone === "warning" ? "border-status-warning/50 bg-status-warning/10" : "border-border bg-surface-2")}>
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
      <span>{children}</span>
    </div>
  );
}

// ---- Form controls ------------------------------------------------------------------ //

const controlBase =
  "h-10 rounded-lg border border-border-strong bg-surface px-3 text-sm text-ink placeholder:text-ink-3 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30 disabled:opacity-50";

/** Full width unless the caller passes an explicit width utility. */
const controlClass = (className?: string) => clsx(controlBase, !/w-/.test(className ?? "") && "w-full", className);

export function Field({ label, hint, children, className }: { label: ReactNode; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={clsx("flex flex-col gap-1", className)}>
      <span className="text-xs font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="text-[11px] text-ink-3">{hint}</span>}
    </label>
  );
}

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...rest }, ref) {
  return <input ref={ref} className={controlClass(className)} {...rest} />;
});

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select({ className, children, ...rest }, ref) {
  return (
    <select ref={ref} className={clsx(controlClass(className), "pr-8")} {...rest}>
      {children}
    </select>
  );
});

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea({ className, ...rest }, ref) {
  return <textarea ref={ref} className={clsx(controlClass(className), "h-auto min-h-[88px] py-2")} {...rest} />;
});

export function Toggle({ label, checked, onChange, hint }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; hint?: ReactNode }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-3 rounded-lg border border-border px-3 py-2">
      <span className="min-w-0">
        <span className="block text-sm text-ink">{label}</span>
        {hint && <span className="block text-[11px] text-ink-3">{hint}</span>}
      </span>
      <span className="relative mt-0.5 inline-flex h-5 w-9 shrink-0 items-center">
        <input type="checkbox" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="h-5 w-9 rounded-full bg-border-strong transition-colors peer-checked:bg-accent peer-focus-visible:ring-2 peer-focus-visible:ring-accent/40" aria-hidden />
        <span className="absolute left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform peer-checked:translate-x-4" aria-hidden />
      </span>
    </label>
  );
}

// ---- Tabs -------------------------------------------------------------------------- //

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { key: T; label: ReactNode }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div role="tablist" className="flex flex-wrap gap-1 rounded-lg bg-surface-2 p-1">
      {tabs.map((t) => (
        <button
          key={t.key}
          role="tab"
          aria-selected={value === t.key}
          onClick={() => onChange(t.key)}
          className={clsx(
            "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            value === t.key ? "bg-surface text-ink shadow-sm" : "text-ink-2 hover:text-ink",
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

// ---- Stat tile ---------------------------------------------------------------------- //

export function StatTile({ label, value, sub, icon: Icon, accent }: { label: string; value: ReactNode; sub?: ReactNode; icon?: LucideIcon; accent?: string }) {
  return (
    <div className="flex items-start justify-between gap-3 rounded-xl border border-border bg-surface p-4 shadow-sm">
      <div className="min-w-0">
        <p className="text-xs text-ink-2">{label}</p>
        <p className="mt-1 truncate text-2xl font-semibold text-ink">{value}</p>
        {sub && <p className="mt-1 text-[11px] text-ink-3">{sub}</p>}
      </div>
      {Icon && (
        <span className="rounded-lg bg-surface-2 p-2" style={{ color: accent ?? "var(--text-2)" }}>
          <Icon className="h-4 w-4" aria-hidden />
        </span>
      )}
    </div>
  );
}

// ---- Table ----------------------------------------------------------------------------- //

export function Table({ children, className, compact = false }: { children: ReactNode; className?: string; compact?: boolean }) {
  return (
    <div className={clsx("scrollbar-thin overflow-x-auto rounded-lg border border-border", className)}>
      <table className={clsx("w-full border-collapse text-left text-sm", !compact && "min-w-[560px]")}>{children}</table>
    </div>
  );
}

export function Th({ children, className, align = "left" }: { children?: ReactNode; className?: string; align?: "left" | "right" }) {
  return <th className={clsx("border-b border-border bg-surface-2 px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-ink-2", align === "right" && "text-right", className)}>{children}</th>;
}

export function Td({ children, className, align = "left", mono }: { children?: ReactNode; className?: string; align?: "left" | "right"; mono?: boolean }) {
  return <td className={clsx("border-b border-border px-3 py-2 text-ink", align === "right" && "text-right", mono && "font-mono tabular-nums text-xs", className)}>{children}</td>;
}

// ---- Section header ----------------------------------------------------------------------- //

export function PageHeader({ title, description, actions, badge }: { title: string; description?: ReactNode; actions?: ReactNode; badge?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold tracking-tight text-ink">{title}</h1>
          {badge}
        </div>
        {description && <p className="mt-1 max-w-3xl text-sm text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function SourceTag({ kind }: { kind: "paper" | "ours" }) {
  return kind === "paper" ? <Badge tone="neutral">Research paper results</Badge> : <Badge tone="accent">Your model performance</Badge>;
}

const JOB_TONE: Record<string, BadgeTone> = { QUEUED: "neutral", PROCESSING: "accent", COMPLETED: "good", FAILED: "critical", CANCELLED: "neutral", READY: "good", PRODUCTION: "accent", TRAINING: "neutral", DEPRECATED: "neutral", VALIDATED: "good", INVALID: "critical", UPLOADED: "neutral", ACTIVE: "good", DISABLED: "neutral" };
const JOB_DOT: Record<string, string> = { QUEUED: "var(--text-3)", PROCESSING: "var(--accent)", COMPLETED: "var(--status-good)", FAILED: "var(--status-critical)", READY: "var(--status-good)", PRODUCTION: "var(--accent)", VALIDATED: "var(--status-good)", INVALID: "var(--status-critical)", ACTIVE: "var(--status-good)", DISABLED: "var(--text-3)" };

export function StatusBadge({ status }: { status: string }) {
  return (
    <Badge tone={JOB_TONE[status] ?? "neutral"} dot={JOB_DOT[status]}>
      {status.toLowerCase()}
    </Badge>
  );
}
