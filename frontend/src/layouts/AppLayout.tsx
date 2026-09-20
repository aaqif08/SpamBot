import clsx from "clsx";
import {
  Activity,
  BarChart3,
  BookOpen,
  Boxes,
  Code2,
  Database,
  FlaskConical,
  History,
  Layers,
  LayoutDashboard,
  Lightbulb,
  Menu,
  Moon,
  ScanSearch,
  Settings,
  ShieldCheck,
  Sun,
  X,
} from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { Badge } from "@/components/ui";
import { useHealth } from "@/hooks/useHealth";
import { useTheme } from "@/hooks/useTheme";
import { API_BASE } from "@/services/api";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/analyze", label: "Analyze Account", icon: ScanSearch },
  { to: "/batch-analysis", label: "Batch Analysis", icon: Layers },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/training", label: "Training", icon: FlaskConical },
  { to: "/evaluation", label: "Evaluation", icon: BarChart3 },
  { to: "/explainability", label: "Explainability", icon: Lightbulb },
  { to: "/history", label: "History", icon: History },
  { to: "/research", label: "Research", icon: BookOpen },
  { to: "/architecture", label: "Architecture", icon: Activity },
  { to: "/api-docs", label: "API Docs", icon: Code2 },
  { to: "/settings", label: "Settings", icon: Settings },
];

function BackendStatus() {
  const health = useHealth();
  if (health.loading) return <Badge tone="neutral">Connecting…</Badge>;
  if (health.error || !health.data) return <Badge tone="critical" dot="var(--status-critical)">Backend offline</Badge>;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge tone="good" dot="var(--status-good)">API v{health.data.version}</Badge>
      {health.data.model_available ? (
        <Badge tone={health.data.active_model?.is_demo ? "demo" : "accent"}>
          Model: {health.data.active_model?.name}
          {health.data.active_model?.is_demo ? " (demo)" : ""}
        </Badge>
      ) : (
        <Badge tone="warning" dot="var(--status-warning)">No trained model</Badge>
      )}
    </div>
  );
}

export function AppLayout() {
  const { theme, toggle } = useTheme();
  const [open, setOpen] = useState(false);

  const nav = (
    <nav className="flex flex-col gap-0.5 px-3" aria-label="Main">
      {NAV.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            clsx(
              "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
              isActive ? "bg-accent-soft font-medium text-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink",
            )
          }
        >
          <item.icon className="h-4 w-4 shrink-0" aria-hidden />
          {item.label}
        </NavLink>
      ))}
    </nav>
  );

  return (
    <div className="flex min-h-screen bg-bg">
      {/* Sidebar */}
      <aside className="hidden w-60 shrink-0 flex-col border-r border-border bg-surface lg:flex">
        <div className="flex items-center gap-2 px-5 py-5">
          <span className="rounded-lg bg-accent p-1.5 text-white">
            <ShieldCheck className="h-5 w-5" aria-hidden />
          </span>
          <div className="leading-tight">
            <div className="text-sm font-semibold text-ink">BotShield AI</div>
            <div className="text-[11px] text-ink-3">Interpretable bot detection</div>
          </div>
        </div>
        {nav}
        <div className="mt-auto px-5 py-4 text-[11px] text-ink-3">
          Final-year project · base paper: IEEE Access 2025
          <br />
          DOI 10.1109/ACCESS.2025.3551993
        </div>
      </aside>

      {/* Mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} aria-hidden />
          <aside className="absolute inset-y-0 left-0 w-64 border-r border-border bg-surface py-4">
            <div className="mb-3 flex items-center justify-between px-5">
              <span className="text-sm font-semibold text-ink">BotShield AI</span>
              <button onClick={() => setOpen(false)} aria-label="Close menu" className="rounded-md p-1 text-ink-2 hover:bg-surface-2">
                <X className="h-4 w-4" />
              </button>
            </div>
            {nav}
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-surface/90 px-4 backdrop-blur sm:px-6">
          <button onClick={() => setOpen(true)} className="rounded-md p-1.5 text-ink-2 hover:bg-surface-2 lg:hidden" aria-label="Open menu">
            <Menu className="h-5 w-5" />
          </button>
          <div className="hidden text-xs text-ink-3 sm:block">
            API base: <code className="font-mono">{API_BASE || "(dev proxy → /api)"}</code>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <BackendStatus />
            <button onClick={toggle} className="rounded-md border border-border p-1.5 text-ink-2 hover:bg-surface-2" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} title="Toggle theme">
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </header>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
