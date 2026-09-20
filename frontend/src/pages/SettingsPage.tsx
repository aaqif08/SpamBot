import { KeyRound, Plug, Shield, Trash2, UserPlus, Users } from "lucide-react";
import { useCallback, useState } from "react";

import { Badge, Button, Card, EmptyState, ErrorState, Field, Input, Notice, PageHeader, Select, StatusBadge, Table, Tabs, Td, Th } from "@/components/ui";
import { ConfirmDialog } from "@/components/ui/Dialog";
import { useAction, useApi } from "@/hooks/useApi";
import { useAuth } from "@/hooks/useAuth";
import { api } from "@/services/api";
import type { RoleName, UserPublic } from "@/types/api";
import { dateTime } from "@/utils/format";

function ProfileTab() {
  const { user, refreshUser } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const change = useAction(useCallback((c: string, n: string) => api.changePassword(c, n), []));
  const [done, setDone] = useState(false);
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Your account">
        <dl className="grid grid-cols-[8rem_1fr] gap-y-2 text-sm">
          <dt className="text-ink-3">Email</dt><dd className="text-ink">{user?.email}</dd>
          <dt className="text-ink-3">Name</dt><dd className="text-ink">{user?.full_name || "—"}</dd>
          <dt className="text-ink-3">Role</dt><dd><Badge tone="good">{user?.role}</Badge></dd>
          <dt className="text-ink-3">Organization</dt><dd className="text-ink">{user?.organization_name}</dd>
          <dt className="text-ink-3">Last sign-in</dt><dd className="text-ink">{dateTime(user?.last_login_at)}</dd>
        </dl>
      </Card>
      <Card title="Change password" subtitle="Changing the password signs out every other session.">
        <div className="space-y-3">
          <Field label="Current password"><Input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} /></Field>
          <Field label="New password" hint="At least 12 characters with upper, lower, digit and symbol."><Input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} /></Field>
          <Field label="Confirm new password"><Input type="password" autoComplete="new-password" value={confirm} onChange={(e) => setConfirm(e.target.value)} /></Field>
          <Button icon={KeyRound} loading={change.loading} disabled={!current || !next || next !== confirm} onClick={async () => { setDone(false); const r = await change.run(current, next); if (r !== null) { setDone(true); setCurrent(""); setNext(""); setConfirm(""); await refreshUser(); } }}>Update password</Button>
          {next && confirm && next !== confirm && <p className="text-xs text-status-critical">Passwords do not match.</p>}
          {change.error && <ErrorState message={change.error} />}
          {done && <Notice>Password updated.</Notice>}
        </div>
      </Card>
    </div>
  );
}

function UsersTab() {
  const { user: me } = useAuth();
  const users = useApi(() => api.users(), []);
  const [form, setForm] = useState({ email: "", full_name: "", password: "", role: "ANALYST" as RoleName });
  const create = useAction(useCallback((b: typeof form) => api.createUser(b), []));
  const update = useAction(useCallback((id: string, body: { role?: string; status?: string }) => api.updateUser(id, body), []));
  const reset = useAction(useCallback((id: string, pw: string) => api.adminResetPassword(id, pw), []));
  const [resetFor, setResetFor] = useState<UserPublic | null>(null);
  const [resetPw, setResetPw] = useState("");
  return (
    <div className="space-y-4">
      <ConfirmDialog open={!!resetFor} title={`Set a new password for ${resetFor?.email}`} description="All of this user's sessions are revoked. Share the password through a secure channel; it is never shown again." confirmLabel="Set password" loading={reset.loading} onCancel={() => { setResetFor(null); setResetPw(""); }} onConfirm={async () => { if (resetFor) { const r = await reset.run(resetFor.id, resetPw); if (r !== null) { setResetFor(null); setResetPw(""); } } }}>
        <Field label="New password"><Input type="password" autoComplete="new-password" value={resetPw} onChange={(e) => setResetPw(e.target.value)} /></Field>
        {reset.error && <p className="mt-2 text-xs text-status-critical">{reset.error}</p>}
      </ConfirmDialog>
      <Card title="Invite a user" subtitle="Roles: ADMIN manages users, models and retention · ANALYST runs analyses, batches and training · VIEWER reads everything.">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <Field label="Email"><Input type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} /></Field>
          <Field label="Full name"><Input value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} /></Field>
          <Field label="Initial password"><Input type="password" autoComplete="new-password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></Field>
          <Field label="Role"><Select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as RoleName })}><option>ADMIN</option><option>ANALYST</option><option>VIEWER</option></Select></Field>
          <div className="flex items-end"><Button icon={UserPlus} loading={create.loading} disabled={!form.email || !form.password} onClick={async () => { const u = await create.run(form); if (u) { setForm({ email: "", full_name: "", password: "", role: "ANALYST" }); users.reload(); } }}>Create user</Button></div>
        </div>
        {create.error && <div className="mt-2"><ErrorState message={create.error} /></div>}
      </Card>
      <Card title="Members" padded={false}>
        {users.error && <div className="p-4"><ErrorState message={users.error} onRetry={users.reload} /></div>}
        {update.error && <div className="p-4"><ErrorState message={update.error} /></div>}
        {users.data && (
          <Table className="rounded-none border-0" compact>
            <thead><tr><Th>User</Th><Th>Role</Th><Th>Status</Th><Th>Last sign-in</Th><Th>Created</Th><Th></Th></tr></thead>
            <tbody>{users.data.map((u) => (
              <tr key={u.id}>
                <Td><div className="text-ink">{u.full_name || u.email}</div><div className="text-xs text-ink-3">{u.email}{u.id === me?.id ? " (you)" : ""}</div></Td>
                <Td><Select className="h-8 w-32 text-xs" value={u.role} disabled={u.id === me?.id} onChange={async (e) => { await update.run(u.id, { role: e.target.value }); users.reload(); }}><option>ADMIN</option><option>ANALYST</option><option>VIEWER</option></Select></Td>
                <Td><StatusBadge status={u.status} /></Td><Td className="text-xs">{dateTime(u.last_login_at)}</Td><Td className="text-xs">{dateTime(u.created_at)}</Td>
                <Td><div className="flex justify-end gap-1">
                  <Button size="sm" variant="ghost" icon={KeyRound} onClick={() => setResetFor(u)}>Reset password</Button>
                  {u.id !== me?.id && <Button size="sm" variant="ghost" onClick={async () => { await update.run(u.id, { status: u.status === "ACTIVE" ? "DISABLED" : "ACTIVE" }); users.reload(); }}>{u.status === "ACTIVE" ? "Disable" : "Enable"}</Button>}
                </div></Td>
              </tr>
            ))}</tbody>
          </Table>
        )}
      </Card>
    </div>
  );
}

function OrganizationTab() {
  const { refreshUser } = useAuth();
  const org = useApi(() => api.organization(), []);
  const [name, setName] = useState<string | null>(null);
  const save = useAction(useCallback((n: string) => api.updateOrganization(n), []));
  const purge = useAction(useCallback((days: number) => api.purgeAnalyses(days), []));
  const [days, setDays] = useState(365);
  const [confirmPurge, setConfirmPurge] = useState(false);
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <ConfirmDialog open={confirmPurge} title={`Delete predictions older than ${days} days?`} description="Stored predictions and their explanations older than the cut-off are permanently removed. Models, datasets and audit records are kept." confirmLabel="Purge" destructive loading={purge.loading} onCancel={() => setConfirmPurge(false)} onConfirm={async () => { await purge.run(days); setConfirmPurge(false); }} />
      <Card title="Organization">
        {org.error && <ErrorState message={org.error} onRetry={org.reload} />}
        {org.data && (
          <div className="space-y-3">
            <Field label="Name"><Input value={name ?? org.data.name} onChange={(e) => setName(e.target.value)} maxLength={120} /></Field>
            <p className="text-xs text-ink-3">Slug <code className="font-mono">{org.data.slug}</code> · created {dateTime(org.data.created_at)}</p>
            <Button loading={save.loading} disabled={name === null || name === org.data.name} onClick={async () => { const r = await save.run(name ?? ""); if (r) { setName(null); org.reload(); await refreshUser(); } }}>Save</Button>
            {save.error && <ErrorState message={save.error} />}
          </div>
        )}
      </Card>
      <Card title="Data retention" subtitle="Manual purge of old predictions. Automatic retention runs only when BOTSHIELD_PREDICTION_RETENTION_DAYS is set on the server.">
        <div className="flex flex-wrap items-end gap-2">
          <Field label="Older than (days)"><Input type="number" min={1} max={3650} value={days} onChange={(e) => setDays(Number(e.target.value))} className="w-32" /></Field>
          <Button variant="danger" icon={Trash2} onClick={() => setConfirmPurge(true)} disabled={days < 1}>Purge predictions</Button>
        </div>
        {purge.error && <div className="mt-2"><ErrorState message={purge.error} /></div>}
        {purge.result && <p className="mt-2 text-sm text-ink-2">{purge.result.deleted} prediction(s) deleted.</p>}
      </Card>
    </div>
  );
}

function IntegrationsTab() {
  const providers = useApi(() => api.providers(), []);
  const runtime = useApi(() => api.runtime(), []);
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Data providers" subtitle="Providers convert platform data into the account schema. Configuration is server-side only (environment variables); nothing is stored in the browser.">
        {providers.error && <ErrorState message={providers.error} onRetry={providers.reload} />}
        {providers.data && (
          <ul className="space-y-2">{providers.data.map((p) => (
            <li key={p.name} className="rounded-lg border border-border p-3 text-sm">
              <div className="flex items-center justify-between gap-2"><span className="font-medium text-ink">{p.name}</span>{p.configured ? <Badge tone="good">configured</Badge> : <Badge tone="neutral">not configured</Badge>}</div>
              <p className="mt-1 text-xs text-ink-2">{p.description}</p>
              {!p.configured && p.configuration_hint && <p className="mt-1 text-xs text-ink-3">{p.configuration_hint}</p>}
            </li>
          ))}</ul>
        )}
      </Card>
      <Card title="Server runtime" subtitle="Read-only information reported by the API.">
        {runtime.error && <ErrorState message={runtime.error} onRetry={runtime.reload} />}
        {runtime.data && (
          <dl className="grid grid-cols-[10rem_1fr] gap-y-1.5 text-sm">
            <dt className="text-ink-3">Version</dt><dd className="text-ink">{runtime.data.version}</dd>
            <dt className="text-ink-3">Environment</dt><dd className="text-ink">{runtime.data.environment}</dd>
            <dt className="text-ink-3">Job backend</dt><dd className="text-ink">{runtime.data.job_backend}</dd>
            <dt className="text-ink-3">Storage backend</dt><dd className="text-ink">{runtime.data.storage_backend}</dd>
            <dt className="text-ink-3">Feature version</dt><dd className="text-ink">{runtime.data.feature_version} ({runtime.data.n_features} features)</dd>
            <dt className="text-ink-3">Python</dt><dd className="text-ink">{runtime.data.python}</dd>
            {Object.entries(runtime.data.libraries).map(([k, v]) => <><dt key={`${k}-k`} className="text-ink-3">{k}</dt><dd key={`${k}-v`} className="font-mono text-xs text-ink">{v}</dd></>)}
          </dl>
        )}
      </Card>
    </div>
  );
}

function AuditTab() {
  const [page, setPage] = useState(1);
  const [action, setAction] = useState("");
  const audit = useApi(() => api.audit({ page, page_size: 50, action: action || undefined }), [page, action]);
  const pages = audit.data ? Math.max(1, Math.ceil(audit.data.total / audit.data.page_size)) : 1;
  return (
    <Card title="Audit log" subtitle="Security-relevant actions for this organization. Secrets are never recorded." padded={false} actions={<Input placeholder="filter by action, e.g. auth.login" value={action} onChange={(e) => { setAction(e.target.value); setPage(1); }} className="h-8 w-56 text-xs" />}>
      {audit.error && <div className="p-4"><ErrorState message={audit.error} onRetry={audit.reload} /></div>}
      {audit.data && audit.data.items.length === 0 && <div className="p-4"><EmptyState icon={Shield} title="No audit entries" /></div>}
      {audit.data && audit.data.items.length > 0 && (
        <Table className="rounded-none border-0" compact>
          <thead><tr><Th>Time</Th><Th>Action</Th><Th>Actor</Th><Th>Target</Th><Th>Outcome</Th><Th>IP</Th><Th>Details</Th></tr></thead>
          <tbody>{audit.data.items.map((a) => <tr key={a.id}><Td className="text-xs">{dateTime(a.created_at)}</Td><Td mono className="text-xs">{a.action}</Td><Td className="text-xs">{a.actor_email || "—"}</Td><Td className="text-xs">{a.target_type}{a.target_id ? ` ${a.target_id.slice(0, 8)}` : ""}</Td><Td><Badge tone={a.outcome === "success" ? "good" : "critical"}>{a.outcome}</Badge></Td><Td mono className="text-xs">{a.ip_address}</Td><Td className="max-w-[20rem] truncate font-mono text-[11px] text-ink-2" >{Object.keys(a.details).length ? JSON.stringify(a.details) : ""}</Td></tr>)}</tbody>
        </Table>
      )}
      {audit.data && pages > 1 && <div className="flex items-center justify-end gap-2 border-t border-border px-4 py-2"><span className="text-xs text-ink-2">page {page} / {pages}</span><Button size="sm" variant="secondary" disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous</Button><Button size="sm" variant="secondary" disabled={page >= pages} onClick={() => setPage(page + 1)}>Next</Button></div>}
    </Card>
  );
}

type TabKey = "profile" | "users" | "organization" | "integrations" | "audit";

export function SettingsPage() {
  const { hasRole } = useAuth();
  const isAdmin = hasRole("ADMIN");
  const [tab, setTab] = useState<TabKey>("profile");
  const tabs: { key: TabKey; label: string }[] = [{ key: "profile", label: "Profile" }, { key: "integrations", label: "Integrations" }, ...(isAdmin ? [{ key: "users" as TabKey, label: "Users" }, { key: "organization" as TabKey, label: "Organization & retention" }, { key: "audit" as TabKey, label: "Audit log" }] : [])];
  return (
    <div className="space-y-6">
      <PageHeader title="Settings" description="Account, organization, users and integrations. Only settings the server actually enforces are shown." badge={isAdmin ? <Badge tone="good"><Users className="mr-1 inline h-3 w-3" />admin</Badge> : undefined} />
      <Tabs tabs={tabs} value={tab} onChange={setTab} />
      {tab === "profile" && <ProfileTab />}
      {tab === "integrations" && <IntegrationsTab />}
      {tab === "users" && isAdmin && <UsersTab />}
      {tab === "organization" && isAdmin && <OrganizationTab />}
      {tab === "audit" && isAdmin && <AuditTab />}
      <p className="text-[11px] text-ink-3"><Plug className="mr-1 inline h-3 w-3" />External API credentials, storage and job backends are configured through server environment variables — see docs/deployment.md.</p>
    </div>
  );
}
