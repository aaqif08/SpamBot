/** Typed HTTP client for the BotShield AI API (v1) with bearer auth + silent refresh. */

import type {
  AccountInput,
  AnalysisResponse,
  AuditResponse,
  BatchResponse,
  BenchmarkStatus,
  DashboardResponse,
  DatasetPublic,
  DatasetVersionPublic,
  EvaluationResult,
  GlobalExplanationResponse,
  HealthResponse,
  HistoryFilters,
  HistoryResponse,
  JobResponse,
  LocalExplanationResponse,
  ModelEvaluationResponse,
  ModelListResponse,
  ModelPublic,
  OrganizationPublic,
  ProviderFetchResponse,
  ProviderInfo,
  ReadinessResponse,
  ResearchResponse,
  RuntimeInfo,
  SetupStatus,
  TokenResponse,
  TrainRequest,
  TrainSubmitted,
  UserPublic,
} from "@/types/api";

export const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";
const V1 = `${API_BASE}/api/v1`;

export const BACKEND_UNAVAILABLE = "The API is not reachable. Check that the backend is running and the API base URL is correct.";

export interface ApiErrorBody {
  detail: string | Record<string, unknown>;
  code?: string;
  errors?: { loc: string[]; msg: string; type: string }[];
}

export class ApiRequestError extends Error {
  status: number;
  code: string;
  errors?: ApiErrorBody["errors"];
  retryAfter?: number;

  constructor(status: number, body: Partial<ApiErrorBody> | null, fallback: string, retryAfter?: number) {
    const detail = body?.detail;
    super(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : fallback);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = body?.code ?? `http_${status}`;
    this.errors = body?.errors;
    this.retryAfter = retryAfter;
  }
}

export function isApiError(e: unknown): e is ApiRequestError {
  return e instanceof ApiRequestError;
}

export function errorMessage(e: unknown): string {
  if (isApiError(e)) {
    if (e.errors?.length) return `${e.message}: ${e.errors.map((x) => `${x.loc.slice(1).join(".")} — ${x.msg}`).join("; ")}`;
    return e.message;
  }
  if (e instanceof TypeError) return BACKEND_UNAVAILABLE;
  if (e instanceof Error) return e.message;
  return "Unknown error";
}

// ---- token store (memory only; refresh token lives in an httpOnly cookie) ---- //

let accessToken: string | null = null;
let onUnauthorized: (() => void) | null = null;
let refreshInFlight: Promise<TokenResponse | null> | null = null;

export const tokenStore = {
  get: () => accessToken,
  set: (t: string | null) => {
    accessToken = t;
  },
  onUnauthorized: (fn: (() => void) | null) => {
    onUnauthorized = fn;
  },
};

async function rawRequest<T>(path: string, init: RequestInit = {}, withAuth = true): Promise<T> {
  const headers = new Headers(init.headers ?? {});
  if (withAuth && accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
  const res = await fetch(`${V1}${path}`, { ...init, headers, credentials: "include" });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!res.ok) {
    // A non-JSON error body means the request never reached the BotShield API
    // (proxy target wrong, service asleep, gateway error) rather than an API-level rejection.
    const fallback = body === null && (res.status >= 500 || res.status === 404) ? BACKEND_UNAVAILABLE : `Request failed (${res.status})`;
    const ra = res.headers.get("Retry-After");
    throw new ApiRequestError(res.status, body as Partial<ApiErrorBody> | null, fallback, ra ? Number(ra) : undefined);
  }
  return body as T;
}

async function refreshSession(): Promise<TokenResponse | null> {
  if (!refreshInFlight) {
    refreshInFlight = rawRequest<TokenResponse>("/auth/refresh", { method: "POST" }, false)
      .then((t) => {
        accessToken = t.access_token;
        return t;
      })
      .catch(() => null)
      .finally(() => {
        refreshInFlight = null;
      });
  }
  return refreshInFlight;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  try {
    return await rawRequest<T>(path, init);
  } catch (e) {
    if (isApiError(e) && e.status === 401 && !path.startsWith("/auth/")) {
      const refreshed = await refreshSession();
      if (refreshed) return rawRequest<T>(path, init);
      accessToken = null;
      onUnauthorized?.();
    }
    throw e;
  }
}

function json<T>(path: string, method: string, payload?: unknown): Promise<T> {
  return request<T>(path, { method, headers: { "Content-Type": "application/json" }, body: payload === undefined ? undefined : JSON.stringify(payload) });
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const sp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null || v === "") return;
    sp.set(k, String(v));
  });
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export const api = {
  // system
  health: () => rawRequest<HealthResponse>("/health", {}, false),
  ready: () => rawRequest<ReadinessResponse>("/health/ready", {}, false),
  runtime: () => request<RuntimeInfo>("/runtime"),
  dashboard: () => request<DashboardResponse>("/dashboard"),
  research: () => rawRequest<ResearchResponse>("/research", {}, false),
  features: () => request<{ feature_version: string; n_features: number; groups: Record<string, string[]>; descriptions: Record<string, string>; order: string[] }>("/features"),

  // auth
  setupStatus: () => rawRequest<SetupStatus>("/auth/setup-status", {}, false),
  login: (email: string, password: string) => rawRequest<TokenResponse>("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) }, false),
  refresh: refreshSession,
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  me: () => request<UserPublic>("/auth/me"),
  changePassword: (current_password: string, new_password: string) => json<void>("/auth/change-password", "POST", { current_password, new_password }),
  requestPasswordReset: (email: string) => rawRequest<{ detail: string }>("/auth/password-reset/request", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email }) }, false),
  confirmPasswordReset: (token: string, new_password: string) => rawRequest<void>("/auth/password-reset/confirm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token, new_password }) }, false),

  // users / organisation (ADMIN)
  users: () => request<UserPublic[]>("/users"),
  createUser: (body: { email: string; password: string; full_name?: string; role: string }) => json<UserPublic>("/users", "POST", body),
  updateUser: (id: string, body: { role?: string; status?: string; full_name?: string }) => json<UserPublic>(`/users/${encodeURIComponent(id)}`, "PATCH", body),
  adminResetPassword: (id: string, new_password: string) => json<void>(`/users/${encodeURIComponent(id)}/reset-password`, "POST", { new_password }),
  organization: () => request<OrganizationPublic>("/organization"),
  updateOrganization: (name: string) => json<OrganizationPublic>("/organization", "PATCH", { name }),
  audit: (params: { action?: string; actor?: string; page?: number; page_size?: number } = {}) => request<AuditResponse>(`/audit${qs(params)}`),

  // analyses
  analyze: (account: AccountInput, opts: { model_id?: string | null; source?: "manual" | "x_api"; explain?: boolean } = {}) =>
    json<AnalysisResponse>("/analyses", "POST", { account, model_id: opts.model_id ?? null, source: opts.source ?? "manual", explain: opts.explain ?? true }),
  history: (filters: HistoryFilters = {}) => request<HistoryResponse>(`/analyses${qs(filters as Record<string, string | number | boolean | undefined>)}`),
  analysis: (id: string) => request<AnalysisResponse>(`/analyses/${encodeURIComponent(id)}`),
  deleteAnalysis: (id: string) => request<void>(`/analyses/${encodeURIComponent(id)}`, { method: "DELETE" }),
  explanation: (id: string, method: "shap" | "lime") => request<LocalExplanationResponse>(`/analyses/${encodeURIComponent(id)}/explanations/${method}`),
  purgeAnalyses: (older_than_days: number) => json<{ deleted: number }>("/analyses/retention/purge", "POST", { older_than_days }),

  // batches
  createBatchFile: (file: File, opts: { model_id?: string | null; name?: string } = {}) => {
    const fd = new FormData();
    fd.append("file", file);
    if (opts.model_id) fd.append("model_id", opts.model_id);
    if (opts.name) fd.append("name", opts.name);
    return request<BatchResponse>("/batches", { method: "POST", body: fd });
  },
  createBatchDataset: (dataset_id: string, opts: { model_id?: string | null; name?: string } = {}) => {
    const fd = new FormData();
    fd.append("dataset_id", dataset_id);
    if (opts.model_id) fd.append("model_id", opts.model_id);
    if (opts.name) fd.append("name", opts.name);
    return request<BatchResponse>("/batches", { method: "POST", body: fd });
  },
  batches: () => request<BatchResponse[]>("/batches"),
  batch: (id: string) => request<BatchResponse>(`/batches/${encodeURIComponent(id)}`),
  deleteBatch: (id: string) => request<void>(`/batches/${encodeURIComponent(id)}`, { method: "DELETE" }),
  downloadBatch: async (id: string): Promise<Blob> => {
    const headers = new Headers();
    if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
    const res = await fetch(`${V1}/batches/${encodeURIComponent(id)}/download`, { headers, credentials: "include" });
    if (!res.ok) throw new ApiRequestError(res.status, null, `Download failed (${res.status})`);
    return res.blob();
  },

  // datasets
  datasets: () => request<DatasetPublic[]>("/datasets"),
  dataset: (id: string) => request<DatasetPublic>(`/datasets/${encodeURIComponent(id)}`),
  datasetVersion: (datasetId: string, versionId: string) => request<DatasetVersionPublic>(`/datasets/${encodeURIComponent(datasetId)}/versions/${encodeURIComponent(versionId)}`),
  uploadDataset: (file: File, opts: { name?: string; description?: string; dataset_id?: string } = {}) => {
    const fd = new FormData();
    fd.append("file", file);
    if (opts.name) fd.append("name", opts.name);
    if (opts.description) fd.append("description", opts.description);
    if (opts.dataset_id) fd.append("dataset_id", opts.dataset_id);
    return request<DatasetPublic>("/datasets", { method: "POST", body: fd });
  },
  deleteDataset: (id: string) => request<void>(`/datasets/${encodeURIComponent(id)}`, { method: "DELETE" }),
  evaluateDataset: (id: string, model_id?: string | null) => json<EvaluationResult>(`/datasets/${encodeURIComponent(id)}/evaluate`, "POST", { model_id: model_id ?? null }),
  benchmarks: () => request<BenchmarkStatus[]>("/datasets/benchmarks"),
  importBenchmark: (kind: string) => request<JobResponse>(`/datasets/benchmarks/${encodeURIComponent(kind)}/import`, { method: "POST" }),
  combinedBenchmark: () => request<DatasetPublic>("/datasets/benchmarks/combined", { method: "POST" }),

  // models / training
  models: () => request<ModelListResponse>("/models"),
  model: (id: string) => request<ModelPublic>(`/models/${encodeURIComponent(id)}`),
  train: (body: TrainRequest) => json<TrainSubmitted>("/models/train", "POST", body),
  activateModel: (id: string) => request<ModelPublic>(`/models/${encodeURIComponent(id)}/activate`, { method: "POST" }),
  deprecateModel: (id: string) => request<ModelPublic>(`/models/${encodeURIComponent(id)}/deprecate`, { method: "POST" }),
  deleteModel: (id: string) => request<void>(`/models/${encodeURIComponent(id)}`, { method: "DELETE" }),
  modelEvaluation: (id: string) => request<ModelEvaluationResponse>(`/models/${encodeURIComponent(id)}/evaluation`),
  globalExplanation: (id: string) => request<GlobalExplanationResponse>(`/models/${encodeURIComponent(id)}/explanation`),

  // jobs
  jobs: (job_type?: string) => request<JobResponse[]>(`/jobs${qs({ job_type })}`),
  job: (id: string) => request<JobResponse>(`/jobs/${encodeURIComponent(id)}`),

  // providers
  providers: () => request<ProviderInfo[]>("/providers"),
  providerFetch: (name: string, identifier: string) => json<ProviderFetchResponse>(`/providers/${encodeURIComponent(name)}/fetch`, "POST", { identifier }),
};

export type Api = typeof api;
