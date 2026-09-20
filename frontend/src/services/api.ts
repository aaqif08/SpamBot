/** Typed HTTP client for the BotShield AI backend. */

import type {
  AccountInput,
  AdapterFetchResponse,
  AdapterInfo,
  ApiError,
  BatchListItem,
  BatchSummary,
  CresciStatus,
  DashboardResponse,
  DatasetDetail,
  DatasetEvaluationResponse,
  DatasetInfo,
  EvaluationResponse,
  FeaturesResponse,
  GlobalExplanationResponse,
  HealthResponse,
  HistoryFilters,
  HistoryResponse,
  LocalExplanationResponse,
  ModelListResponse,
  PredictResponse,
  ResearchResponse,
  SampleAccount,
  TrainingRunInfo,
  TrainJobResponse,
  TrainRequest,
  TrainStatusResponse,
} from "@/types/api";

export const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export const BACKEND_UNAVAILABLE = "Backend unavailable. Start the API server (uvicorn app.main:app) and retry.";

export class ApiRequestError extends Error {
  status: number;
  code: string;
  errors?: ApiError["errors"];

  constructor(status: number, body: Partial<ApiError> | null, fallback: string) {
    super(body?.detail ?? fallback);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = body?.code ?? `http_${status}`;
    this.errors = body?.errors;
  }
}

export function isApiError(e: unknown): e is ApiRequestError {
  return e instanceof ApiRequestError;
}

export function errorMessage(e: unknown): string {
  if (isApiError(e)) {
    if (e.errors?.length) {
      return `${e.message}: ${e.errors.map((x) => `${x.loc.slice(1).join(".")} — ${x.msg}`).join("; ")}`;
    }
    return e.message;
  }
  if (e instanceof TypeError) return BACKEND_UNAVAILABLE;
  if (e instanceof Error) return e.message;
  return "Unknown error";
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}/api${path}`;
  const res = await fetch(url, init);
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!res.ok) {
    // A non-JSON 5xx comes from the dev proxy / reverse proxy, i.e. the API process is not reachable.
    const fallback = body === null && res.status >= 500 ? BACKEND_UNAVAILABLE : `Request failed (${res.status})`;
    throw new ApiRequestError(res.status, body as Partial<ApiError> | null, fallback);
  }
  return body as T;
}

function json<T>(path: string, method: string, payload?: unknown): Promise<T> {
  return request<T>(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
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
  health: () => request<HealthResponse>("/health"),
  dashboard: () => request<DashboardResponse>("/dashboard"),
  features: () => request<FeaturesResponse>("/features"),
  research: () => request<ResearchResponse>("/research"),

  models: () => request<ModelListResponse>("/models"),
  activateModel: (id: string) => json<{ active_model_id: string }>(`/models/${encodeURIComponent(id)}/activate`, "POST"),
  deleteModel: (id: string) => request<void>(`/models/${encodeURIComponent(id)}`, { method: "DELETE" }),

  predict: (account: AccountInput, opts: { model_id?: string | null; source?: "manual" | "sample" | "adapter"; explain?: boolean } = {}) =>
    json<PredictResponse>("/predict", "POST", { account, model_id: opts.model_id ?? null, source: opts.source ?? "manual", explain: opts.explain ?? true, persist: true }),
  sampleAccounts: () => request<SampleAccount[]>("/sample-accounts"),

  predictBatchFile: (file: File, opts: { model_id?: string | null; name?: string; evaluate_if_labelled?: boolean } = {}) => {
    const fd = new FormData();
    fd.append("file", file);
    if (opts.model_id) fd.append("model_id", opts.model_id);
    if (opts.name) fd.append("name", opts.name);
    fd.append("evaluate_if_labelled", String(opts.evaluate_if_labelled ?? true));
    return request<BatchSummary>("/predict/batch", { method: "POST", body: fd });
  },
  predictBatchDataset: (dataset_id: string, opts: { model_id?: string | null; evaluate_if_labelled?: boolean } = {}) => {
    const fd = new FormData();
    fd.append("dataset_id", dataset_id);
    if (opts.model_id) fd.append("model_id", opts.model_id);
    fd.append("evaluate_if_labelled", String(opts.evaluate_if_labelled ?? true));
    return request<BatchSummary>("/predict/batch", { method: "POST", body: fd });
  },
  batches: () => request<BatchListItem[]>("/predict/batches"),
  batch: (id: string) => request<BatchSummary>(`/predict/batch/${encodeURIComponent(id)}`),
  batchDownloadUrl: (id: string) => `${API_BASE}/api/predict/batch/${encodeURIComponent(id)}/download`,

  datasets: () => request<DatasetInfo[]>("/datasets"),
  dataset: (id: string) => request<DatasetDetail>(`/datasets/${encodeURIComponent(id)}`),
  uploadDataset: (file: File, name?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    if (name) fd.append("name", name);
    return request<DatasetDetail>("/datasets/upload", { method: "POST", body: fd });
  },
  deleteDataset: (id: string) => request<void>(`/datasets/${encodeURIComponent(id)}`, { method: "DELETE" }),
  createDemoDataset: (n = 600, seed = 7) => request<DatasetDetail>(`/datasets/demo${qs({ n, seed })}`, { method: "POST" }),
  evaluateDataset: (id: string, model_id?: string | null) =>
    json<DatasetEvaluationResponse>(`/datasets/${encodeURIComponent(id)}/evaluate`, "POST", { model_id: model_id ?? null }),
  cresciStatus: () => request<CresciStatus[]>("/datasets/cresci/status"),
  importCresci: (kind: string, use_cache = true) => request<TrainJobResponse>(`/datasets/cresci/${kind}/import${qs({ use_cache })}`, { method: "POST" }),
  jobStatus: (jobId: string) => request<TrainStatusResponse>(`/datasets/jobs/${encodeURIComponent(jobId)}`),

  train: (body: TrainRequest) => json<TrainJobResponse>("/train", "POST", body),
  trainStatus: (jobId: string) => request<TrainStatusResponse>(`/train/status/${encodeURIComponent(jobId)}`),
  trainRuns: () => request<TrainingRunInfo[]>("/train/runs"),
  trainStages: () => request<string[]>("/train/stages"),

  evaluation: (modelId = "active") => request<EvaluationResponse>(`/evaluation/${encodeURIComponent(modelId)}`),
  globalExplanation: (modelId = "active") => request<GlobalExplanationResponse>(`/explain/global/${encodeURIComponent(modelId)}`),
  shapLocal: (predictionId: string) => request<LocalExplanationResponse>(`/explain/shap/${encodeURIComponent(predictionId)}`),
  limeLocal: (predictionId: string) => request<LocalExplanationResponse>(`/explain/lime/${encodeURIComponent(predictionId)}`),

  history: (filters: HistoryFilters = {}) => request<HistoryResponse>(`/history${qs(filters as Record<string, string | number | boolean | undefined>)}`),
  historyDetail: (id: string) => request<PredictResponse>(`/history/${encodeURIComponent(id)}`),
  deleteHistory: (id: string) => request<void>(`/history/${encodeURIComponent(id)}`, { method: "DELETE" }),

  adapters: () => request<AdapterInfo[]>("/adapters"),
  adapterFetch: (name: string, identifier: string) => json<AdapterFetchResponse>(`/adapters/${encodeURIComponent(name)}/fetch`, "POST", { identifier }),
};

export type Api = typeof api;
