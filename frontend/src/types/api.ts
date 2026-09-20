/** TypeScript mirrors of the backend Pydantic schemas (app/schemas/api.py, API v1). */

export type PredictionLabel = "BOT" | "HUMAN";
export type RiskBand = "critical" | "high" | "medium" | "low" | "minimal";
export type RoleName = "ADMIN" | "ANALYST" | "VIEWER";
export type JobStatus = "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "CANCELLED";
export type ModelStatus = "TRAINING" | "READY" | "PRODUCTION" | "DEPRECATED" | "FAILED";

// ---- auth ----------------------------------------------------------------- //

export interface UserPublic {
  id: string;
  email: string;
  full_name: string;
  role: RoleName;
  status: "ACTIVE" | "DISABLED";
  organization_id: string;
  organization_name: string;
  last_login_at: string | null;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: UserPublic;
}

export interface SetupStatus {
  initialized: boolean;
  self_signup_enabled: boolean;
}

export interface OrganizationPublic {
  id: string;
  name: string;
  slug: string;
  created_at: string;
}

// ---- system --------------------------------------------------------------- //

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ReadinessResponse {
  status: string;
  version: string;
  environment: string;
  checks: Record<string, unknown>;
}

export interface RuntimeInfo {
  version: string;
  environment: string;
  feature_version: string;
  n_features: number;
  job_backend: string;
  storage_backend: string;
  python: string;
  libraries: Record<string, string>;
  production_model: { id: string; name: string; version: number; algorithm: string; n_features: number } | null;
}

// ---- metrics / evaluation ------------------------------------------------- //

export interface MetricSet {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
}

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  false_positive_rate: number;
  false_negative_rate: number;
}

export interface RocCurve {
  fpr: number[];
  tpr: number[];
  auc: number | null;
}

export interface PrCurve {
  precision: number[];
  recall: number[];
  average_precision: number | null;
}

export interface Evaluation {
  metrics: MetricSet;
  confusion_matrix: ConfusionMatrix;
  roc_curve: RocCurve;
  pr_curve: PrCurve;
  probability_histogram: HistogramBin[];
  n_samples: number;
  n_positive: number;
  n_negative: number;
}

export interface CvResults {
  folds: MetricSet[];
  summary: Record<keyof MetricSet, { mean: number; std: number }>;
  n_folds: number;
  fit_time_mean: number;
}

export interface TrainingMetrics {
  holdout: Evaluation;
  train: MetricSet;
  cross_validation: CvResults;
  split: { test_size: number; train_size: number; test_size_n: number; cv_folds: number; stratified: boolean; seed: number };
  class_distribution: { train: { human: number; bot: number }; test: { human: number; bot: number } };
  training_seconds: number;
  best_params: Record<string, unknown>;
  hyperparameter_search: boolean;
}

export interface ImportanceRow {
  feature: string;
  group: string;
  mean_abs_shap: number;
  mean_shap: number;
  rank: number;
}

// ---- models --------------------------------------------------------------- //

export interface ModelPublic {
  id: string;
  name: string;
  version: number;
  algorithm: string;
  status: ModelStatus;
  is_production: boolean;
  dataset_id: string | null;
  dataset_version_id: string | null;
  dataset_name: string;
  feature_version: string;
  feature_names: string[];
  n_features: number;
  training_seconds: number;
  trained_at: string | null;
  created_at: string;
  created_by: string | null;
  job_id: string | null;
  notes: string;
  artifact_checksum: string;
  test_metrics?: Partial<MetricSet> | null;
  validation_metrics?: { mean?: Partial<MetricSet>; std?: Partial<MetricSet>; folds?: number } | null;
  params?: Record<string, unknown> | null;
}

export interface AlgorithmChoice {
  key: string;
  display_name: string;
  family: string;
  available: boolean;
  unavailable_reason: string;
  supports_tree_shap: boolean;
}

export interface ModelListResponse {
  production_model_id: string | null;
  models: ModelPublic[];
  supported_algorithms: AlgorithmChoice[];
}

export interface EvaluationRunPublic {
  id: string;
  kind: "holdout" | "cross_validation" | "dataset";
  dataset_version_id: string | null;
  n_samples: number;
  metrics: MetricSet;
  details: Partial<TrainingMetrics> & { dataset_name?: string; dataset_id?: string; holdout?: Evaluation };
  created_at: string;
}

export interface FeatureMetadata {
  feature_version: string;
  feature_names: string[];
  n_features: number;
  groups: Record<string, string[]>;
  feature_selection: { enabled: boolean; top_k?: number; selected?: string[]; dropped?: string[] };
  dropped_constant_features?: string[];
  scaling: string;
  imputation: string;
  raw_feature_ranges: Record<string, { min: number; max: number; median: number }>;
}

export interface ModelEvaluationResponse {
  model: ModelPublic;
  feature_metadata: FeatureMetadata;
  evaluations: EvaluationRunPublic[];
  feature_importance: ImportanceRow[];
}

export interface BeeswarmPoint {
  shap: number;
  value_scaled: number;
  value: number;
}

export interface ShapGlobal {
  explainer: string;
  output_scale: "probability" | "log_odds";
  base_value: number;
  n_samples: number;
  importance: ImportanceRow[];
  top_features: string[];
  group_importance: Record<string, number>;
  beeswarm: { feature: string; points: BeeswarmPoint[] }[];
}

export interface GlobalExplanationResponse {
  model: ModelPublic;
  shap_global: ShapGlobal;
  feature_descriptions: Record<string, string>;
  feature_groups: Record<string, string[]>;
}

export interface TrainRequest {
  dataset_id: string;
  dataset_version_id?: string | null;
  algorithm: string;
  test_size: number;
  cv_folds: number;
  hyperparameter_search: boolean;
  search_iterations: number;
  feature_selection: boolean;
  feature_selection_top_k: number;
  seed: number;
  activate: boolean;
  notes: string;
}

export interface TrainSubmitted {
  job_id: string;
  model_id: string;
  status: JobStatus;
}

// ---- jobs ------------------------------------------------------------------ //

export interface TrainingJobResult {
  model_id: string;
  model: ModelPublic;
  metrics: TrainingMetrics;
  feature_importance: ImportanceRow[];
  explainer: string;
  output_scale: string;
  activated: boolean;
}

export interface JobResponse {
  id: string;
  job_type: "TRAINING" | "BATCH_PREDICTION" | "DATASET_IMPORT" | "EXPLANATION";
  status: JobStatus;
  stage: string;
  progress: number;
  message: string;
  log: string[];
  error: string | null;
  result: Record<string, unknown> | null;
  target_type: string;
  target_id: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

// ---- analyses -------------------------------------------------------------- //

export interface TweetInput {
  text: string;
  retweet_count?: number;
  reply_count?: number;
  favorite_count?: number;
}

export interface AccountInput {
  account_id?: string | null;
  screen_name?: string | null;
  name?: string | null;
  verified: boolean;
  friends_count: number;
  followers_count: number;
  listed_count: number;
  favorites_count: number;
  statuses_count: number;
  description?: string | null;
  location?: string | null;
  url?: string | null;
  default_profile: boolean;
  default_profile_image: boolean;
  geo_enabled: boolean;
  profile_background_tile: boolean;
  has_profile_banner: boolean;
  hashtag_count?: number | null;
  mentions_count?: number | null;
  retweet_count?: number | null;
  reply_count?: number | null;
  url_count?: number | null;
  favorite_count_received?: number | null;
  tweets_observed?: number | null;
  tweets: (TweetInput | string)[];
}

export interface ModelRef {
  id: string | null;
  name: string;
  version: number;
}

export interface TopFeature {
  feature: string;
  group: string;
  description: string;
  value: number | null;
  impact: number | null;
  direction: "BOT" | "HUMAN" | "NEUTRAL" | "GLOBAL";
}

export interface ShapContribution {
  feature: string;
  group: string;
  description: string;
  value: number;
  value_scaled: number;
  shap: number;
  direction: "BOT" | "HUMAN" | "NEUTRAL";
  cumulative: number;
}

export interface ShapLocal {
  explainer: string;
  output_scale: "probability" | "log_odds";
  base_value: number;
  model_output: number;
  sum_positive: number;
  sum_negative: number;
  contributions: ShapContribution[];
}

export interface LimeItem {
  feature: string;
  group: string;
  description: string;
  rule: string;
  weight: number;
  direction: "BOT" | "HUMAN" | "NEUTRAL";
  value: number | null;
  value_scaled: number | null;
}

export interface LimeLocal {
  class_names: string[];
  prediction_probabilities: { HUMAN: number; BOT: number };
  intercept: number | null;
  local_prediction: number | null;
  surrogate_r2: number | null;
  num_samples: number;
  bot_contribution: number;
  human_contribution: number;
  bot_indicators: LimeItem[];
  human_indicators: LimeItem[];
  items: LimeItem[];
}

export interface Interpretation {
  summary: string;
  recommendation: string;
  bot_indicators: { feature: string; phrase: string; impact: number; value: number }[];
  human_indicators: { feature: string; phrase: string; impact: number; value: number }[];
  disclaimer: string;
}

export interface AnalysisResponse {
  prediction_id: string;
  account_identifier: string;
  account_ref: string | null;
  prediction: PredictionLabel;
  bot_probability: number;
  human_probability: number;
  confidence: number;
  risk_score: number;
  risk_band: RiskBand;
  risk_score_note: string;
  model: ModelRef;
  features: Record<string, number>;
  feature_groups: Record<string, Record<string, number>>;
  auxiliary: Record<string, number>;
  input_summary: Record<string, unknown>;
  top_features: TopFeature[];
  shap_explanation: ShapLocal | null;
  lime_explanation: LimeLocal | null;
  explanation_errors: Record<string, string>;
  explanation_status: Record<string, string>;
  interpretation: Interpretation;
  source: string;
  status: string;
  batch_id: string | null;
  label_true: string | null;
  inference_ms: number;
  created_at: string;
  created_by: string | null;
}

export interface LocalExplanationResponse {
  prediction_id: string;
  method: "shap" | "lime";
  explanation: ShapLocal | LimeLocal;
  computed_now: boolean;
}

export interface HistoryItem {
  id: string;
  account_identifier: string;
  prediction: PredictionLabel;
  bot_probability: number;
  risk_score: number;
  risk_band: RiskBand;
  model_id: string | null;
  model_name: string;
  model_version: number;
  source: string;
  status: string;
  batch_id: string | null;
  label_true: string | null;
  created_at: string;
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface HistoryFilters {
  prediction?: PredictionLabel | "";
  min_risk?: number;
  model_id?: string;
  source?: string;
  batch_id?: string;
  date_from?: string;
  date_to?: string;
  search?: string;
  sort?: string;
  page?: number;
  page_size?: number;
}

// ---- batches --------------------------------------------------------------- //

export interface BatchResponse {
  id: string;
  name: string;
  status: JobStatus;
  job_id: string | null;
  model: ModelRef | null;
  dataset_version_id: string | null;
  total_rows: number;
  processed_rows: number;
  failed_rows: number;
  n_bots: number;
  n_humans: number;
  avg_bot_probability: number;
  high_risk: number;
  summary: {
    risk_band_distribution?: Record<string, number>;
    probability_histogram?: HistogramBin[];
    top_features_exported?: string[];
    label_column?: string | null;
    evaluation?: Evaluation | null;
  };
  error: string | null;
  has_output: boolean;
  created_at: string;
  completed_at: string | null;
}

// ---- datasets --------------------------------------------------------------- //

export interface FeatureAvailability {
  direct: string[];
  derivable: string[];
  missing: string[];
  coverage: number;
}

export interface DatasetSummary {
  n_rows: number;
  n_columns: number;
  duplicate_rows: number;
  total_missing: number;
  missing_by_column: Record<string, number>;
  columns: { name: string; dtype: string; missing: number; unique: number; canonical: string }[];
  label_column: string | null;
  label_valid_rows: number;
  class_distribution: { HUMAN: number; BOT: number; unknown: number } | null;
  feature_availability: FeatureAvailability;
  numeric_summary: Record<string, { min: number; max: number; mean: number; median: number }>;
  preview: Record<string, string | number | null>[];
  warnings: string[];
}

export interface DatasetVersionPublic {
  id: string;
  dataset_id: string;
  version: number;
  original_filename: string;
  size_bytes: number;
  checksum_sha256: string;
  n_rows: number;
  n_columns: number;
  has_label: boolean;
  label_column: string | null;
  status: string;
  validation_errors: string[];
  created_at: string;
  summary?: DatasetSummary | null;
}

export interface DatasetPublic {
  id: string;
  name: string;
  description: string;
  kind: string;
  status: string;
  created_at: string;
  updated_at: string;
  n_versions: number;
  current_version: DatasetVersionPublic | null;
  n_rows: number;
  n_columns: number;
  has_label: boolean;
  class_distribution: Record<string, number> | null;
  feature_coverage: number | null;
  models_trained: number;
  summary?: DatasetSummary | null;
  versions?: DatasetVersionPublic[] | null;
}

export interface BenchmarkStatus {
  kind: string;
  available: boolean;
  subsets_found: { folder: string; category: string; label: number; has_tweets: boolean }[];
  expected_subsets: string[];
  paper_reported: { citation: string; subsets: { name: string; type: string; accounts: number; tweets: number }[] };
  install_path_hint: string;
}

export interface EvaluationResult {
  model: ModelRef;
  dataset_version_id: string;
  n_samples: number;
  evaluation: Evaluation;
  evaluation_run_id: string;
}

// ---- dashboard ---------------------------------------------------------------- //

export interface ModelComparisonRow {
  id: string;
  name: string;
  version: number;
  algorithm: string;
  status: ModelStatus;
  dataset: string;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  roc_auc: number | null;
  training_seconds: number;
}

export interface DashboardResponse {
  cards: {
    accounts_analyzed: number;
    predictions_total: number;
    bots_classified: number;
    humans_classified: number;
    bot_rate: number | null;
    average_bot_probability: number | null;
    high_risk: number;
    datasets: number;
    models: number;
    jobs_total: number;
    jobs_running: number;
    batches: number;
    latest_analysis_at: string | null;
    production_model: { id: string; name: string; version: number; algorithm: string } | null;
  };
  charts: {
    bot_vs_human: { name: string; value: number }[];
    probability_histogram: HistogramBin[];
    timeline: { date: string; BOT: number; HUMAN: number }[];
    risk_distribution: { band: string; count: number }[];
    model_usage: { model: string; count: number }[];
    feature_importance: ImportanceRow[];
    model_comparison: ModelComparisonRow[];
    production_holdout: { metrics: MetricSet; confusion_matrix: ConfusionMatrix | null; roc_curve: RocCurve | null; n_samples: number; class_distribution: TrainingMetrics["class_distribution"] | null } | null;
  };
  production_model: ModelPublic | null;
  has_model: boolean;
  has_predictions: boolean;
}

// ---- providers / audit / research --------------------------------------------- //

export interface ProviderInfo {
  name: string;
  kind: string;
  configured: boolean;
  description: string;
  fetchable: boolean;
  configuration_hint: string | null;
}

export interface ProviderFetchResponse {
  account: AccountInput;
  source: string;
  fetched_at?: string;
  protected?: boolean;
  tweets_fetched: number;
  tweets_error?: string | null;
  unavailable_fields: string[];
  cached?: boolean;
  notice?: string;
}

export interface AuditItem {
  id: string;
  action: string;
  actor_email: string;
  target_type: string;
  target_id: string;
  outcome: string;
  ip_address: string;
  details: Record<string, unknown>;
  created_at: string;
}

export interface AuditResponse {
  items: AuditItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface PaperCitation {
  title: string;
  authors: string[];
  venue: string;
  year: number;
  doi: string;
  license: string;
  protocol: string;
}

export interface PaperResultRow {
  algorithm: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
}

export interface PaperReported {
  source: string;
  citation: PaperCitation;
  datasets: Record<string, { table: string; results: PaperResultRow[]; highlight: string }>;
  baselines: Record<string, { cite: string; accuracy: number; f1: number }[]>;
}

export interface ResearchResponse {
  citation: PaperCitation;
  paper_reported: PaperReported;
  feature_groups: Record<string, string[]>;
  feature_descriptions: Record<string, string>;
  classifiers: { key: string; name: string; family: string }[];
  pipeline: string[];
  engineering_adaptations: string[];
}
