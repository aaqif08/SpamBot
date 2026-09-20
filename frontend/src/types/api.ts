/** TypeScript mirrors of the backend Pydantic schemas (app/schemas/api.py). */

export type PredictionLabel = "BOT" | "HUMAN";
export type RiskBand = "critical" | "high" | "medium" | "low" | "minimal";

export interface ApiError {
  detail: string;
  code: string;
  errors?: { loc: string[]; msg: string; type: string }[];
}

export interface HealthResponse {
  status: string;
  app: string;
  version: string;
  environment: string;
  model_available: boolean;
  active_model: ModelEntry | null;
  datasets_available: number;
  demo_mode_enabled: boolean;
  feature_version: string;
  n_features: number;
  python: string;
  libraries: Record<string, string>;
}

export interface MetricSet {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number | null;
}

export interface ModelEntry {
  id: string;
  name: string;
  algorithm: string;
  version: string;
  dataset_id: string;
  dataset_name: string;
  feature_version: string;
  n_features: number;
  feature_names: string[];
  metrics: { holdout: MetricSet; cv: MetricSet; cv_std?: MetricSet };
  trained_at: string;
  training_seconds: number;
  params: { config?: Record<string, unknown>; best_params?: Record<string, unknown> };
  is_demo: boolean;
  notes: string;
  is_active?: boolean;
  source?: string;
}

export interface AlgorithmChoice {
  key: string;
  display_name: string;
  family: string;
  available: boolean;
  unavailable_reason: string;
  supports_tree_shap: boolean;
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

export interface PaperCitation {
  title: string;
  authors: string[];
  venue: string;
  year: number;
  doi: string;
  license: string;
  protocol: string;
}

export interface ModelListResponse {
  active_model_id: string | null;
  models: ModelEntry[];
  supported_algorithms: AlgorithmChoice[];
  paper_reported: PaperReported;
}

// ---- prediction ---------------------------------------------------------- //

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

export interface ModelInfo {
  id: string;
  name: string;
  algorithm: string;
  version: string;
  feature_version: string;
  n_features: number;
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

export interface PredictResponse {
  prediction_id: string | null;
  account_identifier: string;
  prediction: PredictionLabel;
  bot_probability: number;
  human_probability: number;
  confidence: number;
  risk_score: number;
  risk_band: RiskBand;
  risk_score_note: string;
  model: ModelInfo;
  features: Record<string, number>;
  feature_groups: Record<string, Record<string, number>>;
  auxiliary: Record<string, number>;
  top_features: TopFeature[];
  shap_explanation: ShapLocal | null;
  lime_explanation: LimeLocal | null;
  explanation_errors: Record<string, string>;
  interpretation: Interpretation;
  is_demo: boolean;
  created_at: string | null;
  input?: Record<string, unknown>;
  source?: string;
  batch_id?: string | null;
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

export interface BatchSummary {
  batch_id: string;
  name: string;
  model: ModelInfo;
  total_accounts: number;
  predicted_bots: number;
  predicted_humans: number;
  average_bot_probability: number;
  high_risk_accounts: number;
  risk_band_distribution: Record<string, number>;
  probability_histogram: HistogramBin[];
  preview: Record<string, string | number | null>[];
  evaluation: Evaluation | null;
  is_demo: boolean;
  download_url: string;
  created_at: string;
}

export interface BatchListItem {
  batch_id: string;
  name: string;
  model_name: string;
  total_accounts: number;
  predicted_bots: number;
  predicted_humans: number;
  average_bot_probability: number;
  high_risk_accounts: number;
  is_demo: boolean;
  created_at: string;
}

// ---- datasets ------------------------------------------------------------- //

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

export interface DatasetInfo {
  id: string;
  name: string;
  kind: string;
  original_filename: string;
  n_rows: number;
  n_columns: number;
  has_label: boolean;
  is_demo: boolean;
  class_distribution: Record<string, number> | null;
  feature_coverage: number | null;
  created_at: string;
}

export interface DatasetDetail extends DatasetInfo {
  summary: DatasetSummary;
}

export interface CresciStatus {
  kind: string;
  available: boolean;
  imported: boolean;
  dataset_id: string | null;
  subsets_found: { folder: string; category: string; label: number; has_tweets: boolean }[];
  expected_subsets: string[];
  paper_reported: { citation: string; subsets: { name: string; type: string; accounts: number; tweets: number }[] };
  install_path_hint: string;
}

export interface DatasetEvaluationResponse {
  dataset_id: string;
  model: ModelInfo;
  n_rows: number;
  evaluation: Evaluation;
  is_demo: boolean;
}

// ---- training ------------------------------------------------------------- //

export interface TrainRequest {
  dataset_id: string;
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

export interface TrainJobResponse {
  job_id: string;
  status: string;
  stage: string;
  progress: number;
  message: string;
  run_id: string;
}

export interface TrainStatusResponse {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  progress: number;
  message: string;
  log: string[];
  error: string | null;
  result: TrainResult | null;
}

export interface TrainResult {
  model_id: string;
  model: ModelEntry;
  metrics: EvaluationMetrics;
  shap_global: { importance: ImportanceRow[]; explainer: string; output_scale: string };
  is_demo: boolean;
  dataset?: DatasetDetail;
}

export interface CvResults {
  folds: MetricSet[];
  summary: Record<keyof MetricSet, { mean: number; std: number }>;
  n_folds: number;
  fit_time_mean: number;
}

export interface EvaluationMetrics {
  holdout: Evaluation;
  train: MetricSet;
  cross_validation: CvResults;
  split: { test_size: number; train_size: number; test_size_n: number; cv_folds: number; stratified: boolean; seed: number };
  class_distribution: { train: { human: number; bot: number }; test: { human: number; bot: number } };
  training_seconds: number;
  best_params: Record<string, unknown>;
  hyperparameter_search: boolean;
}

export interface TrainingRunInfo {
  id: string;
  job_id: string;
  model: string;
  dataset: string;
  dataset_id: string;
  status: string;
  stage: string;
  progress: number;
  params: Record<string, unknown>;
  metrics: { holdout: MetricSet; cv: MetricSet; training_seconds: number } | null;
  model_id: string | null;
  error: string | null;
  is_demo: boolean;
  created_at: string;
  completed_at: string | null;
}

// ---- evaluation / explainability ------------------------------------------- //

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

export interface EvaluationResponse {
  model: ModelEntry;
  dataset: { id: string; name: string; is_demo: boolean };
  feature_metadata: FeatureMetadata;
  metrics: EvaluationMetrics;
  is_demo: boolean;
  source: string;
}

export interface ImportanceRow {
  feature: string;
  group: string;
  mean_abs_shap: number;
  mean_shap: number;
  rank: number;
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
  model: ModelEntry;
  shap_global: ShapGlobal;
  feature_descriptions: Record<string, string>;
  feature_groups: Record<string, string[]>;
  is_demo: boolean;
}

export interface LocalExplanationResponse {
  prediction_id: string;
  method: "shap" | "lime";
  prediction: PredictionLabel;
  bot_probability: number;
  model_name: string;
  explanation: ShapLocal | LimeLocal;
  features: Record<string, number>;
}

// ---- history / dashboard ---------------------------------------------------- //

export interface HistoryItem {
  id: string;
  account_identifier: string;
  prediction: PredictionLabel;
  bot_probability: number;
  risk_score: number;
  risk_band: RiskBand;
  model_name: string;
  model_version: string;
  source: string;
  is_demo: boolean;
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
  high_risk?: boolean;
  min_risk?: number;
  model?: string;
  source?: string;
  date_from?: string;
  date_to?: string;
  search?: string;
  page?: number;
  page_size?: number;
}

export interface ModelComparisonRow {
  id: string;
  name: string;
  algorithm: string;
  dataset: string;
  is_demo: boolean;
  is_active: boolean;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  roc_auc: number | null;
  training_seconds: number;
}

export interface DashboardResponse {
  cards: {
    total_accounts_analyzed: number;
    bots_detected: number;
    humans_detected: number;
    average_bot_probability: number | null;
    high_risk_accounts: number;
    current_model: string | null;
    current_model_algorithm: string | null;
    datasets_registered: number;
    demo_predictions: number;
  };
  charts: {
    bot_vs_human: { name: string; value: number }[];
    probability_histogram: HistogramBin[];
    feature_importance: ImportanceRow[];
    model_comparison: ModelComparisonRow[];
    confusion_matrix: ConfusionMatrix | null;
    roc_curve: RocCurve | null;
    dataset_class_distribution: {
      dataset: string;
      train: { human: number; bot: number } | null;
      test: { human: number; bot: number } | null;
      is_demo: boolean;
    } | null;
    timeline: { date: string; BOT: number; HUMAN: number }[];
  };
  active_model: ModelEntry | null;
  has_model: boolean;
  has_predictions: boolean;
  notices: string[];
}

// ---- misc --------------------------------------------------------------------- //

export interface AdapterInfo {
  name: string;
  kind: string;
  configured: boolean;
  description: string;
  is_sample: boolean;
}

export interface AdapterFetchResponse {
  account: AccountInput;
  source: string;
  notice?: string;
  sample_id?: string;
  fetched_at?: string;
  user_id?: string;
  created_at?: string;
  protected?: boolean;
  tweets_fetched: number;
  tweets_error?: string | null;
  unavailable_fields: string[];
  cached?: boolean;
}

export interface SampleAccount {
  sample_id: string;
  label_hint: string;
  notice: string;
  account: AccountInput;
}

export interface FeaturesResponse {
  feature_version: string;
  n_features: number;
  groups: Record<string, string[]>;
  descriptions: Record<string, string>;
  order: string[];
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
