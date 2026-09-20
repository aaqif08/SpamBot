export const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);

export const int = (v: number | null | undefined): string =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Math.round(v).toLocaleString();

export const compact = (v: number | null | undefined): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
};

export const signed = (v: number, digits = 3): string => `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;

export const seconds = (v: number | null | undefined): string =>
  v === null || v === undefined ? "—" : v < 1 ? `${(v * 1000).toFixed(0)} ms` : `${v.toFixed(1)} s`;

export const dateTime = (iso: string | null | undefined): string => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

export const featureLabel = (name: string): string => name.replace(/_/g, " ");

const ALGORITHM_LABELS: Record<string, string> = {
  random_forest: "Random Forest",
  svm: "SVM",
  decision_tree: "Decision Tree",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  logistic_regression: "Logistic Regression",
  extra_trees: "Extra Trees",
  naive_bayes: "Naive Bayes",
  adaboost: "AdaBoost",
};

export const algorithmLabel = (key: string): string => ALGORITHM_LABELS[key] ?? key;

const GROUP_LABELS: Record<string, string> = {
  user_profile: "User profile",
  content: "Content",
  engagement: "Engagement",
  linguistic: "Linguistic",
  profile_attributes: "Profile attributes",
  sentiment: "Sentiment",
};

export const groupLabel = (g: string): string => GROUP_LABELS[g] ?? g;
