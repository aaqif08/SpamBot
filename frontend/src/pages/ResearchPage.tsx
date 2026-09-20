import { ArrowRight, BookOpen, ExternalLink } from "lucide-react";
import { Fragment } from "react";

import { Badge, Card, ErrorState, PageHeader, Skeleton, SourceTag, Table, Td, Th } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import { api } from "@/services/api";
import { algorithmLabel, featureLabel, groupLabel, num } from "@/utils/format";

const SECTIONS: { title: string; body: string[] }[] = [
  {
    title: "Problem statement",
    body: [
      "Social platforms such as X (Twitter) are infiltrated by automated accounts — spambots and fake followers — that spread misinformation, inflate popularity and manipulate opinion, especially around elections.",
      "Most bot-detection systems are black boxes: they can be accurate, yet give no reason for a decision, which limits trust, debugging, bias detection and adaptation to evolving bots.",
    ],
  },
  {
    title: "Motivation",
    body: [
      "Interpretable machine learning (SHAP, LIME) makes the contribution of every feature visible, so analysts can verify that a model relies on meaningful behavioural signals rather than noise.",
      "A compact, well-chosen feature set keeps both training and explanation computationally tractable — SHAP cost grows with the number of features.",
    ],
  },
  {
    title: "Research gap",
    body: [
      "Existing methods either use very large feature sets (≈1,200 in Botometer-style systems) that hurt scalability, or narrow network/profile feature sets that ignore linguistic, temporal and sentiment signals.",
      "Few approaches combine competitive accuracy with per-prediction explanations across different bot families (traditional spambots, social spambots, fake followers).",
    ],
  },
  {
    title: "Objectives",
    body: [
      "Build an interpretable bot-detection model for spambots and fake followers using SHAP and LIME.",
      "Analyse the influence of X account features on the decision across multiple bot types using established benchmark datasets (Cresci-15, Cresci-17).",
      "Validate that explainability can be combined with state-of-the-art performance using a compact 31-feature set.",
    ],
  },
];

const IMPLEMENTATION_MAP: { topic: string; paper: string; ours: string }[] = [
  { topic: "Datasets", paper: "Cresci-15 (humans + fake followers) and Cresci-17 (genuine, social spambots 1–3, traditional spambots, fake followers); users.csv + tweets.csv per subset.", ours: "scripts/fetch_datasets.py downloads the public user-level mirror of both datasets (row counts match Tables 2-3; provenance + SHA-256 recorded). The importer aggregates tweets.csv per user when the authors' tweet files are added; without them the 11 tweet-derived features are constant and dropped (20 of 31 features used)." },
  { topic: "Preprocessing", paper: "Null description → 'missing' (length 0); emoji → text; URLs/mentions/punctuation removed only for the sentiment path; stop-word removal; shuffling.", ours: "Two text paths implemented exactly as described (ml/preprocessing.py); median imputation + min–max scaling fitted on the training split." },
  { topic: "Feature engineering", paper: "31 features in six groups (Table 4): user profile, content, engagement, linguistic, profile attributes, sentiment.", ours: "FEATURE_GROUPS in ml/features.py is the single source of truth; derived-feature formulas are documented in docs/methodology.md because the paper does not state them." },
  { topic: "Feature selection", paper: "Shapley feature selection — rank features by mean |SHAP| and keep the most relevant subset.", ours: "Optional training step: screening model → SHAP ranking → keep top-k." },
  { topic: "Sentiment analysis", paper: "avg_polarity and avg_subjectivity from cleaned tweet/description text.", ours: "TextBlob PatternAnalyzer (its output names match the paper's feature names)." },
  { topic: "Classifiers", paper: "Random Forest, SVM, Decision Tree, XGBoost, LightGBM, Logistic Regression, Extra Trees, Naïve Bayes, AdaBoost; hyperparameters optimised via cross-validation.", ours: "Same nine classifiers in one sklearn Pipeline; randomised search scored by F1 with stratified k-fold CV (grids are not published, so compact search spaces are used)." },
  { topic: "Evaluation", paper: "Stratified split (75/25; 70/30 also mentioned), 5-fold CV; accuracy, precision, recall, F1, AUC.", ours: "Stratified hold-out + stratified k-fold CV (both configurable); same metrics plus confusion matrix, ROC and PR curves. Results are labelled 'Reproduced by this implementation'." },
  { topic: "SHAP", paper: "TreeExplainer; beeswarm summary of the top-20 features; mean |SHAP| ranking.", ours: "TreeExplainer for tree models (probability scale where supported), KernelExplainer otherwise; global beeswarm/bar data and local waterfall per prediction — all computed by the shap library." },
  { topic: "LIME", paper: "LimeTabular explanation with Human/Bot prediction probabilities and two-sided feature-rule contributions.", ours: "LimeTabularExplainer on the min–max scaled feature space (rules like '0.00 < ffratio <= 0.01' match the paper's figures)." },
  { topic: "Reported results", paper: "Cresci-15: LightGBM acc 0.991 / F1 0.993. Cresci-17: XGBoost & LightGBM acc 0.990 / F1 0.993 (5-fold CV).", ours: "Displayed only as 'Reported in base paper'. Our own numbers are shown separately and only after a real training run." },
];

export function ResearchPage() {
  const { data, loading, error, reload } = useApi(() => api.research(), []);
  if (loading && !data) return <div className="space-y-4"><PageHeader title="Research" /><Skeleton className="h-96" /></div>;
  if (error || !data) return <div className="space-y-4"><PageHeader title="Research" /><ErrorState message={error ?? ""} onRetry={reload} /></div>;
  const c = data.citation;

  return (
    <div className="space-y-6">
      <PageHeader title="Research" description="Academic context of BotShield AI: the base paper, its methodology, and how this system implements and extends it." badge={<Badge tone="accent">Final-year project</Badge>} />

      <Card title="Base paper" actions={<a className="inline-flex items-center gap-1 text-xs text-accent hover:underline" href={`https://doi.org/${c.doi}`} target="_blank" rel="noreferrer">doi.org/{c.doi} <ExternalLink className="h-3 w-3" /></a>}>
        <div className="flex items-start gap-3">
          <span className="rounded-lg bg-accent-soft p-2 text-ink"><BookOpen className="h-5 w-5" /></span>
          <div>
            <p className="text-base font-semibold text-ink">{c.title}</p>
            <p className="text-sm text-ink-2">{c.authors.join(", ")}</p>
            <p className="text-sm text-ink-2">{c.venue}, {c.year} · DOI {c.doi} · {c.license}</p>
            <p className="mt-1 text-xs text-ink-3">Experimental protocol reported: {c.protocol}</p>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {SECTIONS.map((s) => (
          <Card key={s.title} title={s.title}>
            <ul className="list-disc space-y-1.5 pl-4 text-sm text-ink-2">{s.body.map((b) => <li key={b}>{b}</li>)}</ul>
          </Card>
        ))}
      </div>

      <Card title="Methodology pipeline (paper Figure 4)" subtitle="Dataset → Preprocessing → Feature Engineering → Feature Selection → Sentiment Analysis → ML Model → Prediction → SHAP/LIME">
        <ol className="flex flex-wrap items-center gap-2">
          {data.pipeline.map((step, i) => (
            <Fragment key={step}>
              <li className="rounded-lg border border-border bg-surface-2 px-3 py-2 text-xs font-medium text-ink">{i + 1}. {step}</li>
              {i < data.pipeline.length - 1 && <ArrowRight className="h-4 w-4 text-ink-3" aria-hidden />}
            </Fragment>
          ))}
        </ol>
      </Card>

      <Card title="Feature engineering — the 31-feature compact set (paper Table 4)" subtitle="Grouped exactly as in the paper; implemented in backend/ml/features.py">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(data.feature_groups).map(([g, fs]) => (
            <div key={g} className="rounded-lg border border-border p-3">
              <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-2">{groupLabel(g)} <span className="text-ink-3">({fs.length})</span></div>
              <ul className="space-y-1 text-xs">
                {fs.map((f) => <li key={f} className="text-ink"><span className="font-medium">{featureLabel(f)}</span> <span className="text-ink-3">— {data.feature_descriptions[f]}</span></li>)}
              </ul>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Machine learning & explainable AI" subtitle="Nine classifiers compared with stratified cross-validation; SHAP for global/local attribution and feature selection; LIME for local surrogate explanations">
        <div className="flex flex-wrap gap-2">{data.classifiers.map((m) => <Badge key={m.key}>{m.name} · {m.family}</Badge>)}</div>
        <p className="mt-3 text-sm text-ink-2">SHAP (Shapley additive explanations) assigns each feature its marginal contribution to a prediction, averaged over feature coalitions; the mean absolute value across accounts ranks features globally. LIME fits a sparse linear surrogate around one account by perturbing its features and reports which feature ranges push toward Bot or Human.</p>
      </Card>

      <Card title="Base paper methodology vs. our system implementation" subtitle="Engineering adaptations are explicit — nothing in the methodology was changed silently">
        <Table>
          <thead><tr><Th>Topic</Th><Th>Base paper methodology</Th><Th>Our system implementation</Th></tr></thead>
          <tbody>
            {IMPLEMENTATION_MAP.map((r) => <tr key={r.topic}><Td className="font-medium">{r.topic}</Td><Td className="text-xs text-ink-2">{r.paper}</Td><Td className="text-xs text-ink-2">{r.ours}</Td></tr>)}
          </tbody>
        </Table>
        <ul className="mt-3 list-disc space-y-1 pl-4 text-xs text-ink-2">{data.engineering_adaptations.map((a) => <li key={a}>{a}</li>)}</ul>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        {(["cresci-15", "cresci-17"] as const).map((ds) => (
          <Card key={ds} title={`Results reported in base paper — ${ds.toUpperCase()} (${data.paper_reported.datasets[ds].table})`} subtitle={data.paper_reported.datasets[ds].highlight} actions={<SourceTag kind="paper" />}>
            <Table>
              <thead><tr><Th>Classifier</Th><Th align="right">Acc</Th><Th align="right">Prec</Th><Th align="right">Rec</Th><Th align="right">F1</Th><Th align="right">AUC</Th></tr></thead>
              <tbody>
                {data.paper_reported.datasets[ds].results.map((r) => <tr key={r.algorithm}><Td className="text-xs">{algorithmLabel(r.algorithm)}</Td><Td align="right" mono>{num(r.accuracy)}</Td><Td align="right" mono>{num(r.precision)}</Td><Td align="right" mono>{num(r.recall)}</Td><Td align="right" mono>{num(r.f1)}</Td><Td align="right" mono>{num(r.roc_auc)}</Td></tr>)}
              </tbody>
            </Table>
            <p className="mt-2 text-[11px] text-ink-3">Baselines compiled by the paper (Tables 7–8): {data.paper_reported.baselines[ds].map((b) => `${b.cite} acc ${num(b.accuracy)} / F1 ${num(b.f1)}`).join("; ")}.</p>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Evaluation criteria (paper §IV-A)">
          <ul className="list-disc space-y-1 pl-4 text-sm text-ink-2">
            <li>Accuracy, precision, recall, F1 (harmonic mean of precision and recall), AUC.</li>
            <li>Interpretability — the ability to explain decisions with SHAP and LIME.</li>
            <li>5-fold cross-validation with shuffling; stratified train/test split.</li>
          </ul>
        </Card>
        <Card title="Limitations & future scope">
          <ul className="list-disc space-y-1 pl-4 text-sm text-ink-2">
            <li>SHAP/LIME are computationally expensive at Cresci scale; the compact feature set mitigates but does not remove this.</li>
            <li>A fixed feature set may not transfer to new-generation bots that closely mimic humans.</li>
            <li>Future work (paper): adaptive/continual learning and graph neural networks over the social graph combined with XAI.</li>
            <li>This implementation: no live X API (adapter contract only); the public Cresci mirror is user-level, so tweet-derived features require the authors' tweets.csv files; models trained without them use 20 of 31 features and are not directly comparable to the paper.</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
