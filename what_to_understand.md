# What to understand — BotShield AI study guide

A defence-preparation guide for this project. Read top to bottom once, then use §9 (likely questions) to rehearse. Every number below is either **measured by this code** (marked *ours*) or **reported by the base paper** (marked *paper*) — never mix the two when you speak.

---

## 1. The 30-second pitch

BotShield AI is an interpretable machine-learning system that decides whether a social-network account is a **bot / spambot / fake follower** or a **legitimate human**, and — unlike black-box detectors — explains *why* for every single prediction using **SHAP** and **LIME**. It implements the methodology of Javed et al. (IEEE Access, 2025): a compact 31-feature set in six groups, nine classifiers compared with stratified cross-validation, and SHAP-based feature analysis. It is trained on the paper's own benchmark data (Cresci-2015 / Cresci-2017, real Twitter accounts) and deployed as a full web application: single-account analysis, batch CSV scoring, dataset analytics, model training and evaluation, an explainability dashboard, prediction history, and an adapter for the live X API.

## 2. The base paper — what you must know cold

| Item | Answer |
|---|---|
| Title | Identification of Spambots and Fake Followers on Social Network via Interpretable AI-Based Machine Learning |
| Authors / venue | Javed, Jhanjhi, Khan, Ray, Al-Dhaqm, Kebande — IEEE Access vol. 13, 2025, DOI 10.1109/ACCESS.2025.3551993 |
| Problem | Bots spread misinformation and manipulate opinion; existing detectors are black boxes (no transparency), or use huge feature sets (Botometer ≈ 1,200) that don't scale |
| Contribution | Interpretable ML framework: compact 31-feature set, 9 classifiers tuned by cross-validation, SHAP (global + feature selection) and LIME (local) explanations |
| Datasets | Cresci-15 (humans TFP, E13; fake followers FSF, INT, TWT — 5,301 accounts) and Cresci-17 (genuine 3,474; social spambots 1–3; traditional spambots 1,000; fake followers 3,351 — 12,737 accounts). Each subset = `users.csv` + `tweets.csv` |
| Preprocessing | Null description → `"missing"` (length 0); emoji → text; URLs/mentions/punctuation removed *only for the sentiment path* (they are features otherwise); stop-word removal; shuffling |
| Split | Stratified 75/25 (70/30 also mentioned) + 5-fold CV; results in Tables 5–6 are 5-fold CV |
| Metrics | Accuracy, precision, recall, F1, AUC (+ "interpretability" as a qualitative criterion) |
| Headline results (*paper*) | Cresci-15: LightGBM acc 0.991 / F1 0.993. Cresci-17: XGBoost & LightGBM acc 0.990 / F1 0.993 |
| Top SHAP features (*paper*) | favorites_count, avg_sentence_length, ffratio, friends_count, mentions_count, avg_mentions, unique_word_use |
| Limitations (*paper*) | SHAP/LIME are computationally expensive; fixed feature set may not transfer to new-generation bots; future work: adaptive learning, graph neural networks |

## 3. The 31 features (paper Table 4) and how we compute them

Six groups. Know the groups, a couple of examples per group, and the formulas we chose (the paper does *not* give formulas — say so if asked).

| Group | Features | Our definition |
|---|---|---|
| User profile (5) | verified, friends_count, followers_count, listed_count, favorites_count | raw profile counts |
| Content (6) | hashtag_count, mentions_count, retweet_count, reply_count, url_count, statuses_count | sums over observed tweets (from `num_hashtags` etc. or regex on text); statuses = profile count |
| Engagement (7) | ffratio, avg_hashtag, avg_retweets, avg_replies, avg_mentions, avg_url, avg_user_engagement | `ffratio = followers / (friends + 1)`; `avg_X = X / n_tweets`; `avg_user_engagement = (retweets + replies + likes received) / n_tweets` |
| Linguistic (5) | unique_word_count, unique_word_use, punctuation_count, avg_sentence_length, punctuation_density | `unique_word_use = unique / total words`; `avg_sentence_length = words / sentences`; `punctuation_density = punct chars / total chars` |
| Profile attributes (6) | profile_completeness, description_binary, default_profile, default_profile_image, geo_enabled, profile_background_tile | completeness = fraction filled of {name, description, location, url, picture, banner} |
| Sentiment (2) | avg_polarity, avg_subjectivity | TextBlob PatternAnalyzer on the cleaned text (paper's feature names are TextBlob's output names) |

Single source of truth: `FEATURE_GROUPS` in [backend/ml/features.py](backend/ml/features.py). Everything (training, prediction, SHAP, LIME, API, UI) reads the list from there.

**Important nuance:** the public Cresci mirror we could obtain contains only `users.csv` (profiles). The `tweets.csv` files are distributed by the dataset authors on request. Without tweets, the 11 tweet-derived features (the content counts and the per-tweet averages) are constant zero, so the training pipeline drops them automatically and records it → **our production models use 20 of the 31 features**. Linguistic and sentiment features are computed from the profile description instead of tweets. The importer supports the full 31 the moment `tweets.csv` is added.

## 4. The pipeline (paper Fig. 4 ↔ our code)

```
Dataset (Cresci users.csv [+ tweets.csv])         ml/datasets.py, scripts/fetch_datasets.py
  → Preprocessing (two text paths, imputation)     ml/preprocessing.py, ml/sentiment.py
  → Feature engineering (31 → constant-dropped)    ml/features.py, ml/train.py
  → [optional] SHAP feature selection (top-k)      ml/train.py
  → Stratified 75/25 split, shuffle                 ml/train.py
  → Pipeline(median imputer → min-max scaler → clf) ml/train.py
  → Randomised hyperparameter search, 5-fold CV    ml/train.py (RandomizedSearchCV, F1)
  → Hold-out evaluation (acc/prec/rec/F1/AUC, CM, ROC, PR)  ml/evaluation.py
  → Global SHAP (mean |SHAP|, beeswarm)             ml/explain.py
  → Model registry (joblib + metadata + metrics)    ml/model_registry.py, backend/models/
Prediction: account → features → P(bot) → risk score → local SHAP + LIME → database  ml/predict.py, app/services/prediction_service.py
```

Why min–max scaling? The paper's LIME figures show rules like `0.00 < ffratio <= 0.01`, i.e. features in [0, 1]. Why one sklearn `Pipeline`? So the exact same artefact serves prediction, SHAP and LIME (no train/serve skew).

## 5. The nine classifiers

Random Forest, SVM (RBF, `probability=True`), Decision Tree, XGBoost, LightGBM, Logistic Regression, Extra Trees, Naive Bayes (Gaussian), AdaBoost. Be able to say one sentence about each family:

- **Bagging ensembles** (RF, Extra Trees): many decorrelated trees, vote; robust, little tuning.
- **Boosting** (XGBoost, LightGBM, AdaBoost): trees added sequentially to fix previous errors; LightGBM grows leaf-wise with histograms → fast and usually best here.
- **Linear / probabilistic** (LR, NB): fast baselines; NB assumes feature independence → weakest.
- **SVM**: max-margin with RBF kernel; slow on ~10k rows because `probability=True` runs internal CV.

Hyperparameters: the paper says "optimised through cross-validation" but publishes no grids → we use compact randomised search spaces scored by F1 with stratified 5-fold CV (say this is *our* choice).

## 6. Explainability — the heart of the project

**SHAP (SHapley Additive exPlanations).** Game theory: each feature is a "player"; its Shapley value is its average marginal contribution to the prediction over all feature coalitions. Properties: *local accuracy* (base value + Σ contributions = model output — our tests assert this), *consistency*. We use `TreeExplainer` for tree models (exact, fast) and `KernelExplainer` (model-agnostic, sampled) for SVM/LR/NB/AdaBoost.
- *Global*: mean |SHAP| over a training sample → feature ranking and the beeswarm plot (one dot per account, x = SHAP value, colour = feature value). The paper uses SHAP this way both to explain and to *select* the compact feature set.
- *Local*: per-account contributions toward BOT (+) or HUMAN (−), shown as bars and as a waterfall from the base value to the model output.
- Output scale: probability for scikit-learn trees and LightGBM; XGBoost falls back to log-odds with the installed library versions — the UI always states the scale.

**LIME (Local Interpretable Model-agnostic Explanations).** Perturb the account's features (3,000 samples), get the black-box model's predictions, fit a weighted sparse linear model locally; its coefficients are the explanation. Output matches the paper's figures: prediction probabilities (Human/Bot) and two-sided feature-range rules (`favorites_count > 0.29` → Human −0.457). `surrogate_r2` tells how faithful the local fit is.

**SHAP vs LIME in one line:** SHAP has theoretical guarantees and gives global + local views; LIME is cheaper conceptually, purely local, and can be unstable between runs. The paper uses both; so do we.

**Risk score** = `round(100 × P(bot))`, banded minimal/low/medium/high/critical. It is an application-level presentation of the probability — *not* a verdict that an account is malicious. Say this sentence in the viva.

## 7. Our results (*ours* — from `docs/results.md`, real Cresci user-level data, 20 features, stratified 25 % hold-out)

| Model | Trained on | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|
| LightGBM (**production / active**) | Cresci-15 + 17 combined (14,687 accounts) | 0.980 | 0.987 | 0.981 | 0.984 | 0.995 |
| LightGBM | Cresci-17 (12,737) | 0.986 | 0.994 | 0.987 | 0.991 | 0.997 |
| XGBoost | Cresci-15 (5,301) | 0.982 | 0.990 | 0.981 | 0.986 | 0.997 |
| Naive Bayes (weakest) | Cresci-17 | 0.906 | 0.902 | 0.977 | 0.938 | 0.933 |

Cross-dataset generalisation (*ours*): Cresci-17 model → Cresci-15 unseen: F1 0.947. Cresci-15 model → Cresci-17: F1 0.607 (it only ever saw fake followers, so it misses social spambots). That gap is *why* the production model is trained on both.

How to compare with the paper honestly: the ordering of classifiers matches (boosting/RF top, NB/LR/SVM bottom); our absolute numbers are slightly lower and are **not one-to-one comparable** because (a) we lack the 11 tweet-derived features, (b) our headline is a hold-out split while the paper reports 5-fold CV means (we report CV means too — e.g. combined LightGBM CV F1 0.986 ± 0.002), (c) search spaces and derived-feature formulas are ours.

## 8. Architecture — enough to draw it on a whiteboard

```
React + TypeScript + Vite + Tailwind (frontend/src/pages …)
        │  JSON over /api/*  (typed client: services/api.ts)
FastAPI  app/api/*  →  app/services/*  →  ml/*  (no FastAPI imports inside ml/)
        │                                   ├─ features.py, preprocessing.py, sentiment.py
        │                                   ├─ train.py, evaluation.py, explain.py, predict.py
PostgreSQL / SQLite (users, orgs, datasets,  └─ model_registry.py → storage org/<org>/models/<id>/
        models, evaluations, predictions,            pipeline.joblib · feature_metadata.json · metrics.json ·
        explanations, batches, jobs, audit)          shap_global.json · background.npy · lime_sample.npy · checksums.json
```

- **Async jobs**: `POST /api/v1/models/train` (and batches, benchmark imports) create a row in the `jobs` table and return `202 {job_id}`; a thread pool or Celery worker executes it; the UI polls `GET /api/v1/jobs/{id}` and shows stage/progress/log (queued → preprocessing → feature engineering → training → cross-validation → SHAP analysis → saving → completed).
- **Providers**: `Provider.fetch_account(identifier) → AccountInput`. `manual`, `csv` and `x_api` (real X API v2: `/2/users/by/username`, `/2/users/:id/tweets`; needs `BOTSHIELD_X_BEARER_TOKEN`; tweets need the Basic tier). Without a token the API answers 409 "External data integration is not configured".
- **Security**: bcrypt passwords + policy + lockout, JWT access tokens + rotating hashed refresh cookies, RBAC (ADMIN/ANALYST/VIEWER), organization scoping on every query, CSV extension/MIME/size/content checks, path-traversal guard, checksum-verified model artefacts, rate limiters, CORS from env, security headers, audit log, no secrets in code or frontend, typed error envelope `{detail, code}` with request ids.
- **Production mode**: `BOTSHIELD_ENVIRONMENT=production` requires PostgreSQL and `BOTSHIELD_SECRET_KEY`, disables docs, forces secure cookies and JSON logs; migrations are checked at start-up; there is no demo mode and no default account (`python -m app.cli create-admin`).

## 9. Likely examiner questions — and honest answers

1. *Did you reproduce the paper's 99 % accuracy?* — We reproduce the methodology and reach 0.98–0.99 accuracy / 0.98–0.99 F1 on the same benchmark datasets with 20 of the 31 features. The paper's exact numbers are shown in the app only under "Reported in base paper". Differences are explained by missing tweet-level features and evaluation protocol.
2. *Why only 20 features?* — The tweet files are not publicly redistributed; the profile-level mirror makes the 11 tweet-derived features constant, and constant features carry no information. The pipeline drops and records them; adding the authors' `tweets.csv` restores all 31 with no code change.
3. *Why LightGBM as the production model?* — Highest or joint-highest F1/AUC on both datasets in our runs, trains in seconds, supports exact TreeSHAP on the probability scale.
4. *How do you know SHAP values are real?* — Computed by the `shap` library on the fitted pipeline; a unit test asserts additivity (base + Σ = model output = `predict_proba`).
5. *What is the difference between global and local explanation?* — Global = which features matter on average (mean |SHAP| over many accounts). Local = why *this* account got *this* prediction (per-account SHAP contributions / LIME rules).
6. *Is a risk score of 90 proof of a bot?* — No. It is 100 × the model's estimated probability under the training distribution; it supports review, it doesn't replace it.
7. *What are the false-positive consequences?* — Blocking a genuine user. The paper stresses precision; our production model's hold-out precision is 0.987 with FPR ≈ 2 %.
8. *Does it generalise to new bots?* — Partially: Cresci-17 → Cresci-15 F1 0.947; the reverse is 0.607. Bots evolve (paper's own limitation); retraining and richer features (tweets, timing, graph) are the mitigation.
9. *Why not deep learning / GNNs?* — The paper's aim is interpretability with a compact feature set; tree ensembles are accurate here and explainable with exact SHAP. GNNs are listed as future work.
10. *What is synthetic in the system?* — Nothing in the application. The only generated data lives in `backend/tests/fixtures` and is used exclusively by the automated tests; `scripts/check_no_demo_data.py` fails the build if demo/sample/mock patterns appear in production code.
11. *Where does the live data come from?* — The X API v2 adapter with the deployer's bearer token; nothing is fabricated when the token is missing (the UI says "not configured").
12. *How would you deploy it?* — `docker compose -f docker-compose.prod.yml up -d --build` (PostgreSQL, Redis, gunicorn API, Celery worker, nginx frontend behind a TLS proxy) or the Render Blueprint; see `docs/deployment.md`.

## 10. Things you should never claim

- That the app achieved the paper's exact numbers.
- That accounts entered manually are real X data (only the X provider fetches live data, and only when configured).
- That a prediction "proves" an account is a bot.
- That the model uses tweet content when the active model card says 20 features.

## 11. Five-minute demo script

1. **Dashboard** — production model, real-data metrics, empty/real prediction counts.
2. **Analyze Account** → enter an account's profile counts manually *or* fetch `@handle` via the X provider (when configured) → Analyze → walk through prediction, probability, risk band, SHAP bars/waterfall, LIME rules, interpretation text.
3. **Batch Analysis** → dataset "CRESCI-15 (imported…)" → run → evaluation against labels → download `predictions.csv`.
4. **Models** → trained versions with hold-out/CV metrics; promote/deprecate lifecycle; the paper's table shown separately as "Research paper results".
5. **Evaluation** → confusion matrix, ROC/PR, 5-fold CV table, feature set + "11 constant features dropped" notice.
6. **Explainability** → global beeswarm and group importance; pick a stored prediction → local SHAP + LIME.
7. **Research / Architecture** → paper mapping table and engineering adaptations.

## 12. Key files to be able to open on request

| Question | Open |
|---|---|
| Where are the 31 features defined? | `backend/ml/features.py` (`FEATURE_GROUPS`) |
| Where is the preprocessing? | `backend/ml/preprocessing.py`, `backend/ml/sentiment.py` |
| Where is training / CV / search? | `backend/ml/train.py` |
| Where are SHAP and LIME? | `backend/ml/explain.py` |
| Where do the paper's numbers live? | `backend/ml/paper_results.py` (reference only) |
| Where are our measured results? | Evaluation page per model (`evaluation_runs` table), `docs/results.md` via `scripts/report_results.py` |
| Dataset provenance? | `backend/data/datasets/PROVENANCE.md`, `scripts/fetch_datasets.py` |
| Methodology mapping and adaptations? | `docs/methodology.md`, `docs/paper-analysis.md` |
| Tests? | `backend/tests/` (58), `frontend/src/test/` (16) |
