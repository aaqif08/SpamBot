# BotShield AI — System Architecture

Interpretable AI-Based Social Bot and Fake Follower Detection System.

## 1. High-level view

```
┌───────────────────────────────┐        ┌──────────────────────────────────────────┐
│  Frontend (React + TS + Vite) │  HTTP  │  Backend (FastAPI)                       │
│  Tailwind, Recharts, Lucide   │◄──────►│  app/api/*  → app/services/* → ml/*      │
│  pages: dashboard, analyze,   │  JSON  │  SQLite (SQLAlchemy)  models/ (joblib)   │
│  batch, datasets, models, …   │        │  background job runner (training)        │
└───────────────────────────────┘        └──────────────────────────────────────────┘
```

Two independent packages in one repository:

- `frontend/` — knows nothing about ML; consumes typed JSON from `/api/*`.
- `backend/` — `app/` (HTTP layer) is a thin shell around `ml/` (pure Python ML package with no FastAPI imports).

## 2. Backend architecture

```
backend/
├── app/
│   ├── main.py            FastAPI app factory, CORS, exception handlers, router mounting
│   ├── core/
│   │   ├── config.py      pydantic-settings (env vars), paths
│   │   ├── logging.py     structured logging config
│   │   └── security.py    filename sanitisation, upload validation, simple rate limiter
│   ├── db/
│   │   ├── database.py    SQLAlchemy engine/session (SQLite)
│   │   └── models.py      ORM tables: predictions, datasets, models, training_runs
│   ├── schemas/           Pydantic request/response models (API contract)
│   ├── services/          orchestration: prediction, dataset, training, dashboard, explanation
│   └── api/               routers: health, models, predict, datasets, train, evaluation, explain, history, dashboard, adapters
├── ml/                    framework-independent ML package
│   ├── features.py        FEATURE_GROUPS (31 features), FeatureExtractor
│   ├── preprocessing.py   text cleaning (feature path / sentiment path), imputation, scaling
│   ├── sentiment.py       polarity/subjectivity (TextBlob) with emoji→text
│   ├── train.py           model zoo, CV, hyperparameter search, training orchestration
│   ├── evaluation.py      metrics, confusion matrix, ROC/PR curves
│   ├── explain.py         SHAP (global/local) + LIME (local)
│   ├── predict.py         Predictor: load registry model → features → proba → risk score
│   ├── model_registry.py  registry.json management under models/
│   ├── datasets.py        Cresci loader (users.csv + tweets.csv), CSV schema inference
│   ├── demo_data.py       synthetic demo generator (labelled DEMO)
│   ├── schemas.py         dataclasses for ML-level results
│   └── utils.py           helpers (json-safe, timers, seeds)
├── models/                trained artefacts (joblib + metadata + registry.json)
├── data/                  datasets/ (cresci-15, cresci-17, demo, uploads/)
└── tests/
```

### 2.1 Data flow — single prediction

```
POST /api/predict (AccountInput)
  → PredictionService.predict()
      → FeatureExtractor.transform(account)          # 31-feature vector
      → Predictor.predict_proba(vector)               # scaler + model from registry
      → risk_score = round(p_bot * 100)
      → Explainer.shap_local(vector)  (TreeExplainer for tree models, else KernelExplainer w/ background)
      → Explainer.lime_local(vector)  (LimeTabularExplainer trained on the model's training features)
      → persist Prediction row (+ explanation JSON) in SQLite
  ← PredictionResponse
```

### 2.2 Data flow — training (async job)

```
POST /api/train {dataset_id, model_name, test_size, cv_folds, feature_selection}
  → JobManager.submit(train_job) → job_id           # background thread
  GET /api/train/status/{job_id}                     # stages: queued → preprocessing → feature_engineering
                                                     #   → training → cross_validation → shap_analysis → saving → completed
  on completion: models/<run_id>/ {model.joblib, scaler.joblib, feature_metadata.json, metrics.json,
                                   shap_global.json, background.npy}  + registry.json entry + training_runs row
```

### 2.3 Data flow — dataset upload

```
POST /api/datasets/upload (multipart CSV, ≤ MAX_UPLOAD_MB)
  → validate extension/MIME/size → sanitise filename → save to data/uploads/<uuid>.csv
  → DatasetService.inspect(): rows, columns, dtypes, missing, duplicates, label detection,
                               feature availability against the 31-feature schema, class distribution
  → Dataset row in SQLite
POST /api/predict/batch {dataset_id} or multipart CSV
  → FeatureExtractor.transform_frame() → Predictor.predict_frame() → summary + downloadable CSV
```

## 3. Database schema (SQLite)

```
predictions
  id TEXT PK, account_identifier TEXT, prediction TEXT (BOT|HUMAN), bot_probability REAL,
  human_probability REAL, risk_score INTEGER, model_name TEXT, model_version TEXT,
  features_json TEXT, shap_json TEXT, lime_json TEXT, input_json TEXT, source TEXT,
  is_demo BOOLEAN, created_at DATETIME

datasets
  id TEXT PK, name TEXT, kind TEXT (cresci-15|cresci-17|upload|demo), path TEXT,
  n_rows INTEGER, n_columns INTEGER, has_label BOOLEAN, summary_json TEXT,
  is_demo BOOLEAN, created_at DATETIME

models
  id TEXT PK, name TEXT, algorithm TEXT, version TEXT, dataset_id TEXT, dataset_name TEXT,
  feature_version TEXT, n_features INTEGER, metrics_json TEXT, path TEXT,
  is_active BOOLEAN, trained_at DATETIME

training_runs
  id TEXT PK, job_id TEXT, model TEXT, dataset TEXT, status TEXT, stage TEXT, progress REAL,
  params_json TEXT, metrics_json TEXT, error TEXT, created_at DATETIME, completed_at DATETIME
```

## 4. API contract (summary — full detail in `docs/api.md`)

| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | liveness + model/dataset availability flags |
| GET | /api/models | registry listing (+ paper-reported results, separately keyed) |
| POST | /api/models/{id}/activate | set active model |
| POST | /api/predict | single account prediction with SHAP + LIME |
| POST | /api/predict/batch | CSV upload or dataset_id → batch summary + download token |
| GET | /api/predict/batch/{batch_id}/download | predictions.csv |
| GET/POST | /api/datasets, /api/datasets/upload | list / upload |
| GET | /api/datasets/{id} | inspection summary |
| POST | /api/datasets/{id}/evaluate | evaluate active model on labelled dataset |
| POST | /api/datasets/import-cresci | scan data/datasets for Cresci folders |
| POST | /api/train | submit async training job |
| GET | /api/train/status/{job_id} | poll |
| GET | /api/train/runs | history of training runs |
| GET | /api/evaluation/{model_id} | metrics, CM, ROC, PR, CV results |
| GET | /api/explain/global/{model_id} | SHAP global importance + beeswarm sample |
| GET | /api/explain/shap/{prediction_id} | stored local SHAP |
| GET | /api/explain/lime/{prediction_id} | stored local LIME |
| GET | /api/history, /api/history/{id} | prediction history |
| GET | /api/dashboard | aggregate cards + chart data |
| GET | /api/adapters, POST /api/adapters/{name}/fetch | social-network adapter interface (sample adapter only) |
| GET | /api/sample-account | labelled demo account |

Errors use `{"detail": str, "code": str}` with proper 4xx/5xx codes.

## 5. ML architecture

- **FeatureExtractor** is the single source of truth for the 31 features (`FEATURE_GROUPS` in `ml/features.py`). It accepts either an *account dict* (profile counts + description + list of tweets) or a *pre-aggregated row* (CSV with count columns) and produces the same ordered vector.
- **Preprocessing** applies the paper's two text paths; numeric imputation (median from training) + min–max scaling (fit at training time, persisted as `scaler.joblib`).
- **Model zoo** (`ml/train.py`): 9 estimators with small randomised search spaces; every model wrapped in an sklearn `Pipeline(imputer → scaler → clf)` so the same artefact serves prediction, SHAP and LIME.
- **Explainers**: `shap.TreeExplainer` for RF/ET/DT/XGB/LGBM/AdaBoost-with-trees (falls back to `shap.Explainer` / `KernelExplainer` with a stored background sample for SVM/LR/NB). LIME via `lime.lime_tabular.LimeTabularExplainer` with training data statistics stored per model.
- **Risk score** = `round(100 × P(bot))` — an application-level view of the model probability, not a verdict.

## 6. Frontend architecture

```
frontend/src/
├── layouts/AppLayout.tsx        sidebar + topbar + theme toggle + backend status
├── pages/                       one file per route
├── components/ui/               Card, Button, Badge, Table, EmptyState, ErrorState, Spinner, Tabs …
├── components/charts/           Recharts wrappers (Bar, Pie, Line/ROC, Histogram, ConfusionMatrix, ShapBar, ShapWaterfall, Beeswarm)
├── hooks/                       useApi (loading/error/data), useTheme, usePolling
├── services/api.ts              typed fetch client for every endpoint
├── types/                       TS mirrors of Pydantic schemas
└── utils/                       formatters, download helpers
```

State is per-page (`useApi`) with a global theme + backend-health context. Every page has loading, error, and empty states; a global error boundary prevents blank pages.

## 7. Security decisions

- Upload: extension + content sniff + size cap; filenames replaced by UUID; original name stored only as metadata.
- Paths never returned to the client; datasets referenced by id.
- CORS origins from env; no secrets in code; `.env.example` documents all settings.
- In-memory token-bucket rate limiter on predict/upload/train routes (configurable).
- Pydantic validation with bounds on all numeric fields; CSV column validation before inference.

## 8. Engineering adaptations from the paper (documented, not silent)

1. Derived-feature formulas are not given in the paper; we use standard definitions (see `docs/methodology.md`).
2. Hyperparameter grids are not published; we use compact randomised searches per model.
3. TextBlob used for sentiment (paper does not name the library; feature names match TextBlob output).
4. Production data: `scripts/fetch_datasets.py` downloads the public user-level mirror of Cresci-2015/2017 (exact paper subsets); the tweet files are only distributed by the authors, so tweet-derived features are dropped as constant until they are added. A labelled synthetic demo dataset exists only behind `BOTSHIELD_DEMO_MODE_ENABLED=true`.
5. Paper-reported numbers are stored as static reference data and rendered only under "Reported in base paper".
