# API reference

Base URL: `http://localhost:8000/api`. Interactive docs: `/docs` (Swagger UI), `/redoc`, `/openapi.json`.

All error responses have the shape `{"detail": string, "code": string}`; validation errors add `errors: [{loc, msg, type}]` (HTTP 422).

## Health & metadata

| Method | Path | Response |
|---|---|---|
| GET | `/health` | `HealthResponse` — status, version, `model_available`, `active_model`, dataset count, feature version, library versions |
| GET | `/features` | feature groups, descriptions, canonical order |
| GET | `/research` | citation, paper-reported results, pipeline, engineering adaptations |
| GET | `/dashboard` | `DashboardResponse` — cards + charts (SQLite aggregates, active-model artefacts) |

## Models

| Method | Path | Notes |
|---|---|---|
| GET | `/models` | `{active_model_id, models[], supported_algorithms[], paper_reported}` — `paper_reported.source` explicitly says the numbers are from the paper |
| GET | `/models/algorithms` | availability of the nine classifiers |
| POST | `/models/{id}/activate` | set active model |
| DELETE | `/models/{id}` | remove model + artefacts (204) |

## Prediction

### `POST /predict`

```json
{
  "account": {
    "account_id": "string", "verified": false,
    "friends_count": 120, "followers_count": 15, "listed_count": 0, "favorites_count": 50, "statuses_count": 500,
    "description": "...", "location": "", "url": "",
    "default_profile": false, "default_profile_image": false, "geo_enabled": false, "profile_background_tile": false, "has_profile_banner": false,
    "hashtag_count": 25, "mentions_count": 40, "retweet_count": 300, "reply_count": 2, "url_count": 20,
    "favorite_count_received": 0, "tweets_observed": null,
    "tweets": ["text", {"text": "…", "retweet_count": 1, "reply_count": 0, "favorite_count": 2}]
  },
  "model_id": null, "explain": true, "source": "manual", "persist": true
}
```

Content counts are optional — when omitted they are derived from `tweets`. Response (`PredictResponse`):

```json
{
  "prediction_id": "…", "account_identifier": "…",
  "prediction": "BOT", "bot_probability": 0.94, "human_probability": 0.06, "confidence": 0.94,
  "risk_score": 94, "risk_band": "critical", "risk_score_note": "Risk Score is an application-level representation …",
  "model": {"id": "…", "name": "LightGBM", "algorithm": "lightgbm", "version": "…", "feature_version": "paper-31-v1", "n_features": 31},
  "features": {"verified": 0, "…": 0}, "feature_groups": {"user_profile": {…}, "…": {…}}, "auxiliary": {"avg_favorites": 0, "n_tweets": 16},
  "top_features": [{"feature": "avg_url", "group": "engagement", "value": 1.0, "impact": 0.09, "direction": "BOT"}],
  "shap_explanation": {"explainer": "TreeExplainer", "output_scale": "probability", "base_value": 0.36, "model_output": 0.94, "contributions": [{"feature": "…", "value": 1, "value_scaled": 0.2, "shap": 0.09, "direction": "BOT", "cumulative": 0.45}]},
  "lime_explanation": {"prediction_probabilities": {"HUMAN": 0.06, "BOT": 0.94}, "bot_indicators": [{"rule": "avg_url > 0.29", "weight": 0.39}], "human_indicators": [], "items": [], "surrogate_r2": 0.42, "num_samples": 3000},
  "explanation_errors": {}, "interpretation": {"summary": "…", "recommendation": "…"}, "is_demo": false, "created_at": "…"
}
```

Status codes: 200, 404 (unknown `model_id`), 422 (validation), 503 (no trained model).

### Batch

| Method | Path | Notes |
|---|---|---|
| POST | `/predict/batch` | multipart: `file` (CSV) **or** `dataset_id`; optional `model_id`, `name`, `evaluate_if_labelled` → `BatchSummary` |
| GET | `/predict/batches` | list |
| GET | `/predict/batch/{batch_id}` | summary |
| GET | `/predict/batch/{batch_id}/download` | `predictions.csv` with `account_id, [label,] prediction, bot_probability, human_probability, risk_score, risk_band` + top-10 SHAP-ranked features |
| GET | `/sample-accounts` | DEMO accounts (labelled as such) |

## Datasets

| Method | Path | Notes |
|---|---|---|
| GET | `/datasets` | list (`DatasetInfo`) |
| POST | `/datasets/upload` | multipart `file` (.csv, ≤ `BOTSHIELD_MAX_UPLOAD_MB`) → `DatasetDetail` incl. inspection summary (rows, columns, missing, duplicates, label column, class distribution, feature availability, preview, warnings). 400/413/415/422 on invalid input |
| GET | `/datasets/{id}` | detail |
| DELETE | `/datasets/{id}` | 204 |
| POST | `/datasets/{id}/evaluate` | `{model_id?}` → full evaluation of a model on a labelled dataset |
| POST | `/datasets/demo?n=600&seed=7` | generate synthetic demo dataset |
| GET | `/datasets/cresci/status` | local availability + paper-reported subset statistics |
| POST | `/datasets/cresci/{kind}/import` | featurise local Cresci files (202, job) |
| GET | `/datasets/jobs/{job_id}` | import job status |

## Training

| Method | Path | Notes |
|---|---|---|
| POST | `/train` | `TrainRequest` `{dataset_id, algorithm, test_size, cv_folds, hyperparameter_search, search_iterations, feature_selection, feature_selection_top_k, seed, activate, notes}` → 202 `{job_id, run_id, …}` |
| GET | `/train/status/{job_id}` | `{status: queued|running|completed|failed, stage, progress, message, log[], error, result}`; stages: queued → preprocessing → feature_engineering → training → cross_validation → shap_analysis → saving → completed |
| GET | `/train/stages` | stage list |
| GET | `/train/runs` | persisted run history |

## Evaluation & explainability

| Method | Path | Notes |
|---|---|---|
| GET | `/evaluation/{model_id}` | `active` allowed; hold-out metrics, confusion matrix, ROC/PR curves, probability histogram, CV folds, split info, feature metadata |
| GET | `/explain/global/{model_id}` | SHAP importance ranking, top-20 beeswarm sample, group importance |
| GET | `/explain/shap/{prediction_id}` | stored local SHAP; computed on demand for batch rows |
| GET | `/explain/lime/{prediction_id}` | stored local LIME; computed on demand for batch rows |

## History

| Method | Path | Notes |
|---|---|---|
| GET | `/history` | filters: `prediction`, `high_risk`, `min_risk`, `model`, `source`, `date_from`, `date_to`, `search`, `page`, `page_size` |
| GET | `/history/{id}` | full stored analysis (`PredictResponse` + `input`, `source`, `batch_id`) |
| DELETE | `/history/{id}` | 204 |

## Adapters

| Method | Path | Notes |
|---|---|---|
| GET | `/adapters` | `sample` (bundled, demo data) and `x_api` (contract only; `configured` reflects `BOTSHIELD_X_BEARER_TOKEN`) |
| POST | `/adapters/{name}/fetch` | `{identifier}` → `{account, source, notice}`; 409 if not configured, 501 if not implemented |

## Security

- Uploads: `.csv` only, content-type whitelist, size cap, binary sniffing, filenames replaced by UUIDs; server paths never returned.
- In-memory rate limiter on predict / upload / train (`BOTSHIELD_RATE_LIMIT_PER_MINUTE`).
- CORS origins from `BOTSHIELD_CORS_ORIGINS`; no secrets in code.
