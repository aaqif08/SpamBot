# System architecture

BotShield AI is three cooperating parts: a React/TypeScript single-page application that only consumes typed JSON, a FastAPI backend that owns authentication, authorization, tenancy, persistence, jobs and storage, and a framework-independent ML package (`backend/ml`) that implements the paper's method and knows nothing about HTTP or databases.

## 1. Runtime topology

```
browser ──HTTPS──► reverse proxy (TLS) ──► frontend (nginx, static SPA)
                                              │ /api/*  (same-origin proxy)
                                              ▼
                                     api (gunicorn + uvicorn workers)
                                       ├── PostgreSQL  (system of record)
                                       ├── object storage / volume  (datasets, model artefacts, batch outputs)
                                       └── Redis ──► worker (Celery)  — or thread pool inside the API
```

Development runs the same code on SQLite, local storage and an in-process thread pool (`docker-compose.yml` or `uvicorn --reload`). Production uses `docker-compose.prod.yml` or the Render Blueprint; see `docs/deployment.md`.

## 2. Backend layout

```
backend/app
├── main.py            app factory; RequestContextMiddleware (X-Request-ID, security headers, HSTS, body-size guard, access log),
│                      CORS, proxy headers, exception handlers ({detail, code}), lifespan (migration check, job recovery, retention)
├── cli.py             migrate · create-admin · check-config · purge-expired-sessions · apply-retention
├── worker.py          Celery app; task botshield.run_job(job_id) → services.jobs.execute_job
├── api/
│   ├── deps.py        get_current_user (Bearer JWT, org + password-change checks), require_roles, rate-limit dependencies
│   └── v1/            auth.py · analyses.py (analyses + batches) · datasets.py · models.py · system.py
├── core/
│   ├── config.py      Settings (BOTSHIELD_*), production validation (no SQLite, SECRET_KEY, cookie rules, URL normalisation)
│   ├── security.py    bcrypt, password policy, JWT, opaque refresh tokens (HMAC-hashed), SlidingWindowLimiter, upload validation
│   ├── storage.py     Storage ABC → LocalStorage (traversal-safe) | S3Storage (boto3, local cache for artefact loading)
│   ├── logging.py     JSON/plain logging, request/user context vars, secret scrubbing
│   └── timeutil.py
├── db/
│   ├── database.py    engine/session factory, connection + migration-revision checks
│   └── models.py      ORM (below)
├── schemas/api.py     Pydantic v2 request/response models (mirrored in frontend/src/types/api.ts)
└── services/
    ├── auth_service.py        organizations, users, login (lockout), token issue/rotate/revoke, password change/reset
    ├── prediction_service.py  analyze → persist Prediction (+ Explanation rows) → interpretation text; history; purge
    ├── dataset_service.py     upload validation/profiling, versions, benchmark import, evaluation on labelled data
    ├── model_service.py       lifecycle TRAINING→READY→PRODUCTION→DEPRECATED, checksum-verified loading + cache, persistence
    ├── jobs.py / handlers.py  jobs table, thread or Celery dispatch, handlers: training, batch_prediction, dataset_import
    ├── dashboard_service.py   aggregates per organization (no placeholders: empty → empty state)
    ├── audit.py               audit rows + security log, sensitive keys scrubbed
    ├── providers.py           manual · csv · x_api (X API v2; 409 when not configured)
    └── retention.py           scheduled purge when retention windows are configured
```

### Request flow — single analysis

```
POST /api/v1/analyses  (Bearer token, role ≥ ANALYST, ML rate limit)
  → schemas.AnalyzeRequest (AccountInput validated)
  → ModelService.resolve(org, model_id | production)      # 409 when no production model
  → ModelService.load(model)                               # verify SHA-256 checksums → joblib pipeline (cached)
  → InferenceEngine.predict_account                        # FeatureExtractor (31) → pipeline → P(bot) → risk
        ├── SHAP local (TreeExplainer / KernelExplainer)
        └── LIME local (LimeTabularExplainer on stored sample)
  → PredictionService: upsert AnalyzedAccount, insert Prediction + Explanation rows, audit "analysis.created"
  → 201 AnalysisResponse (prediction, probabilities, risk_score + note, features, SHAP, LIME, interpretation)
```

### Job flow — training

```
POST /api/v1/models/train → Job(QUEUED) + MLModel(TRAINING) → dispatch (thread pool | Celery)
  handlers.training_job: load DatasetVersion from storage → ml.train.train_model
     (label coercion → 31 features → drop constant → stratified split → Pipeline(imputer→MinMax→clf)
      → RandomizedSearchCV(F1, k-fold) → hold-out evaluation → global SHAP → artefacts + checksums)
  → ModelService.persist_training_result: upload artefacts to storage, EvaluationRun(holdout, cross_validation),
    FeatureImportance rows, status READY (or PRODUCTION when activate=true and caller is ADMIN)
GET /api/v1/jobs/{id} → progress/stage/log/result
```

Batch prediction and benchmark import follow the same pattern; interrupted jobs are marked FAILED at start-up.

## 3. Data model

| Table | Purpose |
|---|---|
| `organizations`, `users`, `refresh_sessions`, `password_reset_tokens` | tenancy, accounts (bcrypt hash, role, status, lockout, `password_changed_at`), rotating refresh sessions (hashed), reset tokens (hashed) |
| `datasets`, `dataset_versions` | named datasets with immutable versions: storage key, SHA-256, rows/columns, label column, validation status, profile JSON |
| `ml_models` | version, algorithm, status, dataset link, feature metadata, `artifact_prefix`, `artifact_checksums_json`, test/validation metrics |
| `evaluation_runs`, `feature_importance` | hold-out / cross-validation / dataset evaluations with full detail JSON; global mean |SHAP| per feature |
| `analyzed_accounts`, `predictions`, `explanations` | one row per scored account (features, probabilities, risk, input summary — no raw tweet text), SHAP/LIME payloads by method |
| `batches` | batch metadata, storage keys of input/output, summary JSON |
| `jobs` | type, status, progress, stage, message, log, result, error, target |
| `audit_log` | action, actor, organization, target, outcome, IP, scrubbed details |

Every business table carries `organization_id`; services filter on it unconditionally. Migrations live in `backend/alembic/versions` (initial schema `20260919_aea49a2b381c`).

## 4. Storage layout

```
org/<organization_id>/
├── datasets/<dataset_id>/v<n>/<checksum>.csv
├── models/<model_id>/pipeline.joblib · feature_metadata.json · metrics.json · shap_global.json ·
│                     background.npy · lime_sample.npy · lime_sample_raw.npy · checksums.json
└── batches/<batch_id>/input.csv · predictions.csv
```

Model artefacts are loaded only after every file's SHA-256 matches the checksum stored in `ml_models.artifact_checksums_json`; a mismatch raises `ArtifactIntegrityError` and the model cannot be activated or used.

## 5. Frontend

- `src/services/api.ts` — typed client for `/api/v1`; access token in memory, silent `/auth/refresh` on 401 (single in-flight refresh), `credentials: "include"` for the cookie.
- `src/hooks/useAuth.tsx` — `AuthProvider`, `RequireAuth` (route guard with optional roles), `hasRole`.
- Pages render empty states from real API responses (no placeholder charts); jobs are polled through `useJobPolling` → `/jobs/{id}`.
- Paper-reported numbers and measured numbers are rendered with distinct `SourceTag`s ("Research paper results" vs "Your model performance").

## 6. Security controls (summary)

bcrypt + password policy + lockout · JWT HS256 (30 min) + rotating hashed refresh tokens in httpOnly/Secure/SameSite cookies · tokens invalidated on password change · RBAC dependencies · organization scoping · per-IP, per-user and login rate limits with `Retry-After` · upload validation (extension, MIME, content sniffing, size, row limits) · path-traversal-safe local storage · checksum-verified artefacts (no blind joblib/pickle loads) · request ids + structured logs with secret scrubbing · security headers, HSTS, `Cache-Control: no-store` on API responses · docs disabled in production by default · no default accounts, `.env` ignored by git.

## 7. Related documents

`docs/deployment.md` (operations), `docs/api.md` (endpoints), `docs/methodology.md` (ML method and adaptations), `docs/paper-analysis.md` (what the paper says), `docs/production-audit.md` and `docs/production-completion-report.md` (readiness).
