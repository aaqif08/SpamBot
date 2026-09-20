# BotShield AI — Interpretable Social Bot & Fake Follower Detection

A production web platform that implements the methodology of

> D. Javed, N. Z. Jhanjhi, N. A. Khan, S. K. Ray, A. Al-Dhaqm, V. R. Kebande,
> **"Identification of Spambots and Fake Followers on Social Network via Interpretable AI-Based Machine Learning"**,
> IEEE Access, vol. 13, 2025. DOI [10.1109/ACCESS.2025.3551993](https://doi.org/10.1109/ACCESS.2025.3551993)

as a multi-tenant service: organizations upload labelled account data, train and promote models (31 features, nine classifiers, stratified hold-out + 5-fold CV), analyze accounts one by one or in batches, and get real SHAP and LIME explanations for every prediction. Everything shown in the application is computed from the organization's own data; nothing is seeded, sampled or mocked.

**Integrity rules.** Figures from the paper are only ever displayed under *"Research paper results"*; figures produced by this software under *"Your model performance"*. The two are never mixed. A *risk score* is `round(100 × P(bot))` — a model output, not a verified fact — and results are worded as "Classified as BOT/HUMAN" and "Estimated bot probability".

---

## 1. What is in the repository

```
SpamBot/
├── backend/
│   ├── app/
│   │   ├── main.py             FastAPI app: request ids, security headers, body limits, CORS, error handlers, lifespan
│   │   ├── cli.py              python -m app.cli migrate | create-admin | check-config | purge-expired-sessions | apply-retention
│   │   ├── worker.py           Celery application (job backend "celery")
│   │   ├── api/v1/             auth · analyses (+batches) · datasets · models · system (health, jobs, providers, audit, research)
│   │   ├── core/               config (BOTSHIELD_* env), security (bcrypt, JWT, limiter, upload validation), storage (local/S3), logging
│   │   ├── db/                 SQLAlchemy 2 models (users, organizations, refresh sessions, datasets/versions, models, evaluations,
│   │   │                       predictions, explanations, batches, jobs, audit) + engine
│   │   ├── schemas/api.py      Pydantic request/response models
│   │   └── services/           auth · prediction · dataset · model lifecycle · jobs + handlers · dashboard · audit · providers · retention
│   ├── alembic/                migrations (PostgreSQL in production, SQLite in development/tests)
│   ├── ml/                     framework-independent ML package: features (31), preprocessing, sentiment, train, evaluation,
│   │                           explain (SHAP/LIME), predict, model_registry (artefacts + SHA-256 checksums), datasets, paper_results
│   ├── tests/                  pytest suite (API v1, auth/RBAC/tenancy, ML, storage); tests/fixtures generates TEST-ONLY data
│   ├── Dockerfile · docker-entrypoint.sh   gunicorn API or Celery worker, migrations at boot
│   └── requirements.txt
├── frontend/                   React 18 + TypeScript + Vite + Tailwind; typed API client with silent token refresh
│   ├── src/pages/              Login, Dashboard, Analyze, Batch, Datasets, Models, Training, Evaluation, Explainability,
│   │                           History, Research, Architecture, API Docs, Settings
│   └── Dockerfile · nginx.conf static SPA + /api/ proxy
├── scripts/                    train_model.py · report_results.py · fetch_datasets.py ·
│                               check_no_demo_data.(py|sh|ps1) · verify_production_readiness.(py|sh|ps1)
├── docs/                       deployment · api · architecture · methodology · paper-analysis · results ·
│                               production-audit · production-completion-report
├── docker-compose.yml          development stack (SQLite, thread jobs)
├── docker-compose.prod.yml     production stack (PostgreSQL, Redis, API, worker, nginx)
├── render.yaml                 Render Blueprint (free tier)
├── .env.example · .env.production.example
└── what_to_understand.md       guided tour of the codebase
```

## 2. Requirements

- Python 3.10–3.12 (3.13+ lacks prebuilt wheels for `shap`/`lightgbm`), Node 20+
- Development: nothing else (SQLite, local storage, in-process jobs)
- Production: PostgreSQL 14+, an S3-compatible bucket (recommended) and Redis if you run Celery workers

## 3. Local development

```bash
# backend
cd backend
python -m venv .venv && . .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cd .. && cp .env.example .env                       # development profile, SQLite, local storage
cd backend
python -m app.cli migrate                           # alembic upgrade head
python -m app.cli create-admin                      # first administrator (interactive; enforces the password policy)
uvicorn app.main:app --reload --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev                                         # http://localhost:5173 → proxies /api to :8000
```

Sign in with the administrator you just created. There are **no built-in accounts**.

Docker alternative: `docker compose up --build` (API on :8000, UI on :8080), then `docker compose exec backend python -m app.cli create-admin`.

## 4. Workflow inside the application

1. **Datasets** — upload a CSV (one account per row: `followers_count`, `friends_count`, `statuses_count`, `favourites_count`, `listed_count`, `verified`, `default_profile`, `description`, … plus a label column such as `label`/`is_bot`/`class` with `bot|human` or `1|0`). Uploads are size/MIME/content validated, checksummed, versioned and profiled (column mapping, missing values, class balance, feature coverage). Admins can also import the public Cresci-2015/2017 user-level benchmarks.
2. **Training** — pick a labelled dataset and one of nine classifiers; the job runs in the background (thread pool or Celery) with progress, log and stored results: hold-out metrics, confusion matrix, ROC/PR curves, CV folds, hyper-parameters, global SHAP importance.
3. **Models** — promote a `READY` model to `PRODUCTION` (artefact checksums are verified first); deprecate or delete old versions. The paper's tables are displayed separately, labelled as research results.
4. **Analyze / Batch** — score accounts entered manually, fetched from the X API (only when `BOTSHIELD_X_BEARER_TOKEN` is configured — otherwise the UI states *"External API integration is not configured"*), or uploaded as a CSV batch (async job, downloadable `predictions.csv`, evaluation when labels are present).
5. **Explainability / History** — global SHAP (bar, beeswarm, groups) per model; local SHAP waterfall + LIME per prediction, computed on demand for batch rows and persisted.
6. **Settings** — profile & password, users and roles (`ADMIN`, `ANALYST`, `VIEWER`), organization name, retention purge, providers, audit log.

## 5. Command-line pipeline

```bash
cd backend
python ../scripts/train_model.py --email admin@example.org --csv ../data/my_accounts.csv --algorithm lightgbm --activate
python ../scripts/train_model.py --email admin@example.org --benchmark cresci-17 --all        # nine classifiers
python ../scripts/report_results.py --email admin@example.org                                 # → docs/results.md
python ../scripts/fetch_datasets.py --help                                                    # public Cresci mirror
```

Scripts act as the given user inside their organization and go through the same services as the API (jobs, audit, checksums).

## 6. Tests and verification

```bash
cd backend && python -m pytest -q                       # API v1, auth/RBAC/tenancy, jobs, ML, storage (empty temp DB per run)
cd frontend && npx tsc -b && npx vitest run && npm run build
python scripts/check_no_demo_data.py                    # fails on demo/mock/sample/seed patterns in production code
python scripts/verify_production_readiness.py [--url https://app.example.com --strict]
```

## 7. Production deployment

See **[docs/deployment.md](docs/deployment.md)** for the full guide (topology, every environment variable, TLS, backups/recovery, Render). Short version:

```bash
cp .env.production.example .env        # set SECRET_KEY, DATABASE_URL, CORS_ORIGINS, storage, Postgres password …
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec api python -m app.cli create-admin
```

Security properties: bcrypt password hashing with a policy and login lockout; JWT access tokens + rotating hashed refresh tokens in httpOnly/Secure cookies; RBAC and organization scoping on every query; upload validation (extension, MIME, sniffing, size); artefacts loaded only after SHA-256 verification; rate limiting; structured JSON logs with request ids and secret scrubbing; security headers and HSTS; no stack traces to clients; no secrets in the frontend; `.env` never committed.

## 8. Results

Measured results for your own models appear on the Models/Evaluation pages and can be exported to `docs/results.md` with `scripts/report_results.py`. Numbers obtained on the public Cresci user-level mirrors during development (stratified hold-out) are recorded in [docs/results.md](docs/results.md) with their exact configuration; the paper's numbers are quoted there separately and are not directly comparable (the public mirror lacks the tweet files, so 11 tweet-derived features are constant and dropped, leaving 20 of 31).

## 9. Known limitations

- Public Cresci mirrors contain profiles only; the authors' `tweets.csv` files (available on request from MIB) are needed for the 11 tweet-derived features.
- X API v2 omits four v1.1 profile flags used by the paper; those are imputed and listed under `unavailable_fields`.
- Kernel SHAP (SVM, LR, NB, AdaBoost) is slower than TreeSHAP; global importance uses a capped background sample.
- Password-reset tokens are written to the security log for operators to deliver; there is no e-mail integration.
- Render free tier has no persistent disk — use S3-compatible storage there (see deployment guide).

## License / attribution

The base paper is CC BY 4.0. Cresci datasets: Cresci et al. 2015 (*DSS* 80) and 2017 (*WWW Companion*), academic terms of the original authors. This project is an independent implementation; the paper's authors are not affiliated with it.
