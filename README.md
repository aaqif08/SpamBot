# BotShield AI — Interpretable AI-Based Social Bot and Fake Follower Detection System

An end-to-end web application that implements the methodology of the base paper

> D. Javed, N. Z. Jhanjhi, N. A. Khan, S. K. Ray, A. Al-Dhaqm, V. R. Kebande,
> **"Identification of Spambots and Fake Followers on Social Network via Interpretable AI-Based Machine Learning"**,
> IEEE Access, vol. 13, 2025. DOI [10.1109/ACCESS.2025.3551993](https://doi.org/10.1109/ACCESS.2025.3551993)

and extends it into a usable academic project: a FastAPI + scikit-learn/XGBoost/LightGBM backend with real SHAP and LIME explanations, a React/TypeScript dashboard, SQLite persistence, dataset analytics, batch prediction, asynchronous training and reproducible scripts.

**Academic integrity.** Numbers from the paper are only ever shown as *"Reported in base paper"*. Numbers produced by this code are shown as *"Reproduced by this implementation"*. They are never mixed. Synthetic demo data is labelled **DEMO DATA — NOT REAL SOCIAL MEDIA DATA** everywhere and is never used to claim research accuracy. The *Risk Score* is `round(100 × P(bot))` — an application-level view of the model probability, not a statement that an account is malicious.

---

## 1. Project structure

```
SpamBot/
├── backend/
│   ├── app/                 FastAPI layer
│   │   ├── main.py          app factory, CORS, exception handlers
│   │   ├── api/             routers: health, models, predict, datasets, train, evaluation, explain, history, misc
│   │   ├── core/            config (env), logging, security (uploads, path guard, rate limit)
│   │   ├── db/              SQLAlchemy engine + ORM tables (predictions, datasets, models, training_runs, batches)
│   │   ├── schemas/api.py   Pydantic request/response models
│   │   └── services/        prediction, dataset, training (job manager), dashboard, adapters
│   ├── ml/                  framework-independent ML package
│   │   ├── features.py      FEATURE_GROUPS (31 paper features) + FeatureExtractor
│   │   ├── preprocessing.py two text paths (feature / sentiment), imputation rules
│   │   ├── sentiment.py     TextBlob polarity / subjectivity
│   │   ├── train.py         9 classifiers, stratified split, k-fold CV, randomised search, SHAP selection
│   │   ├── evaluation.py    metrics, confusion matrix, ROC / PR curves
│   │   ├── explain.py       SHAP (Tree/Kernel) global + local, LIME local
│   │   ├── predict.py       Predictor, risk score
│   │   ├── model_registry.py registry.json + artefact layout
│   │   ├── datasets.py      Cresci importer, CSV inspection
│   │   ├── demo_data.py     synthetic demo generator (labelled DEMO)
│   │   └── paper_results.py paper-reported tables (reference only)
│   ├── models/              trained artefacts (created by training)
│   ├── data/                SQLite DB, uploads, datasets/, exports/
│   ├── tests/               pytest suite (features, ML, SHAP, LIME, API)
│   └── requirements.txt
├── frontend/                React 18 + TypeScript + Vite + Tailwind v4 + Recharts + Lucide
│   └── src/{pages,components,layouts,hooks,services,types,utils,test}
├── scripts/                 fetch_datasets.py · train_model.py · evaluate_model.py · report_results.py · purge_demo.py · seed_demo.py
├── docs/                    paper-analysis.md · architecture.md · methodology.md · api.md · results.md
├── base-paper.pdf
├── docker-compose.yml · .env.example
```

## 2. Requirements

- **Python 3.10 – 3.12** (3.13/3.14 lack prebuilt wheels for `shap`/`lightgbm` on Windows). On Windows with several interpreters use `py -3.12`.
- **Node.js 18+** (tested with Node 24, npm 11).
- Optional: Docker Desktop.

## 3. Installation

### Backend

Windows (PowerShell / cmd):

```powershell
cd backend
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

### Environment

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

All variables are optional; see `.env.example` (CORS origins, upload limit, rate limit, demo mode, tweets-per-user for Cresci import).

## 4. Run

Backend (from `backend/`, venv active):

```bash
uvicorn app.main:app --reload            # http://localhost:8000  ·  Swagger: /docs  ·  ReDoc: /redoc
```

Frontend (from `frontend/`):

```bash
npm run dev                              # http://localhost:5173  (proxies /api → localhost:8000)
```

Docker (frontend on http://localhost:8080, backend on :8000, SQLite + models persisted in `backend/data` and `backend/models`):

```bash
docker compose up --build
```

## 5. Production setup (real data)

The production model is trained on the **real Cresci-2015 / Cresci-2017 benchmark data** (the base paper's datasets). A public, user-level mirror of the MIB `users.csv` files is downloaded, verified (SHA-256 recorded in `backend/data/datasets/PROVENANCE.md`) and split into the per-subset folder layout the importer expects. Row counts match the paper's Tables 2–3 exactly.

```bash
# repository root, backend venv active
python scripts/fetch_datasets.py                       # Cresci-15 + Cresci-17 (add --external for a 37k-account CC BY-SA dataset)
python scripts/train_model.py --dataset cresci-17 --all # all nine classifiers on Cresci-17 (paper's main dataset)
python scripts/train_model.py --dataset cresci-15 --all
python scripts/train_model.py --dataset cresci-combined --algorithm lightgbm   # production model on both datasets (activated)
python scripts/report_results.py                        # writes docs/results.md (measured numbers only)
```

Then start the backend and frontend (section 4). `.env` ships with `BOTSHIELD_ENVIRONMENT=production` and `BOTSHIELD_DEMO_MODE_ENABLED=false`, so no synthetic data can be generated; `python scripts/purge_demo.py` removes anything left over from demo runs.

**What the public mirror contains — and what it does not.** The mirror has account profiles only. The tweet files (`tweets.csv`, ~2.5 GB) are distributed by the dataset authors on request (MIB project) and are not redistributed here. Without them the 11 tweet-derived features of the paper (hashtag/mention/URL/retweet/reply counts and per-tweet averages) are constant and are dropped automatically, so production models use **20 of the 31 features**; linguistic and sentiment features come from the profile description. To restore the full 31-feature setup, place each subset's `tweets.csv` next to its `users.csv` and re-import (`--reimport`). Every model card states the features actually used.

## 6. Datasets

| Dataset | Source | Rows | Notes |
|---|---|---|---|
| Cresci-15 | MIB (Cresci et al. 2015), public user-level mirror | 5,301 | TFP, E13 (humans) · FSF, INT, TWT (fake followers) |
| Cresci-17 | MIB (Cresci et al. 2017), public user-level mirror | 12,737 | genuine, social spambots 1–3, traditional spambots #1, fake followers (paper Table 3) |
| Cresci-17-extra | same mirror | 1,631 | traditional spambots 2–4 — kept aside, not in the paper's composition |
| Twitter Human Bots (optional) | Hugging Face `airt-ml/twitter-human-bots`, CC BY-SA 3.0 | 37,438 | independent real dataset for cross-dataset evaluation (`--external`) |

Citations: Cresci et al., "Fame for sale: Efficient detection of fake Twitter followers", *DSS* 80 (2015); Cresci et al., "The paradigm-shift of social spambots", *WWW Companion* (2017). Terms of use are those of the original authors (academic/research).

Any other labelled CSV works too: one account per row with raw counts (`followers_count`, `friends_count`, `statuses_count`, `favourites_count`, `num_hashtags`, …) or pre-computed features, plus a label column (`label`, `is_bot`, `class`, `account_type`; values such as `bot/human`, `1/0`). Upload it on the Datasets page or pass `--csv` to the training script.

Layout expected by the importer (also usable for the authors' full distribution):

```
backend/data/datasets/cresci-15/{TFP,E13,FSF,INT,TWT}/users.csv[, tweets.csv]
backend/data/datasets/cresci-17/{genuine_accounts,social_spambots_1..3,traditional_spambots_1,fake_followers}/users.csv[, tweets.csv]
```

## 7. Preprocess, train, evaluate (CLI)

```bash
python scripts/train_model.py --dataset cresci-15 --all                 # imports (once) + trains all nine classifiers
python scripts/train_model.py --dataset cresci-17 --algorithm xgboost --feature-selection --top-k 20
python scripts/train_model.py --csv path/to/labelled.csv --algorithm random_forest --test-size 0.3 --cv-folds 5

python scripts/evaluate_model.py                                        # active model: stored hold-out + CV metrics
python scripts/evaluate_model.py --dataset cresci-15 --out eval.json    # active model on a dataset
python scripts/evaluate_model.py --model <model_id> --csv labelled.csv
```

Training stages: preprocessing → feature engineering (optional SHAP selection) → training (randomised search, stratified CV) → cross-validation → hold-out evaluation → SHAP analysis → saving. Artefacts: `backend/models/<model_id>/{pipeline.joblib, scaler.joblib, feature_metadata.json, metrics.json, shap_global.json, background.npy, lime_sample.npy}` plus `backend/models/registry.json` and convenience copies `best_model.joblib`, `scaler.joblib`, `feature_metadata.json` for the active model.

## 7b. Live X (Twitter) API integration

The Analyze page can pull a real account straight from X through the bundled **X API v2 adapter** (`backend/app/services/x_api_adapter.py`):

1. Create a project/app at https://developer.x.com and copy the app-only **Bearer Token** (Keys and tokens).
2. Put it in `.env`: `BOTSHIELD_X_BEARER_TOKEN=...` and restart the backend. Settings → Integrations shows `x_api · configured`.
3. Analyze Account → *Fetch a live account* → adapter "X (Twitter) API v2" → `@username` → Fetch → Analyze.

What it does: `GET /2/users/by/username/{username}` (profile + public metrics) and `GET /2/users/{id}/tweets?max_results=100` (recent tweets with entities and engagement) → mapped to the account schema → the usual 31-feature extraction, prediction, SHAP and LIME. Responses are cached for 10 minutes; 401/403/404/429 are surfaced with the API's reason (429 includes `Retry-After`).

Access-level caveats (stated in the UI, never hidden): reading other accounts' tweets requires the **Basic** tier or higher — on the Free tier the profile is fetched and the tweets error is shown; protected accounts expose no tweets. API v2 does not expose `default_profile`, `geo_enabled`, `profile_background_tile` or the banner, so these are sent as `false` and listed as `unavailable_fields`.

## 8. Tests

```bash
cd backend && .venv\Scripts\python -m pytest        # Windows   (52 tests: features, preprocessing, ML, SHAP, LIME, API, security)
cd backend && python -m pytest                      # macOS/Linux
cd frontend && npm test                             # vitest: API client, state handling, components
cd frontend && npm run typecheck && npm run build
```

## 9. Application pages

| Route | What it does |
|---|---|
| `/` Dashboard | cards (accounts analysed, bots, humans, avg P(bot), high-risk, current model) + charts from SQLite / active model artefacts; empty states when nothing is trained |
| `/analyze` | AI analysis console: account info, behavioural metrics, tweet text → prediction, confidence, risk, SHAP (bars/waterfall), LIME, key indicators, interpretation; *Load sample account* (DEMO); sample adapter |
| `/batch-analysis` | CSV or registered dataset → batch prediction, distribution, evaluation if labelled, `predictions.csv` download |
| `/datasets` | upload/inspect CSVs (schema, missing, duplicates, class distribution, feature availability), demo generator, Cresci import |
| `/models` | nine classifiers, registry with measured metrics, comparison chart, paper-reported tables (separate) |
| `/training` | select dataset/model/test size/CV folds/search/SHAP selection → async job with stage tracker → results |
| `/evaluation` | metrics, confusion matrix, ROC, PR, CV folds, split info, feature set |
| `/explainability` | global SHAP (bar, beeswarm, group importance, feature distribution) + local SHAP/LIME for any stored prediction |
| `/history` | filterable prediction history; row → full analysis |
| `/research` · `/architecture` · `/api-docs` · `/settings` | academic context, system diagrams, Swagger, configuration |

## 10. Results

Measured results (hold-out + 5-fold CV per classifier and dataset, cross-dataset evaluation) are generated into [docs/results.md](docs/results.md) by `scripts/report_results.py` and shown live on the Models / Evaluation pages under "Reproduced by this implementation". Headline numbers from the current registry (stratified 25 % hold-out, real Cresci user-level data, 20 features):

| Model | Trained on | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| LightGBM **(active, production)** | Cresci-15 + Cresci-17 combined (14,687 accounts) | 0.980 | 0.987 | 0.981 | 0.984 | 0.995 |
| LightGBM | Cresci-17 (12,737) | 0.986 | 0.994 | 0.987 | 0.991 | 0.997 |
| XGBoost | Cresci-15 (5,301) | 0.982 | 0.990 | 0.981 | 0.986 | 0.997 |

Cross-dataset: the Cresci-17 model reaches F1 0.947 on Cresci-15 unseen, while the Cresci-15 model (fake followers only) reaches F1 0.607 on Cresci-17 — which is why the production model is trained on both. The paper's numbers (Cresci-15 LightGBM 0.991 / 0.993; Cresci-17 XGBoost 0.990 / 0.993, 5-fold CV with all 31 features) are shown separately under "Reported in base paper" and are not comparable one-to-one (docs/results.md §4).

## 11. How the implementation maps to the paper

See `docs/paper-analysis.md` (what the paper says), `docs/methodology.md` (what we implemented, formulas, adaptations) and `docs/architecture.md`. Summary: the exact 31-feature set of Table 4 in six groups; the paper's two-path preprocessing; nine classifiers with stratified split + 5-fold CV and CV-driven hyperparameter search; SHAP for feature selection and global/local attribution; LIME for local explanations; Cresci-15/17 as the intended benchmark datasets.

## 12. Known limitations

- The public Cresci mirror is user-level only; the 11 tweet-derived features need the authors' `tweets.csv` files (request from MIB). Production models therefore use 20 of 31 features until those files are added.
- Cresci text-derived features (when tweets are available) use up to N tweets per account (configurable) for tractability.
- XGBoost SHAP values are reported in log-odds when the installed shap/xgboost pair cannot compute probability-scale values; the scale is shown in the UI.
- Kernel SHAP (SVM, LR, NB, AdaBoost) is slower and uses capped samples for global importance.
- Live X integration depends on your developer tier (tweets need Basic+) and X API v2 omits four v1.1 profile flags used by the paper.
- Single-worker in-process training job queue (sufficient for a local academic deployment; not horizontally scalable).
- No authentication (local/academic use). Uploads are validated and rate-limited, but the API is not multi-tenant.

## 13. Future improvements

- OAuth user-context for the X adapter (reads protected accounts the user can access) and a scheduled re-scan of watch-listed accounts.
- Temporal features (tweet timing) and graph features, which the paper lists as future work, plus adaptive/continual retraining.
- Persisted job queue (e.g. RQ/Celery) and authentication for shared deployments.
- Cross-dataset generalisation experiments (train on Cresci-15, test on Cresci-17) exposed in the UI.

## Appendix — Demo Mode (optional, off by default)

Set `BOTSHIELD_DEMO_MODE_ENABLED=true` to allow generating a small synthetic dataset (`python scripts/seed_demo.py` or Datasets → Generate demo dataset). Everything derived from it is flagged `is_demo` and banner-labelled **DEMO DATA — NOT REAL SOCIAL MEDIA DATA**; it exists only for classroom demonstrations without data and is never a research result. `python scripts/purge_demo.py` removes all demo artefacts.

## License / attribution

The base paper is CC BY 4.0. This project is an academic implementation; the paper's authors are not affiliated with it.
