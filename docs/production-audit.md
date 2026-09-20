# Production audit — BotShield AI (pre-conversion state)

Audit date: 2026-09-19. Scope: the complete repository as it existed before the production conversion (academic prototype built on 2026-09-18/19). Every finding below is tied to a file; the "Recommended change" column became the implementation plan.

## A. Current architecture

| Layer | Implementation | Verdict |
|---|---|---|
| Frontend | React 18 + TypeScript + Vite 6 + Tailwind v4 + Recharts + Lucide; 13 pages; typed fetch client `src/services/api.ts`; no auth | Structure sound, no auth/tenancy, demo affordances |
| Backend | FastAPI (`backend/app`), routers under `/api/*` (unversioned), Pydantic v2 schemas, services layer | Sound skeleton; no authentication, no authorization, no tenancy |
| ML | Framework-independent `backend/ml` package: 31-feature extractor, 9 classifiers, stratified CV + randomised search, SHAP (Tree/Kernel), LIME, evaluation | Production quality; keep |
| Persistence | SQLite via SQLAlchemy 2; `Base.metadata.create_all` at start-up; no migrations | Not production grade |
| Model registry | `backend/models/registry.json` (filesystem JSON) + joblib artefacts; global (no owner) | Needs DB-backed metadata, ownership, checksums |
| Jobs | In-process `ThreadPoolExecutor(max_workers=1)`; job state in memory + `training_runs` row | Lost on restart; not horizontally scalable |
| Storage | Hard-coded local paths under `backend/data` and `backend/models` | Needs storage abstraction |
| Deployment | Dockerfiles + `docker-compose.yml` (backend uvicorn, frontend nginx, SQLite volume) | No Postgres, no Redis/worker, no TLS layer |

## B. Demo / mock / sample data locations

| Location | What | Executes in production? |
|---|---|---|
| `backend/ml/demo_data.py` | synthetic account generator (`generate_demo_accounts`), hand-written `sample_accounts()` | Yes — reachable through `POST /api/datasets/demo` (gated by `BOTSHIELD_DEMO_MODE_ENABLED`) and `GET /api/sample-accounts` (always on) |
| `backend/app/services/adapter_service.py` `SampleAdapter` | returns hand-written accounts through `POST /api/adapters/sample/fetch` | Yes |
| `scripts/seed_demo.py` | seeds synthetic dataset + model + predictions into the live DB | Manual, but writes fake business data |
| `scripts/purge_demo.py` | removes the above | n/a |
| `frontend/src/pages/AnalyzePage.tsx` | "Load sample account (DEMO)…" selector; sample adapter option | Yes |
| `frontend/src/components/ui/index.tsx` `DemoBanner` | demo banner component | Yes (rendered when `is_demo`) |
| `frontend/src/pages/{Datasets,Training,Models,Dashboard,Evaluation,Explainability}.tsx` | demo generation buttons, demo badges/notices, "DEMO data" comparison series | Yes |
| `backend/data/datasets/demo/` (git-ignored) | generated demo CSV | dev artefact |
| `is_demo` columns on `predictions`, `datasets`, `models`, `training_runs`, `batches` | flags for demo provenance | schema debt |

Not demo (keep, relabel): `backend/ml/paper_results.py` — the paper's published tables, rendered only under "Research paper results"; `backend/data/datasets/cresci-*` and `PROVENANCE.md` — real benchmark data; `scripts/fetch_datasets.py` — real data download.

## C. Hard-coded values

- `frontend/src/pages/ResearchPage.tsx` `IMPLEMENTATION_MAP` / `SECTIONS` — static academic text (acceptable: documentation, not metrics).
- `frontend/src/pages/ApiDocsPage.tsx` `ENDPOINTS` — static endpoint list (must be regenerated for `/api/v1`).
- `frontend/src/pages/SettingsPage.tsx` env-var table — static documentation of variables; several rows had no backend implementation behind them.
- No hard-coded metrics, charts, timestamps or prediction outputs were found in production code paths; all chart data comes from `/api/dashboard`, `/api/evaluation`, `/api/explain/*`.
- `X_API` adapter description strings; Cresci paper statistics in `ml/datasets.py` `CRESCI_PAPER_STATS` (labelled "reported in base paper").

## D. Production blockers

1. No authentication or authorization — every endpoint public, including training and deletion.
2. No tenant boundary — all records global.
3. SQLite + `create_all` — no migrations, no PostgreSQL.
4. Demo/sample code paths reachable at runtime.
5. In-memory job queue — job state lost on restart; single process.
6. Filesystem registry with no ownership, no artefact checksum.
7. Unversioned API (`/api/*`).
8. No audit trail; unstructured logs; no request IDs.
9. Secrets/config: `cors_origins` string, no `SECRET_KEY`, no cookie/proxy settings.
10. Frontend: `window.confirm` dialogs, no login, `is_demo` sprinkled through types.

## E. Security issues

- Public endpoints for expensive ML operations (train, batch, explain) — abuse/DoS risk; only an in-memory per-IP limiter.
- No CSRF consideration (no cookies yet), no security headers, no body-size limit middleware beyond upload cap.
- `DELETE /api/models/{id}` and dataset deletion unauthenticated.
- Model artefacts loaded with `joblib.load` from the registry directory without checksum verification.
- Error envelope OK (`{detail, code}`), stack traces hidden — keep.
- Upload validation (extension, MIME, size, binary sniff, UUID names, path-traversal guard) — keep.

## F. Database issues

- SQLite only; `create_all` at start-up; no Alembic.
- No `updated_at`, no ownership (`organization_id`, `created_by`), no `users`, `organizations`, `audit_logs`, `jobs`, `dataset_versions`, `model_versions`, `explanations`, `system_settings`.
- Hex UUID strings without type annotation of intent; fine for cross-DB portability — keep as CHAR(36)/String(36).
- `predictions.result_json` duplicates several columns.

## G. ML issues

- Pipeline itself is correct and tested (52 tests). Issues are integration-level:
  - `ml/model_registry.py` mixes artefact storage with "active model" state and convenience copies (`best_model.joblib`) — state belongs in the DB, per organisation.
  - `ml/demo_data.py` lives in the production package.
  - No artefact checksum; no feature-schema compatibility check at load time.

## H. API issues

- Unversioned; no auth; `GET /api/sample-accounts`, `POST /api/datasets/demo`, `POST /api/adapters/sample/fetch` are demo endpoints.
- Batch prediction runs synchronously inside the request.
- History filters exist; no sort parameter; page size max 200 (fine).
- OpenAPI has no security scheme.

## I. Frontend issues

- No login/session; API base unversioned.
- Demo affordances (above); `window.confirm` (not accessible); `DemoBanner`.
- Empty states exist on most pages but Dashboard renders empty charts when there are zero predictions in some tiles.
- Settings page lists variables that the backend does not implement (`BOTSHIELD_X_MAX_TWEETS` fine; notifications none).
- Language: "BOT / spambot-like" headline is acceptable; some copy ("Bots detected") should read "Model-classified bots".

## J. Deployment issues

- Compose runs uvicorn directly (no gunicorn/worker count), SQLite volume, no Postgres/Redis/worker, no TLS proxy, no `.env.production.example`.
- `.gitignore` covers `.env`, data and models — keep.

## K. Testing gaps

- No auth/authz tests, no tenancy isolation tests, no migration test, no audit-log tests, no storage tests, no frontend routing/auth tests.

## L. Recommended changes (implemented in this conversion)

1. Delete demo generator/sample adapter/seed scripts from production code; move the synthetic generator to `backend/tests/fixtures/` for tests only.
2. Add `organizations`, `users`, `refresh_sessions`, `password_reset_tokens`, `audit_logs`, `jobs`, `datasets` + `dataset_versions`, `models` (+ versions), `predictions`, `explanations`, `batches`, `system_settings` with `organization_id`, `created_by`, `created_at`, `updated_at`; Alembic migrations; PostgreSQL in production, SQLite for local dev/tests.
3. JWT access tokens + httpOnly refresh cookie sessions, bcrypt password hashing, roles ADMIN/ANALYST/VIEWER, login rate limiting, audit logging.
4. Storage abstraction (`LocalStorage`, `S3Storage`), artefacts keyed per organisation, SHA-256 checksums verified on load.
5. Job system with persisted status (`jobs` table) executed by Celery workers (Redis broker) or, for single-node deployments, an in-process executor selected by `BOTSHIELD_JOB_BACKEND`.
6. `/api/v1/*` versioned routers; `/health` liveness and `/api/v1/health/ready` readiness.
7. Frontend: login, auth context with silent refresh, protected routes, role-aware UI, real empty states, accessible dialogs, no demo affordances.
8. Structured JSON logs with request IDs; security and ML job loggers.
9. Docker: postgres, redis, api (gunicorn/uvicorn), worker, frontend (nginx), Caddy TLS proxy example; production env template.
10. Scripts: `create_admin`, `check_no_demo_data`, `verify_production_readiness`, retained `fetch_datasets`, `train_model` (org-scoped), `report_results`.

## M. Production-readiness checklist (tracked in `docs/production-completion-report.md`)

- [ ] No demo/sample code paths in production
- [ ] Authentication + RBAC + tenancy enforced server-side
- [ ] Migrations, PostgreSQL support, empty first boot
- [ ] Persisted asynchronous jobs
- [ ] Storage abstraction + artefact checksums
- [ ] Audit logs + structured logs + readiness probe
- [ ] Versioned, documented API with security scheme
- [ ] Frontend: auth, empty states, accessibility, production build clean
- [ ] Docker production topology + TLS guidance + backups
- [ ] Tests: backend/frontend/integration green
- [ ] Empty-database verification walkthrough executed
