# Deployment guide — BotShield AI

This document describes how to run BotShield AI in production: topology, environment variables, first-run workflow, TLS, backups and recovery, and the two supported deployment paths (Docker Compose and Render).

## 1. Topology

```
                     ┌────────────────────────────────────────────────┐
  HTTPS              │  reverse proxy / TLS termination (Caddy, Traefik, LB)
  ───────────────────►  https://app.example.com                        │
                     └───────────────┬────────────────────────────────┘
                                     │ HTTP
                     ┌───────────────▼───────────────┐
                     │ frontend (nginx, static SPA)   │  /api/* → api:8000
                     └───────────────┬───────────────┘
                                     │
   ┌────────────┐    ┌───────────────▼───────────────┐    ┌─────────────┐
   │ PostgreSQL │◄───┤ api (gunicorn + uvicorn)       ├───►│ Redis       │
   │ 16         │    │ Alembic migrations at boot     │    │ (Celery)    │
   └─────▲──────┘    └───────────────────────────────┘    └──────▲──────┘
         │           ┌───────────────────────────────┐           │
         └───────────┤ worker (Celery)                ├───────────┘
                     │ training · batch · imports     │
                     └───────────────┬───────────────┘
                                     ▼
                     object storage (S3-compatible) or shared volume
                     org/<org>/datasets|models|batches/...
```

| Service | Image | Role |
|---|---|---|
| `frontend` | `frontend/Dockerfile` (nginx) | Serves the built SPA, proxies `/api/` to the API so cookies stay first-party |
| `api` | `backend/Dockerfile` (`CMD api`) | Runs `alembic upgrade head`, optional first-admin creation, then gunicorn with uvicorn workers |
| `worker` | `backend/Dockerfile` (`CMD worker`) | Celery worker executing jobs from the `jobs` table (training, batch prediction, benchmark import) |
| `db` | `postgres:16-alpine` | System of record: users, organizations, datasets, models, predictions, explanations, jobs, audit |
| `redis` | `redis:7-alpine` | Celery broker only (no application state) |

Single-container installs can run with `BOTSHIELD_JOB_BACKEND=thread` (jobs execute in the API process) and `BOTSHIELD_STORAGE_BACKEND=local` on a persistent volume. That is acceptable for small teams; the Compose file above is the recommended topology.

## 2. Environment variables

All settings are read from environment variables prefixed `BOTSHIELD_` (see `backend/app/core/config.py`). Templates: `.env.example` (development) and `.env.production.example` (production). **Never commit a real `.env`.**

| Variable | Required (prod) | Default | Notes |
|---|---|---|---|
| `BOTSHIELD_ENVIRONMENT` | yes | `development` | `production` enables JSON logs, secure cookies, disables docs, forbids SQLite |
| `BOTSHIELD_DATABASE_URL` | yes | SQLite (dev only) | `postgresql+psycopg://user:pass@host:5432/db`; `postgres://` URLs are normalised automatically |
| `BOTSHIELD_DB_SCHEMA` | no | – | PostgreSQL schema to own all BotShield tables (created by `migrate`); set when the database is shared with another application |
| `BOTSHIELD_SECRET_KEY` | yes | — | JWT signing + refresh-token hashing. `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Rotating it logs every user out |
| `BOTSHIELD_CORS_ORIGINS` | yes | — | Exact browser origins, comma-separated. Not needed when the SPA proxies `/api/` same-origin, but set it anyway |
| `BOTSHIELD_COOKIE_SECURE` | yes | true in prod | Refresh cookie `Secure` flag; requires HTTPS |
| `BOTSHIELD_COOKIE_SAMESITE` | no | `strict` | `none` only when SPA and API are on different sites (then `COOKIE_SECURE` must be true) |
| `BOTSHIELD_COOKIE_DOMAIN` | no | — | Set when the API is on a sub-domain of the SPA |
| `BOTSHIELD_TRUSTED_PROXIES` | no | — | `*` or a list of proxy IPs; enables `X-Forwarded-*` handling (rate limits, HTTPS detection) |
| `BOTSHIELD_ACCESS_TOKEN_MINUTES` / `BOTSHIELD_REFRESH_TOKEN_DAYS` | no | 30 / 7 | Token lifetimes |
| `BOTSHIELD_STORAGE_BACKEND` | yes | `local` | `s3` recommended. `local` needs a persistent volume at `BOTSHIELD_STORAGE_LOCAL_ROOT` shared by api and worker |
| `BOTSHIELD_S3_BUCKET`, `_S3_ENDPOINT_URL`, `_S3_REGION`, `_S3_ACCESS_KEY_ID`, `_S3_SECRET_ACCESS_KEY`, `_S3_PREFIX` | when s3 | — | Any S3-compatible store (AWS S3, MinIO, Cloudflare R2, Backblaze B2) |
| `BOTSHIELD_JOB_BACKEND` | yes | `thread` | `celery` requires `BOTSHIELD_REDIS_URL` and the worker service |
| `BOTSHIELD_REDIS_URL` | when celery | — | `redis://redis:6379/0` |
| `BOTSHIELD_DOCS_ENABLED` | no | false in prod | Exposes Swagger/ReDoc |
| `BOTSHIELD_RATE_LIMIT_PER_MINUTE` / `BOTSHIELD_ML_RATE_LIMIT_PER_MINUTE` / `BOTSHIELD_LOGIN_RATE_LIMIT_PER_MINUTE` | no | 120 / 30 / 10 | Per IP / per user / per IP+email |
| `BOTSHIELD_MAX_UPLOAD_MB` / `BOTSHIELD_MAX_REQUEST_BODY_MB` | no | 50 / 64 | Upload limits; keep the proxy's `client_max_body_size` ≥ these |
| `BOTSHIELD_LOG_LEVEL` / `BOTSHIELD_LOG_JSON` | no | INFO / true in prod | Structured logs with request ids; secrets are never logged |
| `BOTSHIELD_RETENTION_PREDICTIONS_DAYS` / `BOTSHIELD_RETENTION_AUDIT_DAYS` | no | unset | Automatic purge at startup and daily; unset = keep |
| `BOTSHIELD_X_BEARER_TOKEN` | no | — | Enables the X API v2 provider. Without it the UI says "External API integration is not configured" |
| `BOTSHIELD_ADMIN_EMAIL` / `_ADMIN_PASSWORD` / `_ADMIN_ORG` | first boot only | — | Consumed once by `create-admin`; remove afterwards |
| `WEB_CONCURRENCY`, `GUNICORN_TIMEOUT`, `CELERY_CONCURRENCY`, `PORT` | no | 2 / 120 / 2 / 8000 | Process tuning (entrypoint) |

Frontend: only `VITE_API_BASE_URL` exists and it is **not secret**. Leave it empty when nginx/Render proxies `/api/` same-origin; set it to the API origin only for split-domain deployments (then `BOTSHIELD_COOKIE_SAMESITE=none`).

## 3. First run (any platform)

```bash
# 1. secrets
cp .env.production.example .env          # fill every CHANGE_ME; generate SECRET_KEY
# 2. start
docker compose -f docker-compose.prod.yml up -d --build
# 3. migrations run automatically; verify
curl -s https://app.example.com/api/v1/health/ready
# 4. first administrator (interactive; refuses if any user exists)
docker compose -f docker-compose.prod.yml exec api python -m app.cli create-admin
# 5. sign in, upload a labelled dataset, train, promote to production
```

`python -m app.cli check-config` prints the effective configuration with secrets masked. `python -m app.cli migrate` applies migrations manually (the API refuses to start in production when the schema is behind `alembic head`).

There are **no default accounts**. The first admin is created only by `create-admin`, which enforces the password policy. Additional users are created by an admin from Settings → Users; the initial password is shown to the admin once and never stored in clear.

## 4. TLS / reverse proxy

The containers speak plain HTTP; terminate TLS in front of `frontend:80`. Caddy example (automatic Let's Encrypt):

```
app.example.com {
    encode gzip
    reverse_proxy frontend:80
}
```

nginx/Traefik work the same way. Requirements:

- forward `X-Forwarded-Proto` and `X-Forwarded-For`, and set `BOTSHIELD_TRUSTED_PROXIES` so the API trusts them;
- keep `client_max_body_size` ≥ `BOTSHIELD_MAX_REQUEST_BODY_MB`;
- the API sets `Strict-Transport-Security` when `COOKIE_SECURE` is true, plus `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, and `Cache-Control: no-store` on API responses.

## 5. Render (free tier)

`render.yaml` is a Render Blueprint: managed PostgreSQL, the backend as a Docker web service and the frontend as a static site whose `/api/*` route is rewritten to the backend (same-origin, so the strict refresh cookie works).

1. Push to GitHub, then Render → **New → Blueprint** → select the repository → **Apply**.
2. In the backend service set `BOTSHIELD_ADMIN_EMAIL`, `BOTSHIELD_ADMIN_PASSWORD` (≥ 12 chars, mixed) and `BOTSHIELD_ADMIN_ORG` *before* the first boot — or run `python -m app.cli create-admin` from the service shell afterwards.
3. If Render gives the backend another hostname than `botshield-backend.onrender.com`, edit the rewrite destination and `BOTSHIELD_CORS_ORIGINS` in `render.yaml` and re-apply.

Free-tier caveats (documented, not hidden):

- **No persistent disk** → with `STORAGE_BACKEND=local` uploaded datasets and model artefacts are lost on redeploy/restart, although database rows survive. Provide an S3-compatible bucket (`BOTSHIELD_S3_*`) and set `BOTSHIELD_STORAGE_BACKEND=s3` for durable storage.
- **No Redis/worker** → `JOB_BACKEND=thread`; training runs inside the single 512 MB web instance. Use `hyperparameter_search=false` or few iterations for large datasets.
- Free web services sleep after inactivity (first request is slow) and the free database expires after 90 days unless upgraded.

## 6. Backups and recovery

What must be backed up:

| Data | Where | Method |
|---|---|---|
| Relational data (users, orgs, datasets/versions metadata, models metadata, predictions, explanations, jobs, audit) | PostgreSQL | `pg_dump` |
| Artefacts (dataset CSVs, model pipelines + checksums, batch outputs) | S3 bucket or `api-data` volume | bucket versioning / snapshot of the volume |
| Secrets | `.env` | secret manager; not in git |

Daily backup example (Compose):

```bash
docker compose -f docker-compose.prod.yml exec -T db pg_dump -U "$POSTGRES_USER" -Fc botshield > backups/botshield-$(date +%F).dump
# local storage backend only:
docker run --rm -v spambot_api-data:/data -v "$PWD/backups":/b alpine tar czf /b/storage-$(date +%F).tgz -C /data storage
```

Restore:

```bash
docker compose -f docker-compose.prod.yml stop api worker
docker compose -f docker-compose.prod.yml exec -T db pg_restore -U "$POSTGRES_USER" -d botshield --clean --if-exists < backups/botshield-YYYY-MM-DD.dump
# restore the storage snapshot / bucket to the same prefix, then
docker compose -f docker-compose.prod.yml start api worker
```

Model artefacts are verified against the SHA-256 checksums stored in the database before loading; a restore with mismatched artefacts fails closed (the model cannot be activated or used) instead of loading a tampered file.

Migrations are forward-only in production; `alembic downgrade` exists for development. Roll back a release by restoring the database dump taken before deployment.

## 7. Operations checklist

- `GET /api/v1/health` (liveness) and `GET /api/v1/health/ready` (database, migrations, storage) for probes.
- Logs are JSON lines with `request_id`, `user_id`, `organization_id`, latency and status; the `security` logger records auth events and password-reset tokens (delivery is the operator's job — there is no e-mail integration).
- Rotate `BOTSHIELD_SECRET_KEY` to invalidate every session; use Settings → Users → *Reset password* for a single account.
- `python -m app.cli purge-expired-sessions` can run from cron; retention purges run automatically when the retention variables are set.
- `python scripts/verify_production_readiness.py --url https://app.example.com --strict` runs the static and runtime readiness checks (see `docs/production-completion-report.md`).
