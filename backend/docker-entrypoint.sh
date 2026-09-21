#!/bin/sh
# Entrypoint for the BotShield API / worker image.
#   api    -> run Alembic migrations, optionally create the first admin, then serve with gunicorn
#   worker -> Celery worker (requires BOTSHIELD_JOB_BACKEND=celery and BOTSHIELD_REDIS_URL)
#   *      -> exec the given command
set -e

case "$1" in
  api)
    echo "[entrypoint] applying database migrations"
    python -m app.cli migrate
    if [ -n "$BOTSHIELD_ADMIN_EMAIL" ] && [ -n "$BOTSHIELD_ADMIN_PASSWORD" ]; then
      # Idempotent: exit 3 means users already exist (fine); anything else is a real problem
      # (e.g. password policy) and must be visible in the deploy log, but must not block serving.
      set +e
      python -m app.cli create-admin
      rc=$?
      set -e
      case "$rc" in
        0) echo "[entrypoint] initial administrator created; remove BOTSHIELD_ADMIN_* from the environment" ;;
        3) echo "[entrypoint] users already exist; BOTSHIELD_ADMIN_* ignored" ;;
        *) echo "[entrypoint] WARNING: create-admin failed (exit $rc) - fix BOTSHIELD_ADMIN_* or run 'python -m app.cli create-admin' in a shell" ;;
      esac
    else
      echo "[entrypoint] BOTSHIELD_ADMIN_EMAIL/PASSWORD not set; create the first administrator with: python -m app.cli create-admin"
    fi
    WORKERS="${WEB_CONCURRENCY:-2}"
    PORT="${PORT:-8000}"
    echo "[entrypoint] starting gunicorn on :$PORT with $WORKERS workers"
    exec gunicorn app.main:app \
      --worker-class uvicorn.workers.UvicornWorker \
      --workers "$WORKERS" \
      --bind "0.0.0.0:$PORT" \
      --timeout "${GUNICORN_TIMEOUT:-120}" \
      --graceful-timeout 30 \
      --access-logfile - --error-logfile - \
      --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}"
    ;;
  worker)
    echo "[entrypoint] starting celery worker"
    exec celery -A app.worker.celery_app worker \
      --loglevel "${CELERY_LOG_LEVEL:-INFO}" \
      --concurrency "${CELERY_CONCURRENCY:-2}" \
      --max-tasks-per-child 20
    ;;
  *)
    exec "$@"
    ;;
esac
