# BotShield AI — API / worker image (production)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/app/nltk_data

# libgomp is required by LightGBM / XGBoost wheels; curl for the health check
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system botshield && useradd --system --gid botshield --home /app botshield

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && python -m nltk.downloader -d /app/nltk_data vader_lexicon punkt stopwords

COPY alembic.ini .
COPY alembic ./alembic
COPY app ./app
COPY ml ./ml
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/data/storage /app/data/cache /app/data/datasets \
    && chown -R botshield:botshield /app

USER botshield
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/api/v1/health" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
# "api" runs migrations then gunicorn; "worker" runs the Celery worker; anything else is exec'd as-is
CMD ["api"]
