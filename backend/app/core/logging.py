"""Structured logging.

* ``app`` logger family — application logs
* ``security`` logger — authentication / authorization events (OWASP logging)
* ``ml.jobs`` logger — background job progress

In staging/production records are emitted as single-line JSON with timestamp,
level, logger, message, request id, and any extra fields; in development a
readable text format is used. Secrets are never logged: the request-logging
middleware records method, path, status, latency and the request id only.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")
job_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("job_id", default="-")

_RESERVED = set(logging.LogRecord("x", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}
_SENSITIVE_KEYS = {"password", "token", "secret", "authorization", "cookie", "refresh_token", "access_token", "bearer"}


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("***" if any(s in str(k).lower() for s in _SENSITIVE_KEYS) else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        uid = user_id_var.get()
        if uid != "-":
            payload["user_id"] = uid
        jid = job_id_var.get()
        if jid != "-":
            payload["job_id"] = jid
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = _scrub(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{time.strftime('%H:%M:%S', time.localtime(record.created))} | {record.levelname:<7} | {record.name} | rid={request_id_var.get()} | {record.getMessage()}"
        extras = {k: _scrub(v) for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
        if extras:
            base += " | " + json.dumps(extras, default=str)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "shap", "lime", "matplotlib", "numba", "httpx", "httpcore", "botocore", "boto3", "urllib3", "celery.utils.functional"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


app_log = logging.getLogger("app")
security_log = logging.getLogger("security")
job_log = logging.getLogger("ml.jobs")
