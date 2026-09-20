"""Security primitives: password hashing, JWT access tokens, opaque refresh
tokens, rate limiting, upload validation, filename sanitisation.

Rate limiting is an in-memory sliding window per process. It is sufficient for
single-node deployments; behind several API replicas the limits apply per
replica (documented in docs/deployment.md).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bcrypt
import jwt
from fastapi import HTTPException, Request, UploadFile, status

from app.core.config import get_settings

# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #

_PASSWORD_UPPER = re.compile(r"[A-Z]")
_PASSWORD_LOWER = re.compile(r"[a-z]")
_PASSWORD_DIGIT = re.compile(r"\d")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def validate_password_strength(password: str) -> list[str]:
    """Return a list of problems (empty = acceptable)."""
    s = get_settings()
    problems: list[str] = []
    if len(password) < s.password_min_length:
        problems.append(f"at least {s.password_min_length} characters")
    if not _PASSWORD_UPPER.search(password):
        problems.append("an uppercase letter")
    if not _PASSWORD_LOWER.search(password):
        problems.append("a lowercase letter")
    if not _PASSWORD_DIGIT.search(password):
        problems.append("a digit")
    if len(password) > 256:
        problems.append("at most 256 characters")
    return problems


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #

JWT_ALGORITHM = "HS256"


def create_access_token(user_id: str, organization_id: str, role: str, minutes: int | None = None) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "org": organization_id,
        "role": role,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes or s.access_token_minutes)).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, s.secret_key, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    s = get_settings()
    try:
        payload = jwt.decode(token, s.secret_key, algorithms=[JWT_ALGORITHM], options={"require": ["exp", "sub", "org", "role"]})
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired", headers={"WWW-Authenticate": "Bearer"}) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"}) from exc
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})
    return payload


def new_opaque_token() -> str:
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    """Refresh/reset tokens are stored hashed (HMAC-SHA256 keyed with the app secret)."""
    return hmac.new(get_settings().secret_key.encode("utf-8"), token.encode("utf-8"), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, int]:
        """Register a hit; return (allowed, retry_after_seconds)."""
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, int(self.window - (now - q[0])) + 1
            q.append(now)
            return True, 0

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


def client_ip(request: Request) -> str:
    """Client address, honouring X-Forwarded-For only when the immediate peer is a trusted proxy."""
    s = get_settings()
    peer = request.client.host if request.client else "unknown"
    trusted = s.trusted_proxy_list
    if trusted and ("*" in trusted or peer in trusted):
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()[:64]
    return peer


def raise_rate_limited(retry_after: int) -> None:
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded; try again shortly",
        headers={"Retry-After": str(retry_after)},
    )


# --------------------------------------------------------------------------- #
# Uploads / paths
# --------------------------------------------------------------------------- #

ALLOWED_EXTENSIONS = {".csv"}
ALLOWED_CONTENT_TYPES = {"text/csv", "application/vnd.ms-excel", "application/csv", "text/plain", "application/octet-stream"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    base = Path(name or "upload.csv").name
    base = _SAFE_NAME_RE.sub("_", base).strip("._") or "upload.csv"
    return base[:120]


def safe_join(root: Path, *parts: str) -> Path:
    root = root.resolve()
    target = root.joinpath(*parts).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")
    return target


async def read_validated_csv(upload: UploadFile, max_mb: int) -> bytes:
    filename = sanitize_filename(upload.filename or "")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Only .csv files are accepted")
    if upload.content_type and upload.content_type.split(";")[0].strip() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=f"Unsupported content type '{upload.content_type}'")
    limit = max_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File exceeds the {max_mb} MB limit")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
    head = data[:4096]
    if b"\x00" in head or head.startswith(b"PK") or head.startswith(b"%PDF"):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File does not look like text/CSV")
    return data


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
