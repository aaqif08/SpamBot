"""Small shared helpers for the ML package."""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises and never returns NaN/inf."""
    try:
        if denominator == 0 or denominator is None:
            return default
        value = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def to_float(value: Any, default: float = 0.0) -> float:
    """Coerce arbitrary CSV/JSON values to a finite float."""
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        return default if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("", "nan", "none", "null", "na", "n/a"):
            return default
        if s in ("true", "yes", "y", "t"):
            return 1.0
        if s in ("false", "no", "n", "f"):
            return 0.0
        try:
            f = float(s)
            return default if (math.isnan(f) or math.isinf(f)) else f
        except ValueError:
            return default
    try:
        f = float(value)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return not math.isnan(float(value)) and float(value) != 0.0
        except (TypeError, ValueError):
            return False
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "t", "1.0")
    return bool(value)


def to_text(value: Any) -> str:
    """Return a clean string, treating NaN / None / 'nan' as empty."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value)
    if s.strip().lower() in ("nan", "none", "null"):
        return ""
    return s


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy / non-serialisable values to plain Python."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(json_safe(obj), ensure_ascii=False)


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager measuring wall-clock seconds: ``with timer() as t: ...; t['seconds']``."""
    result: dict[str, float] = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
