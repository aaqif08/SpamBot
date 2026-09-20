"""Lightweight ML-level data structures shared between ``ml`` modules.

The HTTP layer has its own Pydantic schemas in ``app/schemas``; these
dataclasses keep the ML package independent of FastAPI/Pydantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionOutcome:
    prediction: str
    bot_probability: float
    human_probability: float
    risk_score: int
    model_id: str
    features: dict[str, float] = field(default_factory=dict)
    shap: dict[str, Any] | None = None
    lime: dict[str, Any] | None = None


@dataclass
class DatasetSummary:
    n_rows: int
    n_columns: int
    has_label: bool
    class_distribution: dict[str, int] | None
    feature_coverage: float


@dataclass
class JobState:
    job_id: str
    status: str = "queued"  # queued | running | completed | failed
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    log: list[str] = field(default_factory=list)
