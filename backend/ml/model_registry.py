"""Model artefact conventions.

Model *metadata* lives in the application database (``models`` table); this
module only defines the artefact layout produced by training and helpers to
checksum / verify it. Artefacts are persisted through the storage abstraction
by the service layer.

Artefact directory layout::

    pipeline.joblib          sklearn Pipeline(imputer -> scaler -> clf)
    scaler.joblib            preprocessing sub-pipeline
    feature_metadata.json    ordered feature list, version, dropped features, ranges
    metrics.json             hold-out + CV metrics, curves, confusion matrix
    shap_global.json         mean |SHAP| ranking + beeswarm sample
    background.npy           scaled background rows for explainers
    lime_sample.npy          scaled training rows for LIME
    lime_sample_raw.npy      raw training rows (for display)
    CHECKSUMS.json           sha256 of every file above
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ARTIFACT_FILES: tuple[str, ...] = (
    "pipeline.joblib",
    "scaler.joblib",
    "feature_metadata.json",
    "metrics.json",
    "shap_global.json",
    "background.npy",
    "lime_sample.npy",
    "lime_sample_raw.npy",
)
CHECKSUM_FILE = "CHECKSUMS.json"


@dataclass
class ModelCard:
    """Framework-independent description of a trained model version."""

    name: str
    algorithm: str
    feature_version: str
    feature_names: list[str]
    holdout_metrics: dict[str, float]
    cv_metrics: dict[str, float]
    cv_std: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)
    training_seconds: float = 0.0
    trained_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(artifact_dir: Path) -> dict[str, str]:
    checksums = {name: sha256_of(artifact_dir / name) for name in ARTIFACT_FILES if (artifact_dir / name).exists()}
    (artifact_dir / CHECKSUM_FILE).write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return checksums


class ArtifactIntegrityError(RuntimeError):
    pass


def verify_checksums(artifact_dir: Path, expected: dict[str, str]) -> None:
    """Refuse to load artefacts whose content differs from the recorded checksums."""
    if not expected:
        raise ArtifactIntegrityError("No checksums recorded for this model; refusing to load")
    for name, digest in expected.items():
        path = artifact_dir / name
        if not path.exists():
            raise ArtifactIntegrityError(f"Artefact missing: {name}")
        actual = sha256_of(path)
        if actual != digest:
            raise ArtifactIntegrityError(f"Checksum mismatch for {name}")
