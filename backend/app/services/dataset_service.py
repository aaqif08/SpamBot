"""Datasets: upload → validate → inspect → version; benchmark import; deletion."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import sanitize_filename, sha256_bytes
from app.core.storage import get_storage
from app.db.models import Dataset, DatasetStatus, DatasetVersion, MLModel, User
from app.services import audit
from ml.datasets import CRESCI_PAPER_STATS, CRESCI_SUBSETS, build_cresci_features, inspect_dataframe, scan_cresci
from ml.features import FEATURE_NAMES
from ml.utils import json_safe

log = logging.getLogger(__name__)

MAX_COLUMNS = 500


def _read_csv_bytes(data: bytes, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), nrows=nrows, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", low_memory=False)


class DatasetService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.storage = get_storage()

    # ---- presentation ------------------------------------------------------- #

    @staticmethod
    def version_public(v: DatasetVersion, with_summary: bool = False) -> dict[str, Any]:
        d = {
            "id": v.id,
            "dataset_id": v.dataset_id,
            "version": v.version,
            "original_filename": v.original_filename,
            "size_bytes": v.size_bytes,
            "checksum_sha256": v.checksum_sha256,
            "n_rows": v.n_rows,
            "n_columns": v.n_columns,
            "has_label": v.has_label,
            "label_column": v.label_column,
            "status": v.status.value,
            "validation_errors": json.loads(v.validation_errors_json or "[]"),
            "created_at": v.created_at,
        }
        if with_summary:
            d["summary"] = json.loads(v.summary_json or "{}")
        return d

    def dataset_public(self, ds: Dataset, with_summary: bool = False) -> dict[str, Any]:
        versions = sorted(ds.versions, key=lambda v: v.version)
        current = next((v for v in versions if v.id == ds.current_version_id), versions[-1] if versions else None)
        summary = json.loads(current.summary_json or "{}") if current else {}
        model_count = self.db.query(MLModel).filter(MLModel.dataset_id == ds.id).count()
        d = {
            "id": ds.id,
            "name": ds.name,
            "description": ds.description,
            "kind": ds.kind,
            "status": ds.status.value,
            "created_at": ds.created_at,
            "updated_at": ds.updated_at,
            "n_versions": len(versions),
            "current_version": self.version_public(current) if current else None,
            "n_rows": current.n_rows if current else 0,
            "n_columns": current.n_columns if current else 0,
            "has_label": bool(current and current.has_label),
            "class_distribution": summary.get("class_distribution"),
            "feature_coverage": (summary.get("feature_availability") or {}).get("coverage"),
            "models_trained": model_count,
        }
        if with_summary:
            d["summary"] = summary
            d["versions"] = [self.version_public(v) for v in versions]
        return d

    # ---- queries ------------------------------------------------------------- #

    def list(self, organization_id: str) -> list[dict[str, Any]]:
        rows = self.db.query(Dataset).filter(Dataset.organization_id == organization_id, Dataset.status != DatasetStatus.ARCHIVED).order_by(Dataset.created_at.desc()).all()
        return [self.dataset_public(r) for r in rows]

    def get(self, organization_id: str, dataset_id: str) -> Dataset:
        ds = self.db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.organization_id == organization_id).first()
        if ds is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
        return ds

    def current_version(self, ds: Dataset) -> DatasetVersion:
        v = next((v for v in ds.versions if v.id == ds.current_version_id), None) or (sorted(ds.versions, key=lambda x: x.version)[-1] if ds.versions else None)
        if v is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset has no versions")
        return v

    def get_version(self, organization_id: str, version_id: str) -> DatasetVersion:
        v = self.db.query(DatasetVersion).filter(DatasetVersion.id == version_id, DatasetVersion.organization_id == organization_id).first()
        if v is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset version not found")
        return v

    def load_frame(self, version: DatasetVersion) -> pd.DataFrame:
        try:
            data = self.storage.get_bytes(version.storage_key)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=status.HTTP_410_GONE, detail="Dataset file is no longer available in storage") from exc
        return _read_csv_bytes(data)

    # ---- validation ----------------------------------------------------------- #

    @staticmethod
    def validate_frame(df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        if df.shape[1] == 0:
            errors.append("The CSV has no columns")
        if len(df) == 0:
            errors.append("The CSV has no data rows")
        if df.shape[1] > MAX_COLUMNS:
            errors.append(f"Too many columns ({df.shape[1]} > {MAX_COLUMNS})")
        if len(df) > get_settings().max_batch_rows:
            errors.append(f"Too many rows ({len(df):,} > {get_settings().max_batch_rows:,})")
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if len(unnamed) == df.shape[1]:
            errors.append("No header row detected")
        return errors

    # ---- upload ----------------------------------------------------------------- #

    def upload(self, user: User, data: bytes, original_filename: str, *, name: str | None = None, description: str = "", dataset_id: str | None = None, request: Request | None = None) -> Dataset:
        """Create a dataset (version 1) or add a new version to an existing dataset."""
        try:
            df = _read_csv_bytes(data)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Could not parse CSV: {exc}") from exc
        errors = self.validate_frame(df)
        summary = inspect_dataframe(df) if not errors else {"warnings": errors}
        avail = summary.get("feature_availability") or {}
        if not errors and avail.get("coverage", 0) == 0:
            errors.append("None of the required account columns were recognised (e.g. followers_count, friends_count, statuses_count)")

        if dataset_id:
            ds = self.get(user.organization_id, dataset_id)
            version_no = (max((v.version for v in ds.versions), default=0)) + 1
        else:
            ds = Dataset(
                organization_id=user.organization_id,
                created_by=user.id,
                name=(name or Path(sanitize_filename(original_filename)).stem or "dataset")[:200],
                description=description[:2000],
                kind="upload",
                status=DatasetStatus.UPLOADED,
            )
            self.db.add(ds)
            self.db.flush()
            version_no = 1

        key = f"org/{user.organization_id}/datasets/{ds.id}/v{version_no}.csv"
        self.storage.put_bytes(key, data, content_type="text/csv")
        version = DatasetVersion(
            dataset_id=ds.id,
            organization_id=user.organization_id,
            created_by=user.id,
            version=version_no,
            storage_key=key,
            original_filename=sanitize_filename(original_filename),
            size_bytes=len(data),
            checksum_sha256=sha256_bytes(data),
            n_rows=int(len(df)),
            n_columns=int(df.shape[1]),
            has_label=bool(summary.get("label_column")) and bool(summary.get("label_valid_rows")),
            label_column=summary.get("label_column"),
            status=DatasetStatus.INVALID if errors else DatasetStatus.VALIDATED,
            summary_json=json.dumps(json_safe(summary)),
            validation_errors_json=json.dumps(errors),
        )
        self.db.add(version)
        self.db.flush()
        ds.current_version_id = version.id
        ds.status = DatasetStatus.INVALID if errors else DatasetStatus.VALIDATED
        self.db.commit()
        self.db.refresh(ds)  # relationship 'versions' must include the new row (expire_on_commit=False)
        audit.record(self.db, "dataset.uploaded", actor=user, target_type="dataset", target_id=ds.id, details={"version": version_no, "rows": version.n_rows, "valid": not errors, "filename": version.original_filename}, request=request)
        return ds

    def delete(self, user: User, dataset_id: str, request: Request | None = None) -> None:
        ds = self.get(user.organization_id, dataset_id)
        in_use = self.db.query(MLModel).filter(MLModel.dataset_id == ds.id).count()
        for v in ds.versions:
            try:
                self.storage.delete(v.storage_key)
            except Exception:  # noqa: BLE001
                log.warning("could not delete storage object %s", v.storage_key)
        self.db.delete(ds)
        self.db.commit()
        audit.record(self.db, "dataset.deleted", actor=user, target_type="dataset", target_id=dataset_id, details={"models_referenced": in_use}, request=request)

    # ---- benchmark datasets (real Cresci data placed on the server) -------------- #

    def benchmark_status(self) -> list[dict[str, Any]]:
        out = []
        for kind in CRESCI_SUBSETS:
            scan = scan_cresci(self.settings.datasets_dir / kind, kind)
            out.append(
                {
                    "kind": kind,
                    "available": scan.available or scan.features_cache is not None,
                    "subsets_found": [{"folder": s.key, "category": s.category, "label": s.label, "has_tweets": s.tweets_file is not None} for s in scan.subsets],
                    "expected_subsets": list(CRESCI_SUBSETS[kind].keys()),
                    "paper_reported": CRESCI_PAPER_STATS[kind],
                    "install_path_hint": f"backend/data/datasets/{kind}/<subset>/users.csv (+ tweets.csv)",
                }
            )
        return out

    def import_benchmark(self, user_id: str | None, organization_id: str, kind: str, progress: Callable[[str], None] | None = None, use_cache: bool = True) -> Dataset:
        """Featurise locally installed Cresci files and register them as a dataset for this organisation."""
        if kind not in CRESCI_SUBSETS:
            raise ValueError(f"Unknown benchmark '{kind}'")
        scan = scan_cresci(self.settings.datasets_dir / kind, kind)
        if not scan.available and scan.features_cache is None:
            raise FileNotFoundError(f"{kind} files not found under backend/data/datasets/{kind}/")
        df = build_cresci_features(scan, max_tweets_per_user=self.settings.max_tweets_per_user, progress=progress, use_cache=use_cache)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        data = buf.getvalue().encode("utf-8")
        user = self.db.get(User, user_id) if user_id else None
        existing = self.db.query(Dataset).filter(Dataset.organization_id == organization_id, Dataset.kind == f"benchmark-{kind}").first()
        if user is None:
            raise ValueError("A user is required to register a benchmark dataset")
        ds = self.upload(user, data, f"{kind}-features.csv", name=f"{kind.upper()} benchmark (user-level)", description=f"Cresci {kind} benchmark featurised from local files. Provenance: backend/data/datasets/PROVENANCE.md", dataset_id=existing.id if existing else None)
        ds.kind = f"benchmark-{kind}"
        self.db.commit()
        return ds

    def combined_benchmark(self, user: User, request: Request | None = None) -> Dataset:
        """Union of the organisation's imported Cresci-15 and Cresci-17 datasets."""
        frames = []
        for kind in ("cresci-15", "cresci-17"):
            ds = self.db.query(Dataset).filter(Dataset.organization_id == user.organization_id, Dataset.kind == f"benchmark-{kind}").first()
            if ds is None:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Import {kind} first")
            df = self.load_frame(self.current_version(ds))
            df["source_dataset"] = kind
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        if "id" in combined.columns:
            combined = combined.drop_duplicates(subset=["id"], keep="first")
        buf = io.StringIO()
        combined.to_csv(buf, index=False)
        existing = self.db.query(Dataset).filter(Dataset.organization_id == user.organization_id, Dataset.kind == "benchmark-combined").first()
        ds = self.upload(user, buf.getvalue().encode("utf-8"), "cresci-combined-features.csv", name="CRESCI-15 + CRESCI-17 combined (user-level)", description="Union of both imported Cresci benchmarks, de-duplicated by account id.", dataset_id=existing.id if existing else None, request=request)
        ds.kind = "benchmark-combined"
        self.db.commit()
        return ds

    @staticmethod
    def feature_columns() -> list[str]:
        return list(FEATURE_NAMES)
