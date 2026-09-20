"""ORM schema.

Every client-owned table carries ``organization_id`` (tenant boundary) and, where
meaningful, ``created_by``. All ids are random UUID4 strings so they are safe to
expose and portable across SQLite/PostgreSQL. Timestamps are timezone-aware UTC.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    ADMIN = "ADMIN"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"


class UserStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobType(str, enum.Enum):
    TRAINING = "TRAINING"
    BATCH_PREDICTION = "BATCH_PREDICTION"
    DATASET_IMPORT = "DATASET_IMPORT"
    EXPLANATION = "EXPLANATION"


class ModelStatus(str, enum.Enum):
    TRAINING = "TRAINING"
    READY = "READY"
    PRODUCTION = "PRODUCTION"
    DEPRECATED = "DEPRECATED"
    FAILED = "FAILED"


class DatasetStatus(str, enum.Enum):
    UPLOADED = "UPLOADED"
    VALIDATED = "VALIDATED"
    INVALID = "INVALID"
    ARCHIVED = "ARCHIVED"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


# --------------------------------------------------------------------------- #
# Identity & tenancy
# --------------------------------------------------------------------------- #


class Organization(TimestampMixin, Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    users: Mapped[list["User"]] = relationship(back_populates="organization")


class User(TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Role] = mapped_column(Enum(Role, name="role_enum"), default=Role.VIEWER, nullable=False)
    status: Mapped[UserStatus] = mapped_column(Enum(UserStatus, name="user_status_enum"), default=UserStatus.ACTIVE, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    password_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    organization: Mapped[Organization] = relationship(back_populates="users")


class RefreshSession(Base):
    """One row per issued refresh token (hashed). Revoked on logout / rotation."""

    __tablename__ = "refresh_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    user_agent: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_org_created", "organization_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    actor_email: Mapped[str] = mapped_column(String(320), default="", nullable=False)
    action: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    target_type: Mapped[str] = mapped_column(String(60), default="", nullable=False)
    target_id: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    outcome: Mapped[str] = mapped_column(String(20), default="success", nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    details_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


class SystemSetting(TimestampMixin, Base):
    """Per-organisation key/value settings (non-secret)."""

    __tablename__ = "system_settings"
    __table_args__ = (UniqueConstraint("organization_id", "key", name="uq_settings_org_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(80), nullable=False)
    value_json: Mapped[str] = mapped_column(Text, default="null", nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_org_status", "organization_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    job_type: Mapped[JobType] = mapped_column(Enum(JobType, name="job_type_enum"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, name="job_status_enum"), default=JobStatus.QUEUED, nullable=False)
    stage: Mapped[str] = mapped_column(String(60), default="queued", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    message: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    params_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    target_type: Mapped[str] = mapped_column(String(60), default="", nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    worker: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"
    __table_args__ = (Index("ix_datasets_org_created", "organization_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    kind: Mapped[str] = mapped_column(String(40), default="upload", nullable=False)  # upload | benchmark-cresci-15 | benchmark-cresci-17 | benchmark-combined
    status: Mapped[DatasetStatus] = mapped_column(Enum(DatasetStatus, name="dataset_status_enum"), default=DatasetStatus.UPLOADED, nullable=False)
    current_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="dataset", cascade="all, delete-orphan", order_by="DatasetVersion.version")


class DatasetVersion(TimestampMixin, Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "version", name="uq_dataset_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_columns: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    has_label: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    label_column: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[DatasetStatus] = mapped_column(Enum(DatasetStatus, name="dataset_status_enum"), default=DatasetStatus.UPLOADED, nullable=False)
    summary_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    validation_errors_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)

    dataset: Mapped[Dataset] = relationship(back_populates="versions")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class MLModel(TimestampMixin, Base):
    """A trained model version. Exactly one PRODUCTION model per organisation."""

    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", "version", name="uq_models_org_name_version"),
        Index("ix_models_org_status", "organization_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    algorithm: Mapped[str] = mapped_column(String(60), nullable=False)
    status: Mapped[ModelStatus] = mapped_column(Enum(ModelStatus, name="model_status_enum"), default=ModelStatus.TRAINING, nullable=False)
    dataset_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True)
    dataset_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    feature_version: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    feature_names_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    n_features: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    validation_metrics_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)  # cross-validation
    test_metrics_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)  # hold-out
    params_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    artifact_prefix: Mapped[str] = mapped_column(String(500), default="", nullable=False)  # storage key prefix
    artifact_checksums_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)  # file → sha256
    training_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)


class EvaluationRun(TimestampMixin, Base):
    """Evaluation of a model version on a dataset version (hold-out at training time, or later)."""

    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    model_id: Mapped[str] = mapped_column(String(36), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True)
    kind: Mapped[str] = mapped_column(String(30), default="holdout", nullable=False)  # holdout | cross_validation | dataset
    n_samples: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    details_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)  # confusion, curves, folds, feature importance


class FeatureImportance(Base):
    __tablename__ = "feature_importance"
    __table_args__ = (UniqueConstraint("model_id", "feature", name="uq_feature_importance"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_id: Mapped[str] = mapped_column(String(36), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True)
    feature: Mapped[str] = mapped_column(String(80), nullable=False)
    feature_group: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    mean_abs_shap: Mapped[float] = mapped_column(Float, nullable=False)
    mean_shap: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #


class AnalyzedAccount(TimestampMixin, Base):
    """The identity being analysed (no raw tweet text is stored — only the derived feature vector on the prediction)."""

    __tablename__ = "analyzed_accounts"
    __table_args__ = (UniqueConstraint("organization_id", "identifier", name="uq_accounts_org_identifier"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    identifier: Mapped[str] = mapped_column(String(200), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    source: Mapped[str] = mapped_column(String(40), default="manual", nullable=False)  # manual | csv | x_api
    last_analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    analysis_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class Batch(TimestampMixin, Base):
    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
    dataset_version_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus, name="job_status_enum"), default=JobStatus.QUEUED, nullable=False)
    input_storage_key: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    output_storage_key: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    total_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_bots: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_humans: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    avg_bot_probability: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    high_risk: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Prediction(TimestampMixin, Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_org_created", "organization_id", "created_at"),
        Index("ix_predictions_org_label", "organization_id", "prediction"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    account_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("analyzed_accounts.id", ondelete="SET NULL"), nullable=True, index=True)
    account_identifier: Mapped[str] = mapped_column(String(200), default="", nullable=False, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("batches.id", ondelete="CASCADE"), nullable=True, index=True)
    model_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("models.id", ondelete="SET NULL"), nullable=True, index=True)
    model_name: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    model_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source: Mapped[str] = mapped_column(String(40), default="manual", nullable=False)  # manual | csv | batch | x_api
    status: Mapped[str] = mapped_column(String(20), default="COMPLETED", nullable=False)
    prediction: Mapped[str] = mapped_column(String(8), nullable=False)
    bot_probability: Mapped[float] = mapped_column(Float, nullable=False)
    human_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_band: Mapped[str] = mapped_column(String(12), default="", nullable=False)
    features_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    auxiliary_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    input_summary_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)  # profile counts/flags only, no tweet text
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    label_true: Mapped[str | None] = mapped_column(String(8), nullable=True)  # ground truth when supplied in a batch

    explanations: Mapped[list["Explanation"]] = relationship(back_populates="prediction", cascade="all, delete-orphan")


class Explanation(TimestampMixin, Base):
    __tablename__ = "explanations"
    __table_args__ = (UniqueConstraint("prediction_id", "method", name="uq_explanation_prediction_method"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    prediction_id: Mapped[str] = mapped_column(String(36), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    model_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
    method: Mapped[str] = mapped_column(String(10), nullable=False)  # shap | lime
    status: Mapped[str] = mapped_column(String(20), default="COMPLETED", nullable=False)
    explainer: Mapped[str] = mapped_column(String(60), default="", nullable=False)
    output_scale: Mapped[str] = mapped_column(String(20), default="", nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    compute_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    prediction: Mapped[Prediction] = relationship(back_populates="explanations")
