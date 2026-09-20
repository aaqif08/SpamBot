"""Pydantic schemas — the public API contract (v1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

PredictionLabel = Literal["BOT", "HUMAN"]
RoleName = Literal["ADMIN", "ANALYST", "VIEWER"]


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"


# --------------------------------------------------------------------------- #
# Auth / users / organisation
# --------------------------------------------------------------------------- #


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: "UserPublic"


class UserPublic(BaseModel):
    id: str
    email: str
    full_name: str
    role: RoleName
    status: str
    organization_id: str
    organization_name: str
    last_login_at: datetime | None
    created_at: datetime


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str = Field(min_length=10, max_length=200)
    new_password: str = Field(min_length=1, max_length=256)


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)
    full_name: str = Field(default="", max_length=200)
    role: RoleName = "VIEWER"


class UpdateUserRequest(BaseModel):
    role: RoleName | None = None
    status: Literal["ACTIVE", "DISABLED"] | None = None
    full_name: str | None = Field(default=None, max_length=200)


class AdminResetPasswordRequest(BaseModel):
    new_password: str = Field(min_length=1, max_length=256)


class OrganizationPublic(BaseModel):
    id: str
    name: str
    slug: str
    created_at: datetime


class UpdateOrganizationRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class SetupStatus(BaseModel):
    initialized: bool
    self_signup_enabled: bool


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    version: str
    environment: str
    checks: dict[str, Any]


class RuntimeInfo(BaseModel):
    version: str
    environment: str
    feature_version: str
    n_features: int
    job_backend: str
    storage_backend: str
    python: str
    libraries: dict[str, str]
    production_model: dict[str, Any] | None


# --------------------------------------------------------------------------- #
# Accounts / prediction
# --------------------------------------------------------------------------- #


class TweetInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(default="", max_length=5000)
    retweet_count: int = Field(default=0, ge=0, le=10_000_000)
    reply_count: int = Field(default=0, ge=0, le=10_000_000)
    favorite_count: int = Field(default=0, ge=0, le=10_000_000)


class AccountInput(BaseModel):
    """Account profile as accepted by ``POST /api/v1/analyses``. Derived features are computed server-side."""

    model_config = ConfigDict(extra="ignore")

    account_id: str | None = Field(default=None, max_length=200)
    screen_name: str | None = Field(default=None, max_length=100)
    name: str | None = Field(default=None, max_length=200)
    verified: bool = False
    friends_count: int = Field(default=0, ge=0, le=100_000_000)
    followers_count: int = Field(default=0, ge=0, le=1_000_000_000)
    listed_count: int = Field(default=0, ge=0, le=10_000_000)
    favorites_count: int = Field(default=0, ge=0, le=100_000_000)
    statuses_count: int = Field(default=0, ge=0, le=100_000_000)
    description: str | None = Field(default=None, max_length=2000)
    location: str | None = Field(default=None, max_length=200)
    url: str | None = Field(default=None, max_length=500)
    default_profile: bool = False
    default_profile_image: bool = False
    geo_enabled: bool = False
    profile_background_tile: bool = False
    has_profile_banner: bool = False
    hashtag_count: int | None = Field(default=None, ge=0, le=100_000_000)
    mentions_count: int | None = Field(default=None, ge=0, le=100_000_000)
    retweet_count: int | None = Field(default=None, ge=0, le=1_000_000_000)
    reply_count: int | None = Field(default=None, ge=0, le=1_000_000_000)
    url_count: int | None = Field(default=None, ge=0, le=100_000_000)
    favorite_count_received: int | None = Field(default=None, ge=0, le=1_000_000_000)
    tweets_observed: int | None = Field(default=None, ge=0, le=100_000_000)
    tweets: list[TweetInput | str] = Field(default_factory=list, max_length=500)

    @field_validator("tweets", mode="before")
    @classmethod
    def _coerce_tweets(cls, v: Any) -> Any:
        return [] if v is None else v

    def to_account_dict(self) -> dict[str, Any]:
        return self.model_dump()


class AnalyzeRequest(BaseModel):
    account: AccountInput
    model_id: str | None = None
    explain: bool = True
    source: Literal["manual", "x_api"] = "manual"


class ModelRef(BaseModel):
    id: str | None
    name: str
    version: int


class TopFeature(BaseModel):
    feature: str
    group: str
    description: str = ""
    value: float | None = None
    impact: float | None = None
    direction: str


class AnalysisResponse(BaseModel):
    prediction_id: str
    account_identifier: str
    account_ref: str | None
    prediction: PredictionLabel
    bot_probability: float
    human_probability: float
    confidence: float
    risk_score: int
    risk_band: str
    risk_score_note: str
    model: ModelRef
    features: dict[str, float]
    feature_groups: dict[str, dict[str, float]]
    auxiliary: dict[str, float]
    input_summary: dict[str, Any]
    top_features: list[TopFeature]
    shap_explanation: dict[str, Any] | None
    lime_explanation: dict[str, Any] | None
    explanation_errors: dict[str, str]
    explanation_status: dict[str, str]
    interpretation: dict[str, Any]
    source: str
    status: str
    batch_id: str | None
    label_true: str | None
    inference_ms: float
    created_at: datetime
    created_by: str | None


class ExplanationResponse(BaseModel):
    prediction_id: str
    method: Literal["shap", "lime"]
    explanation: dict[str, Any]
    computed_now: bool


class HistoryItem(BaseModel):
    id: str
    account_identifier: str
    prediction: PredictionLabel
    bot_probability: float
    risk_score: int
    risk_band: str
    model_id: str | None
    model_name: str
    model_version: int
    source: str
    status: str
    batch_id: str | None
    label_true: str | None
    created_at: datetime


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


# --------------------------------------------------------------------------- #
# Jobs / batches
# --------------------------------------------------------------------------- #


class JobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    stage: str
    progress: float
    message: str
    log: list[str]
    error: str | None
    result: dict[str, Any] | None
    target_type: str
    target_id: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class BatchResponse(BaseModel):
    id: str
    name: str
    status: str
    job_id: str | None
    model: ModelRef | None
    dataset_version_id: str | None
    total_rows: int
    processed_rows: int
    failed_rows: int
    n_bots: int
    n_humans: int
    avg_bot_probability: float
    high_risk: int
    summary: dict[str, Any]
    error: str | None
    has_output: bool
    created_at: datetime
    completed_at: datetime | None


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class DatasetVersionPublic(BaseModel):
    id: str
    dataset_id: str
    version: int
    original_filename: str
    size_bytes: int
    checksum_sha256: str
    n_rows: int
    n_columns: int
    has_label: bool
    label_column: str | None
    status: str
    validation_errors: list[str]
    created_at: datetime
    summary: dict[str, Any] | None = None


class DatasetPublic(BaseModel):
    id: str
    name: str
    description: str
    kind: str
    status: str
    created_at: datetime
    updated_at: datetime
    n_versions: int
    current_version: DatasetVersionPublic | None
    n_rows: int
    n_columns: int
    has_label: bool
    class_distribution: dict[str, int] | None
    feature_coverage: float | None
    models_trained: int
    summary: dict[str, Any] | None = None
    versions: list[DatasetVersionPublic] | None = None


class BenchmarkStatus(BaseModel):
    kind: str
    available: bool
    subsets_found: list[dict[str, Any]]
    expected_subsets: list[str]
    paper_reported: dict[str, Any]
    install_path_hint: str


class DatasetEvaluateRequest(BaseModel):
    model_id: str | None = None


class EvaluationResult(BaseModel):
    model: ModelRef
    dataset_version_id: str
    n_samples: int
    evaluation: dict[str, Any]
    evaluation_run_id: str


# --------------------------------------------------------------------------- #
# Models / training
# --------------------------------------------------------------------------- #


class TrainRequest(BaseModel):
    dataset_id: str
    dataset_version_id: str | None = None
    algorithm: str = "lightgbm"
    test_size: float = Field(default=0.25, ge=0.05, le=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    hyperparameter_search: bool = True
    search_iterations: int = Field(default=6, ge=1, le=30)
    feature_selection: bool = False
    feature_selection_top_k: int = Field(default=20, ge=5, le=31)
    seed: int = Field(default=42, ge=0, le=1_000_000)
    activate: bool = False
    notes: str = Field(default="", max_length=500)


class TrainSubmitted(BaseModel):
    job_id: str
    model_id: str
    status: str


class ModelPublic(BaseModel):
    id: str
    name: str
    version: int
    algorithm: str
    status: str
    is_production: bool
    dataset_id: str | None
    dataset_version_id: str | None
    dataset_name: str
    feature_version: str
    feature_names: list[str]
    n_features: int
    training_seconds: float
    trained_at: datetime | None
    created_at: datetime
    created_by: str | None
    job_id: str | None
    notes: str
    artifact_checksum: str
    test_metrics: dict[str, Any] | None = None
    validation_metrics: dict[str, Any] | None = None
    params: dict[str, Any] | None = None


class ModelListResponse(BaseModel):
    production_model_id: str | None
    models: list[ModelPublic]
    supported_algorithms: list[dict[str, Any]]


class ModelEvaluationResponse(BaseModel):
    model: ModelPublic
    feature_metadata: dict[str, Any]
    evaluations: list[dict[str, Any]]
    feature_importance: list[dict[str, Any]]


class GlobalExplanationResponse(BaseModel):
    model: ModelPublic
    shap_global: dict[str, Any]
    feature_descriptions: dict[str, str]
    feature_groups: dict[str, list[str]]


# --------------------------------------------------------------------------- #
# Providers / settings / audit / dashboard
# --------------------------------------------------------------------------- #


class ProviderInfo(BaseModel):
    name: str
    kind: str
    configured: bool
    description: str
    fetchable: bool
    configuration_hint: str | None


class ProviderFetchRequest(BaseModel):
    identifier: str = Field(min_length=1, max_length=200)


class AuditItem(BaseModel):
    id: str
    action: str
    actor_email: str
    target_type: str
    target_id: str
    outcome: str
    ip_address: str
    details: dict[str, Any]
    created_at: datetime


class AuditResponse(BaseModel):
    items: list[AuditItem]
    total: int
    page: int
    page_size: int


class RetentionRequest(BaseModel):
    older_than_days: int = Field(ge=1, le=3650)


class DashboardResponse(BaseModel):
    cards: dict[str, Any]
    charts: dict[str, Any]
    production_model: dict[str, Any] | None
    has_model: bool
    has_predictions: bool


class ResearchResponse(BaseModel):
    citation: dict[str, Any]
    paper_reported: dict[str, Any]
    feature_groups: dict[str, list[str]]
    feature_descriptions: dict[str, str]
    classifiers: list[dict[str, Any]]
    pipeline: list[str]
    engineering_adaptations: list[str]


TokenResponse.model_rebuild()
