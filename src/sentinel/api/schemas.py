"""Pydantic v2 request/response models for all API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response from the ``GET /health`` endpoint."""

    status: str = Field(
        description="Overall status: 'healthy' or 'degraded'.",
        examples=["healthy"],
    )
    api: str = Field(
        default="ok",
        description="API subsystem status.",
    )
    ollama: str = Field(
        default="unknown",
        description="Ollama LLM subsystem status: 'ok' or 'unreachable'.",
    )
    version: str = Field(description="Sentinel package version.")


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(description="Human-readable error message.")
    code: str = Field(
        default="UNKNOWN",
        description="Machine-readable error code.",
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class DatasetSummary(BaseModel):
    """Summary metadata for a single dataset."""

    dataset_id: str = Field(description="UUID identifier for the dataset.")
    name: str = Field(default="", description="Original file name.")
    source: str = Field(default="upload", description="Ingestion source.")
    rows: int = Field(default=0, description="Number of rows.")
    columns: int = Field(default=0, description="Number of columns.")
    features: list[str] = Field(
        default_factory=list,
        description="Feature column names.",
    )
    time_range: dict[str, str] = Field(
        default_factory=dict,
        description="Start and end timestamps as ISO strings.",
    )
    uploaded_at: str = Field(default="", description="Upload timestamp (ISO).")


class DatasetListResponse(BaseModel):
    """Paginated list of datasets."""

    items: list[DatasetSummary] = Field(
        default_factory=list,
        description="Dataset summaries for the requested page.",
    )
    total: int = Field(default=0, description="Total number of datasets.")
    page: int = Field(default=1, description="Current page number.")


class DatasetUploadResponse(BaseModel):
    """Response after a successful dataset upload."""

    dataset_id: str
    shape: list[int] = Field(description="[rows, columns].")
    features: list[str]
    time_range: dict[str, str]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """Request body for ``POST /api/train``."""

    config_path: str = Field(description="Path to YAML config file.")
    data_path: str | None = Field(
        default=None,
        description="Optional data file override.",
    )


class TrainJobResponse(BaseModel):
    """Response for training job submission and status polling."""

    job_id: str = Field(description="Unique job identifier.")
    status: str = Field(
        description="Job status: pending, running, completed, failed, cancelled.",
    )
    model_name: str = Field(default="", description="Model being trained.")
    poll_url: str = Field(default="", description="URL to poll for status.")
    progress_pct: float | None = Field(
        default=None,
        description="Progress percentage (0-100).",
    )
    metrics: dict[str, float | None] | None = Field(
        default=None,
        description="Evaluation metrics (available when completed).",
    )
    run_id: str | None = Field(
        default=None,
        description="Experiment run_id (available when completed).",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details (available when failed).",
    )
    duration_s: float | None = Field(
        default=None,
        description="Wall-clock duration in seconds.",
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class DetectRequest(BaseModel):
    """Request body for ``POST /api/detect``."""

    data_path: str = Field(description="Path to data file.")
    model_path: str = Field(description="Path to saved model directory.")


class DetectResponse(BaseModel):
    """Response from batch anomaly detection."""

    scores: list[float] = Field(description="Per-sample anomaly scores.")
    labels: list[int] = Field(description="Per-sample binary labels (0/1).")
    threshold: float = Field(description="Applied anomaly threshold.")
    model_name: str = Field(default="", description="Model used for detection.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluateResponse(BaseModel):
    """Evaluation metrics for a training run."""

    run_id: str
    model_name: str = Field(default="")
    metrics: dict[str, float | None] = Field(
        default_factory=dict,
        description="Metric name to value mapping.",
    )


# ---------------------------------------------------------------------------
# Models (registry)
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """Information about a registered model."""

    name: str = Field(description="Registered model name.")
    category: str = Field(default="", description="Model category.")
    description: str = Field(default="", description="Brief model description.")


class ModelListResponse(BaseModel):
    """List of all registered models."""

    models: list[ModelInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Summary of a single experiment run."""

    run_id: str
    model_name: str = Field(default="")
    created_at: str = Field(default="")
    metrics: dict[str, float | None] = Field(default_factory=dict)


class RunListResponse(BaseModel):
    """Paginated list of experiment runs."""

    items: list[RunSummary] = Field(default_factory=list)
    total: int = Field(default=0)
    page: int = Field(default=1)


class RunCompareResponse(BaseModel):
    """Comparison of metrics across experiment runs."""

    runs: list[RunSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt (LLM)
# ---------------------------------------------------------------------------


class PromptRequest(BaseModel):
    """Request body for ``POST /api/prompt``."""

    message: str = Field(description="Natural language prompt.")


class PromptResponse(BaseModel):
    """Response from LLM-powered prompt processing."""

    response: str = Field(description="Assembled response text.")
    tools_called: list[str] = Field(
        default_factory=list,
        description="Names of MCP tools that were invoked.",
    )
