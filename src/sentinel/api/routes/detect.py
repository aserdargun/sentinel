"""Detection routes: batch anomaly detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException

from sentinel.api.schemas import DetectRequest, DetectResponse, ErrorResponse
from sentinel.core.registry import get_model_class, list_models
from sentinel.data.loaders import load_file
from sentinel.data.validators import get_feature_columns, validate_dataframe

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post(
    "",
    response_model=DetectResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def batch_detect(body: DetectRequest) -> DetectResponse:
    """Run batch anomaly detection on a dataset.

    Loads the saved model from ``model_path``, loads data from
    ``data_path``, scores the data, and returns scores with labels.

    Args:
        body: Request with data_path and model_path.

    Returns:
        Detection results with scores, labels, and threshold.
    """
    # Validate data path.
    data_path = Path(body.data_path)
    if not data_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Data file not found: {body.data_path}",
        )

    # Validate model path.
    model_path = Path(body.model_path)
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model path not found: {body.model_path}",
        )

    # Load data.
    try:
        df = load_file(data_path)
        df = validate_dataframe(df)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Data validation failed: {exc}"
        ) from exc

    # Separate features.
    feature_cols = get_feature_columns(df)
    X = df.select(feature_cols).to_numpy().astype(np.float64)

    # Determine the model name from the saved config.
    import json

    config_file = model_path / "config.json"
    meta_file = model_path / "meta.json"

    model_name = ""
    if config_file.exists():
        config_data = json.loads(config_file.read_text())
        model_name = config_data.get("model", config_data.get("model_name", ""))
    elif meta_file.exists():
        meta_data = json.loads(meta_file.read_text())
        model_name = meta_data.get("model_name", "")

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="Cannot determine model name from saved artifacts. "
            "Expected config.json or meta.json with a 'model' or "
            "'model_name' field.",
        )

    # Verify model is registered.
    registry = list_models()
    if model_name not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' is not registered. "
            f"Available: {', '.join(sorted(registry.keys()))}",
        )

    # Instantiate and load the model.
    try:
        model_cls = get_model_class(model_name)
        model = model_cls()
        model.load(str(model_path))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {exc}",
        ) from exc

    # Score the data.
    try:
        scores = model.score(X)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Scoring failed: {exc}",
        ) from exc

    # Apply a default threshold (95th percentile) if not stored.
    threshold = float(np.percentile(scores, 95))
    labels = (scores > threshold).astype(np.int32)

    logger.info(
        "detect.batch_complete",
        model_name=model_name,
        n_samples=len(scores),
        n_anomalies=int(labels.sum()),
    )

    return DetectResponse(
        scores=scores.tolist(),
        labels=labels.tolist(),
        threshold=threshold,
        model_name=model_name,
    )
