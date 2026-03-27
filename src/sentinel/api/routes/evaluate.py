"""Evaluation routes: retrieve metrics for experiment runs."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from sentinel.api.deps import get_tracker
from sentinel.api.schemas import ErrorResponse, EvaluateResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.get(
    "/{run_id}",
    response_model=EvaluateResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_evaluation(run_id: str) -> EvaluateResponse:
    """Return evaluation metrics for a training run.

    Reads the experiment tracker to load stored metrics for the given
    run_id.

    Args:
        run_id: The experiment run identifier.

    Returns:
        Evaluation response with run_id, model_name, and metrics dict.
    """
    tracker = get_tracker()

    try:
        run_data = tracker.get_run(run_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {run_id}",
        )

    metrics = run_data.get("metrics", {})
    model_name = run_data.get("model_name", "")

    logger.info(
        "evaluate.retrieved",
        run_id=run_id,
        model_name=model_name,
        n_metrics=len(metrics),
    )

    return EvaluateResponse(
        run_id=run_id,
        model_name=model_name,
        metrics=metrics,
    )
