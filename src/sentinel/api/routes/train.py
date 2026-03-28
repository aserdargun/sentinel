"""Training routes: submit async jobs and poll status."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request

from sentinel.api.deps import resolve_safe_path
from sentinel.api.schemas import ErrorResponse, TrainJobResponse, TrainRequest
from sentinel.core.config import RunConfig
from sentinel.core.exceptions import ConfigError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/train", tags=["training"])


@router.post(
    "",
    response_model=TrainJobResponse,
    status_code=202,
    responses={400: {"model": ErrorResponse}},
)
async def submit_training_job(
    body: TrainRequest,
    request: Request,
) -> TrainJobResponse:
    """Submit an asynchronous training job.

    The config is validated before the job is queued.  Returns a
    ``job_id`` that can be polled via ``GET /api/train/{job_id}``.

    Args:
        body: Training request with config_path and optional data_path.
        request: FastAPI request (provides access to app state).

    Returns:
        Job submission response with job_id and poll URL.
    """
    job_manager = request.app.state.job_manager

    # Sanitize and validate config path.
    config_path = resolve_safe_path(body.config_path)
    if not config_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Config file not found: {body.config_path}",
        )

    # Parse and validate config before queuing.
    try:
        config = RunConfig.from_yaml(config_path)
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not config.model:
        raise HTTPException(
            status_code=400,
            detail="Config must specify a 'model' field.",
        )

    # Sanitize and validate data path if provided.
    if body.data_path is not None:
        data_path_obj = resolve_safe_path(body.data_path)
        if not data_path_obj.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Data path not found: {body.data_path}",
            )

    # Submit to the background job manager.
    job_id = job_manager.submit_job(config, data_path=body.data_path)

    logger.info(
        "train.job_submitted",
        job_id=job_id,
        model=config.model,
        config_path=str(config_path),
    )

    return TrainJobResponse(
        job_id=job_id,
        status="pending",
        model_name=config.model,
        poll_url=f"/api/train/{job_id}",
    )


@router.delete(
    "/{job_id}",
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def cancel_training_job(
    job_id: str,
    request: Request,
) -> dict[str, str]:
    """Cancel a queued training job.

    Only jobs that have not yet started can be cancelled.  Returns 409
    if the job is already completed or running.

    Args:
        job_id: The job identifier returned by ``POST /api/train``.
        request: FastAPI request (provides access to app state).

    Returns:
        Confirmation message.
    """
    job_manager = request.app.state.job_manager

    try:
        status = job_manager.get_job_status(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
        )

    if status["status"] in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} already {status['status']}.",
        )

    cancelled = job_manager.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already running and cannot be cancelled.",
        )

    logger.info("train.job_cancelled", job_id=job_id)
    return {"message": f"Job {job_id} cancelled."}


@router.get(
    "/{job_id}",
    response_model=TrainJobResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(
    job_id: str,
    request: Request,
) -> TrainJobResponse:
    """Poll the status of a training job.

    Args:
        job_id: The job identifier returned by ``POST /api/train``.
        request: FastAPI request (provides access to app state).

    Returns:
        Current job status with metrics if completed.
    """
    job_manager = request.app.state.job_manager

    try:
        status = job_manager.get_job_status(job_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
        )

    return TrainJobResponse(
        job_id=status["job_id"],
        status=status["status"],
        model_name=status["model_name"],
        poll_url=f"/api/train/{job_id}",
        progress_pct=status.get("progress_pct"),
        metrics=status.get("metrics"),
        run_id=status.get("run_id"),
        error_message=status.get("error_message"),
        duration_s=status.get("duration_s"),
    )
