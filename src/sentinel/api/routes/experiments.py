"""Experiment routes: list runs and compare metrics."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query

from sentinel.api.deps import get_tracker
from sentinel.api.schemas import (
    ErrorResponse,
    RunCompareResponse,
    RunListResponse,
    RunSummary,
)
from sentinel.tracking.comparison import compare_runs

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get(
    "",
    response_model=RunListResponse,
)
async def list_experiments(
    page: int = Query(default=1, ge=1, description="Page number."),
    limit: int = Query(default=50, ge=1, le=200, description="Items per page."),
) -> RunListResponse:
    """List all experiment runs with pagination.

    Scans the experiment directory and returns run metadata sorted
    by creation time (newest first).

    Args:
        page: Page number (1-based).
        limit: Number of items per page.

    Returns:
        Paginated list of experiment run summaries.
    """
    tracker = get_tracker()
    all_runs = tracker.list_runs()

    items: list[RunSummary] = []
    for run in all_runs:
        items.append(
            RunSummary(
                run_id=run.get("run_id", ""),
                model_name=run.get("model_name", ""),
                created_at=run.get("created_at", ""),
                metrics=run.get("metrics", {}),
            )
        )

    total = len(items)
    start = (page - 1) * limit
    end = start + limit
    page_items = items[start:end]

    logger.info("experiments.listed", total=total, page=page)
    return RunListResponse(items=page_items, total=total, page=page)


@router.get(
    "/compare",
    response_model=RunCompareResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def compare_experiments(
    ids: str = Query(description="Comma-separated run IDs to compare (e.g. 'a,b,c')."),
) -> RunCompareResponse:
    """Compare metrics across multiple experiment runs.

    Loads metrics for each specified run and returns them in a
    single response for side-by-side comparison.

    Args:
        ids: Comma-separated list of run IDs.

    Returns:
        Comparison response with metrics for each run.
    """
    run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]

    if not run_ids:
        raise HTTPException(
            status_code=400,
            detail="No run IDs provided. Use ?ids=a,b,c format.",
        )

    tracker = get_tracker()

    # Validate all run IDs exist before comparing.
    for rid in run_ids:
        try:
            tracker.get_run(rid)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"Run not found: {rid}",
            )

    # Use the comparison utility to get a DataFrame.
    try:
        comparison_df = compare_runs(
            run_ids=run_ids,
            base_dir=str(tracker.base_dir),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Convert to response schema.
    runs: list[RunSummary] = []
    for row in comparison_df.to_dicts():
        # Extract standard fields and put the rest into metrics.
        metrics: dict[str, Any] = {}
        for key, value in row.items():
            if key not in ("run_id", "model_name", "created_at"):
                metrics[key] = value

        runs.append(
            RunSummary(
                run_id=row.get("run_id", ""),
                model_name=row.get("model_name", ""),
                created_at=row.get("created_at", ""),
                metrics=metrics,
            )
        )

    logger.info("experiments.compared", n_runs=len(runs))
    return RunCompareResponse(runs=runs)
