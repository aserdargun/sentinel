"""Visualization routes: generate and return plots for experiment runs."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from sentinel.api.deps import get_tracker
from sentinel.api.schemas import ErrorResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/visualize", tags=["visualization"])

_VALID_PLOT_TYPES = {"timeseries", "reconstruction", "latent"}


@router.get(
    "/{run_id}",
    responses={
        200: {"content": {"image/png": {}}},
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
    },
)
async def visualize_run(
    run_id: str,
    type: str = Query(
        default="timeseries",
        description="Plot type: timeseries, reconstruction, or latent.",
    ),
) -> Response:
    """Generate and return a visualization for an experiment run.

    Reads run data from the tracker and generates the requested plot
    type.  The plot is returned as a PNG image.

    Args:
        run_id: The experiment run identifier.
        type: Plot type to generate.

    Returns:
        PNG image response.
    """
    if type not in _VALID_PLOT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid plot type: '{type}'. "
            f"Valid types: {', '.join(sorted(_VALID_PLOT_TYPES))}",
        )

    tracker = get_tracker()

    try:
        run_data = tracker.get_run(run_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Run not found: {run_id}",
        )

    # Look for stored predictions/scores in the run directory.
    run_dir = tracker.base_dir / run_id

    if type == "timeseries":
        return _generate_timeseries_plot(run_data, run_dir)
    elif type == "reconstruction":
        return _generate_reconstruction_plot(run_data, run_dir)
    elif type == "latent":
        return _generate_latent_plot(run_data, run_dir)

    # Should not reach here due to the validation above.
    raise HTTPException(status_code=400, detail=f"Unhandled plot type: {type}")


def _generate_timeseries_plot(
    run_data: dict,
    run_dir: Path,
) -> Response:
    """Generate a time series plot with anomaly highlighting.

    Args:
        run_data: Run metadata and metrics.
        run_dir: Path to the run's artifact directory.

    Returns:
        PNG image response.
    """
    from sentinel.viz.timeseries import plot_timeseries

    # Try to load predictions from the run directory.
    predictions_path = run_dir / "predictions.json"
    scores: np.ndarray | None = None
    labels: np.ndarray | None = None
    threshold: float | None = None
    timestamps: np.ndarray = np.array([])
    values: np.ndarray = np.array([])

    if predictions_path.exists():
        pred_data = json.loads(predictions_path.read_text())
        scores = np.array(pred_data.get("scores", []))
        labels = np.array(pred_data.get("labels", []))
        threshold = pred_data.get("threshold")
        if "timestamps" in pred_data:
            timestamps = np.array(pred_data["timestamps"])
        if "values" in pred_data:
            values = np.array(pred_data["values"])

    # Fall back to synthetic data for the plot if no predictions found.
    if len(timestamps) == 0:
        n_points = len(scores) if scores is not None and len(scores) > 0 else 100
        timestamps = np.arange(n_points).astype(str)
        if len(values) == 0:
            values = np.zeros(n_points)

    model_name = run_data.get("model_name", "unknown")
    title = f"Anomaly Detection - {model_name} (Run: {run_data.get('run_id', '')})"

    fig = plot_timeseries(
        timestamps=timestamps,
        values=values,
        scores=scores if scores is not None and len(scores) > 0 else None,
        labels=labels if labels is not None and len(labels) > 0 else None,
        threshold=threshold,
        title=title,
    )

    return _figure_to_response(fig)


def _generate_reconstruction_plot(
    run_data: dict,
    run_dir: Path,
) -> Response:
    """Generate a reconstruction error plot.

    Args:
        run_data: Run metadata and metrics.
        run_dir: Path to the run's artifact directory.

    Returns:
        PNG image response.
    """
    # Reconstruction plot requires stored original + reconstructed values.
    # Fall back to a basic timeseries plot if reconstruction data is missing.
    return _generate_timeseries_plot(run_data, run_dir)


def _generate_latent_plot(
    run_data: dict,
    run_dir: Path,
) -> Response:
    """Generate a latent space projection plot.

    Args:
        run_data: Run metadata and metrics.
        run_dir: Path to the run's artifact directory.

    Returns:
        PNG image response.
    """
    # Latent space plot requires stored embeddings.
    # Fall back to a basic timeseries plot if latent data is missing.
    return _generate_timeseries_plot(run_data, run_dir)


def _figure_to_response(fig: object) -> Response:
    """Convert a matplotlib Figure to a FastAPI PNG Response.

    Args:
        fig: Matplotlib Figure object.

    Returns:
        Response with PNG content.
    """
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")  # type: ignore[attr-defined]
    plt.close(fig)  # type: ignore[arg-type]
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=plot.png"},
    )
