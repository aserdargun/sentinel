"""Compare metrics across multiple experiment runs.

Loads metrics from each run directory and assembles them into a single
Polars DataFrame for side-by-side comparison and ranking.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import structlog

from sentinel.tracking.experiment import LocalTracker

logger = structlog.get_logger(__name__)


def compare_runs(
    run_ids: list[str],
    base_dir: str = "data/experiments",
) -> pl.DataFrame:
    """Compare metrics across multiple experiment runs.

    Loads each run's metadata and metrics, then assembles them into a
    Polars DataFrame sorted by ``f1`` descending (nulls last).

    Args:
        run_ids: List of run identifiers to compare.
        base_dir: Root directory for experiments.

    Returns:
        DataFrame with columns ``run_id``, ``model_name``,
        ``created_at``, and one column per metric key found across
        all runs.  Sorted by ``f1`` descending with nulls last.

    Raises:
        FileNotFoundError: If any run directory does not exist.
    """
    tracker = LocalTracker(base_dir=base_dir)
    rows: list[dict[str, Any]] = []

    for run_id in run_ids:
        run_data = tracker.get_run(run_id)
        row: dict[str, Any] = {
            "run_id": run_data.get("run_id", run_id),
            "model_name": run_data.get("model_name", ""),
            "created_at": run_data.get("created_at", ""),
        }

        metrics = run_data.get("metrics", {})
        for key, value in metrics.items():
            row[key] = value

        rows.append(row)

    if not rows:
        logger.warning("comparison.no_runs", run_ids=run_ids)
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "model_name": pl.Utf8,
                "created_at": pl.Utf8,
            }
        )

    df = pl.DataFrame(rows)

    # Sort by f1 descending with nulls last, if the column exists.
    if "f1" in df.columns:
        df = df.sort("f1", descending=True, nulls_last=True)

    logger.info(
        "comparison.complete",
        n_runs=len(run_ids),
        columns=df.columns,
    )
    return df
