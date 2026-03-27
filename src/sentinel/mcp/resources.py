"""MCP resource definitions for Sentinel.

Resources expose read-only structured data about experiments, models,
and datasets.  Each function returns JSON-serializable dicts suitable
for MCP resource responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_EXPERIMENTS_DIR = "data/experiments"
_DEFAULT_METADATA_FILE = "data/datasets.json"


def experiments_list() -> list[dict[str, Any]]:
    """List all experiment runs with metadata and key metrics.

    Scans ``data/experiments/`` for run directories and reads their
    ``meta.json`` and ``metrics.json`` files.

    Returns:
        List of dicts, each with ``run_id``, ``model_name``,
        ``created_at``, and ``metrics``.
    """
    logger.info("resource.experiments_list")
    try:
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=_DEFAULT_EXPERIMENTS_DIR)
        return tracker.list_runs()
    except Exception as exc:
        logger.error("resource.experiments_list.failed", error=str(exc))
        return []


def experiment_detail(run_id: str) -> dict[str, Any]:
    """Get full details for a single experiment run.

    Args:
        run_id: Experiment run identifier.

    Returns:
        Dict with ``run_id``, ``model_name``, ``created_at``,
        ``config``, ``metrics``, ``params``, and ``artifact_paths``.
        Returns an error dict if the run is not found.
    """
    logger.info("resource.experiment_detail", run_id=run_id)
    try:
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=_DEFAULT_EXPERIMENTS_DIR)
        run_data = tracker.get_run(run_id)

        # Add artifact file listing
        run_dir = Path(_DEFAULT_EXPERIMENTS_DIR) / run_id
        artifact_paths: list[str] = []
        if run_dir.exists():
            artifact_paths = [f.name for f in run_dir.iterdir() if f.is_file()]

        run_data["artifact_paths"] = artifact_paths
        return run_data
    except FileNotFoundError:
        return {"error": f"Run '{run_id}' not found", "code": "RUN_NOT_FOUND"}
    except Exception as exc:
        logger.error("resource.experiment_detail.failed", error=str(exc))
        return {"error": str(exc), "code": "RESOURCE_ERROR"}


def models_registry() -> list[dict[str, Any]]:
    """List all registered models with metadata.

    Triggers model registration and returns the registry contents
    including name, category, description, and config schema hints.

    Returns:
        List of model info dicts.
    """
    logger.info("resource.models_registry")
    try:
        import sentinel.models  # noqa: F401 — trigger registration
        from sentinel.core.registry import list_models

        registry = list_models()
        models: list[dict[str, Any]] = []

        for name, cls in sorted(registry.items()):
            category = ""
            if "statistical" in cls.__module__:
                category = "statistical"
            elif "deep" in cls.__module__:
                category = "deep"
            elif "ensemble" in cls.__module__:
                category = "ensemble"

            description = ""
            if cls.__doc__:
                # First non-empty line of the docstring.
                for line in cls.__doc__.strip().split("\n"):
                    line = line.strip()
                    if line:
                        description = line
                        break

            # Extract constructor parameters for schema hints
            params: dict[str, str] = {}
            try:
                import inspect

                sig = inspect.signature(cls.__init__)
                for pname, param in sig.parameters.items():
                    if pname == "self":
                        continue
                    annotation = (
                        param.annotation.__name__
                        if hasattr(param.annotation, "__name__")
                        else str(param.annotation)
                    )
                    default = (
                        str(param.default)
                        if param.default is not inspect.Parameter.empty
                        else "required"
                    )
                    params[pname] = f"{annotation} (default: {default})"
            except (TypeError, ValueError):
                pass

            models.append(
                {
                    "name": name,
                    "category": category,
                    "description": description,
                    "parameters": params,
                }
            )

        return models
    except Exception as exc:
        logger.error("resource.models_registry.failed", error=str(exc))
        return []


def datasets_list() -> list[dict[str, Any]]:
    """List all uploaded datasets with metadata.

    Reads the ``data/datasets.json`` metadata registry.

    Returns:
        List of dataset summary dicts.
    """
    logger.info("resource.datasets_list")
    try:
        metadata = _load_metadata()
        datasets: list[dict[str, Any]] = []
        for dataset_id, info in metadata.items():
            datasets.append(
                {
                    "id": dataset_id,
                    "name": info.get("original_name", ""),
                    "source": info.get("source", ""),
                    "shape": info.get("shape", []),
                    "features": info.get("feature_names", []),
                    "time_range": info.get("time_range", {}),
                    "uploaded_at": info.get("uploaded_at", ""),
                }
            )
        return datasets
    except Exception as exc:
        logger.error("resource.datasets_list.failed", error=str(exc))
        return []


def dataset_detail(dataset_id: str) -> dict[str, Any]:
    """Get detailed information about a single dataset.

    Reads the Parquet file and computes column-level statistics via
    Polars ``.describe()``.

    Args:
        dataset_id: UUID identifier of the dataset.

    Returns:
        Dict with metadata, column types, basic stats, and null counts.
        Returns an error dict if the dataset is not found.
    """
    logger.info("resource.dataset_detail", dataset_id=dataset_id)
    try:
        metadata = _load_metadata()
        if dataset_id not in metadata:
            return {
                "error": f"Dataset '{dataset_id}' not found",
                "code": "DATASET_NOT_FOUND",
            }

        info = metadata[dataset_id]
        result: dict[str, Any] = {
            "id": dataset_id,
            "name": info.get("original_name", ""),
            "source": info.get("source", ""),
            "shape": info.get("shape", []),
            "features": info.get("feature_names", []),
            "time_range": info.get("time_range", {}),
            "uploaded_at": info.get("uploaded_at", ""),
        }

        # Load the Parquet file for statistics
        parquet_path = Path("data/raw") / f"{dataset_id}.parquet"
        if parquet_path.exists():
            import polars as pl

            df = pl.read_parquet(parquet_path)
            result["column_types"] = {
                col: str(dtype) for col, dtype in df.schema.items()
            }

            # Null counts per column
            result["null_counts"] = {col: df[col].null_count() for col in df.columns}

            # Basic stats via describe (exclude timestamp for numeric stats)
            numeric_cols = [
                c for c in df.columns if c != "timestamp" and df[c].dtype.is_numeric()
            ]
            if numeric_cols:
                desc = df.select(numeric_cols).describe()
                result["stats"] = desc.to_dicts()

        return result
    except Exception as exc:
        logger.error("resource.dataset_detail.failed", error=str(exc))
        return {"error": str(exc), "code": "RESOURCE_ERROR"}


def dataset_preview(dataset_id: str, rows: int = 20) -> dict[str, Any]:
    """Get first N rows of a dataset as structured JSON.

    Args:
        dataset_id: UUID identifier of the dataset.
        rows: Number of rows to return (default 20, max 1000).

    Returns:
        Dict with ``id``, ``rows_returned``, and ``data`` (list of row dicts).
    """
    logger.info("resource.dataset_preview", dataset_id=dataset_id, rows=rows)
    try:
        metadata = _load_metadata()
        if dataset_id not in metadata:
            return {
                "error": f"Dataset '{dataset_id}' not found",
                "code": "DATASET_NOT_FOUND",
            }

        rows = min(rows, 1000)
        parquet_path = Path("data/raw") / f"{dataset_id}.parquet"
        if not parquet_path.exists():
            return {
                "error": f"Parquet file not found for dataset {dataset_id}",
                "code": "FILE_NOT_FOUND",
            }

        import polars as pl

        df = pl.read_parquet(parquet_path)
        preview = df.head(rows)

        return {
            "id": dataset_id,
            "rows_returned": preview.height,
            "data": preview.to_dicts(),
        }
    except Exception as exc:
        logger.error("resource.dataset_preview.failed", error=str(exc))
        return {"error": str(exc), "code": "RESOURCE_ERROR"}


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _load_metadata() -> dict[str, Any]:
    """Load the dataset metadata registry from datasets.json.

    Returns:
        Dict mapping dataset_id to metadata, or empty dict if file
        does not exist.
    """
    path = Path(_DEFAULT_METADATA_FILE)
    if not path.exists():
        return {}
    return json.loads(path.read_text())  # type: ignore[no-any-return]
