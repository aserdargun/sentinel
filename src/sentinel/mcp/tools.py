"""MCP tool definitions for Sentinel.

Each function is a standalone tool that can be registered with FastMCP.
All tools return JSON-serializable dicts.  On error they return
``{"error": str, "code": str}`` instead of raising exceptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Default paths (matching base.yaml conventions)
_DEFAULT_DATA_DIR = "data/raw"
_DEFAULT_METADATA_FILE = "data/datasets.json"
_DEFAULT_EXPERIMENTS_DIR = "data/experiments"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _error(message: str, code: str) -> dict[str, str]:
    """Build a standard error response dict.

    Args:
        message: Human-readable error description.
        code: Machine-readable error code.

    Returns:
        Error dict with ``error`` and ``code`` keys.
    """
    return {"error": message, "code": code}


def _load_metadata() -> dict[str, Any]:
    """Load the dataset metadata registry.

    Returns:
        Dict mapping dataset_id to metadata, or empty dict if file
        does not exist.
    """
    path = Path(_DEFAULT_METADATA_FILE)
    if not path.exists():
        return {}
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/non-standard types to JSON-safe values.

    Args:
        obj: Value to convert.

    Returns:
        JSON-serializable equivalent.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------


def sentinel_train(config_path: str) -> dict[str, Any]:
    """Train an anomaly detection model from a YAML config file.

    Validates the config, runs the full training pipeline via the
    Trainer class, and returns run metadata with metrics.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Dict with ``run_id``, ``metrics``, and ``duration_s`` on
        success, or ``{error, code}`` on failure.
    """
    logger.info("tool.train.start", config_path=config_path)
    try:
        path = Path(config_path)
        if not path.exists():
            return _error(f"Config file not found: {config_path}", "CONFIG_NOT_FOUND")

        from sentinel.core.config import RunConfig
        from sentinel.training.trainer import Trainer

        config = RunConfig.from_yaml(path)
        trainer = Trainer(config)
        result = trainer.run()

        return _make_serializable(
            {
                "run_id": result["run_id"],
                "model_name": result["model_name"],
                "metrics": result["metrics"],
                "duration_s": result["duration_s"],
            }
        )
    except Exception as exc:
        logger.error("tool.train.failed", error=str(exc))
        return _error(str(exc), "TRAIN_FAILED")


def sentinel_detect(model_path: str, data_path: str) -> dict[str, Any]:
    """Run batch anomaly detection with a saved model.

    Loads the model from ``model_path``, scores the data at
    ``data_path``, and returns scores, labels, and threshold.

    Args:
        model_path: Path to saved model directory.
        data_path: Path to data file (CSV or Parquet).

    Returns:
        Dict with ``scores``, ``labels``, and ``threshold``, or
        ``{error, code}`` on failure.
    """
    logger.info("tool.detect.start", model_path=model_path, data_path=data_path)
    try:
        from sentinel.core.registry import get_model_class
        from sentinel.data.loaders import load_file
        from sentinel.data.preprocessors import fill_nan, scale_zscore, to_numpy
        from sentinel.data.validators import separate_labels, validate_dataframe
        from sentinel.training.thresholds import percentile_threshold

        # Load and preprocess data
        df = load_file(data_path)
        df = validate_dataframe(df)
        df, _labels = separate_labels(df)
        df = fill_nan(df)
        df, _stats = scale_zscore(df)
        X = to_numpy(df)

        # Load model metadata and reconstruct
        model_dir = Path(model_path)
        config_file = model_dir / "config.json"
        if not config_file.exists():
            return _error(f"Model config not found at {config_file}", "MODEL_NOT_FOUND")

        model_config = json.loads(config_file.read_text())
        model_name = model_config.get("model_name", "")
        model_cls = get_model_class(model_name)
        model = model_cls()
        model.load(str(model_dir))

        # Score and threshold
        scores = model.score(X)
        threshold = percentile_threshold(scores, 95.0)
        labels = (scores > threshold).astype(np.int32)

        return _make_serializable(
            {
                "scores": scores,
                "labels": labels,
                "threshold": threshold,
                "model_name": model_name,
            }
        )
    except Exception as exc:
        logger.error("tool.detect.failed", error=str(exc))
        return _error(str(exc), "DETECT_FAILED")


def sentinel_list_models() -> dict[str, Any]:
    """List all registered anomaly detection models.

    Triggers model registration by importing ``sentinel.models`` and
    returns the registry contents.

    Returns:
        Dict with ``models`` list, each entry containing ``name``,
        ``category``, and ``description``.
    """
    logger.info("tool.list_models")
    try:
        import sentinel.models  # noqa: F401 — trigger registration
        from sentinel.core.registry import list_models

        registry = list_models()
        models = []
        for name, cls in sorted(registry.items()):
            category = ""
            if hasattr(cls, "category"):
                category = str(cls.category)
            elif "statistical" in cls.__module__:
                category = "statistical"
            elif "deep" in cls.__module__:
                category = "deep"
            elif "ensemble" in cls.__module__:
                category = "ensemble"

            description = ""
            if cls.__doc__:
                description = cls.__doc__.strip().split("\n")[0]

            models.append(
                {
                    "name": name,
                    "category": category,
                    "description": description,
                }
            )
        return {"models": models}
    except Exception as exc:
        logger.error("tool.list_models.failed", error=str(exc))
        return _error(str(exc), "LIST_MODELS_FAILED")


def sentinel_list_datasets() -> dict[str, Any]:
    """List all uploaded datasets from the metadata registry.

    Returns:
        Dict with ``datasets`` list, each containing ``id``, ``name``,
        ``shape``, ``features``, and ``uploaded_at``.
    """
    logger.info("tool.list_datasets")
    try:
        metadata = _load_metadata()
        datasets = []
        for dataset_id, info in metadata.items():
            datasets.append(
                {
                    "id": dataset_id,
                    "name": info.get("original_name", ""),
                    "shape": info.get("shape", []),
                    "features": info.get("feature_names", []),
                    "uploaded_at": info.get("uploaded_at", ""),
                    "source": info.get("source", ""),
                    "time_range": info.get("time_range", {}),
                }
            )
        return {"datasets": datasets}
    except Exception as exc:
        logger.error("tool.list_datasets.failed", error=str(exc))
        return _error(str(exc), "LIST_DATASETS_FAILED")


def sentinel_upload(file_path: str) -> dict[str, Any]:
    """Ingest a data file into the Sentinel data store.

    Validates the file, assigns a dataset_id, and stores it as Parquet.

    Args:
        file_path: Path to a CSV or Parquet file.

    Returns:
        Dict with ``dataset_id``, ``shape``, ``features``, and
        ``time_range``, or ``{error, code}`` on failure.
    """
    logger.info("tool.upload.start", file_path=file_path)
    try:
        if not Path(file_path).exists():
            return _error(f"File not found: {file_path}", "FILE_NOT_FOUND")

        from sentinel.data.ingest import ingest_file

        result = ingest_file(file_path)
        return {
            "dataset_id": result["dataset_id"],
            "shape": result["shape"],
            "features": result["feature_names"],
            "time_range": result["time_range"],
        }
    except Exception as exc:
        logger.error("tool.upload.failed", error=str(exc))
        return _error(str(exc), "UPLOAD_FAILED")


def sentinel_compare_runs(run_ids: list[str]) -> dict[str, Any]:
    """Compare metrics across multiple experiment runs.

    Args:
        run_ids: List of run identifiers to compare.

    Returns:
        Dict with ``runs`` list, each containing ``run_id``,
        ``model_name``, ``created_at``, and metrics.
    """
    logger.info("tool.compare_runs", run_ids=run_ids)
    try:
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=_DEFAULT_EXPERIMENTS_DIR)
        runs = []
        for run_id in run_ids:
            try:
                run_data = tracker.get_run(run_id)
                runs.append(
                    _make_serializable(
                        {
                            "run_id": run_data.get("run_id", run_id),
                            "model_name": run_data.get("model_name", ""),
                            "created_at": run_data.get("created_at", ""),
                            "metrics": run_data.get("metrics", {}),
                        }
                    )
                )
            except FileNotFoundError:
                runs.append(
                    {
                        "run_id": run_id,
                        "error": f"Run '{run_id}' not found",
                    }
                )
        return {"runs": runs}
    except Exception as exc:
        logger.error("tool.compare_runs.failed", error=str(exc))
        return _error(str(exc), "COMPARE_FAILED")


async def sentinel_analyze(
    run_id: str,
    ollama_client: Any | None = None,
) -> dict[str, Any]:
    """Generate a natural-language anomaly analysis report for a run.

    Uses the Ollama LLM to produce a narrative from run metrics.
    Falls back to raw metrics if Ollama is unavailable.

    Args:
        run_id: Experiment run identifier.
        ollama_client: Optional :class:`OllamaClient` instance.  If
            ``None``, falls back to structured data without narrative.

    Returns:
        Dict with ``run_id``, ``report`` (text), and ``metrics``.
    """
    logger.info("tool.analyze.start", run_id=run_id)
    try:
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=_DEFAULT_EXPERIMENTS_DIR)
        run_data = tracker.get_run(run_id)
        metrics = run_data.get("metrics", {})

        scores_summary = {
            k: v
            for k, v in metrics.items()
            if k.startswith("score_") or k == "threshold"
        }
        classification_metrics = {
            k: v for k, v in metrics.items() if k not in scores_summary
        }

        # Try LLM narrative
        report: str | None = None
        if ollama_client is not None:
            from sentinel.mcp.prompts import anomaly_report_prompt

            prompt = anomaly_report_prompt(
                run_id, classification_metrics, scores_summary
            )
            report = await ollama_client.generate(prompt)

        if report is None:
            # Fallback: structured text without LLM
            metric_lines = "\n".join(f"  {k}: {v}" for k, v in metrics.items())
            report = (
                f"Anomaly analysis for run {run_id} "
                f"(LLM unavailable, showing raw metrics):\n{metric_lines}"
            )

        return _make_serializable(
            {
                "run_id": run_id,
                "model_name": run_data.get("model_name", ""),
                "report": report,
                "metrics": metrics,
            }
        )
    except FileNotFoundError:
        return _error(f"Run '{run_id}' not found", "RUN_NOT_FOUND")
    except Exception as exc:
        logger.error("tool.analyze.failed", error=str(exc))
        return _error(str(exc), "ANALYZE_FAILED")


async def sentinel_recommend_model(
    data_path: str,
    ollama_client: Any | None = None,
) -> dict[str, Any]:
    """Recommend anomaly detection models for a dataset.

    Analyses data characteristics (length, features, periodicity, noise)
    and optionally uses the LLM for a narrative recommendation.

    Args:
        data_path: Path to a data file (CSV or Parquet).
        ollama_client: Optional :class:`OllamaClient` instance.

    Returns:
        Dict with ``data_summary`` and ``recommendations``.
    """
    logger.info("tool.recommend.start", data_path=data_path)
    try:
        if not Path(data_path).exists():
            return _error(f"File not found: {data_path}", "FILE_NOT_FOUND")

        from sentinel.data.loaders import load_file
        from sentinel.data.validators import get_feature_columns, validate_dataframe

        df = load_file(data_path)
        df = validate_dataframe(df)

        feature_cols = get_feature_columns(df)
        n_rows = df.height
        n_features = len(feature_cols)

        # Basic data analysis
        data_summary: dict[str, Any] = {
            "n_rows": n_rows,
            "n_features": n_features,
            "has_labels": "is_anomaly" in df.columns,
        }

        # Deterministic recommendations based on data characteristics
        recommendations = _deterministic_recommendations(n_rows, n_features)

        # Try LLM-enhanced recommendations
        llm_response: str | None = None
        if ollama_client is not None:
            from sentinel.mcp.prompts import model_recommendation_prompt

            prompt = model_recommendation_prompt(data_summary)
            llm_response = await ollama_client.generate(prompt)

        result: dict[str, Any] = {
            "data_summary": data_summary,
            "recommendations": recommendations,
        }
        if llm_response is not None:
            result["llm_analysis"] = llm_response

        return _make_serializable(result)
    except Exception as exc:
        logger.error("tool.recommend.failed", error=str(exc))
        return _error(str(exc), "RECOMMEND_FAILED")


def sentinel_delete_run(run_id: str) -> dict[str, Any]:
    """Delete an experiment run and its artifacts.

    Args:
        run_id: Experiment run identifier to delete.

    Returns:
        Dict with ``deleted`` and ``artifacts_removed`` count, or
        ``{error, code}`` on failure.
    """
    logger.info("tool.delete_run", run_id=run_id)
    try:
        import shutil

        run_dir = Path(_DEFAULT_EXPERIMENTS_DIR) / run_id
        if not run_dir.exists():
            return _error(f"Run '{run_id}' not found", "RUN_NOT_FOUND")

        artifact_count = sum(1 for _ in run_dir.iterdir())
        shutil.rmtree(run_dir)

        logger.info("tool.delete_run.complete", run_id=run_id, removed=artifact_count)
        return {"deleted": True, "run_id": run_id, "artifacts_removed": artifact_count}
    except Exception as exc:
        logger.error("tool.delete_run.failed", error=str(exc))
        return _error(str(exc), "DELETE_FAILED")


def sentinel_export_model(
    run_id: str,
    format: str = "native",
) -> dict[str, Any]:
    """Export a trained model from an experiment run.

    Args:
        run_id: Experiment run identifier.
        format: Export format (``"native"`` or ``"onnx"``).

    Returns:
        Dict with ``export_path``, ``format``, and ``size_bytes``.
    """
    logger.info("tool.export", run_id=run_id, format=format)
    try:
        run_dir = Path(_DEFAULT_EXPERIMENTS_DIR) / run_id
        if not run_dir.exists():
            return _error(f"Run '{run_id}' not found", "RUN_NOT_FOUND")

        if format not in ("native", "onnx"):
            return _error(
                f"Unsupported format: {format}. Use 'native' or 'onnx'.",
                "INVALID_FORMAT",
            )

        # For native format, the model artifacts are already in the run dir
        if format == "native":
            model_files = list(run_dir.glob("model.*")) + list(run_dir.glob("*.joblib"))
            if not model_files:
                return _error(
                    f"No model artifacts found in run {run_id}", "NO_ARTIFACTS"
                )

            total_size = sum(f.stat().st_size for f in model_files)
            return {
                "export_path": str(run_dir),
                "format": "native",
                "size_bytes": total_size,
                "files": [f.name for f in model_files],
            }

        # ONNX export is a placeholder for now
        return _error("ONNX export not yet implemented", "NOT_IMPLEMENTED")
    except Exception as exc:
        logger.error("tool.export.failed", error=str(exc))
        return _error(str(exc), "EXPORT_FAILED")


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _deterministic_recommendations(
    n_rows: int,
    n_features: int,
) -> list[dict[str, str]]:
    """Generate rule-based model recommendations from data characteristics.

    Args:
        n_rows: Number of rows in the dataset.
        n_features: Number of feature columns.

    Returns:
        List of recommendation dicts with ``model`` and ``reason``.
    """
    recommendations: list[dict[str, str]] = []

    # Small datasets: prefer statistical methods
    if n_rows < 1000:
        recommendations.append(
            {
                "model": "zscore",
                "reason": "Fast and interpretable for small datasets.",
            }
        )
        recommendations.append(
            {
                "model": "isolation_forest",
                "reason": "Handles multivariate data without temporal ordering.",
            }
        )
    # Medium datasets: good for most models
    elif n_rows < 50000:
        recommendations.append(
            {
                "model": "lstm_ae",
                "reason": "Captures temporal patterns in medium-length sequences.",
            }
        )
        recommendations.append(
            {
                "model": "isolation_forest",
                "reason": "Robust baseline for multivariate anomaly detection.",
            }
        )
        if n_features <= 20:
            recommendations.append(
                {
                    "model": "matrix_profile",
                    "reason": "Subsequence-level discord detection for moderate data.",
                }
            )
    # Large datasets: deep models shine
    else:
        recommendations.append(
            {
                "model": "tranad",
                "reason": "Transformer-based, excels on long multivariate series.",
            }
        )
        recommendations.append(
            {
                "model": "lstm_ae",
                "reason": "Reliable deep model for temporal reconstruction.",
            }
        )
        recommendations.append(
            {
                "model": "vae",
                "reason": "Latent-space modelling captures complex distributions.",
            }
        )

    # Always suggest ensemble if enough variety
    if len(recommendations) >= 2:
        recommendations.append(
            {
                "model": "hybrid_ensemble",
                "reason": "Combines multiple detectors for more robust scoring.",
            }
        )

    return recommendations


# ------------------------------------------------------------------
# Tool metadata (for registration in server.py)
# ------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "sentinel_train",
        "description": "Train an anomaly detection model from a YAML config file.",
        "parameters": {"config_path": "str — Path to YAML config file"},
        "function": sentinel_train,
    },
    {
        "name": "sentinel_detect",
        "description": "Run batch anomaly detection on data with a saved model.",
        "parameters": {
            "model_path": "str — Path to saved model directory",
            "data_path": "str — Path to CSV or Parquet data file",
        },
        "function": sentinel_detect,
    },
    {
        "name": "sentinel_list_models",
        "description": "List all registered anomaly detection models.",
        "parameters": {},
        "function": sentinel_list_models,
    },
    {
        "name": "sentinel_list_datasets",
        "description": "List all uploaded datasets.",
        "parameters": {},
        "function": sentinel_list_datasets,
    },
    {
        "name": "sentinel_upload",
        "description": "Ingest a CSV or Parquet file into the Sentinel data store.",
        "parameters": {"file_path": "str — Path to the data file"},
        "function": sentinel_upload,
    },
    {
        "name": "sentinel_compare_runs",
        "description": "Compare metrics across multiple experiment runs.",
        "parameters": {"run_ids": "list[str] — List of run identifiers"},
        "function": sentinel_compare_runs,
    },
    {
        "name": "sentinel_analyze",
        "description": (
            "Generate a natural-language anomaly analysis report for a run. "
            "Uses LLM if available."
        ),
        "parameters": {"run_id": "str — Experiment run identifier"},
        "function": sentinel_analyze,
    },
    {
        "name": "sentinel_recommend_model",
        "description": (
            "Recommend anomaly detection models based on dataset characteristics."
        ),
        "parameters": {"data_path": "str — Path to the data file to analyse"},
        "function": sentinel_recommend_model,
    },
    {
        "name": "sentinel_delete_run",
        "description": "Delete an experiment run and its artifacts.",
        "parameters": {"run_id": "str — Experiment run identifier"},
        "function": sentinel_delete_run,
    },
    {
        "name": "sentinel_export_model",
        "description": "Export a trained model from an experiment run.",
        "parameters": {
            "run_id": "str — Experiment run identifier",
            "format": "str — Export format: 'native' or 'onnx'",
        },
        "function": sentinel_export_model,
    },
]
