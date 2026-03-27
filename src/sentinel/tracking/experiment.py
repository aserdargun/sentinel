"""Lightweight JSON-backed experiment tracker.

Stores run metadata, configs, metrics, and parameters in per-run
directories under a configurable base path.  All file writes are atomic
(write to ``{path}.tmp`` then ``os.rename()``) to prevent partial
artifacts from corrupted runs.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class LocalTracker:
    """JSON-backed experiment tracker.

    Each experiment run gets a directory ``{base_dir}/{run_id}/`` containing:

    * ``config.json`` -- full run configuration
    * ``metrics.json`` -- evaluation metrics
    * ``params.json`` -- model parameters
    * ``meta.json`` -- run metadata (run_id, model_name, timestamp)

    Args:
        base_dir: Root directory for experiment artifacts.
            Defaults to ``"data/experiments"``.

    Example::

        tracker = LocalTracker(base_dir="data/experiments")
        run_id = tracker.create_run("zscore")
        tracker.log_config(run_id, {"window_size": 30})
        tracker.log_metrics(run_id, {"f1": 0.85, "auc_roc": 0.92})
        run = tracker.get_run(run_id)
    """

    def __init__(self, base_dir: str = "data/experiments") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("tracker.init", base_dir=str(self._base_dir))

    @property
    def base_dir(self) -> Path:
        """Return the base directory for experiments."""
        return self._base_dir

    def create_run(self, model_name: str) -> str:
        """Create a new experiment run.

        Generates a UUID-based run_id, creates the run directory, and
        writes an initial ``meta.json`` with run metadata.

        Args:
            model_name: Name of the model being trained.

        Returns:
            The generated run_id string.
        """
        run_id = uuid.uuid4().hex[:12]
        run_dir = self._base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "run_id": run_id,
            "model_name": model_name,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "created",
        }
        self._write_json(run_dir / "meta.json", meta)

        logger.info(
            "tracker.run_created",
            run_id=run_id,
            model_name=model_name,
        )
        return run_id

    def log_config(self, run_id: str, config: dict[str, Any]) -> None:
        """Write run configuration to ``config.json``.

        Args:
            run_id: The experiment run identifier.
            config: Configuration dictionary to persist.

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """
        run_dir = self._get_run_dir(run_id)
        self._write_json(run_dir / "config.json", config)
        logger.info("tracker.config_logged", run_id=run_id)

    def log_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        """Write evaluation metrics to ``metrics.json``.

        Args:
            run_id: The experiment run identifier.
            metrics: Metrics dictionary to persist.  Values may be
                ``float``, ``int``, ``None``, or ``str``.

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """
        run_dir = self._get_run_dir(run_id)
        serializable = _make_serializable(metrics)
        self._write_json(run_dir / "metrics.json", serializable)
        logger.info("tracker.metrics_logged", run_id=run_id)

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Write model parameters to ``params.json``.

        Args:
            run_id: The experiment run identifier.
            params: Parameters dictionary to persist.

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """
        run_dir = self._get_run_dir(run_id)
        serializable = _make_serializable(params)
        self._write_json(run_dir / "params.json", serializable)
        logger.info("tracker.params_logged", run_id=run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Load all data for a single run.

        Reads ``meta.json``, ``config.json``, ``metrics.json``, and
        ``params.json`` if they exist.

        Args:
            run_id: The experiment run identifier.

        Returns:
            Combined dictionary with keys ``run_id``, ``model_name``,
            ``created_at``, ``config``, ``metrics``, and ``params``.

        Raises:
            FileNotFoundError: If the run directory does not exist.
        """
        run_dir = self._get_run_dir(run_id)

        result: dict[str, Any] = {"run_id": run_id}

        meta = self._read_json(run_dir / "meta.json")
        if meta:
            result["model_name"] = meta.get("model_name", "")
            result["created_at"] = meta.get("created_at", "")
            result["status"] = meta.get("status", "")

        result["config"] = self._read_json(run_dir / "config.json") or {}
        result["metrics"] = self._read_json(run_dir / "metrics.json") or {}
        result["params"] = self._read_json(run_dir / "params.json") or {}

        return result

    def list_runs(self) -> list[dict[str, Any]]:
        """List all experiment runs.

        Scans the base directory for run subdirectories and reads their
        metadata.  Runs are sorted by creation timestamp (newest first).

        Returns:
            List of dicts, each containing ``run_id``, ``model_name``,
            ``created_at``, and ``metrics``.
        """
        runs: list[dict[str, Any]] = []

        if not self._base_dir.exists():
            return runs

        for entry in sorted(self._base_dir.iterdir()):
            if not entry.is_dir():
                continue

            meta_path = entry / "meta.json"
            if not meta_path.exists():
                continue

            meta = self._read_json(meta_path)
            if meta is None:
                continue

            run_info: dict[str, Any] = {
                "run_id": meta.get("run_id", entry.name),
                "model_name": meta.get("model_name", ""),
                "created_at": meta.get("created_at", ""),
            }

            metrics = self._read_json(entry / "metrics.json")
            run_info["metrics"] = metrics or {}

            runs.append(run_info)

        # Sort newest first.
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)

        logger.info("tracker.list_runs", count=len(runs))
        return runs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_run_dir(self, run_id: str) -> Path:
        """Resolve and validate a run directory path.

        Args:
            run_id: The experiment run identifier.

        Returns:
            Path to the run directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        run_dir = self._base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        """Atomically write a JSON file.

        Writes to ``{path}.tmp`` first, then renames to the final path.

        Args:
            path: Target file path.
            data: Dictionary to serialize as JSON.
        """
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.rename(tmp_path, path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        """Read a JSON file, returning None if it does not exist.

        Args:
            path: File path to read.

        Returns:
            Parsed dictionary or None if file is missing.
        """
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)  # type: ignore[no-any-return]


def _make_serializable(data: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy/non-standard types to JSON-serializable values.

    Args:
        data: Dictionary that may contain numpy scalars or arrays.

    Returns:
        Dictionary with all values converted to native Python types.
    """
    import numpy as np

    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, dict):
            result[key] = _make_serializable(value)
        else:
            result[key] = value
    return result
