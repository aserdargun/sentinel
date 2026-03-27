"""Save and load model artifacts and prediction arrays.

Model artifacts are stored in the run directory alongside config and
metrics.  Statistical models use joblib; deep models use
``state_dict`` + ``config.json``.  Predictions (scores + labels) are
persisted as ``.npz`` files for lightweight, fast reloading.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import structlog

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.exceptions import SentinelError
from sentinel.core.registry import get_model_class

logger = structlog.get_logger(__name__)

_MODEL_SUBDIR = "model"
_PREDICTIONS_FILE = "predictions.npz"


def save_model_artifact(
    run_id: str,
    model: BaseAnomalyDetector,
    base_dir: str = "data/experiments",
) -> str:
    """Save a trained model to the experiment run directory.

    Delegates to the model's own ``save()`` method, which handles
    serialization format (joblib for statistical, state_dict for deep).

    Args:
        run_id: Experiment run identifier.
        model: Trained model instance to persist.
        base_dir: Root directory for experiments.

    Returns:
        Absolute path to the saved model directory.

    Raises:
        FileNotFoundError: If the run directory does not exist.
    """
    run_dir = Path(base_dir) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    model_dir = run_dir / _MODEL_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(model_dir))

    model_path = str(model_dir.resolve())
    logger.info(
        "artifacts.model_saved",
        run_id=run_id,
        model_name=model.model_name,
        path=model_path,
    )
    return model_path


def load_model_artifact(
    run_id: str,
    model_name: str,
    base_dir: str = "data/experiments",
) -> BaseAnomalyDetector:
    """Load a model from an experiment run directory.

    Looks up the model class by name from the registry, instantiates it,
    and calls ``load()`` with the saved artifact path.

    Args:
        run_id: Experiment run identifier.
        model_name: Registered model name (e.g. ``"zscore"``).
        base_dir: Root directory for experiments.

    Returns:
        Loaded model instance ready for scoring.

    Raises:
        FileNotFoundError: If the run or model directory does not exist.
        ModelNotFoundError: If the model name is not in the registry.
        SentinelError: If loading fails.
    """
    run_dir = Path(base_dir) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    model_dir = run_dir / _MODEL_SUBDIR
    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifacts not found: {model_dir}")

    model_cls = get_model_class(model_name)
    model = model_cls()

    try:
        model.load(str(model_dir))
    except Exception as exc:
        raise SentinelError(
            f"Failed to load model '{model_name}' from {model_dir}: {exc}"
        ) from exc

    logger.info(
        "artifacts.model_loaded",
        run_id=run_id,
        model_name=model_name,
        path=str(model_dir),
    )
    return model


def save_predictions(
    run_id: str,
    scores: np.ndarray,
    labels: np.ndarray,
    base_dir: str = "data/experiments",
) -> None:
    """Save prediction scores and labels to the run directory.

    Persisted as a compressed ``.npz`` file with keys ``scores`` and
    ``labels``.  The write is atomic (temp file + rename).

    Args:
        run_id: Experiment run identifier.
        scores: 1-D array of anomaly scores.
        labels: 1-D array of predicted binary labels.
        base_dir: Root directory for experiments.

    Raises:
        FileNotFoundError: If the run directory does not exist.
    """
    run_dir = Path(base_dir) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    pred_path = run_dir / _PREDICTIONS_FILE
    # np.savez_compressed appends ".npz" if the path doesn't end with it.
    # Write to a temp name ending in ".npz" so numpy doesn't double-suffix.
    tmp_path = run_dir / "predictions_tmp.npz"

    np.savez_compressed(str(tmp_path), scores=scores, labels=labels)
    os.rename(tmp_path, pred_path)

    logger.info(
        "artifacts.predictions_saved",
        run_id=run_id,
        n_samples=len(scores),
        path=str(pred_path),
    )


def load_predictions(
    run_id: str,
    base_dir: str = "data/experiments",
) -> dict[str, np.ndarray]:
    """Load prediction scores and labels from the run directory.

    Args:
        run_id: Experiment run identifier.
        base_dir: Root directory for experiments.

    Returns:
        Dictionary with ``"scores"`` and ``"labels"`` numpy arrays.

    Raises:
        FileNotFoundError: If the run directory or predictions file
            does not exist.
    """
    run_dir = Path(base_dir) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    pred_path = run_dir / _PREDICTIONS_FILE
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    data = np.load(str(pred_path))

    logger.info(
        "artifacts.predictions_loaded",
        run_id=run_id,
        path=str(pred_path),
    )
    return {"scores": data["scores"], "labels": data["labels"]}
