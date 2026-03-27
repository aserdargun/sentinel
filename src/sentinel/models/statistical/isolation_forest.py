"""Isolation Forest anomaly detector.

Wraps scikit-learn's IsolationForest with the Sentinel BaseAnomalyDetector
interface. Anomaly scores are negated ``score_samples`` so that higher
values indicate more anomalous observations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.exceptions import SentinelError
from sentinel.core.registry import register_model

DEFAULT_N_ESTIMATORS: int = 100
DEFAULT_CONTAMINATION: float = 0.05
DEFAULT_RANDOM_STATE: int = 42

_MODEL_FILENAME = "model.joblib"
_CONFIG_FILENAME = "config.json"


@register_model("isolation_forest")
class IsolationForestDetector(BaseAnomalyDetector):
    """Anomaly detector based on Isolation Forest.

    Isolation Forest isolates anomalies by randomly selecting a feature
    and then randomly selecting a split value between the maximum and
    minimum values of that feature.  Anomalies require fewer splits
    (shorter path length) to be isolated, producing higher anomaly scores.

    Attributes:
        n_estimators: Number of base estimators (trees) in the ensemble.
        contamination: Expected proportion of anomalies in the dataset.
        random_state: Seed for the random number generator.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        contamination: float = DEFAULT_CONTAMINATION,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        """Initialise the Isolation Forest detector.

        Args:
            n_estimators: Number of trees in the forest.
            contamination: Expected fraction of anomalies in training data.
                Must be in (0, 0.5].
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self._model: Any = None  # sklearn IsolationForest instance
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # BaseAnomalyDetector interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, **kwargs: Any) -> None:
        """Fit the Isolation Forest on training data.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            **kwargs: Ignored; accepted for interface compatibility.

        Raises:
            SentinelError: If the input array is empty or has no features.
        """
        from sklearn.ensemble import IsolationForest

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise SentinelError(
                f"Expected 2-D array with at least 1 sample and 1 feature, "
                f"got shape {X.shape}"
            )

        self._n_features = X.shape[1]
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._model.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for each sample.

        Scores are ``-1 * model.score_samples(X)`` so that **higher**
        values correspond to more anomalous observations.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of anomaly scores with length ``n_samples``.

        Raises:
            SentinelError: If the model has not been fitted.
        """
        if self._model is None:
            raise SentinelError("Model has not been fitted. Call fit() first.")

        return -1.0 * self._model.score_samples(X)

    def save(self, path: str) -> None:
        """Persist the fitted model and its configuration to *path*.

        Writes two files inside the directory:

        * ``model.joblib`` -- serialised sklearn model (atomic write).
        * ``config.json``  -- hyperparameters and metadata (atomic write).

        Args:
            path: Directory in which to save artefacts.  Created if it
                does not exist.

        Raises:
            SentinelError: If the model has not been fitted.
        """
        if self._model is None:
            raise SentinelError("Cannot save an unfitted model.")

        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Atomic write: model
        model_path = dir_path / _MODEL_FILENAME
        tmp_model = str(model_path) + ".tmp"
        joblib.dump(self._model, tmp_model)
        os.rename(tmp_model, str(model_path))

        # Atomic write: config
        config_path = dir_path / _CONFIG_FILENAME
        config_data = {
            "model_name": self.model_name,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_features": self._n_features,
        }
        tmp_config = str(config_path) + ".tmp"
        with open(tmp_config, "w") as fh:
            json.dump(config_data, fh, indent=2)
        os.rename(tmp_config, str(config_path))

    def load(self, path: str) -> None:
        """Restore a previously saved model from *path*.

        Args:
            path: Directory containing ``model.joblib`` and ``config.json``.

        Raises:
            SentinelError: If required files are missing or corrupt.
        """
        dir_path = Path(path)

        config_path = dir_path / _CONFIG_FILENAME
        if not config_path.exists():
            raise SentinelError(f"Config file not found: {config_path}")
        try:
            with open(config_path) as fh:
                config_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise SentinelError(f"Failed to read config: {exc}") from exc

        self.n_estimators = config_data.get("n_estimators", DEFAULT_N_ESTIMATORS)
        self.contamination = config_data.get("contamination", DEFAULT_CONTAMINATION)
        self.random_state = config_data.get("random_state", DEFAULT_RANDOM_STATE)
        self._n_features = config_data.get("n_features")

        model_path = dir_path / _MODEL_FILENAME
        if not model_path.exists():
            raise SentinelError(f"Model file not found: {model_path}")
        try:
            self._model = joblib.load(str(model_path))
        except Exception as exc:
            raise SentinelError(f"Failed to load model: {exc}") from exc

    def get_params(self) -> dict[str, Any]:
        """Return the detector's hyperparameters.

        Returns:
            Dictionary with keys ``n_estimators``, ``contamination``, and
            ``random_state``.
        """
        return {
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }
