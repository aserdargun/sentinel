"""Rolling Z-Score anomaly detector.

Flags data points whose rolling z-score exceeds a configurable sigma
threshold. The z-score is computed per feature as |x - rolling_mean| /
rolling_std, and the maximum across features is taken as the sample score.
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.exceptions import SentinelError
from sentinel.core.registry import register_model

# Minimum standard deviation floor to avoid division by zero.
_MIN_STD = 1e-10


@register_model("zscore")
class ZScoreDetector(BaseAnomalyDetector):
    """Rolling Z-Score anomaly detector.

    For each sample, a rolling window of the preceding ``window_size`` points
    is used to compute a local mean and standard deviation per feature.  The
    z-score is ``|x - rolling_mean| / rolling_std``, and the maximum z-score
    across all features becomes the sample's anomaly score.

    During ``fit()``, global per-feature mean and standard deviation are
    stored so that the first ``window_size - 1`` samples (which lack a full
    rolling window) can still receive a reasonable score.

    Args:
        window_size: Number of preceding samples in the rolling window.
        threshold_sigma: Default sigma threshold for ``detect()``.

    Example::

        detector = ZScoreDetector(window_size=30, threshold_sigma=3.0)
        detector.fit(X_train)
        scores = detector.score(X_test)
    """

    def __init__(
        self,
        window_size: int = 30,
        threshold_sigma: float = 3.0,
    ) -> None:
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma

        # Populated by fit().
        self._global_mean: np.ndarray | None = None
        self._global_std: np.ndarray | None = None
        self._n_features: int | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, **kwargs: Any) -> None:
        """Compute and store global per-feature mean and std from training data.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            **kwargs: Ignored; present for interface compatibility.

        Raises:
            SentinelError: If ``X`` has fewer than 2 rows.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] < 2:
            raise SentinelError(
                f"Z-Score requires at least 2 samples, got {X.shape[0]}"
            )

        self._n_features = X.shape[1]
        self._global_mean = np.nanmean(X, axis=0)
        self._global_std = np.nanstd(X, axis=0)
        # Floor tiny std values to avoid division by zero.
        self._global_std = np.maximum(self._global_std, _MIN_STD)
        self._is_fitted = True

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute rolling z-score anomaly scores.

        For each sample *i*, the rolling mean and std are computed over the
        window ``[i - window_size + 1, i]`` (inclusive).  Samples at the
        start of the array that lack a full window use the global statistics
        learned during ``fit()`` as a fallback.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of length ``n_samples`` with anomaly scores (higher
            is more anomalous).

        Raises:
            SentinelError: If the model has not been fitted.
        """
        self._check_fitted()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        scores = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            start = max(0, i - self.window_size + 1)
            window = X[start : i + 1]

            if window.shape[0] >= 2:
                rolling_mean = np.nanmean(window, axis=0)
                rolling_std = np.nanstd(window, axis=0)
                rolling_std = np.maximum(rolling_std, _MIN_STD)
            else:
                # Not enough history -- fall back to global stats.
                rolling_mean = self._global_mean  # type: ignore[assignment]
                rolling_std = self._global_std  # type: ignore[assignment]

            z = np.abs(X[i] - rolling_mean) / rolling_std
            scores[i] = float(np.nanmax(z))

        return scores

    def save(self, path: str) -> None:
        """Save model parameters to disk.

        Writes two files into *path*:

        * ``config.json`` -- JSON with hyperparameters and metadata.
        * ``model.joblib`` -- Serialised numpy arrays (global stats).

        All writes go to a temporary file first and are atomically renamed.

        Args:
            path: Directory in which to store artifacts.

        Raises:
            SentinelError: If the model has not been fitted.
        """
        self._check_fitted()
        os.makedirs(path, exist_ok=True)

        # -- config.json (hyperparams + metadata) -------------------------
        config = {
            "model_name": "zscore",
            "window_size": self.window_size,
            "threshold_sigma": self.threshold_sigma,
            "n_features": self._n_features,
        }
        config_path = os.path.join(path, "config.json")
        tmp_config = config_path + ".tmp"
        with open(tmp_config, "w") as f:
            json.dump(config, f, indent=2)
        os.rename(tmp_config, config_path)

        # -- model.joblib (learned arrays) ---------------------------------
        state = {
            "global_mean": self._global_mean,
            "global_std": self._global_std,
        }
        model_path = os.path.join(path, "model.joblib")
        tmp_model = model_path + ".tmp"
        joblib.dump(state, tmp_model)
        os.rename(tmp_model, model_path)

    def load(self, path: str) -> None:
        """Load a previously saved model from disk.

        Args:
            path: Directory containing ``config.json`` and ``model.joblib``.

        Raises:
            SentinelError: If required files are missing.
        """
        config_path = os.path.join(path, "config.json")
        model_path = os.path.join(path, "model.joblib")

        if not os.path.isfile(config_path):
            raise SentinelError(f"Config not found: {config_path}")
        if not os.path.isfile(model_path):
            raise SentinelError(f"Model file not found: {model_path}")

        with open(config_path) as f:
            config = json.load(f)

        self.window_size = int(config["window_size"])
        self.threshold_sigma = float(config["threshold_sigma"])
        self._n_features = int(config["n_features"])

        state = joblib.load(model_path)
        self._global_mean = state["global_mean"]
        self._global_std = state["global_std"]
        self._is_fitted = True

    def get_params(self) -> dict[str, Any]:
        """Return model parameters.

        Returns:
            Dict containing ``window_size``, ``threshold_sigma``, and
            ``n_features`` (``None`` if not yet fitted).
        """
        return {
            "model_name": "zscore",
            "window_size": self.window_size,
            "threshold_sigma": self.threshold_sigma,
            "n_features": self._n_features,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not self._is_fitted:
            raise SentinelError("ZScoreDetector has not been fitted. Call fit() first.")
