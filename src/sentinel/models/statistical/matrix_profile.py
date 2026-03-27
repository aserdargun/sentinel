"""Matrix Profile anomaly detector using STOMP via stumpy.

Computes subsequence distance profiles to identify anomalous temporal
patterns. For multivariate data, computes per-feature matrix profiles
and combines them. Long series are chunked with overlap to manage
O(n^2) memory.
"""

from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import stumpy

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.exceptions import ValidationError
from sentinel.core.registry import register_model

MIN_LENGTH_MULTIPLIER = 3


@register_model("matrix_profile")
class MatrixProfileDetector(BaseAnomalyDetector):
    """Anomaly detector based on Matrix Profile distance scoring.

    Uses stumpy's STOMP algorithm to compute the matrix profile -- a vector
    of nearest-neighbor distances for every subsequence in a time series.
    Subsequences with large distances (no similar neighbors) are flagged
    as anomalous.

    For multivariate data, per-feature matrix profiles are computed
    independently and averaged. For datasets exceeding ``max_rows``,
    the data is chunked with overlap equal to ``subsequence_length`` to
    keep memory usage bounded.

    Attributes:
        subsequence_length: Length of subsequences for the profile.
        max_rows: Maximum rows before chunking is applied.
        n_features_: Number of features seen during fit (set after fit).
        mean_score_: Mean matrix profile value from training (set after fit).
    """

    def __init__(
        self,
        subsequence_length: int = 50,
        max_rows: int = 100_000,
    ) -> None:
        """Initialize the Matrix Profile detector.

        Args:
            subsequence_length: Length of each subsequence window. Must be
                at least 4 (stumpy minimum).
            max_rows: Maximum number of rows before chunking with overlap.
                Datasets longer than this are split into chunks of size
                ``max_rows`` with ``subsequence_length`` overlap.
        """
        if subsequence_length < 4:
            raise ValidationError(
                f"subsequence_length must be >= 4, got {subsequence_length}"
            )
        self.subsequence_length = subsequence_length
        self.max_rows = max_rows
        self.n_features_: int = 0
        self.mean_score_: float = 0.0

    def fit(self, X: np.ndarray, **kwargs: Any) -> None:
        """Store training data statistics for later scoring.

        Heavy matrix profile computation is deferred to ``score()`` since
        it must operate on the actual data being scored. Fit records the
        feature count and computes a baseline mean score on training data
        for padding purposes.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            **kwargs: Unused.

        Raises:
            ValidationError: If data is too short for the subsequence length.
        """
        self._validate_length(X)
        self.n_features_ = X.shape[1] if X.ndim == 2 else 1

        # Compute baseline mean score for padding and reference
        scores = self._compute_scores(X)
        self.mean_score_ = float(np.nanmean(scores))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores via matrix profile distances.

        For univariate data (single feature), uses ``stumpy.stump()``.
        For multivariate data, computes per-feature profiles and averages
        them. The matrix profile is padded at the front to match input
        length -- the first ``subsequence_length - 1`` entries receive
        the mean score.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            1D array of anomaly scores with length ``n_samples``.
            Higher scores indicate more anomalous subsequences.

        Raises:
            ValidationError: If data is too short for the subsequence length.
        """
        self._validate_length(X)
        return self._compute_scores(X)

    def save(self, path: str) -> None:
        """Save model parameters to disk.

        Writes a JSON config and a joblib state file atomically
        (write to temp, then rename).

        Args:
            path: Directory to save model artifacts into.
        """
        os.makedirs(path, exist_ok=True)

        config = {
            "subsequence_length": self.subsequence_length,
            "max_rows": self.max_rows,
            "n_features_": self.n_features_,
            "mean_score_": self.mean_score_,
        }

        config_path = os.path.join(path, "config.json")
        tmp_config = f"{config_path}.tmp"
        with open(tmp_config, "w") as f:
            json.dump(config, f, indent=2)
        os.rename(tmp_config, config_path)

        state = {
            "subsequence_length": self.subsequence_length,
            "max_rows": self.max_rows,
            "n_features_": self.n_features_,
            "mean_score_": self.mean_score_,
        }
        state_path = os.path.join(path, "model.joblib")
        tmp_state = f"{state_path}.tmp"
        joblib.dump(state, tmp_state)
        os.rename(tmp_state, state_path)

    def load(self, path: str) -> None:
        """Load model parameters from disk.

        Args:
            path: Directory containing saved model artifacts.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        state_path = os.path.join(path, "model.joblib")
        state: dict[str, Any] = joblib.load(state_path)

        self.subsequence_length = state["subsequence_length"]
        self.max_rows = state["max_rows"]
        self.n_features_ = state["n_features_"]
        self.mean_score_ = state["mean_score_"]

    def get_params(self) -> dict[str, Any]:
        """Return model parameters as a dictionary.

        Returns:
            Dict containing ``subsequence_length``, ``max_rows``,
            ``n_features_``, and ``mean_score_``.
        """
        return {
            "subsequence_length": self.subsequence_length,
            "max_rows": self.max_rows,
            "n_features_": self.n_features_,
            "mean_score_": self.mean_score_,
        }

    def _validate_length(self, X: np.ndarray) -> None:
        """Check that data length meets the minimum requirement.

        Args:
            X: Input data array.

        Raises:
            ValidationError: If ``len(data) < subsequence_length * 3``.
        """
        n_samples = X.shape[0]
        min_required = self.subsequence_length * MIN_LENGTH_MULTIPLIER
        if n_samples < min_required:
            raise ValidationError(
                f"Data length ({n_samples}) must be >= "
                f"subsequence_length * {MIN_LENGTH_MULTIPLIER} "
                f"({min_required})"
            )

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute padded matrix profile scores for the input data.

        Handles univariate vs multivariate routing and chunking for
        large datasets.

        Args:
            X: Data of shape ``(n_samples,)`` or ``(n_samples, n_features)``.

        Returns:
            1D scores array of length ``n_samples``.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if n_features == 1:
            raw_profile = self._compute_univariate(X[:, 0])
        else:
            raw_profile = self._compute_multivariate(X)

        return self._pad_profile(raw_profile, n_samples)

    def _compute_univariate(self, series: np.ndarray) -> np.ndarray:
        """Compute matrix profile for a single 1D series.

        Uses chunking if the series exceeds ``max_rows``.

        Args:
            series: 1D time series array.

        Returns:
            Matrix profile distances with length
            ``len(series) - subsequence_length + 1``.
        """
        if len(series) <= self.max_rows:
            result = stumpy.stump(series, m=self.subsequence_length)
            return result[:, 0].astype(np.float64)

        return self._chunked_univariate(series)

    def _chunked_univariate(self, series: np.ndarray) -> np.ndarray:
        """Compute matrix profile in chunks for a long univariate series.

        Each chunk has size ``max_rows`` with overlap of
        ``subsequence_length`` between consecutive chunks. Overlapping
        regions use the minimum distance from both chunks.

        Args:
            series: 1D time series longer than ``max_rows``.

        Returns:
            Stitched matrix profile distances.
        """
        n = len(series)
        overlap = self.subsequence_length
        chunk_size = self.max_rows
        profile_len = n - self.subsequence_length + 1
        combined = np.full(profile_len, np.inf, dtype=np.float64)

        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunk = series[start:end]

            if len(chunk) < self.subsequence_length * MIN_LENGTH_MULTIPLIER:
                break

            result = stumpy.stump(chunk, m=self.subsequence_length)
            chunk_profile = result[:, 0].astype(np.float64)

            profile_start = start
            profile_end = profile_start + len(chunk_profile)
            profile_end = min(profile_end, profile_len)
            actual_len = profile_end - profile_start

            combined[profile_start:profile_end] = np.minimum(
                combined[profile_start:profile_end],
                chunk_profile[:actual_len],
            )

            if end >= n:
                break
            start = end - overlap

        # Replace any remaining inf values with the maximum finite value
        finite_mask = np.isfinite(combined)
        if finite_mask.any():
            max_finite = combined[finite_mask].max()
            combined[~finite_mask] = max_finite

        return combined

    def _compute_multivariate(self, X: np.ndarray) -> np.ndarray:
        """Compute per-feature matrix profiles and average them.

        Args:
            X: 2D array of shape ``(n_samples, n_features)``.

        Returns:
            Averaged profile distances.
        """
        n_features = X.shape[1]
        profiles: list[np.ndarray] = []

        for i in range(n_features):
            profile = self._compute_univariate(X[:, i])
            profiles.append(profile)

        return np.mean(profiles, axis=0)

    def _pad_profile(self, profile: np.ndarray, target_length: int) -> np.ndarray:
        """Pad the matrix profile to match the original data length.

        The first ``subsequence_length - 1`` entries are assigned the
        mean score since the profile is shorter than the input.

        Args:
            profile: Raw matrix profile distances.
            target_length: Desired output length (equals input row count).

        Returns:
            Padded 1D scores array of length ``target_length``.
        """
        pad_length = target_length - len(profile)
        if pad_length <= 0:
            return profile[:target_length]

        mean_val = float(np.nanmean(profile)) if len(profile) > 0 else self.mean_score_
        pad = np.full(pad_length, mean_val, dtype=np.float64)
        return np.concatenate([pad, profile])
