"""Base class for all anomaly detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from sentinel.core.types import DetectionResult


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors.

    All models must implement fit(), score(), save(), load(), and get_params().
    The detect() method is concrete — it combines score() with a threshold.
    """

    model_name: str = ""

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs: Any) -> None:
        """Train the model on data.

        Args:
            X: Training data array of shape (n_samples, n_features).
            **kwargs: Model-specific training parameters.
        """

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for data.

        Args:
            X: Data array of shape (n_samples, n_features).

        Returns:
            1D array of anomaly scores, higher = more anomalous.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Directory path to save model artifacts.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Directory path containing model artifacts.
        """

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model parameters as a dictionary.

        Returns:
            Dict of parameter names to values.
        """

    def detect(self, X: np.ndarray, threshold: float) -> DetectionResult:
        """Score data and apply threshold to produce labels.

        Args:
            X: Data array of shape (n_samples, n_features).
            threshold: Anomaly threshold — scores above this are anomalous.

        Returns:
            DetectionResult with scores, labels, and threshold.
        """
        scores = self.score(X)
        labels = (scores > threshold).astype(np.int32)
        return DetectionResult(scores=scores, labels=labels, threshold=threshold)
