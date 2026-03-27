"""Online anomaly detector with a sliding window buffer.

Maintains a fixed-size window of recent observations.  Each time a new
data point arrives the window shifts forward and the full buffer is
scored using the model's standard batch ``score()`` API.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import structlog

from sentinel.core.base_model import BaseAnomalyDetector

logger = structlog.get_logger(__name__)


class OnlineDetector:
    """Sliding-window online anomaly detector.

    Wraps any :class:`BaseAnomalyDetector` for streaming use.  Internally
    a fixed-length buffer of ``seq_len`` rows is maintained.  When a new
    row arrives via :meth:`update`, the oldest row is evicted (FIFO),
    the buffer is reshaped into a 2-D array ``(seq_len, n_features)``,
    and ``model.score()`` is called on the full window.

    The detector returns ``None`` for :meth:`update` calls until the
    buffer has accumulated at least ``seq_len`` rows.

    Args:
        model: A fitted anomaly detection model.
        seq_len: Number of rows in the sliding window.
        threshold: Anomaly threshold -- scores above this value are
            labelled as anomalies (label ``1``).

    Example::

        detector = OnlineDetector(model, seq_len=50, threshold=0.75)
        for row in incoming_stream:
            result = detector.update(row)
            if result and result["label"] == 1:
                print(f"Anomaly at {result['timestamp']}")
    """

    def __init__(
        self,
        model: BaseAnomalyDetector,
        seq_len: int,
        threshold: float,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")

        self._model = model
        self._seq_len = seq_len
        self._threshold = threshold

        # Deque with a max length acts as a FIFO sliding window.
        self._buffer: deque[list[float]] = deque(maxlen=seq_len)
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Ingest one data point and optionally return a detection result.

        The row dict is expected to contain numeric feature values.  The
        ``timestamp`` key (if present) is preserved in the result but
        excluded from the feature vector.  The ``is_anomaly`` key is
        likewise excluded.

        Args:
            row: Single observation as ``{column_name: value}``.

        Returns:
            A dict with ``score``, ``label``, ``threshold``, and
            ``timestamp`` (if available) when the buffer is full.
            ``None`` while the buffer is still filling.
        """
        timestamp = row.get("timestamp")

        # Extract numeric features, excluding reserved columns.
        feature_names, feature_values = _extract_features(row)

        # Lazily capture the feature order from the first row.
        if self._feature_names is None:
            self._feature_names = feature_names

        self._buffer.append(feature_values)

        if not self.is_ready:
            return None

        # Build the 2-D window array and score.
        window = np.array(list(self._buffer), dtype=np.float64)
        scores = self._model.score(window)

        # Use the last score (corresponding to the newest point).
        score = float(scores[-1])
        label = int(score > self._threshold)

        result: dict[str, Any] = {
            "score": score,
            "label": label,
            "threshold": self._threshold,
        }
        if timestamp is not None:
            result["timestamp"] = timestamp

        return result

    def reset(self) -> None:
        """Clear the sliding window buffer."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Number of rows currently in the buffer."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has accumulated enough rows to score."""
        return len(self._buffer) >= self._seq_len

    @property
    def seq_len(self) -> int:
        """Required window length."""
        return self._seq_len

    @property
    def threshold(self) -> float:
        """Current anomaly threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Update the anomaly threshold.

        Args:
            value: New threshold value.
        """
        self._threshold = value


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_EXCLUDED_KEYS = {"timestamp", "is_anomaly"}


def _extract_features(row: dict[str, Any]) -> tuple[list[str], list[float]]:
    """Extract ordered feature names and numeric values from a row dict.

    Keys in :data:`_EXCLUDED_KEYS` are skipped.  Non-numeric values are
    also skipped so that stray metadata does not corrupt the feature
    vector.

    Args:
        row: Single observation dict.

    Returns:
        Tuple of ``(feature_names, feature_values)`` in insertion order.
    """
    names: list[str] = []
    values: list[float] = []

    for key, val in row.items():
        if key in _EXCLUDED_KEYS:
            continue
        try:
            values.append(float(val))
            names.append(key)
        except (TypeError, ValueError):
            continue

    return names, values
