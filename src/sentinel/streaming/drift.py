"""Concept drift detection and simulation.

Provides the ADWIN (ADaptive WINdowing) algorithm for detecting concept
drift in streaming data, plus a simulator for generating synthetic drift
scenarios (gradual, abrupt, recurring) for testing.
"""

from __future__ import annotations

import math

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ADWINDetector:
    """ADWIN (ADaptive WINdowing) concept drift detector.

    Maintains a variable-length window of recent values.  When a new
    value arrives, the algorithm tests whether any split of the window
    into two sub-windows shows a statistically significant difference
    in means (using the Hoeffding bound).  If drift is detected the
    older sub-window is dropped.

    Args:
        delta: Confidence parameter for the Hoeffding bound.  Smaller
            values require stronger evidence before signaling drift.
            Default ``0.002``.

    Example::

        detector = ADWINDetector(delta=0.002)
        for value in stream:
            if detector.update(value):
                print(f"Drift detected at window size {detector.width}")
    """

    def __init__(self, delta: float = 0.002) -> None:
        if delta <= 0.0 or delta >= 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self._delta = delta
        self._window: list[float] = []
        self._total: float = 0.0
        self._variance_sum: float = 0.0
        self._drift_count: int = 0

    def update(self, value: float) -> bool:
        """Add a new value and check for concept drift.

        Appends ``value`` to the window, then scans all possible
        splits to test whether the mean of the older sub-window
        differs significantly from the newer sub-window.  If drift
        is detected, the older portion is discarded.

        Args:
            value: New observation from the stream.

        Returns:
            ``True`` if concept drift was detected, ``False`` otherwise.
        """
        self._window.append(value)
        self._total += value
        n = len(self._window)

        if n >= 2:
            mean = self._total / n
            self._variance_sum += (value - mean) * (value - mean)

        drift_found = False

        if n < 4:
            return False

        # Scan possible split points from oldest to newest.
        # For each split we compare the sub-window means using the
        # Hoeffding bound.  We iterate from the beginning so that
        # when drift is found we can drop the minimal older portion.
        left_sum = 0.0
        for i in range(1, n):
            left_sum += self._window[i - 1]
            left_n = i
            right_n = n - i

            if left_n < 2 or right_n < 2:
                continue

            left_mean = left_sum / left_n
            right_mean = (self._total - left_sum) / right_n

            # Hoeffding bound for the difference of two sub-window means.
            harmonic = (1.0 / left_n) + (1.0 / right_n)
            epsilon = math.sqrt(0.5 * harmonic * math.log(4.0 / self._delta))

            if abs(left_mean - right_mean) >= epsilon:
                # Drift detected: drop the older sub-window.
                self._window = self._window[i:]
                self._total = sum(self._window)
                self._variance_sum = _compute_variance_sum(self._window)
                self._drift_count += 1
                drift_found = True

                logger.info(
                    "adwin_drift_detected",
                    split_point=i,
                    dropped=left_n,
                    remaining=len(self._window),
                    left_mean=round(left_mean, 4),
                    right_mean=round(right_mean, 4),
                    epsilon=round(epsilon, 4),
                )
                break

        return drift_found

    def reset(self) -> None:
        """Clear the window and reset all internal state."""
        self._window.clear()
        self._total = 0.0
        self._variance_sum = 0.0

    @property
    def delta(self) -> float:
        """Confidence parameter for the Hoeffding bound."""
        return self._delta

    @property
    def mean(self) -> float:
        """Mean of the current window.

        Returns:
            Mean value, or ``0.0`` if the window is empty.
        """
        if not self._window:
            return 0.0
        return self._total / len(self._window)

    @property
    def variance(self) -> float:
        """Variance of the current window (population variance).

        Returns:
            Variance, or ``0.0`` if fewer than 2 values in window.
        """
        n = len(self._window)
        if n < 2:
            return 0.0
        return self._variance_sum / n

    @property
    def width(self) -> int:
        """Number of values currently in the window."""
        return len(self._window)

    @property
    def drift_count(self) -> int:
        """Total number of drift events detected since creation or reset."""
        return self._drift_count


def _compute_variance_sum(values: list[float]) -> float:
    """Compute the running variance numerator for a list of values.

    This is ``sum((x - mean)^2)`` which is used to maintain the
    variance incrementally.

    Args:
        values: List of numeric values.

    Returns:
        Sum of squared deviations from the mean.
    """
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values)


class DriftSimulator:
    """Generates synthetic drift scenarios for testing drift detectors.

    Provides static methods that inject different types of concept
    drift into a 1D numpy array: abrupt, gradual, and recurring.
    The input data is not modified in place; a new array is returned.

    Example::

        data = np.random.randn(1000)
        shifted = DriftSimulator.abrupt_drift(data, change_point=500, magnitude=3.0)
    """

    @staticmethod
    def abrupt_drift(
        data: np.ndarray,
        change_point: int,
        magnitude: float,
    ) -> np.ndarray:
        """Apply an abrupt (sudden) mean shift at a change point.

        All values from ``change_point`` onward are shifted by
        ``magnitude``.

        Args:
            data: 1D array of values.
            change_point: Index at which the shift occurs.
            magnitude: Amount to shift the mean by.

        Returns:
            New array with the abrupt drift applied.

        Raises:
            ValueError: If ``change_point`` is out of bounds.
        """
        _validate_drift_args(data, change_point)
        result = data.copy()
        result[change_point:] += magnitude
        return result

    @staticmethod
    def gradual_drift(
        data: np.ndarray,
        change_point: int,
        magnitude: float,
    ) -> np.ndarray:
        """Apply a gradual mean shift starting at a change point.

        The shift increases linearly from ``0`` at ``change_point`` to
        ``magnitude`` at the end of the array.

        Args:
            data: 1D array of values.
            change_point: Index at which the gradual shift begins.
            magnitude: Final shift amount at the end of the array.

        Returns:
            New array with the gradual drift applied.

        Raises:
            ValueError: If ``change_point`` is out of bounds.
        """
        _validate_drift_args(data, change_point)
        result = data.copy()
        n_remaining = len(data) - change_point
        if n_remaining <= 1:
            result[change_point:] += magnitude
            return result
        ramp = np.linspace(0.0, magnitude, n_remaining)
        result[change_point:] += ramp
        return result

    @staticmethod
    def recurring_drift(
        data: np.ndarray,
        period: int,
        magnitude: float,
    ) -> np.ndarray:
        """Apply periodic mean shifts that alternate on and off.

        The data alternates between the original distribution and a
        shifted distribution every ``period`` samples.  Odd-numbered
        segments are shifted by ``magnitude``; even-numbered segments
        are left unchanged.

        Args:
            data: 1D array of values.
            period: Number of samples in each segment before toggling.
            magnitude: Amount to shift during active segments.

        Returns:
            New array with recurring drift applied.

        Raises:
            ValueError: If ``period`` is not positive.
        """
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        result = data.copy()
        n = len(data)
        for i in range(n):
            segment_index = i // period
            if segment_index % 2 == 1:
                result[i] += magnitude
        return result


def _validate_drift_args(data: np.ndarray, change_point: int) -> None:
    """Validate common drift simulation arguments.

    Args:
        data: Input data array.
        change_point: Index of the change point.

    Raises:
        ValueError: If data is not 1D or change_point is out of bounds.
    """
    if data.ndim != 1:
        raise ValueError(f"data must be 1D, got {data.ndim}D")
    if change_point < 0 or change_point >= len(data):
        raise ValueError(
            f"change_point must be in [0, {len(data) - 1}], got {change_point}"
        )
