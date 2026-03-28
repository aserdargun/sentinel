"""Unit tests for sentinel.training.thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.training.thresholds import best_f1_threshold, percentile_threshold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scores_and_labels(
    n_normal: int = 45,
    n_anomaly: int = 10,
    normal_score: float = 0.1,
    anomaly_score: float = 0.9,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return perfectly-separated scores and binary labels."""
    rng = np.random.default_rng(seed)
    scores = np.concatenate(
        [
            rng.normal(normal_score, 0.01, n_normal),
            rng.normal(anomaly_score, 0.01, n_anomaly),
        ]
    )
    labels = np.concatenate(
        [np.zeros(n_normal, dtype=np.int32), np.ones(n_anomaly, dtype=np.int32)]
    )
    return scores, labels


# ---------------------------------------------------------------------------
# TestPercentileThreshold
# ---------------------------------------------------------------------------


class TestPercentileThreshold:
    """Tests for percentile_threshold().

    Verifies correct percentile computation across typical and edge cases.
    """

    def test_50th_percentile_is_median(self) -> None:
        """50th percentile equals the median of the sorted array."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = percentile_threshold(scores, 50.0)
        assert result == pytest.approx(3.0)

    def test_100th_percentile_is_max(self) -> None:
        """100th percentile equals the maximum score."""
        scores = np.array([0.1, 0.5, 1.0, 3.0])
        assert percentile_threshold(scores, 100.0) == pytest.approx(3.0)

    def test_0th_percentile_is_min(self) -> None:
        """0th percentile equals the minimum score."""
        scores = np.array([0.1, 0.5, 1.0, 3.0])
        assert percentile_threshold(scores, 0.0) == pytest.approx(0.1)

    def test_default_percentile_is_95(self) -> None:
        """Default percentile argument is 95.0."""
        scores = np.arange(1.0, 101.0)
        result = percentile_threshold(scores)
        expected = float(np.percentile(scores, 95.0))
        assert result == pytest.approx(expected)

    def test_matches_numpy_percentile(self) -> None:
        """Result matches np.percentile for arbitrary percentile values."""
        rng = np.random.default_rng(42)
        scores = rng.random(500)
        for p in (10.0, 25.0, 75.0, 90.0, 99.0):
            assert percentile_threshold(scores, p) == pytest.approx(
                float(np.percentile(scores, p))
            )

    def test_returns_python_float(self) -> None:
        """Return type is Python float, not numpy scalar."""
        scores = np.array([1.0, 2.0, 3.0])
        result = percentile_threshold(scores, 50.0)
        assert isinstance(result, float)

    def test_all_same_values(self) -> None:
        """When all scores are equal, every percentile returns that value."""
        scores = np.full(20, 0.5)
        for p in (0.0, 50.0, 95.0, 100.0):
            assert percentile_threshold(scores, p) == pytest.approx(0.5)

    def test_single_value_array(self) -> None:
        """A single-element array returns that element at any percentile."""
        scores = np.array([7.0])
        assert percentile_threshold(scores, 95.0) == pytest.approx(7.0)

    def test_large_array_95th_near_max(self) -> None:
        """95th percentile of a uniform [0, 100] range is near 95."""
        scores = np.linspace(0.0, 100.0, 1000)
        result = percentile_threshold(scores, 95.0)
        assert result > 94.0

    def test_two_element_array_median(self) -> None:
        """Minimal two-element array: 50th percentile is between the two values."""
        scores = np.array([2.0, 8.0])
        result = percentile_threshold(scores, 50.0)
        assert 2.0 <= result <= 8.0

    def test_result_within_score_range(self) -> None:
        """Threshold is always within [min, max] of the score array."""
        rng = np.random.default_rng(7)
        scores = rng.random(200)
        for p in (5.0, 50.0, 95.0):
            t = percentile_threshold(scores, p)
            assert scores.min() <= t <= scores.max()


# ---------------------------------------------------------------------------
# TestBestF1Threshold
# ---------------------------------------------------------------------------


class TestBestF1Threshold:
    """Tests for best_f1_threshold().

    Verifies optimal threshold search with known, manually computable cases.
    """

    def test_returns_tuple_of_two_floats(self) -> None:
        """Return type is a two-element tuple of Python floats."""
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        thresh, f1 = best_f1_threshold(scores, labels)
        assert isinstance(thresh, float)
        assert isinstance(f1, float)

    def test_perfect_separation_f1_equals_one(self) -> None:
        """Perfectly separated classes produce F1 = 1.0."""
        scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        _, f1 = best_f1_threshold(scores, labels)
        assert f1 == pytest.approx(1.0)

    def test_perfect_separation_threshold_between_classes(self) -> None:
        """Optimal threshold lies between normal and anomaly score clusters."""
        scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        thresh, _ = best_f1_threshold(scores, labels)
        assert 0.1 <= thresh <= 0.9

    def test_all_same_scores_returns_zero_f1(self) -> None:
        """Constant scores yield F1 = 0.0 (no separation possible)."""
        scores = np.full(10, 0.5)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        _, f1 = best_f1_threshold(scores, labels)
        assert f1 == pytest.approx(0.0)

    def test_all_same_scores_threshold_equals_that_value(self) -> None:
        """When all scores are identical, threshold equals that constant value."""
        scores = np.full(10, 0.42)
        labels = np.zeros(10, dtype=np.int32)
        thresh, _ = best_f1_threshold(scores, labels)
        assert thresh == pytest.approx(0.42)

    def test_f1_in_unit_range(self) -> None:
        """Best F1 is always in [0.0, 1.0]."""
        rng = np.random.default_rng(42)
        scores = rng.random(50)
        labels = (scores > 0.5).astype(np.int32)
        _, f1 = best_f1_threshold(scores, labels)
        assert 0.0 <= f1 <= 1.0

    def test_threshold_within_score_range(self) -> None:
        """Returned threshold is always within [min(scores), max(scores)]."""
        rng = np.random.default_rng(0)
        scores = rng.random(100)
        labels = (scores > 0.7).astype(np.int32)
        thresh, _ = best_f1_threshold(scores, labels)
        assert scores.min() <= thresh <= scores.max()

    def test_all_positive_labels(self) -> None:
        """All samples positive: every threshold predicts all correctly or not."""
        scores = np.array([0.1, 0.5, 0.9, 0.95])
        labels = np.ones(4, dtype=np.int32)
        thresh, f1 = best_f1_threshold(scores, labels)
        # Threshold at min → all predicted positive → recall=1.
        # Result must still be a valid float pair.
        assert isinstance(thresh, float)
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0

    def test_all_negative_labels(self) -> None:
        """All samples negative: no true positives possible, F1 = 0.0."""
        scores = np.array([0.1, 0.5, 0.9, 0.95])
        labels = np.zeros(4, dtype=np.int32)
        _, f1 = best_f1_threshold(scores, labels)
        assert f1 == pytest.approx(0.0)

    def test_two_candidates_smoke(self) -> None:
        """n_candidates=2 still produces a valid result."""
        scores = np.array([0.0, 0.5, 1.0])
        labels = np.array([0, 0, 1])
        thresh, f1 = best_f1_threshold(scores, labels, n_candidates=2)
        assert isinstance(thresh, float)
        assert 0.0 <= f1 <= 1.0

    def test_higher_n_candidates_does_not_degrade_f1(self) -> None:
        """More candidates should not produce a lower F1 than fewer candidates."""
        scores, labels = _scores_and_labels(seed=5)
        _, f1_coarse = best_f1_threshold(scores, labels, n_candidates=10)
        _, f1_fine = best_f1_threshold(scores, labels, n_candidates=1000)
        # Fine search should find at least as good a threshold
        assert f1_fine >= f1_coarse - 1e-6

    def test_known_input_manually_computable(self) -> None:
        """Manual verification: 3 normals at 0.0, 2 anomalies at 1.0.

        Any threshold in (0.0, 1.0) gives TP=2, FP=0, FN=0 → F1=1.0.
        """
        scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=float)
        labels = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        thresh, f1 = best_f1_threshold(scores, labels, n_candidates=100)
        assert f1 == pytest.approx(1.0)
        # Threshold must sit between 0.0 and 1.0 so that scores > thresh gives TP=2
        assert 0.0 <= thresh < 1.0

    def test_symmetry_does_not_affect_return_types(self) -> None:
        """Swapped class proportions still return correct types."""
        scores = np.array([0.9, 0.9, 0.9, 0.1, 0.1])
        labels = np.array([0, 0, 0, 1, 1])
        thresh, f1 = best_f1_threshold(scores, labels)
        assert isinstance(thresh, float)
        assert isinstance(f1, float)
