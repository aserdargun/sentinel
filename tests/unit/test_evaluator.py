"""Tests for sentinel.training.evaluator and sentinel.training.thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.training.evaluator import Evaluator
from sentinel.training.thresholds import best_f1_threshold, percentile_threshold

# ---------------------------------------------------------------------------
# percentile_threshold
# ---------------------------------------------------------------------------


class TestPercentileThreshold:
    """percentile_threshold() returns the correct score at a given percentile."""

    def test_median_at_50th_percentile(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = percentile_threshold(scores, 50.0)
        assert result == pytest.approx(3.0)

    def test_95th_percentile(self) -> None:
        scores = np.arange(1.0, 101.0)  # 1..100
        result = percentile_threshold(scores, 95.0)
        assert result == pytest.approx(np.percentile(scores, 95.0))

    def test_100th_percentile_is_max(self) -> None:
        scores = np.array([0.1, 0.5, 1.0, 3.0])
        result = percentile_threshold(scores, 100.0)
        assert result == pytest.approx(3.0)

    def test_0th_percentile_is_min(self) -> None:
        scores = np.array([0.1, 0.5, 1.0, 3.0])
        result = percentile_threshold(scores, 0.0)
        assert result == pytest.approx(0.1)

    def test_returns_float(self) -> None:
        scores = np.array([1.0, 2.0, 3.0])
        result = percentile_threshold(scores, 50.0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# best_f1_threshold
# ---------------------------------------------------------------------------


class TestBestF1Threshold:
    """best_f1_threshold() finds the optimal classification threshold."""

    def test_returns_tuple_of_two_floats(self) -> None:
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        thresh, f1 = best_f1_threshold(scores, labels)
        assert isinstance(thresh, float)
        assert isinstance(f1, float)

    def test_perfect_separation_f1_is_one(self) -> None:
        """When anomalies have strictly higher scores, F1 should be 1.0."""
        scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        thresh, f1 = best_f1_threshold(scores, labels)
        assert f1 == pytest.approx(1.0)

    def test_perfect_separation_threshold_between_classes(self) -> None:
        scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        thresh, _ = best_f1_threshold(scores, labels)
        # Threshold must be between 0.1 and 0.9
        assert 0.1 <= thresh <= 0.9

    def test_all_same_scores_returns_zero_f1(self) -> None:
        """When all scores are identical no separation is possible."""
        scores = np.full(10, 0.5)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        _, f1 = best_f1_threshold(scores, labels)
        assert f1 == pytest.approx(0.0)

    def test_threshold_within_score_range(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.random(100)
        labels = (scores > 0.7).astype(np.int32)
        thresh, _ = best_f1_threshold(scores, labels)
        assert scores.min() <= thresh <= scores.max()

    def test_f1_in_zero_one_range(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.random(50)
        labels = (scores > 0.5).astype(np.int32)
        _, f1 = best_f1_threshold(scores, labels)
        assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# Evaluator — score distribution stats (always computed)
# ---------------------------------------------------------------------------


class TestEvaluatorScoreDistribution:
    """Score distribution statistics are always present in the output."""

    DIST_KEYS = {"score_mean", "score_std", "score_p50", "score_p95", "score_p99"}

    def test_all_distribution_keys_present(self) -> None:
        ev = Evaluator()
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = ev.evaluate(scores=scores)
        assert self.DIST_KEYS.issubset(metrics.keys())

    def test_score_mean_correct(self) -> None:
        ev = Evaluator()
        scores = np.array([2.0, 4.0])
        metrics = ev.evaluate(scores=scores)
        assert metrics["score_mean"] == pytest.approx(3.0)

    def test_score_p50_is_median(self) -> None:
        ev = Evaluator()
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = ev.evaluate(scores=scores)
        assert metrics["score_p50"] == pytest.approx(3.0)

    def test_score_p99_close_to_max(self) -> None:
        ev = Evaluator()
        scores = np.linspace(0, 1, 1000)
        metrics = ev.evaluate(scores=scores)
        assert metrics["score_p99"] > 0.98

    def test_threshold_key_always_present(self) -> None:
        ev = Evaluator()
        metrics = ev.evaluate(scores=np.array([1.0, 2.0, 3.0]))
        assert "threshold" in metrics


# ---------------------------------------------------------------------------
# Evaluator — unsupervised mode (no labels)
# ---------------------------------------------------------------------------


class TestEvaluatorUnsupervisedMode:
    """When labels are absent, classification metrics are None."""

    CLASSIFICATION_KEYS = {"precision", "recall", "f1", "auc_roc", "auc_pr"}

    def test_classification_keys_present_when_no_labels(self) -> None:
        ev = Evaluator()
        scores = np.random.default_rng(0).random(50)
        metrics = ev.evaluate(scores=scores)
        for key in self.CLASSIFICATION_KEYS:
            assert key in metrics, f"Key '{key}' missing in unsupervised output"

    def test_classification_metrics_are_none_when_no_labels(self) -> None:
        ev = Evaluator()
        scores = np.random.default_rng(0).random(50)
        metrics = ev.evaluate(scores=scores)
        for key in self.CLASSIFICATION_KEYS:
            assert metrics[key] is None, f"{key} should be None without labels"

    def test_threshold_uses_percentile_when_no_labels(self) -> None:
        ev = Evaluator()
        scores = np.linspace(0, 100, 100)
        metrics = ev.evaluate(scores=scores)
        expected = float(np.percentile(scores, 95.0))
        assert metrics["threshold"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Evaluator — supervised mode with labels
# ---------------------------------------------------------------------------


class TestEvaluatorSupervisedMode:
    """Supervised evaluation with ground-truth labels."""

    def test_precision_recall_f1_present(self) -> None:
        ev = Evaluator()
        # Perfect separation: normals score 0.1, anomalies score 0.9
        scores = np.array([0.1] * 45 + [0.9] * 10, dtype=float)
        labels = np.array([0] * 45 + [1] * 10, dtype=np.int32)
        val_scores = scores.copy()
        val_labels = labels.copy()
        metrics = ev.evaluate(scores, labels, val_scores, val_labels)
        for key in ("precision", "recall", "f1"):
            assert key in metrics
            assert metrics[key] is not None

    def test_perfect_model_f1_is_one(self) -> None:
        ev = Evaluator()
        scores = np.array([0.1] * 45 + [0.9] * 10, dtype=float)
        labels = np.array([0] * 45 + [1] * 10, dtype=np.int32)
        metrics = ev.evaluate(scores, labels, scores, labels)
        assert metrics["f1"] == pytest.approx(1.0, abs=0.05)

    def test_precision_recall_in_unit_range(self) -> None:
        rng = np.random.default_rng(7)
        scores = rng.random(100)
        labels = (scores > 0.6).astype(np.int32)
        # Ensure at least 5 positives for AUC
        metrics = Evaluator().evaluate(scores, labels, scores, labels)
        for key in ("precision", "recall", "f1"):
            v = metrics[key]
            if v is not None:
                assert 0.0 <= v <= 1.0, f"{key}={v}"

    def test_auc_roc_none_with_too_few_positives(self) -> None:
        """AUC-ROC is None when fewer than 5 positives in val set."""
        ev = Evaluator()
        scores = np.array([0.1] * 96 + [0.9, 0.8, 0.7, 0.6], dtype=float)
        labels = np.array([0] * 96 + [1, 1, 1, 1], dtype=np.int32)
        # val set only has 4 positives → AUC should be None
        metrics = ev.evaluate(scores, labels, scores, labels)
        assert metrics["auc_roc"] is None

    def test_auc_pr_none_with_too_few_positives(self) -> None:
        ev = Evaluator()
        scores = np.array([0.1] * 96 + [0.9, 0.8, 0.7, 0.6], dtype=float)
        labels = np.array([0] * 96 + [1, 1, 1, 1], dtype=np.int32)
        metrics = ev.evaluate(scores, labels, scores, labels)
        assert metrics["auc_pr"] is None

    def test_auc_roc_not_none_with_enough_positives(self) -> None:
        ev = Evaluator()
        scores = np.array([0.1] * 45 + [0.9] * 10, dtype=float)
        labels = np.array([0] * 45 + [1] * 10, dtype=np.int32)
        metrics = ev.evaluate(scores, labels, scores, labels)
        assert metrics["auc_roc"] is not None

    def test_auc_roc_in_unit_range(self) -> None:
        ev = Evaluator()
        scores = np.array([0.1] * 45 + [0.9] * 10, dtype=float)
        labels = np.array([0] * 45 + [1] * 10, dtype=np.int32)
        metrics = ev.evaluate(scores, labels, scores, labels)
        if metrics["auc_roc"] is not None:
            assert 0.0 <= metrics["auc_roc"] <= 1.0


# ---------------------------------------------------------------------------
# Evaluator — threshold selection on validation set
# ---------------------------------------------------------------------------


class TestEvaluatorThresholdOnValidation:
    """Threshold is selected from val_scores, not test scores."""

    def test_threshold_uses_val_scores_percentile_when_no_labels(self) -> None:
        ev = Evaluator()
        val_scores = np.linspace(0, 10, 100)
        test_scores = np.linspace(0, 1, 50)
        metrics = ev.evaluate(test_scores, val_scores=val_scores)
        expected = float(np.percentile(val_scores, 95.0))
        assert metrics["threshold"] == pytest.approx(expected)

    def test_threshold_from_val_labels_when_available(self) -> None:
        """With val labels and enough positives, best-F1 threshold is used."""
        ev = Evaluator()
        val_scores = np.array([0.1] * 45 + [0.9] * 10, dtype=float)
        val_labels = np.array([0] * 45 + [1] * 10, dtype=np.int32)
        test_scores = val_scores.copy()
        test_labels = val_labels.copy()
        metrics = ev.evaluate(test_scores, test_labels, val_scores, val_labels)
        # Threshold must separate 0.1 normals from 0.9 anomalies.
        assert 0.1 <= metrics["threshold"] <= 0.9

    def test_threshold_uses_fallback_when_val_labels_have_too_few_positives(
        self,
    ) -> None:
        """Fewer than 5 val positives → falls back to percentile on val_scores."""
        ev = Evaluator()
        val_scores = np.linspace(0, 1, 100)
        val_labels = np.zeros(100, dtype=np.int32)
        val_labels[-3:] = 1  # only 3 positives
        test_scores = val_scores.copy()
        metrics = ev.evaluate(test_scores, val_scores=val_scores, val_labels=val_labels)
        expected = float(np.percentile(val_scores, 95.0))
        assert metrics["threshold"] == pytest.approx(expected)
