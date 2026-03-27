"""Model evaluation: supervised and unsupervised modes."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sentinel.training.thresholds import best_f1_threshold, percentile_threshold

MIN_POSITIVE_SAMPLES = 5


class Evaluator:
    """Evaluates anomaly detection results.

    Two modes:
        - Supervised: when ground-truth labels are available, computes
          precision, recall, F1, AUC-ROC, AUC-PR.
        - Unsupervised: when no labels, reports score distribution stats
          and a percentile-based threshold.
    """

    def evaluate(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
        val_scores: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Evaluate anomaly scores.

        Threshold is selected on validation data (val_scores/val_labels).
        Metrics are reported on test data (scores/labels).

        Args:
            scores: Test set anomaly scores.
            labels: Test set ground truth labels (optional).
            val_scores: Validation set anomaly scores for threshold selection.
            val_labels: Validation set labels for threshold selection.

        Returns:
            Dict of metric names to values. Classification metrics are None
            when labels are absent.
        """
        metrics: dict[str, Any] = {}

        # Score distribution stats (always computed)
        metrics["score_mean"] = float(np.mean(scores))
        metrics["score_std"] = float(np.std(scores))
        metrics["score_p50"] = float(np.percentile(scores, 50))
        metrics["score_p95"] = float(np.percentile(scores, 95))
        metrics["score_p99"] = float(np.percentile(scores, 99))

        # Threshold selection on validation set
        threshold = self._select_threshold(val_scores, val_labels, scores)
        metrics["threshold"] = threshold

        if labels is not None:
            metrics.update(self._supervised_metrics(scores, labels, threshold))
        else:
            metrics["precision"] = None
            metrics["recall"] = None
            metrics["f1"] = None
            metrics["auc_roc"] = None
            metrics["auc_pr"] = None

        return metrics

    def _select_threshold(
        self,
        val_scores: np.ndarray | None,
        val_labels: np.ndarray | None,
        fallback_scores: np.ndarray,
    ) -> float:
        """Select threshold from validation data.

        Uses best-F1 if validation labels available, else percentile.
        """
        if val_scores is not None and val_labels is not None:
            n_positive = int(np.sum(val_labels))
            if n_positive >= MIN_POSITIVE_SAMPLES:
                thresh, _ = best_f1_threshold(val_scores, val_labels)
                return thresh

        target = val_scores if val_scores is not None else fallback_scores
        return percentile_threshold(target, 95.0)

    def _supervised_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> dict[str, float | None]:
        """Compute classification metrics on test data."""
        preds = (scores > threshold).astype(np.int32)
        labels_int = labels.astype(np.int32)

        n_positive = int(np.sum(labels_int))
        metrics: dict[str, float | None] = {}

        metrics["precision"] = float(
            precision_score(labels_int, preds, zero_division=0)
        )
        metrics["recall"] = float(recall_score(labels_int, preds, zero_division=0))
        metrics["f1"] = float(f1_score(labels_int, preds, zero_division=0))

        if n_positive < MIN_POSITIVE_SAMPLES:
            metrics["auc_roc"] = None
            metrics["auc_pr"] = None
        else:
            try:
                metrics["auc_roc"] = float(roc_auc_score(labels_int, scores))
            except ValueError:
                metrics["auc_roc"] = None

            try:
                prec_curve, rec_curve, _ = precision_recall_curve(labels_int, scores)
                metrics["auc_pr"] = float(auc(rec_curve, prec_curve))
            except ValueError:
                metrics["auc_pr"] = None

        return metrics
