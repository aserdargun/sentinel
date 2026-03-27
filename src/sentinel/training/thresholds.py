"""Threshold selection strategies for anomaly detection."""

from __future__ import annotations

import numpy as np


def percentile_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    """Select threshold at a given percentile of score distribution.

    Args:
        scores: 1D array of anomaly scores.
        percentile: Percentile value (0-100).

    Returns:
        Threshold value.
    """
    return float(np.percentile(scores, percentile))


def best_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_candidates: int = 100,
) -> tuple[float, float]:
    """Find threshold that maximizes F1 score via grid search.

    Args:
        scores: 1D array of anomaly scores.
        labels: 1D array of binary ground truth (0/1).
        n_candidates: Number of threshold candidates to evaluate.

    Returns:
        Tuple of (best_threshold, best_f1_score).
    """
    min_score = float(scores.min())
    max_score = float(scores.max())

    if min_score == max_score:
        return min_score, 0.0

    candidates = np.linspace(min_score, max_score, n_candidates)
    best_f1 = 0.0
    best_thresh = candidates[0]

    for thresh in candidates:
        preds = (scores > thresh).astype(np.int32)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh), float(best_f1)
