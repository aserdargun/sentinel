"""Per-feature reconstruction error decomposition and ranking.

Computes element-wise reconstruction error between original and
reconstructed arrays, then ranks features by their contribution to
the total error.  Useful for understanding *which* features drive a
reconstruction-based anomaly score.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class ReconstructionExplainer:
    """Explain anomaly scores via per-feature reconstruction error.

    Unlike SHAP, this does not require model access -- it only needs the
    original and reconstructed arrays.  It decomposes the total
    reconstruction error into per-feature contributions.

    Example::

        explainer = ReconstructionExplainer()
        result = explainer.explain(original, reconstructed, feature_names)
        print(result["feature_ranking"])
    """

    def explain(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute per-feature reconstruction error decomposition.

        For each feature, computes the mean squared error (MSE) across
        all samples.  Features are ranked by descending MSE contribution.

        Args:
            original: Array of shape ``(n_samples, n_features)`` with
                original input values.
            reconstructed: Array of same shape with reconstructed values.
            feature_names: Optional names for each feature.  If ``None``
                features are numbered ``feature_0``, etc.

        Returns:
            Dictionary with:

            * ``"per_feature_mse"`` -- 1-D array of shape
              ``(n_features,)`` with mean squared error per feature.
            * ``"per_feature_mae"`` -- 1-D array of shape
              ``(n_features,)`` with mean absolute error per feature.
            * ``"per_sample_error"`` -- 2-D array of shape
              ``(n_samples, n_features)`` with absolute error per
              sample per feature.
            * ``"total_mse"`` -- scalar total MSE across all features.
            * ``"feature_ranking"`` -- list of dicts sorted by
              descending MSE, each with keys ``"name"``, ``"mse"``,
              ``"mae"``, and ``"contribution_pct"``.

        Raises:
            ValueError: If *original* and *reconstructed* shapes differ.
        """
        if original.shape != reconstructed.shape:
            raise ValueError(
                f"Shape mismatch: original {original.shape} vs "
                f"reconstructed {reconstructed.shape}"
            )

        if original.ndim == 1:
            original = original.reshape(-1, 1)
            reconstructed = reconstructed.reshape(-1, 1)

        n_samples, n_features = original.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Per-sample absolute error.
        abs_error = np.abs(original - reconstructed)

        # Per-feature aggregates.
        per_feature_mse = np.mean((original - reconstructed) ** 2, axis=0)
        per_feature_mae = np.mean(abs_error, axis=0)

        total_mse = float(np.mean(per_feature_mse))

        # Contribution percentages.
        mse_sum = float(np.sum(per_feature_mse))
        if mse_sum > 0:
            contributions = per_feature_mse / mse_sum * 100.0
        else:
            contributions = np.zeros(n_features)

        # Rank features by MSE (descending).
        ranked_indices = np.argsort(per_feature_mse)[::-1]

        feature_ranking = [
            {
                "name": feature_names[i],
                "mse": float(per_feature_mse[i]),
                "mae": float(per_feature_mae[i]),
                "contribution_pct": float(contributions[i]),
            }
            for i in ranked_indices
        ]

        logger.info(
            "reconstruction.explain_complete",
            n_samples=n_samples,
            n_features=n_features,
            total_mse=total_mse,
            top_feature=feature_ranking[0]["name"] if feature_ranking else None,
        )

        return {
            "per_feature_mse": per_feature_mse,
            "per_feature_mae": per_feature_mae,
            "per_sample_error": abs_error,
            "total_mse": total_mse,
            "feature_ranking": feature_ranking,
        }
