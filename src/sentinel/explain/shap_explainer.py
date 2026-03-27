"""SHAP-based model explainer.

Wraps any ``BaseAnomalyDetector.score()`` function with SHAP's
``KernelExplainer`` (or ``TreeExplainer`` for tree-based models like
Isolation Forest) to attribute anomaly scores to individual features.
SHAP is an optional dependency -- this module degrades gracefully if it
is not installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.exceptions import SentinelError

logger = structlog.get_logger(__name__)

_MAX_BACKGROUND_SAMPLES: int = 100


class SHAPExplainer:
    """Explain anomaly scores using SHAP values.

    For tree-based models (e.g. Isolation Forest) the faster
    ``TreeExplainer`` is used automatically.  For all other models a
    ``KernelExplainer`` wraps the model's ``score()`` method.

    Args:
        model: A fitted ``BaseAnomalyDetector`` instance.
        max_features: Maximum number of top features to include in the
            ranking.  Defaults to 10.

    Example::

        explainer = SHAPExplainer(model, max_features=5)
        result = explainer.explain(X_test, sample_indices=[0, 10, 20])
        print(result["feature_ranking"])
    """

    def __init__(
        self,
        model: BaseAnomalyDetector,
        max_features: int = 10,
    ) -> None:
        self._model = model
        self._max_features = max_features
        self._is_tree_model = _is_tree_model(model)

    def explain(
        self,
        X: np.ndarray,
        sample_indices: list[int],
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute SHAP values for selected samples.

        Args:
            X: Full dataset array of shape ``(n_samples, n_features)``.
                A random subset (up to 100 rows) is used as the
                background dataset for ``KernelExplainer``.
            sample_indices: Row indices to explain.  Maximum 50 indices.
            feature_names: Optional names for each feature column.  If
                ``None``, features are numbered ``feature_0``, etc.

        Returns:
            Dictionary with:

            * ``"shap_values"`` -- 2-D array of shape
              ``(len(sample_indices), n_features)`` with SHAP values.
            * ``"feature_ranking"`` -- list of dicts, each with keys
              ``"name"`` and ``"importance"`` (mean absolute SHAP),
              sorted descending, truncated to *max_features*.

        Raises:
            SentinelError: If SHAP is not installed or indices are out
                of range.
        """
        try:
            import shap  # type: ignore[import-untyped]
        except ImportError as exc:
            raise SentinelError(
                "SHAP is not installed. Install it with: uv add --group explain shap"
            ) from exc

        if len(sample_indices) > 50:
            raise SentinelError(
                f"Maximum 50 sample indices allowed, got {len(sample_indices)}"
            )

        n_samples, n_features = X.shape
        for idx in sample_indices:
            if idx < 0 or idx >= n_samples:
                raise SentinelError(f"Sample index {idx} out of range [0, {n_samples})")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        X_explain = X[sample_indices]

        if self._is_tree_model:
            shap_values = self._explain_tree(shap, X_explain)
        else:
            shap_values = self._explain_kernel(shap, X, X_explain)

        # Build feature ranking from mean |SHAP|.
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        top_k = min(self._max_features, n_features)
        top_indices = np.argsort(mean_abs)[::-1][:top_k]

        feature_ranking = [
            {
                "name": feature_names[i],
                "importance": float(mean_abs[i]),
            }
            for i in top_indices
        ]

        logger.info(
            "shap.explain_complete",
            n_samples=len(sample_indices),
            top_feature=feature_ranking[0]["name"] if feature_ranking else None,
        )

        return {
            "shap_values": shap_values,
            "feature_ranking": feature_ranking,
        }

    def _explain_tree(
        self,
        shap: Any,
        X_explain: np.ndarray,
    ) -> np.ndarray:
        """Use TreeExplainer for tree-based models.

        Args:
            shap: The imported shap module.
            X_explain: Samples to explain.

        Returns:
            SHAP values array.
        """
        # Access the underlying sklearn model.
        tree_model = self._model._model  # type: ignore[attr-defined]
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_explain)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return np.asarray(shap_values)

    def _explain_kernel(
        self,
        shap: Any,
        X_full: np.ndarray,
        X_explain: np.ndarray,
    ) -> np.ndarray:
        """Use KernelExplainer for generic models.

        Args:
            shap: The imported shap module.
            X_full: Full dataset for background sampling.
            X_explain: Samples to explain.

        Returns:
            SHAP values array.
        """
        n = X_full.shape[0]
        if n > _MAX_BACKGROUND_SAMPLES:
            rng = np.random.default_rng(42)
            bg_indices = rng.choice(n, _MAX_BACKGROUND_SAMPLES, replace=False)
            background = X_full[bg_indices]
        else:
            background = X_full

        def score_fn(x: np.ndarray) -> np.ndarray:
            return self._model.score(x)

        explainer = shap.KernelExplainer(score_fn, background)
        shap_values = explainer.shap_values(X_explain, silent=True)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return np.asarray(shap_values)


def _is_tree_model(model: BaseAnomalyDetector) -> bool:
    """Check whether the model wraps a tree-based sklearn estimator.

    Specifically looks for ``model._model`` being an instance of
    ``sklearn.ensemble.IsolationForest``.

    Args:
        model: An anomaly detector instance.

    Returns:
        ``True`` if the underlying model is tree-based.
    """
    try:
        from sklearn.ensemble import IsolationForest

        inner = getattr(model, "_model", None)
        return isinstance(inner, IsolationForest)
    except ImportError:
        return False
