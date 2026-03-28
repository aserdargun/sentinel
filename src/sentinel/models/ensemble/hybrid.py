"""Hybrid Ensemble anomaly detector.

Combines anomaly scores from multiple sub-models using one of three
strategies: weighted average, majority voting, or stacking (logistic
regression meta-learner).  All sub-model scores are min-max normalised
to [0, 1] before combination.  If a sub-model fails during scoring,
it is excluded and the remaining weights are renormalised.
"""

from __future__ import annotations

import inspect
import json
import os
from typing import Any

import joblib
import numpy as np
import structlog

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.exceptions import ConfigError, SentinelError
from sentinel.core.registry import get_model_class, register_model

logger = structlog.get_logger(__name__)

# Valid combination strategies.
VALID_STRATEGIES = ("weighted_average", "majority_voting", "stacking")

# Default voting threshold applied per sub-model before majority vote.
DEFAULT_VOTING_THRESHOLD = 0.5

# Config / artifact filenames.
_CONFIG_FILENAME = "config.json"
_META_LEARNER_FILENAME = "meta_learner.joblib"
_SCORE_STATS_FILENAME = "score_stats.joblib"

# Minimum standard deviation floor to avoid numerical issues.
_MIN_RANGE = 1e-10


def _min_max_normalise(
    scores: np.ndarray,
    score_min: float | None = None,
    score_max: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Normalise scores to [0, 1] via min-max scaling.

    Args:
        scores: 1-D array of raw anomaly scores.
        score_min: Pre-computed minimum (from training). If ``None``, computed
            from *scores*.
        score_max: Pre-computed maximum (from training). If ``None``, computed
            from *scores*.

    Returns:
        Tuple of (normalised_scores, score_min, score_max).
    """
    if score_min is None:
        score_min = float(np.min(scores))
    if score_max is None:
        score_max = float(np.max(scores))

    range_val = score_max - score_min
    if range_val < _MIN_RANGE:
        # Constant scores -- return zeros.
        return np.zeros_like(scores, dtype=np.float64), score_min, score_max

    normalised = (scores - score_min) / range_val
    # Clip in case inference scores fall outside training range.
    normalised = np.clip(normalised, 0.0, 1.0)
    return normalised, score_min, score_max


@register_model("hybrid_ensemble")
class HybridEnsemble(BaseAnomalyDetector):
    """Hybrid Ensemble anomaly detector.

    Instantiates multiple sub-models from the model registry, fits them
    independently, and combines their normalised anomaly scores according
    to the chosen strategy.

    Three combination strategies are supported:

    * **weighted_average** -- weighted sum of min-max normalised scores.
    * **majority_voting** -- each sub-model votes (score > per-model
      threshold), votes are weighted, majority wins.
    * **stacking** -- a logistic regression meta-learner is trained on
      the matrix of sub-model scores (requires validation data with
      labels).

    If a sub-model raises during ``score()``, it is silently excluded
    and the remaining weights are renormalised.  A warning is logged.
    If *all* sub-models fail, the ensemble raises ``SentinelError``.

    Args:
        sub_models: List of registered model names to combine.
        weights: Per-model weights (must match length of *sub_models*).
        strategy: One of ``"weighted_average"``, ``"majority_voting"``,
            or ``"stacking"``.
        voting_threshold: Score threshold used per sub-model when
            ``strategy="majority_voting"``.

    Example::

        ensemble = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.6, 0.4],
            strategy="weighted_average",
        )
        ensemble.fit(X_train)
        scores = ensemble.score(X_test)
    """

    def __init__(
        self,
        sub_models: list[str] | None = None,
        weights: list[float] | None = None,
        strategy: str = "weighted_average",
        voting_threshold: float = DEFAULT_VOTING_THRESHOLD,
        device: str = "auto",
    ) -> None:
        if sub_models is None:
            sub_models = ["zscore", "isolation_forest"]
        if weights is None:
            weights = [1.0 / len(sub_models)] * len(sub_models)

        if len(sub_models) != len(weights):
            raise ConfigError(
                f"sub_models ({len(sub_models)}) and weights "
                f"({len(weights)}) must have the same length"
            )
        if strategy not in VALID_STRATEGIES:
            raise ConfigError(
                f"Invalid strategy '{strategy}'. Must be one of {VALID_STRATEGIES}"
            )
        if len(sub_models) == 0:
            raise ConfigError("sub_models must contain at least one model")

        self.sub_model_names: list[str] = list(sub_models)
        self.weights: list[float] = list(weights)
        self.strategy: str = strategy
        self.voting_threshold: float = voting_threshold
        self.device_str: str = device

        # Populated during fit().
        self._sub_model_instances: list[BaseAnomalyDetector] = []
        self._score_stats: list[dict[str, float]] = []
        self._meta_learner: Any = None  # LogisticRegression for stacking
        self._is_fitted: bool = False
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, **kwargs: Any) -> None:
        """Fit all sub-models on *X*.

        For the ``"stacking"`` strategy, ``val_X`` and ``val_labels``
        keyword arguments are required to train the logistic regression
        meta-learner.  If they are absent, the strategy silently falls
        back to ``"weighted_average"``.

        Args:
            X: Training data of shape ``(n_samples, n_features)``.
            **kwargs: Optional keyword arguments:

                * ``val_X`` -- Validation data array for stacking.
                * ``val_labels`` -- Binary anomaly labels for stacking.
                * Any additional kwargs are forwarded to sub-model
                  ``fit()`` calls.

        Raises:
            SentinelError: If *X* is empty or has wrong dimensionality.
            SentinelError: If all sub-models fail to fit.
        """
        if X.ndim != 2 or X.shape[0] == 0:
            raise SentinelError(
                f"Expected 2-D array with at least 1 sample, got shape {X.shape}"
            )

        self._n_features = X.shape[1]
        self._sub_model_instances = []
        self._score_stats = []

        # Instantiate and fit each sub-model.
        fit_failures: list[str] = []
        for name in self.sub_model_names:
            try:
                cls = get_model_class(name)
                # Pass device to sub-model if its constructor accepts it.
                sig = inspect.signature(cls.__init__)
                sub_params: dict[str, Any] = {}
                if "device" in sig.parameters:
                    sub_params["device"] = self.device_str
                instance = cls(**sub_params)
                instance.fit(X, **kwargs)
                # Score the training data to capture min/max for normalisation.
                raw_scores = instance.score(X)
                s_min = float(np.min(raw_scores))
                s_max = float(np.max(raw_scores))
                self._sub_model_instances.append(instance)
                self._score_stats.append({"min": s_min, "max": s_max})
            except Exception as exc:
                logger.warning("Sub-model '%s' failed during fit: %s", name, exc)
                fit_failures.append(name)

        if len(self._sub_model_instances) == 0:
            raise SentinelError(
                "All sub-models failed to fit: " + ", ".join(fit_failures)
            )

        # If some sub-models failed, rebuild the active lists.
        if fit_failures:
            self._rebuild_active_lists(fit_failures)

        # Train meta-learner for stacking strategy.
        if self.strategy == "stacking":
            self._fit_meta_learner(X, kwargs)

        self._is_fitted = True

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute combined anomaly scores.

        Each sub-model produces raw scores which are min-max normalised
        using the statistics captured during ``fit()``.  The normalised
        scores are then combined according to the configured strategy.

        Args:
            X: Data of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of combined anomaly scores in [0, 1].

        Raises:
            SentinelError: If the model has not been fitted.
            SentinelError: If all sub-models fail during scoring.
        """
        self._check_fitted()

        normalised_scores: list[np.ndarray] = []
        active_weights: list[float] = []
        active_indices: list[int] = []

        for idx, (instance, stats) in enumerate(
            zip(self._sub_model_instances, self._score_stats)
        ):
            try:
                raw = instance.score(X)
                normed, _, _ = _min_max_normalise(raw, stats["min"], stats["max"])
                normalised_scores.append(normed)
                active_weights.append(self.weights[idx])
                active_indices.append(idx)
            except Exception as exc:
                logger.warning(
                    "Sub-model '%s' (index %d) failed during score, excluding: %s",
                    self.sub_model_names[idx],
                    idx,
                    exc,
                )

        if len(normalised_scores) == 0:
            raise SentinelError(
                "All sub-models failed during scoring. Cannot produce ensemble scores."
            )

        # Renormalise weights if some models were excluded.
        w = np.array(active_weights, dtype=np.float64)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones_like(w) / len(w)

        score_matrix = np.column_stack(normalised_scores)

        if self.strategy == "weighted_average":
            return self._combine_weighted_average(score_matrix, w)
        elif self.strategy == "majority_voting":
            return self._combine_majority_voting(score_matrix, w)
        elif self.strategy == "stacking":
            return self._combine_stacking(score_matrix, active_indices)
        else:
            # Should not happen after __init__ validation.
            raise SentinelError(f"Unknown strategy: {self.strategy}")

    def save(self, path: str) -> None:
        """Save ensemble configuration and all sub-models to disk.

        Each sub-model is saved into a subdirectory ``submodel_{i}/``
        inside *path*.  The ensemble configuration and normalisation
        statistics are stored in ``config.json`` and ``score_stats.joblib``.

        Args:
            path: Directory to store ensemble artifacts.

        Raises:
            SentinelError: If the ensemble has not been fitted.
        """
        self._check_fitted()
        os.makedirs(path, exist_ok=True)

        # Save ensemble config.
        config = {
            "model_name": "hybrid_ensemble",
            "sub_model_names": self.sub_model_names,
            "weights": self.weights,
            "strategy": self.strategy,
            "voting_threshold": self.voting_threshold,
            "device": self.device_str,
            "n_features": self._n_features,
            "n_active_models": len(self._sub_model_instances),
        }
        config_path = os.path.join(path, _CONFIG_FILENAME)
        tmp_config = config_path + ".tmp"
        with open(tmp_config, "w") as f:
            json.dump(config, f, indent=2)
        os.rename(tmp_config, config_path)

        # Save score normalisation statistics.
        stats_path = os.path.join(path, _SCORE_STATS_FILENAME)
        tmp_stats = stats_path + ".tmp"
        joblib.dump(self._score_stats, tmp_stats)
        os.rename(tmp_stats, stats_path)

        # Save each sub-model in its own subdirectory.
        for i, instance in enumerate(self._sub_model_instances):
            subdir = os.path.join(path, f"submodel_{i}")
            os.makedirs(subdir, exist_ok=True)
            instance.save(subdir)

        # Save meta-learner if stacking.
        if self._meta_learner is not None:
            ml_path = os.path.join(path, _META_LEARNER_FILENAME)
            tmp_ml = ml_path + ".tmp"
            joblib.dump(self._meta_learner, tmp_ml)
            os.rename(tmp_ml, ml_path)

    def load(self, path: str) -> None:
        """Load a previously saved ensemble from disk.

        Args:
            path: Directory containing ensemble artifacts.

        Raises:
            SentinelError: If required files are missing or corrupt.
        """
        config_path = os.path.join(path, _CONFIG_FILENAME)
        if not os.path.isfile(config_path):
            raise SentinelError(f"Ensemble config not found: {config_path}")

        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise SentinelError(f"Failed to read ensemble config: {exc}") from exc

        self.sub_model_names = config["sub_model_names"]
        self.weights = config["weights"]
        self.strategy = config["strategy"]
        self.voting_threshold = config.get("voting_threshold", DEFAULT_VOTING_THRESHOLD)
        self.device_str = config.get("device", "auto")
        self._n_features = config.get("n_features")
        n_active = config.get("n_active_models", len(self.sub_model_names))

        # Load score stats.
        stats_path = os.path.join(path, _SCORE_STATS_FILENAME)
        if not os.path.isfile(stats_path):
            raise SentinelError(f"Score stats not found: {stats_path}")
        try:
            self._score_stats = joblib.load(stats_path)
        except Exception as exc:
            raise SentinelError(f"Failed to load score stats: {exc}") from exc

        # Load each sub-model.
        self._sub_model_instances = []
        for i in range(n_active):
            subdir = os.path.join(path, f"submodel_{i}")
            if not os.path.isdir(subdir):
                raise SentinelError(f"Sub-model directory not found: {subdir}")
            # Read the sub-model config to determine its type.
            sub_config_path = os.path.join(subdir, _CONFIG_FILENAME)
            if not os.path.isfile(sub_config_path):
                raise SentinelError(f"Sub-model config not found: {sub_config_path}")
            with open(sub_config_path) as f:
                sub_config = json.load(f)

            model_name = sub_config.get("model_name", self.sub_model_names[i])
            cls = get_model_class(model_name)
            instance = cls()
            instance.load(subdir)
            self._sub_model_instances.append(instance)

        # Load meta-learner if stacking.
        ml_path = os.path.join(path, _META_LEARNER_FILENAME)
        if os.path.isfile(ml_path):
            try:
                self._meta_learner = joblib.load(ml_path)
            except Exception as exc:
                raise SentinelError(f"Failed to load meta-learner: {exc}") from exc

        self._is_fitted = True

    def get_params(self) -> dict[str, Any]:
        """Return ensemble parameters.

        Returns:
            Dict containing sub-model names, weights, strategy, and
            voting threshold.
        """
        return {
            "model_name": "hybrid_ensemble",
            "sub_models": self.sub_model_names,
            "weights": self.weights,
            "strategy": self.strategy,
            "voting_threshold": self.voting_threshold,
            "device": self.device_str,
            "n_features": self._n_features,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the ensemble has not been fitted."""
        if not self._is_fitted:
            raise SentinelError("HybridEnsemble has not been fitted. Call fit() first.")

    def _rebuild_active_lists(self, failed_names: list[str]) -> None:
        """Rebuild sub_model_names and weights after fit failures.

        Removes failed models from the name and weight lists, keeping
        only those that successfully fitted.  Weights are renormalised.

        Args:
            failed_names: Model names that failed during fit.
        """
        active_names: list[str] = []
        active_weights: list[float] = []
        for name, weight in zip(self.sub_model_names, self.weights):
            if name not in failed_names:
                active_names.append(name)
                active_weights.append(weight)

        # Renormalise weights.
        w_sum = sum(active_weights)
        if w_sum > 0:
            active_weights = [w / w_sum for w in active_weights]
        else:
            n = len(active_weights)
            active_weights = [1.0 / n for _ in range(n)]

        self.sub_model_names = active_names
        self.weights = active_weights

    def _fit_meta_learner(
        self,
        X_train: np.ndarray,
        kwargs: dict[str, Any],
    ) -> None:
        """Train the logistic regression meta-learner for stacking.

        Requires ``val_X`` and ``val_labels`` in *kwargs*.  If they are
        absent, logs a warning and falls back to weighted_average.

        Args:
            X_train: Training data (used as fallback if no val data).
            kwargs: Must contain ``val_X`` and ``val_labels`` for
                stacking.
        """
        from sklearn.linear_model import LogisticRegression

        val_X = kwargs.get("val_X")
        val_labels = kwargs.get("val_labels")

        if val_X is None or val_labels is None:
            logger.warning(
                "Stacking strategy requires val_X and val_labels. "
                "Falling back to weighted_average for scoring."
            )
            self.strategy = "weighted_average"
            return

        # Build the meta-feature matrix from validation scores.
        meta_features = self._build_meta_features(val_X)
        if meta_features is None:
            logger.warning(
                "Could not build meta-features for stacking. "
                "Falling back to weighted_average."
            )
            self.strategy = "weighted_average"
            return

        # Ensure labels are a 1-D integer array.
        val_labels = np.asarray(val_labels, dtype=np.int32).ravel()

        # Check that both classes are present.
        unique_classes = np.unique(val_labels)
        if len(unique_classes) < 2:
            logger.warning(
                "val_labels contains only one class (%s). "
                "Cannot train meta-learner. Falling back to weighted_average.",
                unique_classes,
            )
            self.strategy = "weighted_average"
            return

        self._meta_learner = LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        )
        self._meta_learner.fit(meta_features, val_labels)

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray | None:
        """Score X with all sub-models and stack normalised scores.

        Args:
            X: Data array of shape ``(n_samples, n_features)``.

        Returns:
            2-D array of shape ``(n_samples, n_sub_models)`` with
            normalised scores, or ``None`` if all sub-models fail.
        """
        columns: list[np.ndarray] = []
        for instance, stats in zip(self._sub_model_instances, self._score_stats):
            try:
                raw = instance.score(X)
                normed, _, _ = _min_max_normalise(raw, stats["min"], stats["max"])
                columns.append(normed)
            except Exception as exc:
                logger.warning("Sub-model failed during meta-feature build: %s", exc)
                # Use zeros as placeholder to keep column alignment.
                columns.append(np.zeros(X.shape[0], dtype=np.float64))

        if not columns:
            return None

        return np.column_stack(columns)

    def _combine_weighted_average(
        self,
        score_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Combine scores via weighted average.

        Args:
            score_matrix: Shape ``(n_samples, n_active_models)``.
            weights: Normalised weight vector of length ``n_active_models``.

        Returns:
            1-D array of combined scores.
        """
        return score_matrix @ weights

    def _combine_majority_voting(
        self,
        score_matrix: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Combine scores via weighted majority voting.

        Each sub-model's normalised score is thresholded to produce a
        binary vote.  Votes are weighted and summed.  The combined
        score is the weighted fraction of models that voted anomalous.

        Args:
            score_matrix: Shape ``(n_samples, n_active_models)``.
            weights: Normalised weight vector.

        Returns:
            1-D array of combined scores in [0, 1].
        """
        votes = (score_matrix > self.voting_threshold).astype(np.float64)
        return votes @ weights

    def _combine_stacking(
        self,
        score_matrix: np.ndarray,
        active_indices: list[int],
    ) -> np.ndarray:
        """Combine scores via stacking meta-learner.

        If the meta-learner was not trained (e.g., missing val data),
        falls back to weighted average.

        Args:
            score_matrix: Shape ``(n_samples, n_active_models)``.
            active_indices: Indices into the original sub-model list
                that are currently active.

        Returns:
            1-D array of anomaly scores (probability of anomaly class).
        """
        if self._meta_learner is None:
            logger.warning(
                "Meta-learner not available. Falling back to weighted_average."
            )
            w = np.array(self.weights, dtype=np.float64)
            w = w / w.sum()
            return self._combine_weighted_average(score_matrix, w)

        # If some models were excluded, we need to pad the feature matrix
        # to match the meta-learner's expected input dimensionality.
        n_total = len(self._score_stats)
        if score_matrix.shape[1] < n_total:
            padded = np.zeros((score_matrix.shape[0], n_total), dtype=np.float64)
            for col_idx, orig_idx in enumerate(active_indices):
                padded[:, orig_idx] = score_matrix[:, col_idx]
            score_matrix = padded

        # Use predict_proba to get a continuous anomaly score.
        probas = self._meta_learner.predict_proba(score_matrix)
        # Anomaly class is assumed to be class 1.
        if probas.shape[1] == 2:
            return probas[:, 1]
        return probas[:, 0]
