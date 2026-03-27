"""Integration test: full Phase 1 pipeline.

Covers: synthetic data -> validate -> preprocess -> fit -> score -> evaluate.
Tests all 3 statistical models and verifies the zscore save/load round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sentinel.models  # noqa: F401 — triggers registration of all built-in models
from sentinel.core.registry import get_model_class
from sentinel.data.preprocessors import (
    chronological_split,
    fill_nan,
    scale_zscore,
    to_numpy,
)
from sentinel.data.synthetic import generate_synthetic
from sentinel.data.validators import separate_labels, validate_dataframe
from sentinel.training.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Expected metric keys
# ---------------------------------------------------------------------------

SCORE_DIST_KEYS = {"score_mean", "score_std", "score_p50", "score_p95", "score_p99"}
CLASSIFICATION_KEYS = {"precision", "recall", "f1", "auc_roc", "auc_pr"}
THRESHOLD_KEY = {"threshold"}
EXPECTED_METRIC_KEYS = SCORE_DIST_KEYS | CLASSIFICATION_KEYS | THRESHOLD_KEY

# ---------------------------------------------------------------------------
# Shared fixture — generated once for all parametrized model tests
# ---------------------------------------------------------------------------

N_FEATURES = 5
LENGTH = 500
SEED = 42

# MatrixProfile default subsequence_length=50 requires 150 rows minimum.
# With 500 rows and a 15% test split (75 rows), we need subsequence_length <= 25.
_MP_KWARGS: dict[str, Any] = {"subsequence_length": 20}


@pytest.fixture(scope="module")
def pipeline_data() -> dict[str, Any]:
    """Run the full Phase 1 preprocessing pipeline once and expose splits.

    Returns a dict with keys:
        df_clean  - validated, label-free DataFrame
        labels    - full is_anomaly Series
        train_np  - numpy training array
        val_np    - numpy validation array
        test_np   - numpy test array
        val_labels_np  - numpy validation labels (int64)
        test_labels_np - numpy test labels (int64)
    """
    # 1. Generate synthetic data
    df = generate_synthetic(n_features=N_FEATURES, length=LENGTH, seed=SEED)

    # 2. Separate labels before validation (is_anomaly must not be a feature)
    df_clean, labels = separate_labels(df)

    # 3. Validate
    df_validated = validate_dataframe(df_clean)

    # 4. Fill NaN then z-score scale
    df_filled = fill_nan(df_validated)
    df_scaled, _stats = scale_zscore(df_filled)

    # 5. Chronological split (70/15/15)
    train_df, val_df, test_df = chronological_split(df_scaled)

    # 6. Convert to numpy
    train_np = to_numpy(train_df)
    val_np = to_numpy(val_df)
    test_np = to_numpy(test_df)

    # Derive label splits from the same positional boundaries
    n = LENGTH
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    assert labels is not None
    labels_np = labels.to_numpy().astype(np.int64)
    val_labels_np = labels_np[train_end:val_end]
    test_labels_np = labels_np[val_end:]

    return {
        "df_clean": df_clean,
        "labels": labels,
        "train_np": train_np,
        "val_np": val_np,
        "test_np": test_np,
        "val_labels_np": val_labels_np,
        "test_labels_np": test_labels_np,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_model(
    model_name: str,
    data: dict[str, Any],
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Instantiate, fit, score, and evaluate a registered model.

    Args:
        model_name: Registry key for the model.
        data: Output of the pipeline_data fixture.
        model_kwargs: Optional constructor keyword arguments passed to the model.

    Returns:
        Tuple of (test_scores, metrics_dict).
    """
    ModelClass = get_model_class(model_name)
    model = ModelClass(**(model_kwargs or {}))

    model.fit(data["train_np"])
    test_scores = model.score(data["test_np"])

    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        scores=test_scores,
        labels=data["test_labels_np"],
        val_scores=model.score(data["val_np"]),
        val_labels=data["val_labels_np"],
    )
    return test_scores, metrics


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhase1Pipeline:
    """Full pipeline tests for all 3 statistical models."""

    # ------------------------------------------------------------------
    # Preprocessing sanity checks (run once, model-agnostic)
    # ------------------------------------------------------------------

    def test_synthetic_data_shape(self, pipeline_data: dict[str, Any]) -> None:
        """Generated data has the expected number of features and rows."""
        assert pipeline_data["train_np"].shape[1] == N_FEATURES
        total_rows = (
            pipeline_data["train_np"].shape[0]
            + pipeline_data["val_np"].shape[0]
            + pipeline_data["test_np"].shape[0]
        )
        assert total_rows == LENGTH

    def test_labels_separated(self, pipeline_data: dict[str, Any]) -> None:
        """is_anomaly column must not appear in the feature array."""
        df_clean = pipeline_data["df_clean"]
        assert "is_anomaly" not in df_clean.columns
        assert pipeline_data["labels"] is not None

    def test_chronological_split_ratios(self, pipeline_data: dict[str, Any]) -> None:
        """Split sizes are within rounding tolerance of 70/15/15."""
        train_n = pipeline_data["train_np"].shape[0]
        val_n = pipeline_data["val_np"].shape[0]
        test_n = pipeline_data["test_np"].shape[0]

        assert train_n == pytest.approx(LENGTH * 0.70, abs=2)
        assert val_n == pytest.approx(LENGTH * 0.15, abs=2)
        assert test_n == pytest.approx(LENGTH * 0.15, abs=2)

    def test_no_nan_after_preprocessing(self, pipeline_data: dict[str, Any]) -> None:
        """fill_nan + scale_zscore must produce NaN-free arrays."""
        for split in ("train_np", "val_np", "test_np"):
            arr = pipeline_data[split]
            assert not np.isnan(arr).any(), f"NaN found in {split}"

    # ------------------------------------------------------------------
    # Z-Score model
    # ------------------------------------------------------------------

    def test_zscore_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """ZScore score() output length equals the test set size."""
        scores, _ = _run_model("zscore", pipeline_data)
        assert scores.shape == (pipeline_data["test_np"].shape[0],)

    def test_zscore_scores_non_negative(self, pipeline_data: dict[str, Any]) -> None:
        """ZScore produces non-negative anomaly scores."""
        scores, _ = _run_model("zscore", pipeline_data)
        assert np.all(scores >= 0.0)

    def test_zscore_metrics_keys(self, pipeline_data: dict[str, Any]) -> None:
        """ZScore evaluation returns all expected metric keys."""
        _, metrics = _run_model("zscore", pipeline_data)
        assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())

    def test_zscore_score_stats_finite(self, pipeline_data: dict[str, Any]) -> None:
        """ZScore score distribution metrics are finite numbers."""
        _, metrics = _run_model("zscore", pipeline_data)
        for key in SCORE_DIST_KEYS:
            assert np.isfinite(metrics[key]), f"{key} is not finite"

    def test_zscore_threshold_positive(self, pipeline_data: dict[str, Any]) -> None:
        """ZScore threshold is a positive number."""
        _, metrics = _run_model("zscore", pipeline_data)
        assert metrics["threshold"] > 0.0

    # ------------------------------------------------------------------
    # Isolation Forest model
    # ------------------------------------------------------------------

    def test_isolation_forest_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """IsolationForest score() output length equals the test set size."""
        scores, _ = _run_model("isolation_forest", pipeline_data)
        assert scores.shape == (pipeline_data["test_np"].shape[0],)

    def test_isolation_forest_metrics_keys(self, pipeline_data: dict[str, Any]) -> None:
        """IsolationForest evaluation returns all expected metric keys."""
        _, metrics = _run_model("isolation_forest", pipeline_data)
        assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())

    def test_isolation_forest_scores_finite(
        self, pipeline_data: dict[str, Any]
    ) -> None:
        """IsolationForest scores are all finite (no Inf or NaN)."""
        scores, _ = _run_model("isolation_forest", pipeline_data)
        assert np.all(np.isfinite(scores))

    def test_isolation_forest_score_stats_finite(
        self, pipeline_data: dict[str, Any]
    ) -> None:
        """IsolationForest score distribution metrics are finite numbers."""
        _, metrics = _run_model("isolation_forest", pipeline_data)
        for key in SCORE_DIST_KEYS:
            assert np.isfinite(metrics[key]), f"{key} is not finite"

    # ------------------------------------------------------------------
    # Matrix Profile model
    # ------------------------------------------------------------------

    def test_matrix_profile_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """MatrixProfile score() output length equals the test set size."""
        scores, _ = _run_model("matrix_profile", pipeline_data, _MP_KWARGS)
        assert scores.shape == (pipeline_data["test_np"].shape[0],)

    def test_matrix_profile_metrics_keys(self, pipeline_data: dict[str, Any]) -> None:
        """MatrixProfile evaluation returns all expected metric keys."""
        _, metrics = _run_model("matrix_profile", pipeline_data, _MP_KWARGS)
        assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())

    def test_matrix_profile_scores_non_negative(
        self, pipeline_data: dict[str, Any]
    ) -> None:
        """MatrixProfile produces non-negative distance scores."""
        scores, _ = _run_model("matrix_profile", pipeline_data, _MP_KWARGS)
        assert np.all(scores >= 0.0)

    def test_matrix_profile_score_stats_finite(
        self, pipeline_data: dict[str, Any]
    ) -> None:
        """MatrixProfile score distribution metrics are finite numbers."""
        _, metrics = _run_model("matrix_profile", pipeline_data, _MP_KWARGS)
        for key in SCORE_DIST_KEYS:
            assert np.isfinite(metrics[key]), f"{key} is not finite"

    # ------------------------------------------------------------------
    # Evaluator output contract (common to all models)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "model_name", ["zscore", "isolation_forest", "matrix_profile"]
    )
    def test_classification_metrics_type(
        self, model_name: str, pipeline_data: dict[str, Any]
    ) -> None:
        """Supervised classification metrics are float or None (never missing)."""
        kwargs = _MP_KWARGS if model_name == "matrix_profile" else None
        _, metrics = _run_model(model_name, pipeline_data, kwargs)
        for key in CLASSIFICATION_KEYS:
            assert key in metrics
            val = metrics[key]
            assert val is None or isinstance(val, float), (
                f"{key} should be float or None, got {type(val)}"
            )

    @pytest.mark.parametrize(
        "model_name", ["zscore", "isolation_forest", "matrix_profile"]
    )
    def test_precision_recall_in_unit_range(
        self, model_name: str, pipeline_data: dict[str, Any]
    ) -> None:
        """Precision and recall are in [0, 1] when not None."""
        kwargs = _MP_KWARGS if model_name == "matrix_profile" else None
        _, metrics = _run_model(model_name, pipeline_data, kwargs)
        for key in ("precision", "recall", "f1"):
            if metrics[key] is not None:
                assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of range"


# ---------------------------------------------------------------------------
# ZScore save/load round-trip
# ---------------------------------------------------------------------------


class TestZScoreSaveLoad:
    """Verify that a fitted ZScoreDetector survives a save/load cycle."""

    def test_save_creates_artifacts(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """save() writes config.json and model.joblib into the target directory."""
        model_dir = str(tmp_path / "zscore_model")
        ModelClass = get_model_class("zscore")
        model = ModelClass()
        model.fit(pipeline_data["train_np"])
        model.save(model_dir)

        assert (Path(model_dir) / "config.json").exists()
        assert (Path(model_dir) / "model.joblib").exists()

    def test_loaded_scores_match(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Scores from a loaded model are identical to the original model."""
        model_dir = str(tmp_path / "zscore_roundtrip")
        ModelClass = get_model_class("zscore")

        original = ModelClass()
        original.fit(pipeline_data["train_np"])
        scores_before = original.score(pipeline_data["test_np"])
        original.save(model_dir)

        restored = ModelClass()
        restored.load(model_dir)
        scores_after = restored.score(pipeline_data["test_np"])

        np.testing.assert_array_almost_equal(scores_before, scores_after)

    def test_loaded_params_match(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Hyperparameters are preserved through save/load."""
        model_dir = str(tmp_path / "zscore_params")
        ModelClass = get_model_class("zscore")

        original = ModelClass(window_size=20, threshold_sigma=2.5)
        original.fit(pipeline_data["train_np"])
        original.save(model_dir)

        restored = ModelClass()
        restored.load(model_dir)

        assert restored.window_size == 20
        assert restored.threshold_sigma == pytest.approx(2.5)
        assert restored._n_features == N_FEATURES

    def test_loaded_model_evaluates(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """A loaded model can be passed directly to Evaluator without error."""
        model_dir = str(tmp_path / "zscore_eval")
        ModelClass = get_model_class("zscore")

        model = ModelClass()
        model.fit(pipeline_data["train_np"])
        model.save(model_dir)

        restored = ModelClass()
        restored.load(model_dir)

        test_scores = restored.score(pipeline_data["test_np"])
        val_scores = restored.score(pipeline_data["val_np"])

        evaluator = Evaluator()
        metrics = evaluator.evaluate(
            scores=test_scores,
            labels=pipeline_data["test_labels_np"],
            val_scores=val_scores,
            val_labels=pipeline_data["val_labels_np"],
        )

        assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())
        assert np.isfinite(metrics["threshold"])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Verify that all Phase 1 models are properly registered."""

    @pytest.mark.parametrize(
        "model_name",
        ["zscore", "isolation_forest", "matrix_profile"],
    )
    def test_model_registered(self, model_name: str) -> None:
        """Each statistical model is retrievable from the registry."""
        cls = get_model_class(model_name)
        assert cls is not None

    @pytest.mark.parametrize(
        "model_name",
        ["zscore", "isolation_forest", "matrix_profile"],
    )
    def test_model_instantiates(self, model_name: str) -> None:
        """Registry class instantiates without arguments."""
        cls = get_model_class(model_name)
        instance = cls()
        assert instance is not None

    @pytest.mark.parametrize(
        "model_name",
        ["zscore", "isolation_forest", "matrix_profile"],
    )
    def test_model_has_get_params(self, model_name: str) -> None:
        """get_params() returns a dict before fitting."""
        cls = get_model_class(model_name)
        instance = cls()
        params = instance.get_params()
        assert isinstance(params, dict)
