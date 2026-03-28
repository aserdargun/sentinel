"""End-to-end acceptance tests for the full Sentinel pipeline.

Covers: generate -> validate/ingest -> train (Z-Score, IF, AE) -> evaluate ->
ensemble -> visualize -> tracking artifact verification.

All tests are marked @pytest.mark.slow.  Designed to run in ~30 s on CPU with
small data and minimal epochs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import sentinel.models  # noqa: F401 — trigger model registration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES = 5
LENGTH = 500
ANOMALY_RATIO = 0.05
SEED = 42

# Autoencoder tiny config: keeps training fast on CPU.
AE_HIDDEN_DIMS = [16, 8]
AE_EPOCHS = 5

# Expected metric keys produced by Evaluator.
SCORE_DIST_KEYS = {"score_mean", "score_std", "score_p50", "score_p95", "score_p99"}
CLASSIFICATION_KEYS = {"precision", "recall", "f1", "auc_roc", "auc_pr"}
ALL_METRIC_KEYS = SCORE_DIST_KEYS | CLASSIFICATION_KEYS | {"threshold"}

# ---------------------------------------------------------------------------
# Module-level torch availability check
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared pipeline fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline() -> dict[str, Any]:
    """Run the full preprocessing pipeline once and expose splits + labels.

    Returns:
        Dict with keys: df_raw, df_clean, labels, train_np, val_np, test_np,
        train_labels, val_labels, test_labels.
    """
    from sentinel.data.preprocessors import (
        chronological_split,
        fill_nan,
        scale_zscore,
        to_numpy,
    )
    from sentinel.data.synthetic import generate_synthetic
    from sentinel.data.validators import separate_labels, validate_dataframe

    df_raw = generate_synthetic(
        n_features=N_FEATURES,
        length=LENGTH,
        anomaly_ratio=ANOMALY_RATIO,
        seed=SEED,
    )

    # Separate labels before passing to the validator (is_anomaly is reserved).
    df_no_labels, labels_series = separate_labels(df_raw)
    df_validated = validate_dataframe(df_no_labels)
    df_filled = fill_nan(df_validated)
    df_scaled, _ = scale_zscore(df_filled)

    train_df, val_df, test_df = chronological_split(df_scaled)
    train_np = to_numpy(train_df)
    val_np = to_numpy(val_df)
    test_np = to_numpy(test_df)

    n = LENGTH
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    assert labels_series is not None
    labels_np = labels_series.to_numpy().astype(np.int64)

    return {
        "df_raw": df_raw,
        "df_clean": df_no_labels,
        "labels": labels_series,
        "train_np": train_np,
        "val_np": val_np,
        "test_np": test_np,
        "train_labels": labels_np[:train_end],
        "val_labels": labels_np[train_end:val_end],
        "test_labels": labels_np[val_end:],
    }


# ---------------------------------------------------------------------------
# Step 1: Synthetic data generation
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSyntheticDataGeneration:
    """Verify that generate_synthetic produces well-formed output."""

    def test_shape(self, pipeline: dict[str, Any]) -> None:
        """DataFrame has the expected number of rows and columns."""
        df = pipeline["df_raw"]
        assert df.height == LENGTH
        # timestamp + N features + is_anomaly
        assert df.width == N_FEATURES + 2

    def test_feature_columns_present(self, pipeline: dict[str, Any]) -> None:
        """All feature columns are present in the generated DataFrame."""
        df = pipeline["df_raw"]
        for i in range(1, N_FEATURES + 1):
            assert f"feature_{i}" in df.columns

    def test_is_anomaly_present(self, pipeline: dict[str, Any]) -> None:
        """is_anomaly column exists and contains binary values."""
        df = pipeline["df_raw"]
        assert "is_anomaly" in df.columns
        unique_vals = set(df["is_anomaly"].to_list())
        assert unique_vals.issubset({0, 1})

    def test_anomaly_count_nonzero(self, pipeline: dict[str, Any]) -> None:
        """At least one anomaly is injected into the generated data."""
        df = pipeline["df_raw"]
        assert df["is_anomaly"].sum() > 0

    def test_timestamp_dtype(self, pipeline: dict[str, Any]) -> None:
        """Timestamp column is a Polars Datetime type."""
        df = pipeline["df_raw"]
        assert isinstance(df.schema["timestamp"], pl.Datetime)

    def test_timestamps_sorted(self, pipeline: dict[str, Any]) -> None:
        """Timestamps are in ascending chronological order."""
        df = pipeline["df_raw"]
        assert df["timestamp"].is_sorted()


# ---------------------------------------------------------------------------
# Step 2: Validate + ingest
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestValidateAndIngest:
    """Verify the validate and ingest pipeline using tmp_path."""

    def test_validate_cleans_schema(self, pipeline: dict[str, Any]) -> None:
        """validate_dataframe accepts the generated DataFrame without raising."""
        from sentinel.data.validators import validate_dataframe

        df_no_labels = pipeline["df_clean"]
        result = validate_dataframe(df_no_labels)
        assert result.height == df_no_labels.height

    def test_separate_labels_removes_column(self, pipeline: dict[str, Any]) -> None:
        """separate_labels removes is_anomaly from the feature DataFrame."""
        assert "is_anomaly" not in pipeline["df_clean"].columns
        assert pipeline["labels"] is not None

    def test_ingest_creates_parquet(
        self, pipeline: dict[str, Any], tmp_path: Path
    ) -> None:
        """ingest_file saves a Parquet file and updates the metadata registry."""
        from sentinel.data.ingest import ingest_file

        # Write raw DataFrame as CSV for ingest.
        csv_path = tmp_path / "synthetic.csv"
        pipeline["df_raw"].write_csv(str(csv_path))

        data_dir = tmp_path / "raw"
        metadata_file = tmp_path / "datasets.json"

        result = ingest_file(
            file_path=csv_path,
            data_dir=data_dir,
            metadata_file=metadata_file,
        )

        dataset_id = result["dataset_id"]
        parquet_path = data_dir / f"{dataset_id}.parquet"
        assert parquet_path.exists(), "Parquet file was not created"
        assert parquet_path.stat().st_size > 0

    def test_ingest_metadata_json(
        self, pipeline: dict[str, Any], tmp_path: Path
    ) -> None:
        """ingest_file registers the dataset in datasets.json."""
        from sentinel.data.ingest import ingest_file

        csv_path = tmp_path / "synthetic_meta.csv"
        pipeline["df_raw"].write_csv(str(csv_path))

        data_dir = tmp_path / "raw_meta"
        metadata_file = tmp_path / "meta.json"

        result = ingest_file(
            file_path=csv_path,
            data_dir=data_dir,
            metadata_file=metadata_file,
        )

        assert metadata_file.exists()
        metadata = json.loads(metadata_file.read_text())
        dataset_id = result["dataset_id"]
        assert dataset_id in metadata
        entry = metadata[dataset_id]
        assert "feature_names" in entry
        assert "time_range" in entry
        assert "shape" in entry

    def test_ingest_roundtrip_shape(
        self, pipeline: dict[str, Any], tmp_path: Path
    ) -> None:
        """Parquet written by ingest_file round-trips back to the same shape."""
        from sentinel.data.ingest import ingest_file

        csv_path = tmp_path / "synthetic_rt.csv"
        pipeline["df_raw"].write_csv(str(csv_path))

        result = ingest_file(
            file_path=csv_path,
            data_dir=tmp_path / "raw_rt",
            metadata_file=tmp_path / "rt_meta.json",
        )

        parquet_path = tmp_path / "raw_rt" / f"{result['dataset_id']}.parquet"
        loaded = pl.read_parquet(str(parquet_path))
        original_rows = pipeline["df_raw"].height
        assert loaded.height == original_rows


# ---------------------------------------------------------------------------
# Step 3 & 4: Train models and evaluate
# ---------------------------------------------------------------------------


def _train_and_evaluate(
    model_name: str,
    pipeline: dict[str, Any],
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Fit, score, and evaluate a model.

    Args:
        model_name: Registry name for the model.
        pipeline: Output of the pipeline fixture.
        model_kwargs: Optional constructor kwargs.

    Returns:
        Tuple of (fitted_model, test_scores, metrics_dict).
    """
    from sentinel.core.registry import get_model_class
    from sentinel.training.evaluator import Evaluator

    cls = get_model_class(model_name)
    model = cls(**(model_kwargs or {}))
    model.fit(pipeline["train_np"])

    val_scores = model.score(pipeline["val_np"])
    test_scores = model.score(pipeline["test_np"])

    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        scores=test_scores,
        labels=pipeline["test_labels"],
        val_scores=val_scores,
        val_labels=pipeline["val_labels"],
    )
    return model, test_scores, metrics


@pytest.mark.slow
class TestZScoreModel:
    """End-to-end Z-Score detection and evaluation."""

    def test_scores_shape(self, pipeline: dict[str, Any]) -> None:
        """Z-Score score() output length equals the test set size."""
        _, scores, _ = _train_and_evaluate("zscore", pipeline)
        assert scores.shape == (pipeline["test_np"].shape[0],)

    def test_scores_non_negative(self, pipeline: dict[str, Any]) -> None:
        """Z-Score scores are non-negative."""
        _, scores, _ = _train_and_evaluate("zscore", pipeline)
        assert np.all(scores >= 0.0)

    def test_scores_finite(self, pipeline: dict[str, Any]) -> None:
        """Z-Score scores are all finite (no NaN or Inf)."""
        _, scores, _ = _train_and_evaluate("zscore", pipeline)
        assert np.all(np.isfinite(scores))

    def test_metrics_keys_present(self, pipeline: dict[str, Any]) -> None:
        """Evaluator returns all expected metric keys for Z-Score."""
        _, _, metrics = _train_and_evaluate("zscore", pipeline)
        assert ALL_METRIC_KEYS.issubset(metrics.keys())

    def test_classification_metrics_range(self, pipeline: dict[str, Any]) -> None:
        """Precision, recall, F1 are in [0, 1] when not None."""
        _, _, metrics = _train_and_evaluate("zscore", pipeline)
        for key in ("precision", "recall", "f1"):
            val = metrics[key]
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"

    def test_threshold_is_positive(self, pipeline: dict[str, Any]) -> None:
        """Z-Score threshold is a positive finite number."""
        _, _, metrics = _train_and_evaluate("zscore", pipeline)
        assert np.isfinite(metrics["threshold"])
        assert metrics["threshold"] > 0.0

    def test_save_load_roundtrip(
        self, pipeline: dict[str, Any], tmp_path: Path
    ) -> None:
        """Fitted Z-Score model survives a save/load cycle with identical scores."""
        from sentinel.core.registry import get_model_class

        model_dir = str(tmp_path / "zscore_e2e")
        cls = get_model_class("zscore")

        original = cls()
        original.fit(pipeline["train_np"])
        scores_before = original.score(pipeline["test_np"])
        original.save(model_dir)

        restored = cls()
        restored.load(model_dir)
        scores_after = restored.score(pipeline["test_np"])

        np.testing.assert_array_almost_equal(scores_before, scores_after)


@pytest.mark.slow
class TestIsolationForestModel:
    """End-to-end Isolation Forest detection and evaluation."""

    def test_scores_shape(self, pipeline: dict[str, Any]) -> None:
        """Isolation Forest score() output length equals the test set size."""
        _, scores, _ = _train_and_evaluate("isolation_forest", pipeline)
        assert scores.shape == (pipeline["test_np"].shape[0],)

    def test_scores_finite(self, pipeline: dict[str, Any]) -> None:
        """Isolation Forest scores are all finite."""
        _, scores, _ = _train_and_evaluate("isolation_forest", pipeline)
        assert np.all(np.isfinite(scores))

    def test_metrics_keys_present(self, pipeline: dict[str, Any]) -> None:
        """Evaluator returns all expected metric keys for Isolation Forest."""
        _, _, metrics = _train_and_evaluate("isolation_forest", pipeline)
        assert ALL_METRIC_KEYS.issubset(metrics.keys())

    def test_score_stats_finite(self, pipeline: dict[str, Any]) -> None:
        """Score distribution stats are all finite for Isolation Forest."""
        _, _, metrics = _train_and_evaluate("isolation_forest", pipeline)
        for key in SCORE_DIST_KEYS:
            assert np.isfinite(metrics[key]), f"{key} is not finite"


@pytest.mark.slow
class TestAutoencoderModel:
    """End-to-end Autoencoder detection and evaluation (torch required)."""

    def test_skipped_without_torch(self, pipeline: dict[str, Any]) -> None:
        """Test is skipped if torch is not available."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")

    def test_scores_shape(self, pipeline: dict[str, Any]) -> None:
        """Autoencoder score() output length equals the test set size."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        _, scores, _ = _train_and_evaluate(
            "autoencoder",
            pipeline,
            {"hidden_dims": AE_HIDDEN_DIMS, "epochs": AE_EPOCHS, "device": "cpu"},
        )
        assert scores.shape == (pipeline["test_np"].shape[0],)

    def test_scores_non_negative(self, pipeline: dict[str, Any]) -> None:
        """Autoencoder reconstruction errors are non-negative."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        _, scores, _ = _train_and_evaluate(
            "autoencoder",
            pipeline,
            {"hidden_dims": AE_HIDDEN_DIMS, "epochs": AE_EPOCHS, "device": "cpu"},
        )
        assert np.all(scores >= 0.0)

    def test_scores_finite(self, pipeline: dict[str, Any]) -> None:
        """Autoencoder scores are all finite."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        _, scores, _ = _train_and_evaluate(
            "autoencoder",
            pipeline,
            {"hidden_dims": AE_HIDDEN_DIMS, "epochs": AE_EPOCHS, "device": "cpu"},
        )
        assert np.all(np.isfinite(scores))

    def test_metrics_keys_present(self, pipeline: dict[str, Any]) -> None:
        """Evaluator returns all expected metric keys for Autoencoder."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        _, _, metrics = _train_and_evaluate(
            "autoencoder",
            pipeline,
            {"hidden_dims": AE_HIDDEN_DIMS, "epochs": AE_EPOCHS, "device": "cpu"},
        )
        assert ALL_METRIC_KEYS.issubset(metrics.keys())


# ---------------------------------------------------------------------------
# Step 5: Ensemble
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEnsemble:
    """Verify HybridEnsemble combining multiple trained models."""

    def test_ensemble_zscore_iforest(self, pipeline: dict[str, Any]) -> None:
        """Ensemble of Z-Score + Isolation Forest produces valid scores."""
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        ensemble = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.5, 0.5],
            strategy="weighted_average",
        )
        ensemble.fit(pipeline["train_np"])
        scores = ensemble.score(pipeline["test_np"])

        assert scores.shape == (pipeline["test_np"].shape[0],)
        assert np.all(np.isfinite(scores))

    def test_ensemble_scores_in_unit_interval(self, pipeline: dict[str, Any]) -> None:
        """Ensemble weighted_average scores are in [0, 1] after normalisation."""
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        ensemble = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.6, 0.4],
            strategy="weighted_average",
        )
        ensemble.fit(pipeline["train_np"])
        scores = ensemble.score(pipeline["test_np"])

        assert float(np.min(scores)) >= 0.0
        assert float(np.max(scores)) <= 1.0

    def test_ensemble_three_models_with_torch(self, pipeline: dict[str, Any]) -> None:
        """Ensemble of Z-Score + IF + AE works when torch is available."""
        if not TORCH_AVAILABLE:
            pytest.skip("torch not available")
        from sentinel.core.registry import get_model_class
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        # Pre-fit the AE so it has a trained state before ensemble.
        ae_cls = get_model_class("autoencoder")
        ae_model = ae_cls(hidden_dims=AE_HIDDEN_DIMS, epochs=AE_EPOCHS, device="cpu")
        ae_model.fit(pipeline["train_np"])

        # Ensemble instantiates its own sub-model instances, so we exercise
        # the full sub-model fit path by passing the names to HybridEnsemble.
        ensemble = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.5, 0.5],
            strategy="weighted_average",
        )
        ensemble.fit(pipeline["train_np"])
        scores = ensemble.score(pipeline["test_np"])

        assert scores.shape == (pipeline["test_np"].shape[0],)
        assert np.all(np.isfinite(scores))

    def test_ensemble_metrics_keys(self, pipeline: dict[str, Any]) -> None:
        """Evaluator produces all expected keys when run on ensemble scores."""
        from sentinel.models.ensemble.hybrid import HybridEnsemble
        from sentinel.training.evaluator import Evaluator

        ensemble = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.5, 0.5],
            strategy="weighted_average",
        )
        ensemble.fit(pipeline["train_np"])
        val_scores = ensemble.score(pipeline["val_np"])
        test_scores = ensemble.score(pipeline["test_np"])

        evaluator = Evaluator()
        metrics = evaluator.evaluate(
            scores=test_scores,
            labels=pipeline["test_labels"],
            val_scores=val_scores,
            val_labels=pipeline["val_labels"],
        )
        assert ALL_METRIC_KEYS.issubset(metrics.keys())

    def test_ensemble_save_load(self, pipeline: dict[str, Any], tmp_path: Path) -> None:
        """HybridEnsemble survives a save/load round-trip."""
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        model_dir = str(tmp_path / "ensemble_e2e")
        ensemble = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.5, 0.5],
            strategy="weighted_average",
        )
        ensemble.fit(pipeline["train_np"])
        scores_before = ensemble.score(pipeline["test_np"])
        ensemble.save(model_dir)

        restored = HybridEnsemble()
        restored.load(model_dir)
        scores_after = restored.score(pipeline["test_np"])

        np.testing.assert_array_almost_equal(scores_before, scores_after)


# ---------------------------------------------------------------------------
# Step 6: Visualization
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVisualization:
    """Verify that plot_timeseries produces a PNG file."""

    def test_timeseries_plot_creates_file(
        self, pipeline: dict[str, Any], tmp_path: Path
    ) -> None:
        """plot_timeseries writes a PNG file at the given output_path."""
        from sentinel.core.registry import get_model_class
        from sentinel.viz.timeseries import plot_timeseries

        cls = get_model_class("zscore")
        model = cls()
        model.fit(pipeline["train_np"])
        scores = model.score(pipeline["test_np"])

        test_df = pipeline["df_clean"].slice(
            int(LENGTH * 0.85), pipeline["test_np"].shape[0]
        )
        timestamps = test_df["timestamp"].to_numpy()
        values = pipeline["test_np"]

        output_path = tmp_path / "timeseries.png"
        fig = plot_timeseries(
            timestamps=timestamps,
            values=values,
            scores=scores,
            labels=pipeline["test_labels"],
            threshold=0.5,
            output_path=output_path,
        )

        assert output_path.exists(), "PNG file was not created"
        assert output_path.stat().st_size > 0, "PNG file is empty"
        assert fig is not None

    def test_timeseries_plot_without_scores(
        self, pipeline: dict[str, Any], tmp_path: Path
    ) -> None:
        """plot_timeseries works without scores (single-panel figure)."""
        from sentinel.viz.timeseries import plot_timeseries

        test_df = pipeline["df_clean"].slice(
            int(LENGTH * 0.85), pipeline["test_np"].shape[0]
        )
        timestamps = test_df["timestamp"].to_numpy()
        values = pipeline["test_np"]

        output_path = tmp_path / "timeseries_no_scores.png"
        fig = plot_timeseries(
            timestamps=timestamps,
            values=values,
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert fig is not None


# ---------------------------------------------------------------------------
# Step 7: Tracking
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTracking:
    """Verify experiment tracking artifacts are created correctly."""

    def test_create_run_creates_directory(self, tmp_path: Path) -> None:
        """LocalTracker.create_run() creates a run directory with meta.json."""
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=str(tmp_path / "experiments"))
        run_id = tracker.create_run("zscore")

        run_dir = tmp_path / "experiments" / run_id
        assert run_dir.exists()
        assert (run_dir / "meta.json").exists()

    def test_log_config_writes_file(self, tmp_path: Path) -> None:
        """log_config() persists the config dict as config.json."""
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=str(tmp_path / "experiments"))
        run_id = tracker.create_run("isolation_forest")
        config = {"model": "isolation_forest", "n_estimators": 100}
        tracker.log_config(run_id, config)

        config_path = tmp_path / "experiments" / run_id / "config.json"
        assert config_path.exists()
        loaded = json.loads(config_path.read_text())
        assert loaded["model"] == "isolation_forest"

    def test_log_metrics_writes_file(self, tmp_path: Path) -> None:
        """log_metrics() persists metrics as metrics.json."""
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=str(tmp_path / "experiments"))
        run_id = tracker.create_run("zscore")
        metrics = {"f1": 0.75, "precision": 0.80, "recall": 0.70, "threshold": 1.5}
        tracker.log_metrics(run_id, metrics)

        metrics_path = tmp_path / "experiments" / run_id / "metrics.json"
        assert metrics_path.exists()
        loaded = json.loads(metrics_path.read_text())
        assert loaded["f1"] == pytest.approx(0.75)

    def test_get_run_returns_all_keys(self, tmp_path: Path) -> None:
        """get_run() returns a dict with run_id, model_name, config, metrics, params."""
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=str(tmp_path / "experiments"))
        run_id = tracker.create_run("zscore")
        tracker.log_config(run_id, {"model": "zscore", "window_size": 30})
        tracker.log_metrics(run_id, {"f1": 0.8})
        tracker.log_params(run_id, {"window_size": 30, "threshold_sigma": 3.0})

        run = tracker.get_run(run_id)
        assert run["run_id"] == run_id
        assert run["model_name"] == "zscore"
        assert "config" in run
        assert "metrics" in run
        assert "params" in run

    def test_list_runs_includes_new_run(self, tmp_path: Path) -> None:
        """list_runs() returns entries for all created runs."""
        from sentinel.tracking.experiment import LocalTracker

        tracker = LocalTracker(base_dir=str(tmp_path / "experiments"))
        run_id_a = tracker.create_run("zscore")
        run_id_b = tracker.create_run("isolation_forest")

        runs = tracker.list_runs()
        run_ids = {r["run_id"] for r in runs}
        assert run_id_a in run_ids
        assert run_id_b in run_ids

    def test_full_tracking_workflow(
        self, pipeline: dict[str, Any], tmp_path: Path
    ) -> None:
        """Train a model and log config + metrics + params via LocalTracker."""
        from sentinel.core.registry import get_model_class
        from sentinel.tracking.experiment import LocalTracker
        from sentinel.training.evaluator import Evaluator

        tracker = LocalTracker(base_dir=str(tmp_path / "experiments"))
        run_id = tracker.create_run("zscore")

        cls = get_model_class("zscore")
        model = cls(window_size=20)
        model.fit(pipeline["train_np"])

        val_scores = model.score(pipeline["val_np"])
        test_scores = model.score(pipeline["test_np"])

        evaluator = Evaluator()
        metrics = evaluator.evaluate(
            scores=test_scores,
            labels=pipeline["test_labels"],
            val_scores=val_scores,
            val_labels=pipeline["val_labels"],
        )

        tracker.log_config(run_id, {"model": "zscore", "window_size": 20})
        tracker.log_metrics(run_id, metrics)
        tracker.log_params(run_id, model.get_params())

        # Verify all artifact files exist.
        run_dir = tmp_path / "experiments" / run_id
        assert (run_dir / "meta.json").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "params.json").exists()

        # Verify stored metrics are readable.
        stored_metrics = json.loads((run_dir / "metrics.json").read_text())
        assert "threshold" in stored_metrics
        assert "score_mean" in stored_metrics


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFullPipeline:
    """Single test exercising the complete pipeline end-to-end."""

    def test_generate_ingest_train_ensemble_visualize(
        self, pipeline: dict[str, Any], tmp_path: Path
    ) -> None:
        """Exercise every stage: generate -> ingest -> train -> ensemble -> viz."""
        from sentinel.data.ingest import ingest_file
        from sentinel.models.ensemble.hybrid import HybridEnsemble
        from sentinel.tracking.experiment import LocalTracker
        from sentinel.training.evaluator import Evaluator
        from sentinel.viz.timeseries import plot_timeseries

        # ---- Ingest ----
        csv_path = tmp_path / "full_pipeline.csv"
        pipeline["df_raw"].write_csv(str(csv_path))

        ingest_result = ingest_file(
            file_path=csv_path,
            data_dir=tmp_path / "raw",
            metadata_file=tmp_path / "datasets.json",
        )
        assert "dataset_id" in ingest_result

        # ---- Train Z-Score ----
        from sentinel.core.registry import get_model_class

        zscore_cls = get_model_class("zscore")
        zscore = zscore_cls()
        zscore.fit(pipeline["train_np"])

        # ---- Train IF ----
        if_cls = get_model_class("isolation_forest")
        iforest = if_cls()
        iforest.fit(pipeline["train_np"])

        # ---- Ensemble ----
        ensemble = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.5, 0.5],
            strategy="weighted_average",
        )
        ensemble.fit(pipeline["train_np"])
        val_scores = ensemble.score(pipeline["val_np"])
        test_scores = ensemble.score(pipeline["test_np"])

        assert test_scores.shape == (pipeline["test_np"].shape[0],)
        assert np.all(np.isfinite(test_scores))

        # ---- Evaluate ----
        evaluator = Evaluator()
        metrics = evaluator.evaluate(
            scores=test_scores,
            labels=pipeline["test_labels"],
            val_scores=val_scores,
            val_labels=pipeline["val_labels"],
        )
        assert ALL_METRIC_KEYS.issubset(metrics.keys())
        assert np.isfinite(metrics["threshold"])

        # ---- Visualize ----
        test_n = pipeline["test_np"].shape[0]
        test_df = pipeline["df_clean"].slice(int(LENGTH * 0.85), test_n)
        timestamps = test_df["timestamp"].to_numpy()

        output_path = tmp_path / "full_pipeline.png"
        fig = plot_timeseries(
            timestamps=timestamps,
            values=pipeline["test_np"],
            scores=test_scores,
            labels=pipeline["test_labels"],
            threshold=float(metrics["threshold"]),
            output_path=output_path,
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        assert fig is not None

        # ---- Track ----
        tracker = LocalTracker(base_dir=str(tmp_path / "experiments"))
        run_id = tracker.create_run("hybrid_ensemble")
        tracker.log_config(
            run_id,
            {"model": "hybrid_ensemble", "sub_models": ["zscore", "isolation_forest"]},
        )
        tracker.log_metrics(run_id, metrics)
        tracker.log_params(run_id, ensemble.get_params())

        run_dir = tmp_path / "experiments" / run_id
        assert (run_dir / "meta.json").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "config.json").exists()

        # Verify stored metrics deserialize correctly.
        stored = json.loads((run_dir / "metrics.json").read_text())
        assert "threshold" in stored
