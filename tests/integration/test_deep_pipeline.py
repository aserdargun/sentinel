"""Integration test: Phase 3 deep learning model pipeline.

Covers: synthetic data -> validate -> preprocess -> fit -> score for
autoencoder, lstm, lstm_ae, and tcn.  Also tests autoencoder save/load
round-trip and Trainer.run() end-to-end with an autoencoder config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Skip the entire module when torch is not installed.
pytest.importorskip("torch")

import sentinel.models  # noqa: F401 — triggers registration of all built-in models  # noqa: E402
from sentinel.core.config import RunConfig  # noqa: E402
from sentinel.core.registry import get_model_class  # noqa: E402
from sentinel.data.preprocessors import (  # noqa: E402
    chronological_split,
    fill_nan,
    scale_zscore,
    to_numpy,
)
from sentinel.data.synthetic import generate_synthetic  # noqa: E402
from sentinel.data.validators import separate_labels, validate_dataframe  # noqa: E402
from sentinel.training.trainer import Trainer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — keep epochs tiny so tests finish in a few seconds
# ---------------------------------------------------------------------------

N_FEATURES = 3
LENGTH = 200
SEED = 42
SEQ_LEN = 10

# Per-model small-param overrides for fast training
_MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "autoencoder": {
        "hidden_dims": [8, 4],
        "epochs": 2,
        "batch_size": 16,
        "device": "cpu",
    },
    "lstm": {
        "hidden_dim": 8,
        "num_layers": 1,
        "seq_len": SEQ_LEN,
        "epochs": 2,
        "batch_size": 16,
        "device": "auto",
    },
    "lstm_ae": {
        "encoder_dim": 8,
        "decoder_dim": 8,
        "latent_dim": 4,
        "seq_len": SEQ_LEN,
        "epochs": 2,
        "batch_size": 16,
        "device": "auto",
    },
    "tcn": {
        "num_channels": [8, 8],
        "kernel_size": 3,
        "seq_len": SEQ_LEN,
        "epochs": 2,
        "batch_size": 16,
        "device": "auto",
    },
}


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the preprocessing pipeline once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_data() -> dict[str, Any]:
    """Run the full preprocessing pipeline once and expose splits.

    Returns a dict with keys:
        train_np  - numpy training array
        val_np    - numpy validation array
        test_np   - numpy test array
        val_labels_np  - numpy validation labels (int64)
        test_labels_np - numpy test labels (int64)
    """
    df = generate_synthetic(n_features=N_FEATURES, length=LENGTH, seed=SEED)

    df_clean, labels = separate_labels(df)
    df_validated = validate_dataframe(df_clean)
    df_filled = fill_nan(df_validated)
    df_scaled, _ = scale_zscore(df_filled)

    train_df, val_df, test_df = chronological_split(df_scaled)

    train_np = to_numpy(train_df)
    val_np = to_numpy(val_df)
    test_np = to_numpy(test_df)

    n = LENGTH
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    assert labels is not None
    labels_np = labels.to_numpy().astype(np.int64)

    return {
        "train_np": train_np,
        "val_np": val_np,
        "test_np": test_np,
        "val_labels_np": labels_np[train_end:val_end],
        "test_labels_np": labels_np[val_end:],
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_model(
    model_name: str,
    data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Instantiate, fit, and score a registered deep model.

    Args:
        model_name: Registry key for the model.
        data: Output of the pipeline_data fixture.

    Returns:
        Tuple of (val_scores, test_scores) as numpy arrays.
    """
    ModelClass = get_model_class(model_name)
    model = ModelClass(**_MODEL_KWARGS[model_name])
    model.fit(data["train_np"])
    val_scores = model.score(data["val_np"])
    test_scores = model.score(data["test_np"])
    return val_scores, test_scores


# ---------------------------------------------------------------------------
# Tests: autoencoder
# ---------------------------------------------------------------------------


class TestAutoencoderPipeline:
    """Full pipeline tests for the Autoencoder model."""

    def test_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """score() output length matches the test set size."""
        _, test_scores = _run_model("autoencoder", pipeline_data)
        assert test_scores.shape == (pipeline_data["test_np"].shape[0],)

    def test_val_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """score() output length matches the validation set size."""
        val_scores, _ = _run_model("autoencoder", pipeline_data)
        assert val_scores.shape == (pipeline_data["val_np"].shape[0],)

    def test_scores_finite(self, pipeline_data: dict[str, Any]) -> None:
        """Autoencoder scores contain no NaN or Inf values."""
        val_scores, test_scores = _run_model("autoencoder", pipeline_data)
        assert np.all(np.isfinite(val_scores)), "NaN/Inf in val_scores"
        assert np.all(np.isfinite(test_scores)), "NaN/Inf in test_scores"

    def test_scores_non_negative(self, pipeline_data: dict[str, Any]) -> None:
        """Reconstruction MSE scores are always >= 0."""
        _, test_scores = _run_model("autoencoder", pipeline_data)
        assert np.all(test_scores >= 0.0)

    def test_scores_dtype(self, pipeline_data: dict[str, Any]) -> None:
        """Returned scores are float64."""
        _, test_scores = _run_model("autoencoder", pipeline_data)
        assert test_scores.dtype == np.float64

    def test_get_params_after_fit(self, pipeline_data: dict[str, Any]) -> None:
        """get_params() reflects n_features after fitting."""
        ModelClass = get_model_class("autoencoder")
        model = ModelClass(**_MODEL_KWARGS["autoencoder"])
        model.fit(pipeline_data["train_np"])
        params = model.get_params()
        assert params["n_features"] == N_FEATURES


# ---------------------------------------------------------------------------
# Tests: lstm
# ---------------------------------------------------------------------------


class TestLSTMPipeline:
    """Full pipeline tests for the LSTM predictor model."""

    def test_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """score() output length matches the test set size."""
        _, test_scores = _run_model("lstm", pipeline_data)
        assert test_scores.shape == (pipeline_data["test_np"].shape[0],)

    def test_val_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """score() output length matches the validation set size."""
        val_scores, _ = _run_model("lstm", pipeline_data)
        assert val_scores.shape == (pipeline_data["val_np"].shape[0],)

    def test_scores_finite(self, pipeline_data: dict[str, Any]) -> None:
        """LSTM scores contain no NaN or Inf values."""
        val_scores, test_scores = _run_model("lstm", pipeline_data)
        assert np.all(np.isfinite(val_scores)), "NaN/Inf in val_scores"
        assert np.all(np.isfinite(test_scores)), "NaN/Inf in test_scores"

    def test_scores_non_negative(self, pipeline_data: dict[str, Any]) -> None:
        """Prediction error MSE scores are always >= 0."""
        _, test_scores = _run_model("lstm", pipeline_data)
        assert np.all(test_scores >= 0.0)

    def test_scores_dtype(self, pipeline_data: dict[str, Any]) -> None:
        """Returned scores are float64."""
        _, test_scores = _run_model("lstm", pipeline_data)
        assert test_scores.dtype == np.float64

    def test_get_params_after_fit(self, pipeline_data: dict[str, Any]) -> None:
        """get_params() reflects n_features and seq_len after fitting."""
        ModelClass = get_model_class("lstm")
        model = ModelClass(**_MODEL_KWARGS["lstm"])
        model.fit(pipeline_data["train_np"])
        params = model.get_params()
        assert params["n_features"] == N_FEATURES
        assert params["seq_len"] == SEQ_LEN


# ---------------------------------------------------------------------------
# Tests: lstm_ae
# ---------------------------------------------------------------------------


class TestLSTMAEPipeline:
    """Full pipeline tests for the LSTM Autoencoder model."""

    def test_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """score() output length matches the test set size."""
        _, test_scores = _run_model("lstm_ae", pipeline_data)
        assert test_scores.shape == (pipeline_data["test_np"].shape[0],)

    def test_val_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """score() output length matches the validation set size."""
        val_scores, _ = _run_model("lstm_ae", pipeline_data)
        assert val_scores.shape == (pipeline_data["val_np"].shape[0],)

    def test_scores_finite(self, pipeline_data: dict[str, Any]) -> None:
        """LSTM-AE scores contain no NaN or Inf values."""
        val_scores, test_scores = _run_model("lstm_ae", pipeline_data)
        assert np.all(np.isfinite(val_scores)), "NaN/Inf in val_scores"
        assert np.all(np.isfinite(test_scores)), "NaN/Inf in test_scores"

    def test_scores_non_negative(self, pipeline_data: dict[str, Any]) -> None:
        """Reconstruction MSE scores are always >= 0."""
        _, test_scores = _run_model("lstm_ae", pipeline_data)
        assert np.all(test_scores >= 0.0)

    def test_scores_numeric_dtype(self, pipeline_data: dict[str, Any]) -> None:
        """Returned scores are a floating-point numpy array."""
        _, test_scores = _run_model("lstm_ae", pipeline_data)
        assert np.issubdtype(test_scores.dtype, np.floating)

    def test_get_params_after_fit(self, pipeline_data: dict[str, Any]) -> None:
        """get_params() reflects n_features and latent_dim after fitting."""
        ModelClass = get_model_class("lstm_ae")
        model = ModelClass(**_MODEL_KWARGS["lstm_ae"])
        model.fit(pipeline_data["train_np"])
        params = model.get_params()
        assert params["n_features"] == N_FEATURES
        assert params["latent_dim"] == 4


# ---------------------------------------------------------------------------
# Tests: tcn
# ---------------------------------------------------------------------------


class TestTCNPipeline:
    """Full pipeline tests for the TCN model."""

    def test_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """score() output length matches the test set size."""
        _, test_scores = _run_model("tcn", pipeline_data)
        assert test_scores.shape == (pipeline_data["test_np"].shape[0],)

    def test_val_scores_shape(self, pipeline_data: dict[str, Any]) -> None:
        """score() output length matches the validation set size."""
        val_scores, _ = _run_model("tcn", pipeline_data)
        assert val_scores.shape == (pipeline_data["val_np"].shape[0],)

    def test_scores_finite(self, pipeline_data: dict[str, Any]) -> None:
        """TCN scores contain no NaN or Inf values."""
        val_scores, test_scores = _run_model("tcn", pipeline_data)
        assert np.all(np.isfinite(val_scores)), "NaN/Inf in val_scores"
        assert np.all(np.isfinite(test_scores)), "NaN/Inf in test_scores"

    def test_scores_non_negative(self, pipeline_data: dict[str, Any]) -> None:
        """Reconstruction MSE scores are always >= 0."""
        _, test_scores = _run_model("tcn", pipeline_data)
        assert np.all(test_scores >= 0.0)

    def test_scores_dtype(self, pipeline_data: dict[str, Any]) -> None:
        """Returned scores are float64."""
        _, test_scores = _run_model("tcn", pipeline_data)
        assert test_scores.dtype == np.float64

    def test_get_params_after_fit(self, pipeline_data: dict[str, Any]) -> None:
        """get_params() reflects n_features and num_channels after fitting."""
        ModelClass = get_model_class("tcn")
        model = ModelClass(**_MODEL_KWARGS["tcn"])
        model.fit(pipeline_data["train_np"])
        params = model.get_params()
        assert params["n_features"] == N_FEATURES
        assert params["num_channels"] == [8, 8]


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestDeepRegistryIntegration:
    """Verify all Phase 3 deep models are registered and instantiable."""

    @pytest.mark.parametrize("model_name", ["autoencoder", "lstm", "lstm_ae", "tcn"])
    def test_model_registered(self, model_name: str) -> None:
        """Each deep model is retrievable from the registry."""
        cls = get_model_class(model_name)
        assert cls is not None

    @pytest.mark.parametrize("model_name", ["autoencoder", "lstm", "lstm_ae", "tcn"])
    def test_model_instantiates(self, model_name: str) -> None:
        """Registry class instantiates without arguments."""
        cls = get_model_class(model_name)
        instance = cls()
        assert instance is not None

    @pytest.mark.parametrize("model_name", ["autoencoder", "lstm", "lstm_ae", "tcn"])
    def test_model_has_get_params(self, model_name: str) -> None:
        """get_params() returns a dict before fitting."""
        cls = get_model_class(model_name)
        instance = cls()
        params = instance.get_params()
        assert isinstance(params, dict)


# ---------------------------------------------------------------------------
# Autoencoder save/load round-trip
# ---------------------------------------------------------------------------


class TestAutoencoderSaveLoad:
    """Verify that a fitted AutoencoderDetector survives a save/load cycle."""

    def test_save_creates_artifacts(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """save() writes config.json and model.pt into the target directory."""
        model_dir = str(tmp_path / "ae_model")
        ModelClass = get_model_class("autoencoder")
        model = ModelClass(**_MODEL_KWARGS["autoencoder"])
        model.fit(pipeline_data["train_np"])
        model.save(model_dir)

        assert (Path(model_dir) / "config.json").exists()
        assert (Path(model_dir) / "model.pt").exists()

    def test_loaded_scores_match(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Scores from a loaded model are identical to the original model."""
        model_dir = str(tmp_path / "ae_roundtrip")
        ModelClass = get_model_class("autoencoder")

        original = ModelClass(**_MODEL_KWARGS["autoencoder"])
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
        model_dir = str(tmp_path / "ae_params")
        ModelClass = get_model_class("autoencoder")

        original = ModelClass(
            hidden_dims=[8, 4],
            epochs=2,
            batch_size=16,
            device="cpu",
        )
        original.fit(pipeline_data["train_np"])
        original.save(model_dir)

        restored = ModelClass()
        restored.load(model_dir)

        assert restored.hidden_dims == [8, 4]
        assert restored._n_features == N_FEATURES

    def test_loaded_model_scores_finite(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """A loaded model produces finite scores on test data."""
        model_dir = str(tmp_path / "ae_loaded_scores")
        ModelClass = get_model_class("autoencoder")

        model = ModelClass(**_MODEL_KWARGS["autoencoder"])
        model.fit(pipeline_data["train_np"])
        model.save(model_dir)

        restored = ModelClass()
        restored.load(model_dir)
        scores = restored.score(pipeline_data["test_np"])

        assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# Trainer.run() integration — autoencoder config written to tmp YAML
# ---------------------------------------------------------------------------


class TestTrainerDeepPipeline:
    """Verify Trainer.run() works end-to-end with an autoencoder config."""

    def test_trainer_run_returns_train_result(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Trainer.run() returns a TrainResult with required keys."""
        # Write a tiny synthetic parquet that Trainer can load.
        data_file = tmp_path / "synth.parquet"
        df = generate_synthetic(n_features=N_FEATURES, length=LENGTH, seed=SEED)
        df.write_parquet(str(data_file))

        # Write a minimal autoencoder YAML config.
        config_file = tmp_path / "ae_test.yaml"
        config_file.write_text(
            f"""\
model: autoencoder
data_path: {data_file}
hidden_dims: [8, 4]
epochs: 2
batch_size: 16
device: cpu
"""
        )

        config = RunConfig.from_yaml(config_file)
        trainer = Trainer(config)
        result = trainer.run()

        assert result["run_id"] != ""
        assert result["model_name"] == "autoencoder"
        assert isinstance(result["metrics"], dict)
        assert result["duration_s"] >= 0.0

    def test_trainer_run_metrics_have_threshold(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Trainer.run() metrics include a finite threshold value."""
        data_file = tmp_path / "synth2.parquet"
        df = generate_synthetic(n_features=N_FEATURES, length=LENGTH, seed=SEED)
        df.write_parquet(str(data_file))

        config_file = tmp_path / "ae_test2.yaml"
        config_file.write_text(
            f"""\
model: autoencoder
data_path: {data_file}
hidden_dims: [8, 4]
epochs: 2
batch_size: 16
device: cpu
"""
        )

        config = RunConfig.from_yaml(config_file)
        trainer = Trainer(config)
        result = trainer.run()

        metrics = result["metrics"]
        assert "threshold" in metrics
        assert np.isfinite(metrics["threshold"])

    def test_trainer_run_score_stats_finite(
        self, pipeline_data: dict[str, Any], tmp_path: Path
    ) -> None:
        """Score distribution stats returned by Trainer.run() are all finite."""
        data_file = tmp_path / "synth3.parquet"
        df = generate_synthetic(n_features=N_FEATURES, length=LENGTH, seed=SEED)
        df.write_parquet(str(data_file))

        config_file = tmp_path / "ae_test3.yaml"
        config_file.write_text(
            f"""\
model: autoencoder
data_path: {data_file}
hidden_dims: [8, 4]
epochs: 2
batch_size: 16
device: cpu
"""
        )

        config = RunConfig.from_yaml(config_file)
        trainer = Trainer(config)
        result = trainer.run()

        metrics = result["metrics"]
        stat_keys = ("score_mean", "score_std", "score_p50", "score_p95", "score_p99")
        for stat_key in stat_keys:
            assert stat_key in metrics, f"Missing key: {stat_key}"
            assert np.isfinite(metrics[stat_key]), f"{stat_key} is not finite"
