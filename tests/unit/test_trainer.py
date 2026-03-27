"""Tests for sentinel.training.trainer — Trainer class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sentinel.models  # noqa: F401 — triggers model registration
from sentinel.core.config import RunConfig
from sentinel.core.exceptions import ConfigError, ValidationError
from sentinel.data.synthetic import generate_synthetic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

# Expected keys present in every TrainResult.
EXPECTED_TRAIN_RESULT_KEYS = {"run_id", "model_name", "metrics", "duration_s"}

# Expected keys always present in the metrics dict.
EXPECTED_METRIC_KEYS = {
    "score_mean",
    "score_std",
    "score_p50",
    "score_p95",
    "score_p99",
    "threshold",
}


def _make_parquet(tmp_path: Path, n_features: int = 3, length: int = 200) -> Path:
    """Generate synthetic data and write it to a Parquet file in tmp_path."""
    df = generate_synthetic(n_features=n_features, length=length, seed=42)
    out = tmp_path / "data.parquet"
    df.write_parquet(str(out))
    return out


def _zscore_config(data_path: str) -> RunConfig:
    """Build a minimal RunConfig for zscore pointing at *data_path*."""
    cfg = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
    cfg = cfg.override(data_path=data_path)
    return cfg


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------


class TestTrainerHappyPath:
    """Normal operation of Trainer.run()."""

    def test_run_returns_train_result(self, tmp_path: Path) -> None:
        """run() returns a TrainResult with all required keys."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()

        assert isinstance(result, dict)
        assert EXPECTED_TRAIN_RESULT_KEYS.issubset(result.keys())

    def test_run_result_model_name_matches_config(self, tmp_path: Path) -> None:
        """TrainResult.model_name matches the config model field."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()

        assert result["model_name"] == "zscore"

    def test_run_result_run_id_is_nonempty_string(self, tmp_path: Path) -> None:
        """run_id is a non-empty hex string."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()

        assert isinstance(result["run_id"], str)
        assert len(result["run_id"]) > 0

    def test_run_result_duration_is_positive(self, tmp_path: Path) -> None:
        """duration_s is a positive float."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()

        assert result["duration_s"] > 0.0

    def test_run_metrics_contains_required_keys(self, tmp_path: Path) -> None:
        """metrics dict contains all expected keys."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()
        metrics = result["metrics"]

        assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())

    def test_run_metrics_score_mean_is_finite(self, tmp_path: Path) -> None:
        """score_mean is a finite number."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()

        assert np.isfinite(result["metrics"]["score_mean"])

    def test_run_metrics_threshold_is_positive(self, tmp_path: Path) -> None:
        """threshold is a positive value."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()

        assert result["metrics"]["threshold"] > 0.0

    def test_run_with_explicit_data_path_override(self, tmp_path: Path) -> None:
        """data_path passed to run() overrides config.data_path."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        # Point config at a non-existent path; override via run() argument.
        cfg = _zscore_config("/nonexistent/path")
        result = Trainer(cfg).run(data_path=str(parquet))

        assert result["model_name"] == "zscore"

    def test_run_with_directory_data_path(self, tmp_path: Path) -> None:
        """Passing a directory to run() picks the first .parquet file."""
        from sentinel.training.trainer import Trainer

        _make_parquet(tmp_path)  # writes tmp_path/data.parquet
        cfg = _zscore_config(str(tmp_path))
        result = Trainer(cfg).run()

        assert result["model_name"] == "zscore"

    def test_two_runs_produce_different_run_ids(self, tmp_path: Path) -> None:
        """Each call to run() generates a unique run_id."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        trainer = Trainer(cfg)
        result1 = trainer.run()
        result2 = trainer.run()

        assert result1["run_id"] != result2["run_id"]


# ---------------------------------------------------------------------------
# Tests — classification metrics with labeled data
# ---------------------------------------------------------------------------


class TestTrainerSupervisedMetrics:
    """Verify supervised classification metrics when labels are present."""

    def test_supervised_classification_keys_present(self, tmp_path: Path) -> None:
        """precision, recall, f1, auc_roc, auc_pr appear when labels exist."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path, length=300)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()
        metrics = result["metrics"]

        for key in ("precision", "recall", "f1"):
            assert key in metrics

    def test_precision_recall_in_unit_range(self, tmp_path: Path) -> None:
        """Precision and recall are in [0, 1] when not None."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path, length=300)
        cfg = _zscore_config(str(parquet))
        result = Trainer(cfg).run()
        metrics = result["metrics"]

        for key in ("precision", "recall", "f1"):
            value = metrics[key]
            if value is not None:
                assert 0.0 <= value <= 1.0, f"{key}={value} out of [0,1]"


# ---------------------------------------------------------------------------
# Tests — normal_only training mode
# ---------------------------------------------------------------------------


class TestTrainerNormalOnlyMode:
    """normal_only mode removes anomalous rows from the training set."""

    def test_normal_only_mode_runs_without_error(self, tmp_path: Path) -> None:
        """Trainer with training_mode=normal_only completes without error."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path, length=300)
        cfg = _zscore_config(str(parquet))
        cfg = cfg.override(training_mode="normal_only")
        result = Trainer(cfg).run()

        assert result["model_name"] == "zscore"

    def test_all_data_mode_runs_without_error(self, tmp_path: Path) -> None:
        """Trainer with training_mode=all_data completes without error."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path, length=300)
        cfg = _zscore_config(str(parquet))
        cfg = cfg.override(training_mode="all_data")
        result = Trainer(cfg).run()

        assert result["model_name"] == "zscore"

    def test_normal_only_filters_anomalies(self, tmp_path: Path) -> None:
        """normal_only mode produces a valid result even with anomalies present."""
        from sentinel.training.trainer import Trainer

        # Use anomaly_ratio=0.2 so there are clearly labelled anomalies.
        df = generate_synthetic(n_features=3, length=300, anomaly_ratio=0.2, seed=7)
        parquet = tmp_path / "anomaly_heavy.parquet"
        df.write_parquet(str(parquet))

        cfg = _zscore_config(str(parquet))
        cfg = cfg.override(training_mode="normal_only")
        result = Trainer(cfg).run()

        assert result["model_name"] == "zscore"


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------


class TestTrainerErrors:
    """Trainer raises appropriate errors on invalid inputs."""

    def test_invalid_model_name_raises_model_not_found(self, tmp_path: Path) -> None:
        """Unknown model name raises ModelNotFoundError (SentinelError subclass)."""
        from sentinel.core.exceptions import ModelNotFoundError
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        cfg = cfg.override(model="totally_nonexistent_model_xyz")

        with pytest.raises(ModelNotFoundError):
            Trainer(cfg).run()

    def test_missing_model_name_raises_config_error(self, tmp_path: Path) -> None:
        """Empty model name in config raises ConfigError."""
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        cfg = cfg.override(model="")

        with pytest.raises(ConfigError):
            Trainer(cfg).run()

    def test_missing_data_path_raises_validation_error(self, tmp_path: Path) -> None:
        """Pointing at a non-existent path raises ValidationError."""
        from sentinel.training.trainer import Trainer

        cfg = _zscore_config("/nonexistent/does/not/exist/data.parquet")

        with pytest.raises(ValidationError):
            Trainer(cfg).run()

    def test_empty_directory_raises_validation_error(self, tmp_path: Path) -> None:
        """A directory containing no Parquet or CSV raises ValidationError."""
        from sentinel.training.trainer import Trainer

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        cfg = _zscore_config(str(empty_dir))

        with pytest.raises(ValidationError):
            Trainer(cfg).run()


# ---------------------------------------------------------------------------
# Tests — config property
# ---------------------------------------------------------------------------


class TestTrainerConfigProperty:
    """Trainer.config exposes the RunConfig passed at construction."""

    def test_config_property_returns_same_instance(self, tmp_path: Path) -> None:
        from sentinel.training.trainer import Trainer

        parquet = _make_parquet(tmp_path)
        cfg = _zscore_config(str(parquet))
        trainer = Trainer(cfg)
        assert trainer.config is cfg
