"""Tests for sentinel.training.callbacks — EarlyStopping and ModelCheckpoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sentinel.training.callbacks import EarlyStopping, ModelCheckpoint, get_callbacks

# ---------------------------------------------------------------------------
# EarlyStopping — basic behavior
# ---------------------------------------------------------------------------


class TestEarlyStoppingInit:
    """Construction and attribute defaults."""

    def test_default_patience_is_ten(self) -> None:
        es = EarlyStopping()
        assert es.patience == 10

    def test_default_mode_is_min(self) -> None:
        es = EarlyStopping()
        assert es.mode == "min"

    def test_default_delta_is_zero(self) -> None:
        es = EarlyStopping()
        assert es.delta == pytest.approx(0.0)

    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="invalid")

    def test_initial_best_is_none(self) -> None:
        es = EarlyStopping()
        assert es.best is None

    def test_initial_counter_is_zero(self) -> None:
        es = EarlyStopping()
        assert es.counter == 0

    def test_initial_stopped_is_false(self) -> None:
        es = EarlyStopping()
        assert es.stopped is False


class TestEarlyStoppingMinMode:
    """EarlyStopping with mode='min' — lower is better."""

    def test_first_check_always_returns_false(self) -> None:
        es = EarlyStopping(patience=3, mode="min")
        assert es.check(1.0) is False

    def test_improvement_resets_counter(self) -> None:
        es = EarlyStopping(patience=3, mode="min")
        es.check(1.0)  # best = 1.0
        es.check(0.9)  # improvement
        assert es.counter == 0

    def test_no_improvement_increments_counter(self) -> None:
        es = EarlyStopping(patience=3, mode="min")
        es.check(1.0)  # best = 1.0
        es.check(1.1)  # no improvement
        assert es.counter == 1

    def test_triggers_after_patience_exceeded(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es.check(1.0)  # best = 1.0
        es.check(1.1)  # counter = 1
        result = es.check(1.2)  # counter = 2 → trigger
        assert result is True

    def test_stopped_property_set_after_trigger(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es.check(1.0)
        es.check(1.1)
        es.check(1.2)
        assert es.stopped is True

    def test_does_not_trigger_with_steady_improvement(self) -> None:
        es = EarlyStopping(patience=3, mode="min")
        for val in [1.0, 0.9, 0.8, 0.7, 0.6]:
            stopped = es.check(val)
        assert stopped is False
        assert es.stopped is False

    def test_best_updated_on_improvement(self) -> None:
        es = EarlyStopping(patience=5, mode="min")
        es.check(1.0)
        es.check(0.5)
        assert es.best == pytest.approx(0.5)

    def test_best_not_updated_on_no_improvement(self) -> None:
        es = EarlyStopping(patience=5, mode="min")
        es.check(1.0)
        es.check(2.0)
        assert es.best == pytest.approx(1.0)


class TestEarlyStoppingMaxMode:
    """EarlyStopping with mode='max' — higher is better."""

    def test_improvement_higher_value_resets_counter(self) -> None:
        es = EarlyStopping(patience=3, mode="max")
        es.check(0.5)  # best = 0.5
        es.check(0.8)  # improvement
        assert es.counter == 0

    def test_no_improvement_lower_value_increments_counter(self) -> None:
        es = EarlyStopping(patience=3, mode="max")
        es.check(0.8)  # best = 0.8
        es.check(0.7)  # no improvement
        assert es.counter == 1

    def test_triggers_after_patience_in_max_mode(self) -> None:
        es = EarlyStopping(patience=2, mode="max")
        es.check(0.8)
        es.check(0.7)
        result = es.check(0.6)
        assert result is True

    def test_steady_increase_never_triggers(self) -> None:
        es = EarlyStopping(patience=2, mode="max")
        for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            stopped = es.check(val)
        assert stopped is False


class TestEarlyStoppingDelta:
    """Delta threshold requires a minimum improvement."""

    def test_improvement_below_delta_not_counted(self) -> None:
        es = EarlyStopping(patience=3, delta=0.1, mode="min")
        es.check(1.0)  # best = 1.0
        es.check(0.95)  # 0.05 improvement < delta=0.1 → no improvement counted
        assert es.counter == 1

    def test_improvement_at_delta_boundary_counted(self) -> None:
        es = EarlyStopping(patience=3, delta=0.1, mode="min")
        es.check(1.0)
        es.check(0.89)  # 0.11 improvement > delta=0.1 → counted as improvement
        assert es.counter == 0


class TestEarlyStoppingReset:
    """reset() restores initial state."""

    def test_reset_clears_counter(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es.check(1.0)
        es.check(2.0)
        es.check(3.0)  # trigger
        es.reset()
        assert es.counter == 0

    def test_reset_clears_best(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es.check(0.5)
        es.reset()
        assert es.best is None

    def test_reset_clears_stopped(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es.check(1.0)
        es.check(2.0)
        es.check(3.0)  # trigger
        assert es.stopped is True
        es.reset()
        assert es.stopped is False

    def test_check_after_reset_behaves_fresh(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es.check(1.0)
        es.check(2.0)
        es.check(3.0)  # trigger
        es.reset()
        # After reset, first check initialises best again.
        result = es.check(0.5)
        assert result is False
        assert es.best == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ModelCheckpoint
# ---------------------------------------------------------------------------


class TestModelCheckpointInit:
    """Construction and attribute defaults."""

    def test_default_mode_is_min(self) -> None:
        ckpt = ModelCheckpoint()
        assert ckpt.mode == "min"

    def test_default_delta_is_zero(self) -> None:
        ckpt = ModelCheckpoint()
        assert ckpt.delta == pytest.approx(0.0)

    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            ModelCheckpoint(mode="bad")

    def test_initial_best_is_none(self) -> None:
        ckpt = ModelCheckpoint()
        assert ckpt.best is None

    def test_initial_save_count_is_zero(self) -> None:
        ckpt = ModelCheckpoint()
        assert ckpt.save_count == 0


class TestModelCheckpointSave:
    """ModelCheckpoint.check() delegates save to the model."""

    def _make_mock_model(self) -> MagicMock:
        """Return a mock that satisfies BaseAnomalyDetector.save()."""
        mock = MagicMock()
        mock.model_name = "mock_model"
        return mock

    def test_first_check_always_saves(self, tmp_path: Path) -> None:
        ckpt = ModelCheckpoint(mode="min")
        mock_model = self._make_mock_model()
        result = ckpt.check(1.0, mock_model, str(tmp_path / "ckpt"))
        assert result is True
        mock_model.save.assert_called_once()

    def test_save_count_increments_on_first_save(self, tmp_path: Path) -> None:
        ckpt = ModelCheckpoint(mode="min")
        mock_model = self._make_mock_model()
        ckpt.check(1.0, mock_model, str(tmp_path / "ckpt"))
        assert ckpt.save_count == 1

    def test_improvement_triggers_save_in_min_mode(self, tmp_path: Path) -> None:
        ckpt = ModelCheckpoint(mode="min")
        mock_model = self._make_mock_model()
        ckpt.check(1.0, mock_model, str(tmp_path / "ckpt"))
        result = ckpt.check(0.5, mock_model, str(tmp_path / "ckpt"))
        assert result is True
        assert ckpt.save_count == 2

    def test_no_improvement_does_not_save(self, tmp_path: Path) -> None:
        ckpt = ModelCheckpoint(mode="min")
        mock_model = self._make_mock_model()
        ckpt.check(1.0, mock_model, str(tmp_path / "ckpt"))
        result = ckpt.check(2.0, mock_model, str(tmp_path / "ckpt"))
        assert result is False
        assert ckpt.save_count == 1

    def test_improvement_triggers_save_in_max_mode(self, tmp_path: Path) -> None:
        ckpt = ModelCheckpoint(mode="max")
        mock_model = self._make_mock_model()
        ckpt.check(0.5, mock_model, str(tmp_path / "ckpt"))
        result = ckpt.check(0.9, mock_model, str(tmp_path / "ckpt"))
        assert result is True

    def test_best_updated_after_improvement(self, tmp_path: Path) -> None:
        ckpt = ModelCheckpoint(mode="min")
        mock_model = self._make_mock_model()
        ckpt.check(1.0, mock_model, str(tmp_path / "ckpt"))
        ckpt.check(0.3, mock_model, str(tmp_path / "ckpt"))
        assert ckpt.best == pytest.approx(0.3)

    def test_creates_save_directory(self, tmp_path: Path) -> None:
        """check() creates the save directory if it does not exist."""
        ckpt = ModelCheckpoint(mode="min")
        mock_model = self._make_mock_model()
        save_dir = tmp_path / "nested" / "subdir"
        ckpt.check(1.0, mock_model, str(save_dir))
        assert save_dir.exists()


class TestModelCheckpointReset:
    """reset() restores initial state."""

    def test_reset_clears_best(self, tmp_path: Path) -> None:
        ckpt = ModelCheckpoint(mode="min")
        mock_model = MagicMock()
        mock_model.model_name = "mock"
        ckpt.check(0.5, mock_model, str(tmp_path / "ckpt"))
        ckpt.reset()
        assert ckpt.best is None

    def test_reset_clears_save_count(self, tmp_path: Path) -> None:
        ckpt = ModelCheckpoint(mode="min")
        mock_model = MagicMock()
        mock_model.model_name = "mock"
        ckpt.check(0.5, mock_model, str(tmp_path / "ckpt"))
        ckpt.reset()
        assert ckpt.save_count == 0


# ---------------------------------------------------------------------------
# get_callbacks helper
# ---------------------------------------------------------------------------


class TestGetCallbacks:
    """get_callbacks() builds callback instances from config dict."""

    def test_empty_config_returns_empty_dict(self) -> None:
        callbacks = get_callbacks({})
        assert callbacks == {}

    def test_early_stopping_created_from_config(self) -> None:
        cfg = {"early_stopping": {"patience": 5, "mode": "min"}}
        callbacks = get_callbacks(cfg)
        assert "early_stopping" in callbacks
        es = callbacks["early_stopping"]
        assert isinstance(es, EarlyStopping)
        assert es.patience == 5

    def test_checkpoint_created_from_config(self) -> None:
        cfg = {"checkpoint": {"mode": "max", "delta": 0.01}}
        callbacks = get_callbacks(cfg)
        assert "checkpoint" in callbacks
        ckpt = callbacks["checkpoint"]
        assert isinstance(ckpt, ModelCheckpoint)
        assert ckpt.mode == "max"

    def test_both_callbacks_created_together(self) -> None:
        cfg = {
            "early_stopping": {"patience": 3},
            "checkpoint": {"mode": "min"},
        }
        callbacks = get_callbacks(cfg)
        assert "early_stopping" in callbacks
        assert "checkpoint" in callbacks

    def test_defaults_applied_when_keys_missing(self) -> None:
        cfg = {"early_stopping": {}}
        callbacks = get_callbacks(cfg)
        es = callbacks["early_stopping"]
        assert es.patience == 10  # default
        assert es.mode == "min"  # default
