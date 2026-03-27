"""Unit tests for the Z-Score anomaly detector."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.core.exceptions import SentinelError
from sentinel.models.statistical.zscore import ZScoreDetector

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def detector() -> ZScoreDetector:
    """Return a default ZScoreDetector."""
    return ZScoreDetector(window_size=10, threshold_sigma=3.0)


@pytest.fixture()
def normal_data() -> np.ndarray:
    """Synthetic normal data: 200 samples x 3 features, Gaussian."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 3))


@pytest.fixture()
def anomalous_data() -> np.ndarray:
    """Normal data with injected spikes at known indices."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((200, 3))
    # Inject obvious anomalies at indices 50, 100, 150.
    for idx in [50, 100, 150]:
        data[idx] = 20.0
    return data


# ------------------------------------------------------------------
# fit() tests
# ------------------------------------------------------------------


class TestFit:
    """Tests for ZScoreDetector.fit()."""

    def test_fit_stores_global_stats(
        self, detector: ZScoreDetector, normal_data: np.ndarray
    ) -> None:
        """fit() should populate global mean and std arrays."""
        detector.fit(normal_data)

        assert detector._is_fitted is True
        assert detector._global_mean is not None
        assert detector._global_std is not None
        assert detector._global_mean.shape == (3,)
        assert detector._global_std.shape == (3,)

    def test_fit_rejects_single_row(self, detector: ZScoreDetector) -> None:
        """fit() should raise on a single-row input."""
        with pytest.raises(SentinelError, match="at least 2 samples"):
            detector.fit(np.array([[1.0, 2.0]]))

    def test_fit_1d_input_reshaped(self, detector: ZScoreDetector) -> None:
        """fit() should accept a 1-D array (single feature)."""
        detector.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert detector._n_features == 1

    def test_fit_minimal_rows(self, detector: ZScoreDetector) -> None:
        """fit() should succeed with exactly 2 rows."""
        detector.fit(np.array([[1.0], [2.0]]))
        assert detector._is_fitted is True


# ------------------------------------------------------------------
# score() tests
# ------------------------------------------------------------------


class TestScore:
    """Tests for ZScoreDetector.score()."""

    def test_score_shape(
        self, detector: ZScoreDetector, normal_data: np.ndarray
    ) -> None:
        """score() should return a 1-D array with one score per sample."""
        detector.fit(normal_data)
        scores = detector.score(normal_data)

        assert scores.ndim == 1
        assert scores.shape[0] == normal_data.shape[0]

    def test_score_not_fitted_raises(self, detector: ZScoreDetector) -> None:
        """score() before fit() should raise."""
        with pytest.raises(SentinelError, match="not been fitted"):
            detector.score(np.zeros((10, 3)))

    def test_anomalies_score_higher(
        self, detector: ZScoreDetector, anomalous_data: np.ndarray
    ) -> None:
        """Injected spikes should produce higher scores than normal points."""
        detector.fit(anomalous_data)
        scores = detector.score(anomalous_data)

        spike_indices = [50, 100, 150]
        normal_indices = list(set(range(200)) - set(spike_indices))

        mean_spike = np.mean(scores[spike_indices])
        mean_normal = np.mean(scores[normal_indices])

        assert mean_spike > mean_normal

    def test_score_single_feature(self, detector: ZScoreDetector) -> None:
        """score() should work with a single feature column."""
        data = np.arange(50, dtype=np.float64).reshape(-1, 1)
        detector.fit(data)
        scores = detector.score(data)

        assert scores.shape == (50,)

    def test_score_1d_input(self, detector: ZScoreDetector) -> None:
        """score() should reshape 1-D input to (n, 1)."""
        data = np.arange(50, dtype=np.float64)
        detector.fit(data)
        scores = detector.score(data)

        assert scores.shape == (50,)

    def test_scores_are_nonnegative(
        self, detector: ZScoreDetector, normal_data: np.ndarray
    ) -> None:
        """Z-scores are absolute values, so all scores must be >= 0."""
        detector.fit(normal_data)
        scores = detector.score(normal_data)

        assert np.all(scores >= 0.0)


# ------------------------------------------------------------------
# save() / load() round-trip
# ------------------------------------------------------------------


class TestSaveLoad:
    """Tests for model persistence."""

    def test_roundtrip(
        self,
        detector: ZScoreDetector,
        normal_data: np.ndarray,
        tmp_path: str,
    ) -> None:
        """save() then load() should restore an equivalent model."""
        detector.fit(normal_data)
        original_scores = detector.score(normal_data)

        save_dir = str(tmp_path)
        detector.save(save_dir)

        loaded = ZScoreDetector()
        loaded.load(save_dir)
        loaded_scores = loaded.score(normal_data)

        np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_save_creates_files(
        self,
        detector: ZScoreDetector,
        normal_data: np.ndarray,
        tmp_path: str,
    ) -> None:
        """save() should create config.json and model.joblib."""
        import os

        detector.fit(normal_data)
        save_dir = str(tmp_path)
        detector.save(save_dir)

        assert os.path.isfile(os.path.join(save_dir, "config.json"))
        assert os.path.isfile(os.path.join(save_dir, "model.joblib"))

    def test_save_not_fitted_raises(
        self,
        detector: ZScoreDetector,
        tmp_path: str,
    ) -> None:
        """save() before fit() should raise."""
        with pytest.raises(SentinelError, match="not been fitted"):
            detector.save(str(tmp_path))

    def test_load_missing_config_raises(
        self,
        detector: ZScoreDetector,
        tmp_path: str,
    ) -> None:
        """load() should raise when config.json is missing."""
        with pytest.raises(SentinelError, match="Config not found"):
            detector.load(str(tmp_path))

    def test_load_missing_model_raises(
        self,
        detector: ZScoreDetector,
        normal_data: np.ndarray,
        tmp_path: str,
    ) -> None:
        """load() should raise when model.joblib is missing."""
        import json
        import os

        save_dir = str(tmp_path)
        # Write only config.json, no model.joblib.
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"window_size": 10, "threshold_sigma": 3.0, "n_features": 3}, f)

        with pytest.raises(SentinelError, match="Model file not found"):
            detector.load(save_dir)

    def test_roundtrip_preserves_params(
        self,
        tmp_path: str,
        normal_data: np.ndarray,
    ) -> None:
        """Loaded model should have identical hyperparameters."""
        original = ZScoreDetector(window_size=20, threshold_sigma=2.5)
        original.fit(normal_data)
        original.save(str(tmp_path))

        loaded = ZScoreDetector()
        loaded.load(str(tmp_path))

        assert loaded.window_size == 20
        assert loaded.threshold_sigma == 2.5
        assert loaded._n_features == normal_data.shape[1]


# ------------------------------------------------------------------
# get_params() tests
# ------------------------------------------------------------------


class TestGetParams:
    """Tests for get_params()."""

    def test_returns_expected_keys(self, detector: ZScoreDetector) -> None:
        """get_params() should contain all expected keys."""
        params = detector.get_params()

        assert "model_name" in params
        assert "window_size" in params
        assert "threshold_sigma" in params
        assert "n_features" in params

    def test_values_before_fit(self, detector: ZScoreDetector) -> None:
        """Before fit, n_features should be None."""
        params = detector.get_params()

        assert params["model_name"] == "zscore"
        assert params["window_size"] == 10
        assert params["threshold_sigma"] == 3.0
        assert params["n_features"] is None

    def test_values_after_fit(
        self, detector: ZScoreDetector, normal_data: np.ndarray
    ) -> None:
        """After fit, n_features should reflect the training data."""
        detector.fit(normal_data)
        params = detector.get_params()

        assert params["n_features"] == 3


# ------------------------------------------------------------------
# detect() (inherited concrete method)
# ------------------------------------------------------------------


class TestDetect:
    """Tests for the inherited detect() method."""

    def test_detect_returns_result(
        self, detector: ZScoreDetector, normal_data: np.ndarray
    ) -> None:
        """detect() should return scores, labels, and threshold."""
        detector.fit(normal_data)
        result = detector.detect(normal_data, threshold=3.0)

        assert "scores" in result
        assert "labels" in result
        assert "threshold" in result
        assert result["scores"].shape == (200,)
        assert result["labels"].shape == (200,)
        assert result["threshold"] == 3.0

    def test_detect_labels_binary(
        self, detector: ZScoreDetector, normal_data: np.ndarray
    ) -> None:
        """Labels from detect() should be 0 or 1."""
        detector.fit(normal_data)
        result = detector.detect(normal_data, threshold=3.0)

        unique_labels = set(np.unique(result["labels"]))
        assert unique_labels.issubset({0, 1})


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and boundary tests."""

    def test_constant_feature(self) -> None:
        """Constant columns should not cause division-by-zero errors."""
        data = np.ones((50, 2))
        data[:, 1] = np.arange(50)  # Second column varies.

        det = ZScoreDetector(window_size=5)
        det.fit(data)
        scores = det.score(data)

        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_large_window_exceeds_data(self) -> None:
        """Window larger than data should still produce scores (using global stats)."""
        data = np.random.default_rng(0).standard_normal((10, 2))
        det = ZScoreDetector(window_size=100)
        det.fit(data)
        scores = det.score(data)

        assert scores.shape == (10,)
        assert not np.any(np.isnan(scores))

    def test_minimal_2_rows(self) -> None:
        """Two-row dataset: fit and score should succeed."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        det = ZScoreDetector(window_size=5)
        det.fit(data)
        scores = det.score(data)

        assert scores.shape == (2,)
