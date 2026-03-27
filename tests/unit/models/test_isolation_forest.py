"""Unit tests for the Isolation Forest anomaly detector."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sentinel.models.statistical.isolation_forest import IsolationForestDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture()
def normal_data(rng: np.random.Generator) -> np.ndarray:
    """200 x 4 array drawn from a standard normal distribution."""
    return rng.standard_normal((200, 4))


@pytest.fixture()
def detector() -> IsolationForestDetector:
    """Default detector instance."""
    return IsolationForestDetector()


@pytest.fixture()
def fitted_detector(
    detector: IsolationForestDetector, normal_data: np.ndarray
) -> IsolationForestDetector:
    """Detector that has been fitted on *normal_data*."""
    detector.fit(normal_data)
    return detector


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    """Tests for IsolationForestDetector.fit()."""

    def test_fit_stores_model(
        self, detector: IsolationForestDetector, normal_data: np.ndarray
    ) -> None:
        """fit() should create the internal sklearn model."""
        assert detector._model is None
        detector.fit(normal_data)
        assert detector._model is not None

    def test_fit_records_n_features(
        self, detector: IsolationForestDetector, normal_data: np.ndarray
    ) -> None:
        """fit() should record the number of input features."""
        detector.fit(normal_data)
        assert detector._n_features == 4

    def test_fit_single_feature(
        self, detector: IsolationForestDetector, rng: np.random.Generator
    ) -> None:
        """fit() should work with a single feature column."""
        X = rng.standard_normal((50, 1))
        detector.fit(X)
        assert detector._n_features == 1

    def test_fit_minimal_rows(
        self, detector: IsolationForestDetector, rng: np.random.Generator
    ) -> None:
        """fit() should work with the minimum viable number of rows."""
        X = rng.standard_normal((2, 3))
        detector.fit(X)
        assert detector._model is not None

    def test_fit_empty_raises(self, detector: IsolationForestDetector) -> None:
        """fit() should raise SentinelError for an empty array."""
        from sentinel.core.exceptions import SentinelError

        with pytest.raises(SentinelError, match="at least 1 sample"):
            detector.fit(np.empty((0, 3)))


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore:
    """Tests for IsolationForestDetector.score()."""

    def test_score_shape(
        self,
        fitted_detector: IsolationForestDetector,
        normal_data: np.ndarray,
    ) -> None:
        """score() should return a 1-D array matching the sample count."""
        scores = fitted_detector.score(normal_data)
        assert scores.shape == (normal_data.shape[0],)

    def test_score_higher_for_anomalies(
        self,
        fitted_detector: IsolationForestDetector,
        rng: np.random.Generator,
    ) -> None:
        """Outliers far from the training distribution should score higher."""
        normal = rng.standard_normal((50, 4))
        anomalous = rng.standard_normal((10, 4)) + 100.0

        normal_scores = fitted_detector.score(normal)
        anomaly_scores = fitted_detector.score(anomalous)

        assert anomaly_scores.mean() > normal_scores.mean()

    def test_score_before_fit_raises(self, detector: IsolationForestDetector) -> None:
        """score() should raise when the model is not fitted."""
        from sentinel.core.exceptions import SentinelError

        with pytest.raises(SentinelError, match="not been fitted"):
            detector.score(np.ones((5, 3)))


# ---------------------------------------------------------------------------
# save() / load() round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for save() and load() serialisation round-trip."""

    def test_round_trip_scores_match(
        self,
        fitted_detector: IsolationForestDetector,
        normal_data: np.ndarray,
        tmp_path: Path,
    ) -> None:
        """A loaded model should produce identical scores to the original."""
        save_dir = str(tmp_path / "iso_model")
        fitted_detector.save(save_dir)

        loaded = IsolationForestDetector()
        loaded.load(save_dir)

        original_scores = fitted_detector.score(normal_data)
        loaded_scores = loaded.score(normal_data)

        np.testing.assert_array_equal(original_scores, loaded_scores)

    def test_round_trip_params_match(
        self,
        fitted_detector: IsolationForestDetector,
        tmp_path: Path,
    ) -> None:
        """Hyperparameters should be preserved across save/load."""
        save_dir = str(tmp_path / "iso_model")
        fitted_detector.save(save_dir)

        loaded = IsolationForestDetector()
        loaded.load(save_dir)

        assert loaded.get_params() == fitted_detector.get_params()

    def test_saved_files_exist(
        self,
        fitted_detector: IsolationForestDetector,
        tmp_path: Path,
    ) -> None:
        """save() should write model.joblib and config.json."""
        save_dir = tmp_path / "iso_model"
        fitted_detector.save(str(save_dir))

        assert (save_dir / "model.joblib").exists()
        assert (save_dir / "config.json").exists()

    def test_config_json_content(
        self,
        fitted_detector: IsolationForestDetector,
        tmp_path: Path,
    ) -> None:
        """config.json should contain expected keys and values."""
        save_dir = tmp_path / "iso_model"
        fitted_detector.save(str(save_dir))

        with open(save_dir / "config.json") as fh:
            data = json.load(fh)

        assert data["model_name"] == "isolation_forest"
        assert data["n_estimators"] == 100
        assert data["contamination"] == 0.05
        assert data["random_state"] == 42
        assert data["n_features"] == 4

    def test_load_missing_dir_raises(
        self, detector: IsolationForestDetector, tmp_path: Path
    ) -> None:
        """load() should raise when the config file is missing."""
        from sentinel.core.exceptions import SentinelError

        with pytest.raises(SentinelError, match="Config file not found"):
            detector.load(str(tmp_path / "nonexistent"))

    def test_save_unfitted_raises(
        self, detector: IsolationForestDetector, tmp_path: Path
    ) -> None:
        """save() should raise when the model has not been fitted."""
        from sentinel.core.exceptions import SentinelError

        with pytest.raises(SentinelError, match="unfitted"):
            detector.save(str(tmp_path / "unfitted"))

    def test_no_tmp_files_left(
        self,
        fitted_detector: IsolationForestDetector,
        tmp_path: Path,
    ) -> None:
        """Atomic writes should not leave .tmp artefacts behind."""
        save_dir = tmp_path / "iso_model"
        fitted_detector.save(str(save_dir))

        tmp_files = list(save_dir.glob("*.tmp"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# get_params()
# ---------------------------------------------------------------------------


class TestGetParams:
    """Tests for get_params()."""

    def test_default_params(self, detector: IsolationForestDetector) -> None:
        """get_params() should return documented default values."""
        params = detector.get_params()
        assert params == {
            "n_estimators": 100,
            "contamination": 0.05,
            "random_state": 42,
        }

    def test_custom_params(self) -> None:
        """get_params() should reflect constructor arguments."""
        det = IsolationForestDetector(
            n_estimators=200, contamination=0.1, random_state=99
        )
        assert det.get_params() == {
            "n_estimators": 200,
            "contamination": 0.1,
            "random_state": 99,
        }

    def test_expected_keys(self, detector: IsolationForestDetector) -> None:
        """get_params() must contain exactly the documented keys."""
        keys = set(detector.get_params().keys())
        assert keys == {"n_estimators", "contamination", "random_state"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Verify model is discoverable through the registry."""

    def test_registered_name(self) -> None:
        """The class attribute model_name should be 'isolation_forest'."""
        assert IsolationForestDetector.model_name == "isolation_forest"

    def test_registry_lookup(self) -> None:
        """get_model_class should find the detector by name."""
        from sentinel.core.registry import get_model_class

        cls = get_model_class("isolation_forest")
        assert cls is IsolationForestDetector
