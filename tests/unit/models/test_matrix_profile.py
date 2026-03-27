"""Unit tests for the Matrix Profile anomaly detector."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sentinel.core.exceptions import ValidationError
from sentinel.models.statistical.matrix_profile import MatrixProfileDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic(
    n_samples: int = 300,
    n_features: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a simple synthetic time series for testing."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_features).astype(np.float64)


def _make_synthetic_with_anomaly(
    n_samples: int = 500,
    n_features: int = 1,
    seed: int = 42,
    anomaly_start: int = 400,
    anomaly_magnitude: float = 10.0,
) -> np.ndarray:
    """Generate data with an obvious anomalous region."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n_samples, n_features).astype(np.float64)
    data[anomaly_start : anomaly_start + 20, :] += anomaly_magnitude
    return data


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for MatrixProfileDetector initialization."""

    def test_default_params(self) -> None:
        """Default parameters are set correctly."""
        model = MatrixProfileDetector()
        assert model.subsequence_length == 50
        assert model.max_rows == 100_000

    def test_custom_params(self) -> None:
        """Custom parameters are stored."""
        model = MatrixProfileDetector(subsequence_length=20, max_rows=5000)
        assert model.subsequence_length == 20
        assert model.max_rows == 5000

    def test_min_subsequence_length(self) -> None:
        """subsequence_length < 4 raises ValidationError."""
        with pytest.raises(ValidationError, match="subsequence_length must be >= 4"):
            MatrixProfileDetector(subsequence_length=3)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


class TestFit:
    """Tests for fit()."""

    def test_fit_stores_n_features(self) -> None:
        """fit() records the number of features."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=3)
        model.fit(X)
        assert model.n_features_ == 3

    def test_fit_stores_mean_score(self) -> None:
        """fit() computes a non-zero mean baseline score."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X)
        assert model.mean_score_ > 0

    def test_fit_too_short_data(self) -> None:
        """fit() raises when data is shorter than subsequence_length * 3."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=25, n_features=1)  # 25 < 10 * 3
        with pytest.raises(ValidationError, match="Data length"):
            model.fit(X)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore:
    """Tests for score()."""

    def test_score_shape_univariate(self) -> None:
        """score() returns array matching input length for 1 feature."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X)
        scores = model.score(X)
        assert scores.shape == (200,)

    def test_score_shape_multivariate(self) -> None:
        """score() returns array matching input length for multiple features."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=4)
        model.fit(X)
        scores = model.score(X)
        assert scores.shape == (200,)

    def test_score_single_feature(self) -> None:
        """Edge case: single feature column works correctly."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X)
        scores = model.score(X)
        assert scores.dtype == np.float64
        assert not np.any(np.isnan(scores))

    def test_score_no_nan(self) -> None:
        """Scores should not contain NaN values."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=2)
        model.fit(X)
        scores = model.score(X)
        assert not np.any(np.isnan(scores))

    def test_score_all_positive(self) -> None:
        """Matrix profile distances are non-negative."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X)
        scores = model.score(X)
        assert np.all(scores >= 0)

    def test_anomalous_region_scores_higher(self) -> None:
        """Injected anomalies should score higher than normal data."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic_with_anomaly(
            n_samples=500,
            anomaly_start=400,
            anomaly_magnitude=15.0,
        )
        model.fit(X)
        scores = model.score(X)

        normal_mean = np.mean(scores[:350])
        anomaly_mean = np.mean(scores[400:420])
        assert anomaly_mean > normal_mean

    def test_score_too_short_data(self) -> None:
        """score() raises when data is shorter than minimum."""
        model = MatrixProfileDetector(subsequence_length=10)
        X_train = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X_train)

        X_short = _make_synthetic(n_samples=25, n_features=1)
        with pytest.raises(ValidationError, match="Data length"):
            model.score(X_short)

    def test_score_minimal_rows(self) -> None:
        """score() works with exactly the minimum required rows."""
        subseq = 10
        min_rows = subseq * 3  # exactly 30
        model = MatrixProfileDetector(subsequence_length=subseq)
        X = _make_synthetic(n_samples=min_rows, n_features=1)
        model.fit(X)
        scores = model.score(X)
        assert scores.shape == (min_rows,)


# ---------------------------------------------------------------------------
# save() / load() round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Tests for save/load round-trip."""

    def test_round_trip(self, tmp_path: object) -> None:
        """Parameters survive a save/load cycle."""
        path = str(tmp_path)
        model = MatrixProfileDetector(subsequence_length=20, max_rows=5000)
        X = _make_synthetic(n_samples=200, n_features=2)
        model.fit(X)

        model.save(path)
        loaded = MatrixProfileDetector()
        loaded.load(path)

        assert loaded.subsequence_length == 20
        assert loaded.max_rows == 5000
        assert loaded.n_features_ == 2
        assert loaded.mean_score_ == pytest.approx(model.mean_score_)

    def test_scores_match_after_load(self, tmp_path: object) -> None:
        """Loaded model produces identical scores to the original."""
        path = str(tmp_path)
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X)
        original_scores = model.score(X)

        model.save(path)
        loaded = MatrixProfileDetector()
        loaded.load(path)
        loaded_scores = loaded.score(X)

        np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_config_json_written(self, tmp_path: object) -> None:
        """save() writes a readable config.json file."""
        path = str(tmp_path)
        model = MatrixProfileDetector(subsequence_length=15)
        X = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X)
        model.save(path)

        config_path = f"{path}/config.json"
        with open(config_path) as f:
            config = json.load(f)

        assert config["subsequence_length"] == 15

    def test_load_nonexistent(self, tmp_path: object) -> None:
        """load() raises FileNotFoundError for missing files."""
        model = MatrixProfileDetector()
        with pytest.raises(FileNotFoundError):
            model.load(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------------
# get_params()
# ---------------------------------------------------------------------------


class TestGetParams:
    """Tests for get_params()."""

    def test_expected_keys(self) -> None:
        """get_params() returns all expected keys."""
        model = MatrixProfileDetector(subsequence_length=25, max_rows=8000)
        params = model.get_params()
        assert "subsequence_length" in params
        assert "max_rows" in params
        assert "n_features_" in params
        assert "mean_score_" in params

    def test_values_match(self) -> None:
        """get_params() values match constructor arguments."""
        model = MatrixProfileDetector(subsequence_length=30, max_rows=9000)
        params = model.get_params()
        assert params["subsequence_length"] == 30
        assert params["max_rows"] == 9000

    def test_params_after_fit(self) -> None:
        """get_params() reflects fitted state."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=3)
        model.fit(X)
        params = model.get_params()
        assert params["n_features_"] == 3
        assert params["mean_score_"] > 0


# ---------------------------------------------------------------------------
# detect() (inherited from BaseAnomalyDetector)
# ---------------------------------------------------------------------------


class TestDetect:
    """Tests for the concrete detect() method from the base class."""

    def test_detect_returns_result(self) -> None:
        """detect() returns a DetectionResult with correct fields."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X)
        result = model.detect(X, threshold=2.0)

        assert "scores" in result
        assert "labels" in result
        assert "threshold" in result
        assert result["scores"].shape == (200,)
        assert result["labels"].shape == (200,)
        assert result["threshold"] == 2.0

    def test_detect_labels_are_binary(self) -> None:
        """Labels from detect() are 0 or 1."""
        model = MatrixProfileDetector(subsequence_length=10)
        X = _make_synthetic(n_samples=200, n_features=1)
        model.fit(X)
        result = model.detect(X, threshold=2.0)
        assert set(np.unique(result["labels"])).issubset({0, 1})


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunking:
    """Tests for large-data chunking behavior."""

    def test_chunked_output_shape(self) -> None:
        """Chunked scoring returns same shape as non-chunked."""
        subseq = 10
        n_samples = 500
        model = MatrixProfileDetector(
            subsequence_length=subseq,
            max_rows=200,
        )
        X = _make_synthetic(n_samples=n_samples, n_features=1)
        model.fit(X)
        scores = model.score(X)
        assert scores.shape == (n_samples,)

    def test_chunked_no_nan(self) -> None:
        """Chunked scoring should not produce NaN values."""
        model = MatrixProfileDetector(
            subsequence_length=10,
            max_rows=150,
        )
        X = _make_synthetic(n_samples=500, n_features=1)
        model.fit(X)
        scores = model.score(X)
        assert not np.any(np.isnan(scores))

    def test_chunked_no_inf(self) -> None:
        """Chunked scoring should not produce infinite values."""
        model = MatrixProfileDetector(
            subsequence_length=10,
            max_rows=150,
        )
        X = _make_synthetic(n_samples=500, n_features=1)
        model.fit(X)
        scores = model.score(X)
        assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Tests that the model is correctly registered in the Sentinel registry."""

    def test_registered_name(self) -> None:
        """Model is registered as 'matrix_profile'."""
        from sentinel.core.registry import get_model_class

        cls = get_model_class("matrix_profile")
        assert cls is MatrixProfileDetector

    def test_model_name_attribute(self) -> None:
        """model_name class attribute is set by the decorator."""
        assert MatrixProfileDetector.model_name == "matrix_profile"
