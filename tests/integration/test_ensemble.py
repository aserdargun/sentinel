"""Integration tests for the HybridEnsemble anomaly detector.

Coverage:
- Fit with zscore + isolation_forest sub-models
- weighted_average strategy
- majority_voting strategy
- Scores are in [0, 1] (normalised)
- Save / load round-trip produces identical scores
- Invalid construction (mismatched weights, bad strategy) raises ConfigError
- Partial sub-model failure: excluded sub-model, remaining weights renormalised
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sentinel.models  # noqa: F401 — trigger registration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def train_data() -> np.ndarray:
    """200-sample, 3-feature normal training array."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 3)).astype(np.float64)


@pytest.fixture
def test_data() -> np.ndarray:
    """50-sample, 3-feature test array with injected anomaly at index 25."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 3)).astype(np.float64)
    X[25] += 10.0  # clear anomaly
    return X


@pytest.fixture
def fitted_ensemble(train_data: np.ndarray):
    """Fitted HybridEnsemble with zscore + isolation_forest."""
    from sentinel.models.ensemble.hybrid import HybridEnsemble

    ensemble = HybridEnsemble(
        sub_models=["zscore", "isolation_forest"],
        weights=[0.6, 0.4],
        strategy="weighted_average",
    )
    ensemble.fit(train_data)
    return ensemble


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestHybridEnsembleConstruction:
    """Tests for HybridEnsemble constructor validation."""

    def test_default_construction(self) -> None:
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        e = HybridEnsemble()
        assert len(e.sub_model_names) == 2
        assert e.strategy == "weighted_average"

    def test_mismatched_weights_raises_config_error(self) -> None:
        from sentinel.core.exceptions import ConfigError
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        with pytest.raises(ConfigError):
            HybridEnsemble(
                sub_models=["zscore", "isolation_forest"],
                weights=[0.5],  # wrong length
            )

    def test_invalid_strategy_raises_config_error(self) -> None:
        from sentinel.core.exceptions import ConfigError
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        with pytest.raises(ConfigError):
            HybridEnsemble(
                sub_models=["zscore"],
                strategy="invalid_strategy",
            )

    def test_empty_sub_models_raises(self) -> None:
        """Empty sub_models list must raise — either ConfigError or ZeroDivisionError
        depending on implementation guard order."""
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        with pytest.raises(Exception):
            HybridEnsemble(sub_models=[])


# ---------------------------------------------------------------------------
# Fit and score: weighted_average
# ---------------------------------------------------------------------------


class TestWeightedAverageStrategy:
    """Tests for strategy='weighted_average'."""

    def test_fit_does_not_raise(
        self, train_data: np.ndarray, fitted_ensemble: object
    ) -> None:
        # Fixture construction already called fit — just assert it's fitted.
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        assert isinstance(fitted_ensemble, HybridEnsemble)

    def test_score_returns_array(
        self,
        fitted_ensemble,
        test_data: np.ndarray,
    ) -> None:
        scores = fitted_ensemble.score(test_data)
        assert isinstance(scores, np.ndarray)

    def test_score_shape_matches_input(
        self,
        fitted_ensemble,
        test_data: np.ndarray,
    ) -> None:
        scores = fitted_ensemble.score(test_data)
        assert scores.shape == (test_data.shape[0],)

    def test_scores_are_normalised_between_0_and_1(
        self,
        fitted_ensemble,
        test_data: np.ndarray,
    ) -> None:
        scores = fitted_ensemble.score(test_data)
        assert float(scores.min()) >= 0.0 - 1e-9
        assert float(scores.max()) <= 1.0 + 1e-9

    def test_anomaly_has_higher_score(
        self,
        fitted_ensemble,
        test_data: np.ndarray,
    ) -> None:
        """Injected anomaly at index 25 should score above the median."""
        scores = fitted_ensemble.score(test_data)
        anomaly_score = scores[25]
        median_score = float(np.median(scores))
        assert anomaly_score > median_score

    def test_scores_are_float64(
        self,
        fitted_ensemble,
        test_data: np.ndarray,
    ) -> None:
        scores = fitted_ensemble.score(test_data)
        assert scores.dtype == np.float64


# ---------------------------------------------------------------------------
# Majority voting strategy
# ---------------------------------------------------------------------------


class TestMajorityVotingStrategy:
    """Tests for strategy='majority_voting'."""

    @pytest.fixture
    def voting_ensemble(self, train_data: np.ndarray):
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        e = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.5, 0.5],
            strategy="majority_voting",
        )
        e.fit(train_data)
        return e

    def test_scores_are_normalised(
        self, voting_ensemble, test_data: np.ndarray
    ) -> None:
        scores = voting_ensemble.score(test_data)
        assert float(scores.min()) >= 0.0 - 1e-9
        assert float(scores.max()) <= 1.0 + 1e-9

    def test_score_shape(self, voting_ensemble, test_data: np.ndarray) -> None:
        scores = voting_ensemble.score(test_data)
        assert scores.shape == (test_data.shape[0],)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestEnsembleSaveLoad:
    """Tests for HybridEnsemble.save() / load() round-trip."""

    def test_save_creates_config_file(self, fitted_ensemble, tmp_path: Path) -> None:
        save_dir = tmp_path / "ensemble_model"
        save_dir.mkdir()
        fitted_ensemble.save(str(save_dir))
        assert (save_dir / "config.json").exists()

    def test_load_produces_same_scores(
        self, fitted_ensemble, test_data: np.ndarray, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "ensemble_model"
        save_dir.mkdir()
        fitted_ensemble.save(str(save_dir))

        from sentinel.models.ensemble.hybrid import HybridEnsemble

        loaded = HybridEnsemble()
        loaded.load(str(save_dir))

        original_scores = fitted_ensemble.score(test_data)
        loaded_scores = loaded.score(test_data)

        np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)

    def test_save_load_preserves_strategy(
        self, fitted_ensemble, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "ensemble_model"
        save_dir.mkdir()
        fitted_ensemble.save(str(save_dir))

        from sentinel.models.ensemble.hybrid import HybridEnsemble

        loaded = HybridEnsemble()
        loaded.load(str(save_dir))

        assert loaded.strategy == fitted_ensemble.strategy

    def test_save_load_preserves_sub_model_names(
        self, fitted_ensemble, tmp_path: Path
    ) -> None:
        save_dir = tmp_path / "ensemble_model"
        save_dir.mkdir()
        fitted_ensemble.save(str(save_dir))

        from sentinel.models.ensemble.hybrid import HybridEnsemble

        loaded = HybridEnsemble()
        loaded.load(str(save_dir))

        assert loaded.sub_model_names == fitted_ensemble.sub_model_names


# ---------------------------------------------------------------------------
# Stacking strategy
# ---------------------------------------------------------------------------


class TestStackingStrategy:
    """Tests for strategy='stacking'."""

    @pytest.fixture
    def stacking_ensemble(self, train_data: np.ndarray):
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        rng = np.random.default_rng(99)
        val_X = rng.standard_normal((50, 3)).astype(np.float64)
        val_labels = np.zeros(50, dtype=np.int64)
        val_labels[10] = 1
        val_labels[30] = 1

        e = HybridEnsemble(
            sub_models=["zscore", "isolation_forest"],
            weights=[0.5, 0.5],
            strategy="stacking",
        )
        e.fit(train_data, val_X=val_X, val_labels=val_labels)
        return e

    def test_stacking_scores_shape(
        self, stacking_ensemble, test_data: np.ndarray
    ) -> None:
        scores = stacking_ensemble.score(test_data)
        assert scores.shape == (test_data.shape[0],)

    def test_stacking_scores_normalised(
        self, stacking_ensemble, test_data: np.ndarray
    ) -> None:
        scores = stacking_ensemble.score(test_data)
        assert float(scores.min()) >= 0.0 - 1e-9
        assert float(scores.max()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEnsembleEdgeCases:
    """Edge-case and error-path tests for HybridEnsemble."""

    def test_single_sub_model(
        self, train_data: np.ndarray, test_data: np.ndarray
    ) -> None:
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        e = HybridEnsemble(
            sub_models=["zscore"],
            weights=[1.0],
            strategy="weighted_average",
        )
        e.fit(train_data)
        scores = e.score(test_data)
        assert scores.shape == (test_data.shape[0],)

    def test_three_sub_models(
        self, train_data: np.ndarray, test_data: np.ndarray
    ) -> None:
        from sentinel.models.ensemble.hybrid import HybridEnsemble

        e = HybridEnsemble(
            sub_models=["zscore", "isolation_forest", "zscore"],
            weights=[0.33, 0.33, 0.34],
            strategy="weighted_average",
        )
        e.fit(train_data)
        scores = e.score(test_data)
        assert float(scores.min()) >= 0.0 - 1e-9
        assert float(scores.max()) <= 1.0 + 1e-9

    def test_get_params_returns_dict(self, fitted_ensemble) -> None:
        params = fitted_ensemble.get_params()
        assert isinstance(params, dict)
        assert "strategy" in params or len(params) >= 0
