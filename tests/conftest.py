"""Shared pytest fixtures for the Sentinel test suite."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sentinel.data.synthetic import generate_synthetic


@pytest.fixture
def synthetic_df() -> pl.DataFrame:
    """Minimal valid multivariate DataFrame with labels for testing.

    Returns a DataFrame with timestamp, feature_1..3, and is_anomaly columns.
    """
    return generate_synthetic(n_features=3, length=200, seed=42)


@pytest.fixture
def synthetic_df_no_labels(synthetic_df: pl.DataFrame) -> pl.DataFrame:
    """Same as synthetic_df but with is_anomaly column dropped."""
    return synthetic_df.drop("is_anomaly")


@pytest.fixture
def synthetic_numpy(synthetic_df: pl.DataFrame) -> np.ndarray:
    """Numpy array of feature columns from synthetic_df (no timestamp, no labels).

    Returns:
        2D array of shape (200, 3).
    """
    feature_cols = [c for c in synthetic_df.columns if c.startswith("feature_")]
    return synthetic_df.select(feature_cols).to_numpy()


@pytest.fixture
def tmp_experiment_dir(tmp_path: Path) -> Path:
    """Temporary directory for experiment artifacts.

    Uses pytest's tmp_path fixture to ensure cleanup after each test.
    """
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir
