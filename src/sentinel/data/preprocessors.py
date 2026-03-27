"""Data preprocessing: scaling, splitting, windowing."""

from __future__ import annotations

import numpy as np
import polars as pl

from sentinel.data.validators import get_feature_columns


def fill_nan(df: pl.DataFrame) -> pl.DataFrame:
    """Forward-fill then zero-fill NaN values in feature columns.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with NaN values filled.
    """
    feature_cols = get_feature_columns(df)
    return df.with_columns(
        [pl.col(c).forward_fill().fill_null(0.0) for c in feature_cols]
    )


ScaleStats = dict[str, tuple[float, float]]


def scale_zscore(df: pl.DataFrame) -> tuple[pl.DataFrame, ScaleStats]:
    """Z-score normalize feature columns: (x - mean) / std.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (scaled DataFrame, dict of column -> (mean, std) stats).
    """
    feature_cols = get_feature_columns(df)
    stats: dict[str, tuple[float, float]] = {}
    exprs = []

    for col_name in feature_cols:
        col = df.get_column(col_name).cast(pl.Float64)
        mean = col.mean() or 0.0
        std = col.std() or 1.0
        if std == 0.0:
            std = 1.0
        stats[col_name] = (mean, std)
        exprs.append(((pl.col(col_name).cast(pl.Float64) - mean) / std).alias(col_name))

    return df.with_columns(exprs), stats


def scale_minmax(df: pl.DataFrame) -> tuple[pl.DataFrame, ScaleStats]:
    """Min-max normalize feature columns to [0, 1].

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (scaled DataFrame, dict of column -> (min, max) stats).
    """
    feature_cols = get_feature_columns(df)
    stats: dict[str, tuple[float, float]] = {}
    exprs = []

    for col_name in feature_cols:
        col = df.get_column(col_name).cast(pl.Float64)
        col_min = col.min() or 0.0
        col_max = col.max() or 1.0
        rng = col_max - col_min
        if rng == 0.0:
            rng = 1.0
        stats[col_name] = (col_min, col_max)
        exprs.append(
            ((pl.col(col_name).cast(pl.Float64) - col_min) / rng).alias(col_name)
        )

    return df.with_columns(exprs), stats


def chronological_split(
    df: pl.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split DataFrame chronologically into train/val/test.

    Args:
        df: Sorted DataFrame.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    n = df.height
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.slice(0, train_end)
    val = df.slice(train_end, val_end - train_end)
    test = df.slice(val_end, n - val_end)

    return train, val, test


def to_numpy(df: pl.DataFrame) -> np.ndarray:
    """Convert feature columns to numpy array.

    Args:
        df: Input DataFrame.

    Returns:
        2D numpy array of shape (n_samples, n_features).
    """
    feature_cols = get_feature_columns(df)
    return df.select(feature_cols).to_numpy().astype(np.float64)


def create_windows(data: np.ndarray, seq_len: int, stride: int = 1) -> np.ndarray:
    """Create sliding windows from a 2D array.

    Args:
        data: 2D array of shape (n_samples, n_features).
        seq_len: Window/sequence length.
        stride: Step size between windows.

    Returns:
        3D array of shape (n_windows, seq_len, n_features).
    """
    n_samples = data.shape[0]
    if n_samples < seq_len:
        raise ValueError(f"Data length ({n_samples}) < seq_len ({seq_len})")

    indices = range(0, n_samples - seq_len + 1, stride)
    windows = np.array([data[i : i + seq_len] for i in indices])
    return windows
