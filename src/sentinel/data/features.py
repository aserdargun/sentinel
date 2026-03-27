"""Feature engineering using native Polars expressions."""

from __future__ import annotations

import polars as pl

from sentinel.data.validators import get_feature_columns


def add_lags(df: pl.DataFrame, lags: list[int]) -> pl.DataFrame:
    """Add lagged feature columns for each feature and lag value.

    For each feature column and each lag value, creates a new column
    ``{col}_lag_{n}`` using ``pl.col().shift(n)``. Leading nulls from
    the shift are forward-filled then zero-filled.

    Args:
        df: Input DataFrame with timestamp and feature columns.
        lags: List of positive integer lag values (number of rows to shift).

    Returns:
        DataFrame with original columns plus lagged feature columns.
    """
    feature_cols = get_feature_columns(df)
    lag_exprs: list[pl.Expr] = []

    for col_name in feature_cols:
        for lag in lags:
            lag_exprs.append(
                pl.col(col_name)
                .shift(lag)
                .forward_fill()
                .fill_null(0.0)
                .alias(f"{col_name}_lag_{lag}")
            )

    if lag_exprs:
        df = df.with_columns(lag_exprs)

    return df


def add_rolling_stats(df: pl.DataFrame, window_size: int) -> pl.DataFrame:
    """Add rolling statistics for each feature column.

    For each feature column, creates four new columns:
      - ``{col}_rolling_mean``  (rolling mean)
      - ``{col}_rolling_std``   (rolling standard deviation)
      - ``{col}_rolling_min``   (rolling minimum)
      - ``{col}_rolling_max``   (rolling maximum)

    Leading nulls from the rolling window are forward-filled then zero-filled.

    Args:
        df: Input DataFrame with timestamp and feature columns.
        window_size: Size of the rolling window in number of rows.

    Returns:
        DataFrame with original columns plus rolling statistic columns.
    """
    feature_cols = get_feature_columns(df)
    rolling_exprs: list[pl.Expr] = []

    for col_name in feature_cols:
        rolling_exprs.extend(
            [
                pl.col(col_name)
                .rolling_mean(window_size=window_size)
                .alias(f"{col_name}_rolling_mean"),
                pl.col(col_name)
                .rolling_std(window_size=window_size)
                .alias(f"{col_name}_rolling_std"),
                pl.col(col_name)
                .rolling_min(window_size=window_size)
                .alias(f"{col_name}_rolling_min"),
                pl.col(col_name)
                .rolling_max(window_size=window_size)
                .alias(f"{col_name}_rolling_max"),
            ]
        )

    if rolling_exprs:
        df = df.with_columns(rolling_exprs)

        # Forward-fill then zero-fill leading nulls from the rolling window
        new_cols = [
            f"{col_name}_{stat}"
            for col_name in feature_cols
            for stat in ("rolling_mean", "rolling_std", "rolling_min", "rolling_max")
        ]
        fill_exprs = [
            pl.col(c).forward_fill().fill_null(0.0).alias(c) for c in new_cols
        ]
        df = df.with_columns(fill_exprs)

    return df


def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extract temporal features from the timestamp column.

    Creates three new columns:
      - ``hour``        : hour of day (0-23)
      - ``day_of_week`` : day of the week (0=Monday .. 6=Sunday)
      - ``is_weekend``  : 1 if Saturday or Sunday, 0 otherwise

    Args:
        df: Input DataFrame with a ``timestamp`` column of type ``pl.Datetime``.

    Returns:
        DataFrame with original columns plus temporal feature columns.
    """
    df = df.with_columns(
        [
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
        ]
    )
    df = df.with_columns(
        (pl.col("day_of_week") >= 6).cast(pl.Int64).alias("is_weekend"),
    )

    return df
