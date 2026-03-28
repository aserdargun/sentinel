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


# Minimum number of data points required for meaningful FFT output.
_MIN_FFT_LENGTH = 4


def add_fft_features(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    top_k: int = 3,
) -> pl.DataFrame:
    """Add FFT-based seasonality features for selected columns.

    For each specified feature column, computes the real FFT via
    ``numpy.fft.rfft`` and extracts:

      - ``{col}_fft_freq_{i}`` : the *i*-th dominant frequency (1-indexed,
        sorted by descending power, excluding DC component).
      - ``{col}_fft_power_{i}`` : the spectral power at that frequency.
      - ``{col}_fft_energy``    : total spectral energy (sum of squared
        magnitudes, excluding DC).

    The extracted values are scalar per column and are broadcast across all
    rows of the returned DataFrame (they characterise the entire series,
    not individual points).

    NaN values in a column are forward-filled then zero-filled before FFT
    computation.  If a column has fewer than ``_MIN_FFT_LENGTH`` non-null
    values after filling, FFT features for that column are set to 0.0.

    Args:
        df: Input DataFrame with timestamp and feature columns.
        columns: Feature column names to process.  When ``None``, all
            feature columns (as determined by ``get_feature_columns``)
            are used.
        top_k: Number of dominant frequencies to extract.  Must be >= 1.

    Returns:
        DataFrame with original columns plus FFT feature columns.

    Raises:
        ValueError: If ``top_k`` < 1.
    """
    import numpy as np

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    feature_cols = columns if columns is not None else get_feature_columns(df)

    if not feature_cols:
        return df

    new_columns: dict[str, list[float]] = {}
    n_rows = df.height

    for col_name in feature_cols:
        values = (
            df.get_column(col_name)
            .forward_fill()
            .fill_null(0.0)
            .fill_nan(0.0)
            .to_numpy()
        )

        if len(values) < _MIN_FFT_LENGTH:
            for i in range(1, top_k + 1):
                new_columns[f"{col_name}_fft_freq_{i}"] = [0.0] * n_rows
                new_columns[f"{col_name}_fft_power_{i}"] = [0.0] * n_rows
            new_columns[f"{col_name}_fft_energy"] = [0.0] * n_rows
            continue

        n = len(values)
        fft_result = np.fft.rfft(values)
        magnitudes = np.abs(fft_result)
        freqs = np.fft.rfftfreq(n)

        magnitudes_no_dc = magnitudes[1:]
        freqs_no_dc = freqs[1:]
        power_no_dc = magnitudes_no_dc**2

        total_energy = float(np.sum(power_no_dc))
        new_columns[f"{col_name}_fft_energy"] = [total_energy] * n_rows

        available_k = min(top_k, len(power_no_dc))
        if available_k > 0:
            top_indices = np.argsort(power_no_dc)[::-1][:available_k]
            top_freqs = freqs_no_dc[top_indices]
            top_powers = power_no_dc[top_indices]
        else:
            top_freqs = np.array([])
            top_powers = np.array([])

        for i in range(1, top_k + 1):
            if i <= available_k:
                new_columns[f"{col_name}_fft_freq_{i}"] = [
                    float(top_freqs[i - 1])
                ] * n_rows
                new_columns[f"{col_name}_fft_power_{i}"] = [
                    float(top_powers[i - 1])
                ] * n_rows
            else:
                new_columns[f"{col_name}_fft_freq_{i}"] = [0.0] * n_rows
                new_columns[f"{col_name}_fft_power_{i}"] = [0.0] * n_rows

    if new_columns:
        df = df.with_columns(
            [pl.Series(name, vals) for name, vals in new_columns.items()]
        )

    return df
