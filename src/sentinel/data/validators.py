"""Data validation using native Polars expressions."""

from __future__ import annotations

import polars as pl

from sentinel.core.exceptions import ValidationError

RESERVED_COLUMNS = {"is_anomaly"}
MIN_ROWS = 2


def validate_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Validate a DataFrame against Sentinel's canonical schema.

    Args:
        df: Input DataFrame to validate.

    Returns:
        The validated DataFrame (timestamp cast to Datetime if needed).

    Raises:
        ValidationError: If any check fails.
    """
    if df.height < MIN_ROWS:
        raise ValidationError(
            f"Dataset must have at least {MIN_ROWS} rows, got {df.height}"
        )

    if df.columns[0] != "timestamp":
        raise ValidationError(
            f"First column must be 'timestamp', got '{df.columns[0]}'"
        )

    df = _ensure_datetime(df)
    _check_sorted(df)
    _check_no_duplicates(df)
    _check_numeric_features(df)
    _check_no_all_nan(df)
    _check_no_constant(df)

    return df


def _ensure_datetime(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure timestamp column is pl.Datetime with UTC timezone."""
    ts_dtype = df.schema["timestamp"]

    if ts_dtype == pl.String or ts_dtype == pl.Utf8:
        try:
            df = df.with_columns(
                pl.col("timestamp").str.to_datetime().alias("timestamp")
            )
        except Exception as e:
            raise ValidationError(f"Cannot parse 'timestamp' as datetime: {e}") from e
        ts_dtype = df.schema["timestamp"]

    if not isinstance(ts_dtype, pl.Datetime):
        raise ValidationError(f"'timestamp' must be Datetime, got {ts_dtype}")

    if ts_dtype.time_zone is None:
        df = df.with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp")
        )
    elif ts_dtype.time_zone != "UTC":
        df = df.with_columns(
            pl.col("timestamp").dt.convert_time_zone("UTC").alias("timestamp")
        )

    if ts_dtype.time_unit != "us":
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")).alias("timestamp")
        )

    return df


def _check_sorted(df: pl.DataFrame) -> None:
    """Check that timestamps are sorted chronologically."""
    ts = df.get_column("timestamp")
    if not ts.is_sorted():
        raise ValidationError("Timestamps must be sorted chronologically")


def _check_no_duplicates(df: pl.DataFrame) -> None:
    """Check for duplicate timestamps."""
    ts = df.get_column("timestamp")
    if ts.n_unique() != ts.len():
        dup_count = ts.len() - ts.n_unique()
        dups = (
            df.group_by("timestamp")
            .len()
            .filter(pl.col("len") > 1)
            .head(5)
            .get_column("timestamp")
            .to_list()
        )
        raise ValidationError(
            f"Found {dup_count} duplicate timestamps. First offenders: {dups}"
        )


def _check_numeric_features(df: pl.DataFrame) -> None:
    """Check that all feature columns are numeric."""
    feature_cols = get_feature_columns(df)
    if not feature_cols:
        raise ValidationError(
            "No feature columns found (need at least one numeric column)"
        )

    for col_name in feature_cols:
        dtype = df.schema[col_name]
        if dtype not in (
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ):
            raise ValidationError(f"Column '{col_name}' must be numeric, got {dtype}")


def _check_no_all_nan(df: pl.DataFrame) -> None:
    """Reject feature columns where every value is null/NaN."""
    feature_cols = get_feature_columns(df)
    for col_name in feature_cols:
        null_count = df.get_column(col_name).null_count()
        if null_count == df.height:
            raise ValidationError(f"Column '{col_name}' is all NaN/null — remove it")


def _check_no_constant(df: pl.DataFrame) -> None:
    """Reject feature columns with zero standard deviation."""
    feature_cols = get_feature_columns(df)
    for col_name in feature_cols:
        col = df.get_column(col_name).drop_nulls()
        if col.len() == 0:
            continue
        std = col.cast(pl.Float64).std()
        if std is not None and std == 0.0:
            raise ValidationError(
                f"Column '{col_name}' is constant (std=0) — remove it"
            )


def get_feature_columns(df: pl.DataFrame) -> list[str]:
    """Get feature column names, excluding timestamp and reserved columns.

    Args:
        df: Input DataFrame.

    Returns:
        List of feature column names.
    """
    return [c for c in df.columns if c != "timestamp" and c not in RESERVED_COLUMNS]


def separate_labels(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.Series | None]:
    """Separate is_anomaly column from features.

    Args:
        df: Input DataFrame possibly containing 'is_anomaly'.

    Returns:
        Tuple of (DataFrame without is_anomaly, is_anomaly Series or None).
    """
    if "is_anomaly" in df.columns:
        labels = df.get_column("is_anomaly")
        df_clean = df.drop("is_anomaly")
        return df_clean, labels
    return df, None
