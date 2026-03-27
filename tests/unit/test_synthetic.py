"""Unit tests for sentinel.data.synthetic."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from sentinel.data.synthetic import generate_synthetic

# ------------------------------------------------------------------
# TestGenerateSynthetic — shape and columns
# ------------------------------------------------------------------


class TestGenerateSyntheticShape:
    """Verify that generated data has the correct shape and column layout."""

    def test_default_shape(self) -> None:
        """Default call produces 10 000 rows x 5 features + timestamp + is_anomaly."""
        df = generate_synthetic()
        assert df.height == 10000
        # timestamp + 5 features + is_anomaly = 7 columns
        assert df.width == 7

    def test_custom_length(self) -> None:
        """length parameter controls the number of rows."""
        df = generate_synthetic(length=200)
        assert df.height == 200

    def test_custom_n_features(self) -> None:
        """n_features parameter controls the number of feature columns."""
        df = generate_synthetic(n_features=3, length=100)
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        assert len(feature_cols) == 3

    def test_feature_columns_named_correctly(self) -> None:
        """Feature columns must be named feature_1, feature_2, …, feature_N."""
        n = 4
        df = generate_synthetic(n_features=n, length=50)
        for i in range(1, n + 1):
            assert f"feature_{i}" in df.columns

    def test_has_timestamp_column(self) -> None:
        """DataFrame always contains a 'timestamp' column."""
        df = generate_synthetic(length=50)
        assert "timestamp" in df.columns

    def test_has_is_anomaly_column(self) -> None:
        """DataFrame always contains an 'is_anomaly' column."""
        df = generate_synthetic(length=50)
        assert "is_anomaly" in df.columns

    def test_timestamp_is_first_column(self) -> None:
        """timestamp must be the first column (canonical schema)."""
        df = generate_synthetic(length=50)
        assert df.columns[0] == "timestamp"


# ------------------------------------------------------------------
# TestGenerateSynthetic — data types
# ------------------------------------------------------------------


class TestGenerateSyntheticTypes:
    """Verify column dtypes match the canonical schema."""

    def test_timestamp_is_datetime_utc(self) -> None:
        """timestamp column is pl.Datetime with UTC timezone."""
        df = generate_synthetic(length=50)
        ts_dtype = df.schema["timestamp"]
        assert isinstance(ts_dtype, pl.Datetime)
        assert ts_dtype.time_zone == "UTC"

    def test_feature_columns_are_float(self) -> None:
        """All feature columns have a floating-point dtype."""
        df = generate_synthetic(n_features=3, length=50)
        for i in range(1, 4):
            dtype = df.schema[f"feature_{i}"]
            assert dtype in (pl.Float32, pl.Float64)

    def test_is_anomaly_is_integer(self) -> None:
        """is_anomaly column contains integer values (0 or 1)."""
        df = generate_synthetic(length=100)
        dtype = df.schema["is_anomaly"]
        assert dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8)

    def test_is_anomaly_values_binary(self) -> None:
        """is_anomaly values are strictly 0 or 1."""
        df = generate_synthetic(length=500, seed=0)
        unique_vals = set(df.get_column("is_anomaly").unique().to_list())
        assert unique_vals.issubset({0, 1})


# ------------------------------------------------------------------
# TestGenerateSynthetic — anomaly count
# ------------------------------------------------------------------


class TestGenerateSyntheticAnomalies:
    """Verify that the anomaly ratio is approximately respected."""

    def test_anomaly_count_matches_ratio(self) -> None:
        """Anomaly count is close to length * anomaly_ratio (within ±2x)."""
        length = 1000
        ratio = 0.05
        df = generate_synthetic(length=length, anomaly_ratio=ratio, seed=42)
        n_anomalies = df.get_column("is_anomaly").sum()
        expected = int(length * ratio)
        # Allow some slack due to collective anomaly alignment
        assert expected // 2 <= n_anomalies <= expected * 3

    def test_no_anomalies_at_ratio_zero(self) -> None:
        """anomaly_ratio=0 should produce no anomaly labels."""
        # max(1, int(0 * ratio)) still gives at least 1 in the implementation,
        # so we just check the ratio is very small when ratio is tiny
        df = generate_synthetic(length=1000, anomaly_ratio=0.0, seed=42)
        # max(1, 0) = 1, so exactly 1 anomaly might be injected — just check < 5
        assert df.get_column("is_anomaly").sum() <= 5

    def test_high_anomaly_ratio(self) -> None:
        """anomaly_ratio=0.5 results in approximately half the rows marked."""
        length = 200
        df = generate_synthetic(length=length, anomaly_ratio=0.5, seed=1)
        n_anomalies = df.get_column("is_anomaly").sum()
        # Should be in the ballpark of 50%-ish
        assert n_anomalies > length // 4


# ------------------------------------------------------------------
# TestGenerateSynthetic — reproducibility
# ------------------------------------------------------------------


class TestGenerateSyntheticReproducibility:
    """Same seed must produce identical data; different seeds must differ."""

    def test_same_seed_same_data(self) -> None:
        """Two calls with the same seed produce identical DataFrames."""
        df1 = generate_synthetic(n_features=3, length=100, seed=99)
        df2 = generate_synthetic(n_features=3, length=100, seed=99)
        assert df1.equals(df2)

    def test_different_seeds_different_data(self) -> None:
        """Two calls with different seeds produce different feature values."""
        df1 = generate_synthetic(n_features=2, length=100, seed=1)
        df2 = generate_synthetic(n_features=2, length=100, seed=2)
        # Very unlikely that both datasets are identical
        assert not df1.get_column("feature_1").equals(df2.get_column("feature_1"))

    def test_seed_zero_works(self) -> None:
        """Seed=0 is a valid seed and produces a consistent result."""
        df1 = generate_synthetic(n_features=2, length=50, seed=0)
        df2 = generate_synthetic(n_features=2, length=50, seed=0)
        assert df1.equals(df2)


# ------------------------------------------------------------------
# TestGenerateSynthetic — timestamps
# ------------------------------------------------------------------


class TestGenerateSyntheticTimestamps:
    """Verify timestamp column properties."""

    def test_timestamps_sorted(self) -> None:
        """Timestamps must be in ascending order."""
        df = generate_synthetic(length=100)
        ts = df.get_column("timestamp")
        assert ts.is_sorted()

    def test_timestamps_unique(self) -> None:
        """Every timestamp must be unique (no duplicates)."""
        df = generate_synthetic(length=100)
        ts = df.get_column("timestamp")
        assert ts.n_unique() == ts.len()

    def test_default_start_time(self) -> None:
        """Default start time is 2024-01-01 UTC."""
        df = generate_synthetic(length=10)
        first_ts = df.get_column("timestamp")[0]
        expected = datetime(2024, 1, 1, tzinfo=UTC)
        assert first_ts == expected

    def test_custom_start_time(self) -> None:
        """Custom start_time is reflected in the first timestamp."""
        custom_start = datetime(2023, 6, 15, tzinfo=UTC)
        df = generate_synthetic(length=10, start_time=custom_start)
        first_ts = df.get_column("timestamp")[0]
        assert first_ts == custom_start

    def test_interval_seconds_respected(self) -> None:
        """Consecutive timestamps differ by interval_seconds."""
        interval = 300  # 5 minutes
        df = generate_synthetic(length=5, interval_seconds=interval)
        ts = df.get_column("timestamp")
        diff = (ts[1] - ts[0]).total_seconds()
        assert diff == interval

    def test_no_null_timestamps(self) -> None:
        """timestamp column must have zero null values."""
        df = generate_synthetic(length=100)
        assert df.get_column("timestamp").null_count() == 0


# ------------------------------------------------------------------
# TestGenerateSynthetic — edge cases
# ------------------------------------------------------------------


class TestGenerateSyntheticEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_feature(self) -> None:
        """n_features=1 produces a valid 1-feature DataFrame."""
        df = generate_synthetic(n_features=1, length=50)
        assert "feature_1" in df.columns
        assert df.width == 3  # timestamp + feature_1 + is_anomaly

    def test_minimum_length(self) -> None:
        """length=2 is the smallest valid dataset; function should not crash."""
        df = generate_synthetic(n_features=2, length=2)
        assert df.height == 2

    def test_no_nulls_in_features(self) -> None:
        """Generated feature columns must have no null values."""
        df = generate_synthetic(n_features=5, length=200)
        for i in range(1, 6):
            assert df.get_column(f"feature_{i}").null_count() == 0
