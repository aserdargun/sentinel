"""Tests for sentinel.data.features module."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from sentinel.data.features import add_lags, add_rolling_stats, add_temporal_features

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df() -> pl.DataFrame:
    """Small DataFrame with two feature columns for deterministic testing."""
    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                [
                    datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 2, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 3, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 4, 0, tzinfo=UTC),
                ]
            ).cast(pl.Datetime("us", "UTC")),
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture
def single_feature_df() -> pl.DataFrame:
    """DataFrame with one feature column."""
    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 3, tzinfo=UTC),
                    datetime(2024, 1, 4, tzinfo=UTC),
                    datetime(2024, 1, 5, tzinfo=UTC),
                ]
            ).cast(pl.Datetime("us", "UTC")),
            "feature_x": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture
def weekend_df() -> pl.DataFrame:
    """DataFrame spanning a weekday–weekend boundary for temporal tests."""
    # 2024-01-05 = Friday, 2024-01-06 = Saturday, 2024-01-07 = Sunday
    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                [
                    datetime(2024, 1, 5, 9, 0, tzinfo=UTC),  # Friday
                    datetime(2024, 1, 6, 9, 0, tzinfo=UTC),  # Saturday
                    datetime(2024, 1, 7, 9, 0, tzinfo=UTC),  # Sunday
                    datetime(2024, 1, 8, 9, 0, tzinfo=UTC),  # Monday
                ]
            ).cast(pl.Datetime("us", "UTC")),
            "feature_1": [1.0, 2.0, 3.0, 4.0],
        }
    )


# ---------------------------------------------------------------------------
# add_lags — column creation
# ---------------------------------------------------------------------------


class TestAddLagsColumns:
    """add_lags() creates the correct column names."""

    def test_creates_lag_columns_for_each_feature_and_lag(
        self, simple_df: pl.DataFrame
    ) -> None:
        result = add_lags(simple_df, lags=[1, 2])
        expected_cols = (
            "feature_a_lag_1",
            "feature_a_lag_2",
            "feature_b_lag_1",
            "feature_b_lag_2",
        )
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, simple_df: pl.DataFrame) -> None:
        result = add_lags(simple_df, lags=[1])
        for col in simple_df.columns:
            assert col in result.columns

    def test_empty_lags_list_returns_unchanged(self, simple_df: pl.DataFrame) -> None:
        result = add_lags(simple_df, lags=[])
        assert result.columns == simple_df.columns

    def test_row_count_unchanged(self, simple_df: pl.DataFrame) -> None:
        result = add_lags(simple_df, lags=[1, 2, 3])
        assert result.height == simple_df.height

    def test_single_lag_column_count(self, single_feature_df: pl.DataFrame) -> None:
        """One feature with one lag creates exactly one new column."""
        result = add_lags(single_feature_df, lags=[1])
        assert result.width == single_feature_df.width + 1


# ---------------------------------------------------------------------------
# add_lags — correctness of shifted values
# ---------------------------------------------------------------------------


class TestAddLagsValues:
    """add_lags() shifts values by the correct number of rows."""

    def test_lag_1_value_at_row_1_equals_original_row_0(
        self, single_feature_df: pl.DataFrame
    ) -> None:
        """Row 1 of lag-1 column should equal row 0 of the original column."""
        result = add_lags(single_feature_df, lags=[1])
        lag_col = result.get_column("feature_x_lag_1").to_list()
        orig_col = single_feature_df.get_column("feature_x").to_list()
        assert lag_col[1] == pytest.approx(orig_col[0])

    def test_lag_2_value_at_row_3_equals_original_row_1(
        self, single_feature_df: pl.DataFrame
    ) -> None:
        result = add_lags(single_feature_df, lags=[2])
        lag_col = result.get_column("feature_x_lag_2").to_list()
        orig_col = single_feature_df.get_column("feature_x").to_list()
        assert lag_col[3] == pytest.approx(orig_col[1])

    def test_lag_1_leading_null_filled_with_zero(
        self, single_feature_df: pl.DataFrame
    ) -> None:
        """The first row of a lag-1 column must not be NaN (filled with 0.0)."""
        result = add_lags(single_feature_df, lags=[1])
        first_val = result.get_column("feature_x_lag_1")[0]
        assert first_val is not None
        assert not (first_val != first_val)  # not NaN

    def test_no_null_values_in_lag_columns(self, simple_df: pl.DataFrame) -> None:
        result = add_lags(simple_df, lags=[1, 2])
        lag_cols = [c for c in result.columns if "_lag_" in c]
        for col in lag_cols:
            assert result.get_column(col).null_count() == 0


# ---------------------------------------------------------------------------
# add_rolling_stats — column creation
# ---------------------------------------------------------------------------


class TestAddRollingStatsColumns:
    """add_rolling_stats() creates four new columns per feature."""

    def test_creates_four_columns_per_feature(self, simple_df: pl.DataFrame) -> None:
        result = add_rolling_stats(simple_df, window_size=3)
        for col in (
            "feature_a_rolling_mean",
            "feature_a_rolling_std",
            "feature_a_rolling_min",
            "feature_a_rolling_max",
        ):
            assert col in result.columns, f"Missing column: {col}"

    def test_creates_rolling_columns_for_all_features(
        self, simple_df: pl.DataFrame
    ) -> None:
        result = add_rolling_stats(simple_df, window_size=3)
        for prefix in ("feature_a", "feature_b"):
            for stat in ("rolling_mean", "rolling_std", "rolling_min", "rolling_max"):
                assert f"{prefix}_{stat}" in result.columns

    def test_original_columns_preserved(self, simple_df: pl.DataFrame) -> None:
        result = add_rolling_stats(simple_df, window_size=3)
        for col in simple_df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self, simple_df: pl.DataFrame) -> None:
        result = add_rolling_stats(simple_df, window_size=3)
        assert result.height == simple_df.height

    def test_no_null_values_after_fill(self, simple_df: pl.DataFrame) -> None:
        result = add_rolling_stats(simple_df, window_size=3)
        rolling_cols = [c for c in result.columns if "rolling_" in c]
        for col in rolling_cols:
            assert result.get_column(col).null_count() == 0


# ---------------------------------------------------------------------------
# add_rolling_stats — correctness
# ---------------------------------------------------------------------------


class TestAddRollingStatsValues:
    """Rolling statistics are computed correctly."""

    def test_rolling_max_final_row_is_max_of_window(
        self, single_feature_df: pl.DataFrame
    ) -> None:
        """Last row of rolling_max with window=3 should be max of last 3 values."""
        result = add_rolling_stats(single_feature_df, window_size=3)
        # Values: 10, 20, 30, 40, 50 — last 3 are 30, 40, 50 → max = 50
        last_max = result.get_column("feature_x_rolling_max")[-1]
        assert last_max == pytest.approx(50.0)

    def test_rolling_min_final_row_is_min_of_window(
        self, single_feature_df: pl.DataFrame
    ) -> None:
        result = add_rolling_stats(single_feature_df, window_size=3)
        # Last 3 are 30, 40, 50 → min = 30
        last_min = result.get_column("feature_x_rolling_min")[-1]
        assert last_min == pytest.approx(30.0)

    def test_rolling_mean_final_row_is_mean_of_window(
        self, single_feature_df: pl.DataFrame
    ) -> None:
        result = add_rolling_stats(single_feature_df, window_size=3)
        # Last 3 are 30, 40, 50 → mean = 40
        last_mean = result.get_column("feature_x_rolling_mean")[-1]
        assert last_mean == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# add_temporal_features — column creation and values
# ---------------------------------------------------------------------------


class TestAddTemporalFeaturesColumns:
    """add_temporal_features() creates the three temporal columns."""

    def test_creates_hour_column(self, simple_df: pl.DataFrame) -> None:
        result = add_temporal_features(simple_df)
        assert "hour" in result.columns

    def test_creates_day_of_week_column(self, simple_df: pl.DataFrame) -> None:
        result = add_temporal_features(simple_df)
        assert "day_of_week" in result.columns

    def test_creates_is_weekend_column(self, simple_df: pl.DataFrame) -> None:
        result = add_temporal_features(simple_df)
        assert "is_weekend" in result.columns

    def test_original_columns_preserved(self, simple_df: pl.DataFrame) -> None:
        result = add_temporal_features(simple_df)
        for col in simple_df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self, simple_df: pl.DataFrame) -> None:
        result = add_temporal_features(simple_df)
        assert result.height == simple_df.height


class TestAddTemporalFeaturesValues:
    """add_temporal_features() produces correct values."""

    def test_hour_values_match_timestamp_hour(self, simple_df: pl.DataFrame) -> None:
        result = add_temporal_features(simple_df)
        hours = result.get_column("hour").to_list()
        # Timestamps go from hour 0 to 4.
        assert hours == [0, 1, 2, 3, 4]

    def test_is_weekend_saturday_is_one(self, weekend_df: pl.DataFrame) -> None:
        result = add_temporal_features(weekend_df)
        is_weekend = result.get_column("is_weekend").to_list()
        # Index 1 = Saturday → 1
        assert is_weekend[1] == 1

    def test_is_weekend_sunday_is_one(self, weekend_df: pl.DataFrame) -> None:
        result = add_temporal_features(weekend_df)
        is_weekend = result.get_column("is_weekend").to_list()
        # Index 2 = Sunday → 1
        assert is_weekend[2] == 1

    def test_is_weekend_friday_is_zero(self, weekend_df: pl.DataFrame) -> None:
        result = add_temporal_features(weekend_df)
        is_weekend = result.get_column("is_weekend").to_list()
        # Index 0 = Friday → 0
        assert is_weekend[0] == 0

    def test_is_weekend_monday_is_zero(self, weekend_df: pl.DataFrame) -> None:
        result = add_temporal_features(weekend_df)
        is_weekend = result.get_column("is_weekend").to_list()
        # Index 3 = Monday → 0
        assert is_weekend[3] == 0

    def test_is_weekend_values_are_zero_or_one(self, weekend_df: pl.DataFrame) -> None:
        result = add_temporal_features(weekend_df)
        is_weekend = result.get_column("is_weekend").to_list()
        for val in is_weekend:
            assert val in (0, 1)

    def test_day_of_week_values_in_valid_range(self, weekend_df: pl.DataFrame) -> None:
        result = add_temporal_features(weekend_df)
        days = result.get_column("day_of_week").to_list()
        for day in days:
            assert 1 <= day <= 7
