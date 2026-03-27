"""Unit tests for sentinel.data.preprocessors."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from sentinel.data.preprocessors import (
    chronological_split,
    create_windows,
    fill_nan,
    scale_minmax,
    scale_zscore,
    to_numpy,
)

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    """10-row multivariate DataFrame with UTC timestamps."""
    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 3, tzinfo=UTC),
                    datetime(2024, 1, 4, tzinfo=UTC),
                    datetime(2024, 1, 5, tzinfo=UTC),
                    datetime(2024, 1, 6, tzinfo=UTC),
                    datetime(2024, 1, 7, tzinfo=UTC),
                    datetime(2024, 1, 8, tzinfo=UTC),
                    datetime(2024, 1, 9, tzinfo=UTC),
                    datetime(2024, 1, 10, tzinfo=UTC),
                ]
            ).cast(pl.Datetime("us", "UTC")),
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )


@pytest.fixture()
def df_with_nans(sample_df: pl.DataFrame) -> pl.DataFrame:
    """DataFrame with NaN values in feature_1 at positions 2 and 4."""
    values = [1.0, 2.0, None, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0]
    return sample_df.with_columns(pl.Series("feature_1", values))


@pytest.fixture()
def df_leading_nan() -> pl.DataFrame:
    """DataFrame where the first value in feature_1 is NaN (tests zero-fill)."""
    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 3, tzinfo=UTC),
                ]
            ).cast(pl.Datetime("us", "UTC")),
            "feature_1": pl.Series([None, 2.0, 3.0], dtype=pl.Float64),
        }
    )


# ------------------------------------------------------------------
# TestFillNan
# ------------------------------------------------------------------


class TestFillNan:
    """fill_nan: forward-fill then zero-fill."""

    def test_no_nans_unchanged(self, sample_df: pl.DataFrame) -> None:
        """DataFrame without nulls should be unchanged."""
        result = fill_nan(sample_df)
        assert (
            result.get_column("feature_1").to_list()
            == sample_df.get_column("feature_1").to_list()
        )

    def test_forward_fills_interior_nans(self, df_with_nans: pl.DataFrame) -> None:
        """Interior nulls should be forward-filled from the preceding value."""
        result = fill_nan(df_with_nans)
        col = result.get_column("feature_1").to_list()
        # Position 2 was None, should become 2.0 (forward-fill from position 1)
        assert col[2] == pytest.approx(2.0)
        # Position 4 was None, should become 4.0 (forward-fill from position 3)
        assert col[4] == pytest.approx(4.0)

    def test_leading_nans_zero_filled(self, df_leading_nan: pl.DataFrame) -> None:
        """Leading nulls (no prior value) should be zero-filled."""
        result = fill_nan(df_leading_nan)
        col = result.get_column("feature_1").to_list()
        assert col[0] == pytest.approx(0.0)
        assert col[1] == pytest.approx(2.0)

    def test_no_nulls_remain(self, df_with_nans: pl.DataFrame) -> None:
        """After fill_nan, no null values should remain in feature columns."""
        result = fill_nan(df_with_nans)
        assert result.get_column("feature_1").null_count() == 0
        assert result.get_column("feature_2").null_count() == 0

    def test_timestamp_column_untouched(self, df_with_nans: pl.DataFrame) -> None:
        """timestamp column must not be modified."""
        original_ts = df_with_nans.get_column("timestamp").to_list()
        result = fill_nan(df_with_nans)
        assert result.get_column("timestamp").to_list() == original_ts

    def test_returns_polars_dataframe(self, sample_df: pl.DataFrame) -> None:
        """Return type is pl.DataFrame."""
        result = fill_nan(sample_df)
        assert isinstance(result, pl.DataFrame)


# ------------------------------------------------------------------
# TestScaleZscore
# ------------------------------------------------------------------


class TestScaleZscore:
    """scale_zscore: each feature column should have mean~0 and std~1."""

    def test_returns_tuple(self, sample_df: pl.DataFrame) -> None:
        """Return value is a 2-tuple (DataFrame, stats dict)."""
        result = scale_zscore(sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_scaled_mean_near_zero(self, sample_df: pl.DataFrame) -> None:
        """After z-score scaling every feature column has mean close to 0."""
        scaled_df, _ = scale_zscore(sample_df)
        for col in ["feature_1", "feature_2"]:
            mean = scaled_df.get_column(col).mean()
            assert mean == pytest.approx(0.0, abs=1e-9)

    def test_scaled_std_near_one(self, sample_df: pl.DataFrame) -> None:
        """After z-score scaling every feature column has std close to 1."""
        scaled_df, _ = scale_zscore(sample_df)
        for col in ["feature_1", "feature_2"]:
            std = scaled_df.get_column(col).std()
            assert std == pytest.approx(1.0, rel=1e-6)

    def test_stats_dict_contains_all_features(self, sample_df: pl.DataFrame) -> None:
        """Stats dict has an entry for every feature column."""
        _, stats = scale_zscore(sample_df)
        assert "feature_1" in stats
        assert "feature_2" in stats
        assert "timestamp" not in stats

    def test_stats_values_are_mean_std_tuples(self, sample_df: pl.DataFrame) -> None:
        """Each stats entry is a (mean, std) tuple of floats."""
        _, stats = scale_zscore(sample_df)
        for col, (mean, std) in stats.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std > 0.0

    def test_timestamp_column_untouched(self, sample_df: pl.DataFrame) -> None:
        """timestamp column must not be scaled."""
        original_ts = sample_df.get_column("timestamp").to_list()
        scaled_df, _ = scale_zscore(sample_df)
        assert scaled_df.get_column("timestamp").to_list() == original_ts


# ------------------------------------------------------------------
# TestScaleMinmax
# ------------------------------------------------------------------


class TestScaleMinmax:
    """scale_minmax: all feature values should lie in [0, 1]."""

    def test_returns_tuple(self, sample_df: pl.DataFrame) -> None:
        """Return value is a 2-tuple (DataFrame, stats dict)."""
        result = scale_minmax(sample_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_min_is_zero(self, sample_df: pl.DataFrame) -> None:
        """Minimum of each feature column after scaling should be 0."""
        scaled_df, _ = scale_minmax(sample_df)
        for col in ["feature_1", "feature_2"]:
            col_min = scaled_df.get_column(col).min()
            assert col_min == pytest.approx(0.0, abs=1e-9)

    def test_max_is_one(self, sample_df: pl.DataFrame) -> None:
        """Maximum of each feature column after scaling should be 1."""
        scaled_df, _ = scale_minmax(sample_df)
        for col in ["feature_1", "feature_2"]:
            col_max = scaled_df.get_column(col).max()
            assert col_max == pytest.approx(1.0, abs=1e-9)

    def test_all_values_in_unit_range(self, sample_df: pl.DataFrame) -> None:
        """Every value in every feature column must be in [0, 1]."""
        scaled_df, _ = scale_minmax(sample_df)
        for col in ["feature_1", "feature_2"]:
            vals = scaled_df.get_column(col).to_numpy()
            assert np.all(vals >= 0.0)
            assert np.all(vals <= 1.0)

    def test_stats_dict_contains_min_max(self, sample_df: pl.DataFrame) -> None:
        """Stats dict holds (min, max) tuples for each feature."""
        _, stats = scale_minmax(sample_df)
        f1_min, f1_max = stats["feature_1"]
        assert f1_min == pytest.approx(1.0)
        assert f1_max == pytest.approx(10.0)

    def test_timestamp_column_untouched(self, sample_df: pl.DataFrame) -> None:
        """timestamp column must not be modified."""
        original_ts = sample_df.get_column("timestamp").to_list()
        scaled_df, _ = scale_minmax(sample_df)
        assert scaled_df.get_column("timestamp").to_list() == original_ts


# ------------------------------------------------------------------
# TestChronologicalSplit
# ------------------------------------------------------------------


class TestChronologicalSplit:
    """chronological_split: correct sizes, no data leak, chronological order."""

    def test_returns_three_dataframes(self, sample_df: pl.DataFrame) -> None:
        """Return value is a 3-tuple of DataFrames."""
        result = chronological_split(sample_df)
        assert len(result) == 3
        assert all(isinstance(df, pl.DataFrame) for df in result)

    def test_total_rows_preserved(self, sample_df: pl.DataFrame) -> None:
        """Sum of all split sizes equals the original row count."""
        train, val, test = chronological_split(sample_df)
        assert train.height + val.height + test.height == sample_df.height

    def test_train_ratio_approximate(self, sample_df: pl.DataFrame) -> None:
        """Training set is approximately 70% of the data."""
        train, _, _ = chronological_split(sample_df)
        # 10 rows * 0.70 = 7 rows
        assert train.height == 7

    def test_val_ratio_approximate(self, sample_df: pl.DataFrame) -> None:
        """Validation set is approximately 15% of the data."""
        _, val, _ = chronological_split(sample_df)
        # 10 rows * 0.15 = 1 row (floor)
        assert val.height == 1

    def test_chronological_order_preserved(self, sample_df: pl.DataFrame) -> None:
        """Last timestamp of train < first timestamp of val < first of test."""
        train, val, test = chronological_split(sample_df)
        if val.height > 0:
            assert train.get_column("timestamp")[-1] < val.get_column("timestamp")[0]
        if test.height > 0 and val.height > 0:
            assert val.get_column("timestamp")[-1] < test.get_column("timestamp")[0]

    def test_no_overlap_between_splits(self, sample_df: pl.DataFrame) -> None:
        """Timestamps are mutually exclusive across splits."""
        train, val, test = chronological_split(sample_df)
        train_ts = set(train.get_column("timestamp").to_list())
        val_ts = set(val.get_column("timestamp").to_list())
        test_ts = set(test.get_column("timestamp").to_list())
        assert train_ts.isdisjoint(val_ts)
        assert train_ts.isdisjoint(test_ts)
        assert val_ts.isdisjoint(test_ts)

    def test_custom_ratios(self) -> None:
        """Custom split ratios are respected."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 21)]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": list(range(1, 21)),
            }
        )
        train, val, test = chronological_split(
            df, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25
        )
        assert train.height == 10
        assert val.height == 5
        assert test.height == 5

    def test_large_dataset_ratios(self) -> None:
        """With 100 rows, default 70/15/15 split gives expected counts."""
        n = 100
        df = pl.DataFrame(
            {
                "timestamp": pl.Series([datetime(2024, 1, 1, tzinfo=UTC)] * n).cast(
                    pl.Datetime("us", "UTC")
                ),
                "feature_1": list(range(n)),
            }
        )
        train, val, test = chronological_split(df)
        assert train.height == 70
        assert val.height == 15
        assert test.height == 15


# ------------------------------------------------------------------
# TestToNumpy
# ------------------------------------------------------------------


class TestToNumpy:
    """to_numpy: correct shape, dtype, and column selection."""

    def test_shape_matches_rows_and_features(self, sample_df: pl.DataFrame) -> None:
        """Output shape is (n_rows, n_features)."""
        arr = to_numpy(sample_df)
        assert arr.shape == (10, 2)

    def test_dtype_is_float64(self, sample_df: pl.DataFrame) -> None:
        """Output dtype must be float64."""
        arr = to_numpy(sample_df)
        assert arr.dtype == np.float64

    def test_timestamp_excluded(self, sample_df: pl.DataFrame) -> None:
        """timestamp column must not appear in the numpy array."""
        arr = to_numpy(sample_df)
        # 2 feature columns, not 3 (timestamp would add a 3rd)
        assert arr.shape[1] == 2

    def test_is_anomaly_excluded(self) -> None:
        """is_anomaly column must not be included in the numpy output."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": [1.0, 2.0],
                "is_anomaly": [0, 1],
            }
        )
        arr = to_numpy(df)
        assert arr.shape == (2, 1)

    def test_values_match_original(self, sample_df: pl.DataFrame) -> None:
        """Numeric values are preserved after conversion."""
        arr = to_numpy(sample_df)
        expected_f1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        np.testing.assert_array_almost_equal(arr[:, 0], expected_f1)

    def test_2d_output(self, sample_df: pl.DataFrame) -> None:
        """Output is always 2-dimensional."""
        arr = to_numpy(sample_df)
        assert arr.ndim == 2


# ------------------------------------------------------------------
# TestCreateWindows
# ------------------------------------------------------------------


class TestCreateWindows:
    """create_windows: sliding window extraction from 2D arrays."""

    def test_output_shape_default_stride(self) -> None:
        """With stride=1, n_windows == n_samples - seq_len + 1."""
        data = np.arange(20, dtype=np.float64).reshape(10, 2)
        windows = create_windows(data, seq_len=3, stride=1)
        # n_windows = 10 - 3 + 1 = 8
        assert windows.shape == (8, 3, 2)

    def test_output_shape_custom_stride(self) -> None:
        """With stride=2 the window count is halved (rounded down)."""
        data = np.arange(30, dtype=np.float64).reshape(10, 3)
        windows = create_windows(data, seq_len=4, stride=2)
        # indices: 0, 2, 4, 6 -> 4 windows
        expected_n = len(range(0, 10 - 4 + 1, 2))
        assert windows.shape == (expected_n, 4, 3)

    def test_window_content_correct(self) -> None:
        """First window matches the first seq_len rows of the input."""
        data = np.arange(12, dtype=np.float64).reshape(6, 2)
        windows = create_windows(data, seq_len=3, stride=1)
        np.testing.assert_array_equal(windows[0], data[:3])

    def test_last_window_content_correct(self) -> None:
        """Last window matches the last seq_len rows of the input."""
        data = np.arange(12, dtype=np.float64).reshape(6, 2)
        windows = create_windows(data, seq_len=3, stride=1)
        np.testing.assert_array_equal(windows[-1], data[3:])

    def test_3d_output(self) -> None:
        """Output array is always 3-dimensional."""
        data = np.zeros((10, 5))
        windows = create_windows(data, seq_len=3)
        assert windows.ndim == 3

    def test_single_feature(self) -> None:
        """Works correctly when the input has only one feature column."""
        data = np.arange(10, dtype=np.float64).reshape(10, 1)
        windows = create_windows(data, seq_len=4, stride=1)
        assert windows.shape == (7, 4, 1)

    def test_seq_len_equals_data_length(self) -> None:
        """seq_len == n_samples produces exactly one window."""
        data = np.ones((5, 3))
        windows = create_windows(data, seq_len=5, stride=1)
        assert windows.shape == (1, 5, 3)

    def test_too_short_raises_value_error(self) -> None:
        """Data shorter than seq_len raises ValueError."""
        data = np.zeros((3, 2))
        with pytest.raises(ValueError, match="seq_len"):
            create_windows(data, seq_len=5)

    def test_stride_larger_than_seq_len(self) -> None:
        """Stride larger than seq_len still produces valid non-overlapping windows."""
        data = np.arange(20, dtype=np.float64).reshape(10, 2)
        windows = create_windows(data, seq_len=3, stride=5)
        # indices: 0, 5 -> 2 windows
        assert windows.shape[0] == 2
        assert windows.shape[1] == 3
        assert windows.shape[2] == 2
