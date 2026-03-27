"""Unit tests for sentinel.data.validators."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from sentinel.core.exceptions import ValidationError
from sentinel.data.validators import (
    get_feature_columns,
    separate_labels,
    validate_dataframe,
)

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def valid_df() -> pl.DataFrame:
    """Minimal valid multivariate DataFrame."""
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
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture()
def valid_df_with_labels(valid_df: pl.DataFrame) -> pl.DataFrame:
    """Valid DataFrame including the is_anomaly column."""
    return valid_df.with_columns(pl.Series("is_anomaly", [0, 0, 1, 0, 0]))


# ------------------------------------------------------------------
# TestValidateDataframe — happy path
# ------------------------------------------------------------------


class TestValidateDataframeHappyPath:
    """Valid DataFrames should pass validation unchanged (modulo UTC cast)."""

    def test_valid_df_passes(self, valid_df: pl.DataFrame) -> None:
        """Well-formed DataFrame returns without raising."""
        result = validate_dataframe(valid_df)
        assert result.height == valid_df.height
        assert result.columns == valid_df.columns

    def test_returns_polars_dataframe(self, valid_df: pl.DataFrame) -> None:
        """Return type is always pl.DataFrame."""
        result = validate_dataframe(valid_df)
        assert isinstance(result, pl.DataFrame)

    def test_valid_with_is_anomaly_column(
        self, valid_df_with_labels: pl.DataFrame
    ) -> None:
        """DataFrame containing is_anomaly should still pass validation."""
        result = validate_dataframe(valid_df_with_labels)
        assert "is_anomaly" in result.columns

    def test_timestamp_cast_to_utc(self, valid_df: pl.DataFrame) -> None:
        """After validation the timestamp column has UTC timezone."""
        result = validate_dataframe(valid_df)
        ts_dtype = result.schema["timestamp"]
        assert isinstance(ts_dtype, pl.Datetime)
        assert ts_dtype.time_zone == "UTC"


# ------------------------------------------------------------------
# TestValidateDataframe — row-count checks
# ------------------------------------------------------------------


class TestRowCountChecks:
    """DataFrames with too few rows must be rejected."""

    def test_empty_df_fails(self) -> None:
        """Empty DataFrame (0 rows) raises ValidationError."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                "feature_1": pl.Series([], dtype=pl.Float64),
            }
        )
        with pytest.raises(ValidationError, match="at least"):
            validate_dataframe(df)

    def test_single_row_df_fails(self) -> None:
        """Single-row DataFrame raises ValidationError."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series([datetime(2024, 1, 1, tzinfo=UTC)]).cast(
                    pl.Datetime("us", "UTC")
                ),
                "feature_1": [1.0],
            }
        )
        with pytest.raises(ValidationError, match="at least"):
            validate_dataframe(df)

    def test_two_rows_passes(self) -> None:
        """Two-row DataFrame meets the minimum and passes."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": [1.0, 2.0],
            }
        )
        result = validate_dataframe(df)
        assert result.height == 2


# ------------------------------------------------------------------
# TestValidateDataframe — column-name checks
# ------------------------------------------------------------------


class TestColumnNameChecks:
    """First column must be named 'timestamp'."""

    def test_wrong_first_column_fails(self) -> None:
        """DataFrame whose first column is not 'timestamp' raises."""
        df = pl.DataFrame(
            {
                "time": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": [1.0, 2.0],
            }
        )
        with pytest.raises(ValidationError, match="timestamp"):
            validate_dataframe(df)

    def test_timestamp_not_first_but_present_fails(self) -> None:
        """'timestamp' in position 1+ (not 0) still fails the first-column check."""
        df = pl.DataFrame(
            {
                "feature_1": [1.0, 2.0],
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
            }
        )
        with pytest.raises(ValidationError, match="timestamp"):
            validate_dataframe(df)


# ------------------------------------------------------------------
# TestValidateDataframe — feature-type checks
# ------------------------------------------------------------------


class TestFeatureTypeChecks:
    """All feature columns must be numeric."""

    def test_string_feature_fails(self) -> None:
        """A non-numeric feature column raises ValidationError."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": ["a", "b"],
            }
        )
        with pytest.raises(ValidationError, match="numeric"):
            validate_dataframe(df)

    def test_bool_feature_fails(self) -> None:
        """Boolean column is not considered numeric by the validator."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": [True, False],
            }
        )
        with pytest.raises(ValidationError, match="numeric"):
            validate_dataframe(df)

    def test_integer_feature_passes(self) -> None:
        """Integer columns are valid numeric features."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": pl.Series([1, 2], dtype=pl.Int64),
            }
        )
        result = validate_dataframe(df)
        assert result.height == 2


# ------------------------------------------------------------------
# TestValidateDataframe — NaN and constant checks
# ------------------------------------------------------------------


class TestNaNAndConstantChecks:
    """All-NaN columns and constant columns must be rejected."""

    def test_all_nan_column_fails(self) -> None:
        """A column where every value is null raises ValidationError."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                        datetime(2024, 1, 3, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": [1.0, 2.0, 3.0],
                "all_nan": pl.Series([None, None, None], dtype=pl.Float64),
            }
        )
        with pytest.raises(ValidationError, match="all NaN"):
            validate_dataframe(df)

    def test_constant_column_fails(self) -> None:
        """A column with std==0 raises ValidationError."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                        datetime(2024, 1, 3, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": [1.0, 2.0, 3.0],
                "constant": [5.0, 5.0, 5.0],
            }
        )
        with pytest.raises(ValidationError, match="constant"):
            validate_dataframe(df)

    def test_partially_nan_column_passes(self, valid_df: pl.DataFrame) -> None:
        """A column with some (not all) nulls is allowed."""
        df = valid_df.with_columns(pl.Series("partial_nan", [1.0, None, 3.0, 4.0, 5.0]))
        result = validate_dataframe(df)
        assert "partial_nan" in result.columns

    def test_is_anomaly_not_subject_to_constant_check(
        self, valid_df: pl.DataFrame
    ) -> None:
        """is_anomaly column that is all-zero should not trigger constant check."""
        df = valid_df.with_columns(
            pl.Series("is_anomaly", [0, 0, 0, 0, 0], dtype=pl.Int64)
        )
        # Should not raise — is_anomaly is reserved, not a feature
        result = validate_dataframe(df)
        assert result.height == 5


# ------------------------------------------------------------------
# TestValidateDataframe — duplicate and sort checks
# ------------------------------------------------------------------


class TestDuplicateAndSortChecks:
    """Timestamps must be unique and sorted."""

    def test_duplicate_timestamps_fail(self) -> None:
        """Repeated timestamps raise ValidationError."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 1, tzinfo=UTC),  # duplicate
                        datetime(2024, 1, 3, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValidationError, match="duplicate"):
            validate_dataframe(df)

    def test_unsorted_timestamps_fail(self) -> None:
        """Out-of-order timestamps raise ValidationError."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 3, tzinfo=UTC),
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                    ]
                ).cast(pl.Datetime("us", "UTC")),
                "feature_1": [3.0, 1.0, 2.0],
            }
        )
        with pytest.raises(ValidationError, match="sorted"):
            validate_dataframe(df)


# ------------------------------------------------------------------
# TestValidateDataframe — timestamp parsing
# ------------------------------------------------------------------


class TestTimestampParsing:
    """String timestamps must be parsed; timezone variants must be normalised."""

    def test_string_timestamps_parsed(self) -> None:
        """String timestamps are converted to Datetime UTC."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-03T00:00:00",
                ],
                "feature_1": [1.0, 2.0, 3.0],
            }
        )
        result = validate_dataframe(df)
        ts_dtype = result.schema["timestamp"]
        assert isinstance(ts_dtype, pl.Datetime)
        assert ts_dtype.time_zone == "UTC"

    def test_naive_timestamps_assumed_utc(self) -> None:
        """Timezone-naive Datetime columns are assigned UTC."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1),
                        datetime(2024, 1, 2),
                        datetime(2024, 1, 3),
                    ]
                ).cast(pl.Datetime("us")),
                "feature_1": [1.0, 2.0, 3.0],
            }
        )
        result = validate_dataframe(df)
        assert result.schema["timestamp"].time_zone == "UTC"

    def test_non_utc_timezone_converted_to_utc(self) -> None:
        """Timestamps in a non-UTC timezone are converted to UTC."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(
                    [
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                        datetime(2024, 1, 3, tzinfo=UTC),
                    ]
                )
                .cast(pl.Datetime("us", "UTC"))
                .dt.convert_time_zone("America/New_York"),
                "feature_1": [1.0, 2.0, 3.0],
            }
        )
        result = validate_dataframe(df)
        assert result.schema["timestamp"].time_zone == "UTC"


# ------------------------------------------------------------------
# TestGetFeatureColumns
# ------------------------------------------------------------------


class TestGetFeatureColumns:
    """get_feature_columns should exclude timestamp and is_anomaly."""

    def test_excludes_timestamp(self, valid_df: pl.DataFrame) -> None:
        """'timestamp' must not appear in the feature list."""
        cols = get_feature_columns(valid_df)
        assert "timestamp" not in cols

    def test_excludes_is_anomaly(self, valid_df_with_labels: pl.DataFrame) -> None:
        """'is_anomaly' must not appear in the feature list."""
        cols = get_feature_columns(valid_df_with_labels)
        assert "is_anomaly" not in cols

    def test_includes_feature_columns(self, valid_df: pl.DataFrame) -> None:
        """Regular feature columns must all be present."""
        cols = get_feature_columns(valid_df)
        assert "feature_1" in cols
        assert "feature_2" in cols

    def test_correct_count(self, valid_df_with_labels: pl.DataFrame) -> None:
        """Count equals total columns minus timestamp and is_anomaly."""
        cols = get_feature_columns(valid_df_with_labels)
        # valid_df has 2 features; with_labels adds is_anomaly -> still 2 features
        assert len(cols) == 2

    def test_single_feature_column(self) -> None:
        """Works when there is exactly one feature column."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series([datetime(2024, 1, 1, tzinfo=UTC)]).cast(
                    pl.Datetime("us", "UTC")
                ),
                "value": [1.0],
            }
        )
        cols = get_feature_columns(df)
        assert cols == ["value"]

    def test_no_feature_columns_returns_empty(self) -> None:
        """DataFrame with only timestamp returns an empty list."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series([datetime(2024, 1, 1, tzinfo=UTC)]).cast(
                    pl.Datetime("us", "UTC")
                ),
            }
        )
        cols = get_feature_columns(df)
        assert cols == []


# ------------------------------------------------------------------
# TestSeparateLabels
# ------------------------------------------------------------------


class TestSeparateLabels:
    """separate_labels should split is_anomaly from features cleanly."""

    def test_extracts_is_anomaly_series(
        self, valid_df_with_labels: pl.DataFrame
    ) -> None:
        """When is_anomaly is present, a Series is returned."""
        df_clean, labels = separate_labels(valid_df_with_labels)
        assert labels is not None
        assert isinstance(labels, pl.Series)
        assert labels.name == "is_anomaly"

    def test_removes_is_anomaly_from_df(
        self, valid_df_with_labels: pl.DataFrame
    ) -> None:
        """The returned DataFrame must not contain is_anomaly."""
        df_clean, _ = separate_labels(valid_df_with_labels)
        assert "is_anomaly" not in df_clean.columns

    def test_label_values_preserved(self, valid_df_with_labels: pl.DataFrame) -> None:
        """is_anomaly values match what was inserted."""
        _, labels = separate_labels(valid_df_with_labels)
        assert labels is not None
        assert labels.to_list() == [0, 0, 1, 0, 0]

    def test_returns_none_when_no_is_anomaly(self, valid_df: pl.DataFrame) -> None:
        """When is_anomaly is absent, the second return value is None."""
        df_clean, labels = separate_labels(valid_df)
        assert labels is None

    def test_df_unchanged_when_no_is_anomaly(self, valid_df: pl.DataFrame) -> None:
        """When is_anomaly is absent the DataFrame is returned as-is."""
        df_clean, _ = separate_labels(valid_df)
        assert df_clean.columns == valid_df.columns
        assert df_clean.height == valid_df.height

    def test_feature_columns_intact_after_separation(
        self, valid_df_with_labels: pl.DataFrame
    ) -> None:
        """Feature columns and timestamp are all still present after separation."""
        df_clean, _ = separate_labels(valid_df_with_labels)
        assert "timestamp" in df_clean.columns
        assert "feature_1" in df_clean.columns
        assert "feature_2" in df_clean.columns
