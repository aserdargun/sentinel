"""Unit tests for sentinel.data.loaders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from sentinel.core.exceptions import ValidationError
from sentinel.data.loaders import load_csv, load_file, load_parquet

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------


def _make_csv(tmp_path: Path, filename: str = "data.csv") -> Path:
    """Write a minimal valid CSV and return its path."""
    content = (
        "timestamp,feature_1,feature_2\n"
        "2024-01-01T00:00:00,1.0,10.0\n"
        "2024-01-02T00:00:00,2.0,20.0\n"
        "2024-01-03T00:00:00,3.0,30.0\n"
    )
    p = tmp_path / filename
    p.write_text(content)
    return p


def _make_parquet(tmp_path: Path, filename: str = "data.parquet") -> Path:
    """Write a minimal valid Parquet file and return its path."""
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
            "feature_2": [10.0, 20.0, 30.0],
        }
    )
    p = tmp_path / filename
    df.write_parquet(p)
    return p


# ------------------------------------------------------------------
# TestLoadCsv
# ------------------------------------------------------------------


class TestLoadCsv:
    """Tests for load_csv()."""

    def test_loads_valid_csv(self, tmp_path: Path) -> None:
        """load_csv returns a non-empty DataFrame for a valid CSV."""
        path = _make_csv(tmp_path)
        df = load_csv(path)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3

    def test_column_names_preserved(self, tmp_path: Path) -> None:
        """Column names from the CSV header are preserved."""
        path = _make_csv(tmp_path)
        df = load_csv(path)
        assert "timestamp" in df.columns
        assert "feature_1" in df.columns
        assert "feature_2" in df.columns

    def test_numeric_columns_loaded(self, tmp_path: Path) -> None:
        """Numeric columns are loaded as numeric dtype, not strings."""
        path = _make_csv(tmp_path)
        df = load_csv(path)
        dtype = df.schema["feature_1"]
        assert dtype in (
            pl.Float32,
            pl.Float64,
            pl.Int32,
            pl.Int64,
            pl.UInt32,
            pl.UInt64,
        )

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """load_csv accepts a plain string path (not only pathlib.Path)."""
        path = _make_csv(tmp_path)
        df = load_csv(str(path))
        assert df.height == 3

    def test_accepts_pathlib_path(self, tmp_path: Path) -> None:
        """load_csv accepts a pathlib.Path object."""
        path = _make_csv(tmp_path)
        df = load_csv(path)
        assert df.height == 3

    def test_missing_file_raises_validation_error(self, tmp_path: Path) -> None:
        """load_csv raises ValidationError when the file does not exist."""
        with pytest.raises(ValidationError, match="not found"):
            load_csv(tmp_path / "nonexistent.csv")

    def test_returns_polars_dataframe(self, tmp_path: Path) -> None:
        """Return type is always pl.DataFrame."""
        path = _make_csv(tmp_path)
        df = load_csv(path)
        assert isinstance(df, pl.DataFrame)

    def test_multirow_csv_all_rows_loaded(self, tmp_path: Path) -> None:
        """All data rows are loaded (header excluded)."""
        path = _make_csv(tmp_path)
        df = load_csv(path)
        assert df.height == 3


# ------------------------------------------------------------------
# TestLoadParquet
# ------------------------------------------------------------------


class TestLoadParquet:
    """Tests for load_parquet()."""

    def test_loads_valid_parquet(self, tmp_path: Path) -> None:
        """load_parquet returns a non-empty DataFrame for a valid Parquet file."""
        path = _make_parquet(tmp_path)
        df = load_parquet(path)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3

    def test_column_names_preserved(self, tmp_path: Path) -> None:
        """Column names written to Parquet are preserved on read."""
        path = _make_parquet(tmp_path)
        df = load_parquet(path)
        assert "timestamp" in df.columns
        assert "feature_1" in df.columns
        assert "feature_2" in df.columns

    def test_dtypes_preserved(self, tmp_path: Path) -> None:
        """Parquet preserves the exact dtype written (Float64 stays Float64)."""
        path = _make_parquet(tmp_path)
        df = load_parquet(path)
        assert df.schema["feature_1"] == pl.Float64
        assert df.schema["feature_2"] == pl.Float64

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """load_parquet accepts a plain string path."""
        path = _make_parquet(tmp_path)
        df = load_parquet(str(path))
        assert df.height == 3

    def test_accepts_pathlib_path(self, tmp_path: Path) -> None:
        """load_parquet accepts a pathlib.Path object."""
        path = _make_parquet(tmp_path)
        df = load_parquet(path)
        assert df.height == 3

    def test_missing_file_raises_validation_error(self, tmp_path: Path) -> None:
        """load_parquet raises ValidationError when the file does not exist."""
        with pytest.raises(ValidationError, match="not found"):
            load_parquet(tmp_path / "nonexistent.parquet")

    def test_returns_polars_dataframe(self, tmp_path: Path) -> None:
        """Return type is always pl.DataFrame."""
        path = _make_parquet(tmp_path)
        df = load_parquet(path)
        assert isinstance(df, pl.DataFrame)

    def test_round_trip_values(self, tmp_path: Path) -> None:
        """Values written to Parquet are identical when read back."""
        path = _make_parquet(tmp_path)
        df = load_parquet(path)
        assert df.get_column("feature_1").to_list() == [1.0, 2.0, 3.0]


# ------------------------------------------------------------------
# TestLoadFile
# ------------------------------------------------------------------


class TestLoadFile:
    """Tests for load_file() — auto-detects format from extension."""

    def test_detects_csv_by_extension(self, tmp_path: Path) -> None:
        """load_file routes .csv files to load_csv."""
        path = _make_csv(tmp_path)
        df = load_file(path)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3

    def test_detects_parquet_by_extension(self, tmp_path: Path) -> None:
        """load_file routes .parquet files to load_parquet."""
        path = _make_parquet(tmp_path)
        df = load_file(path)
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3

    def test_detects_pq_extension(self, tmp_path: Path) -> None:
        """load_file also accepts the .pq shorthand extension."""
        path = _make_parquet(tmp_path, filename="data.pq")
        df = load_file(path)
        assert isinstance(df, pl.DataFrame)

    def test_unsupported_extension_raises_validation_error(
        self, tmp_path: Path
    ) -> None:
        """load_file raises ValidationError for an unknown extension."""
        p = tmp_path / "data.xlsx"
        p.write_bytes(b"not a real xlsx")
        with pytest.raises(ValidationError, match="Unsupported"):
            load_file(p)

    def test_missing_csv_file_raises(self, tmp_path: Path) -> None:
        """load_file propagates ValidationError from load_csv for missing files."""
        with pytest.raises(ValidationError, match="not found"):
            load_file(tmp_path / "ghost.csv")

    def test_missing_parquet_file_raises(self, tmp_path: Path) -> None:
        """load_file propagates ValidationError from load_parquet for missing files."""
        with pytest.raises(ValidationError, match="not found"):
            load_file(tmp_path / "ghost.parquet")

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """load_file accepts a plain string path."""
        path = _make_csv(tmp_path)
        df = load_file(str(path))
        assert df.height == 3

    def test_returns_polars_dataframe_for_csv(self, tmp_path: Path) -> None:
        """Return type is always pl.DataFrame regardless of source format."""
        path = _make_csv(tmp_path)
        df = load_file(path)
        assert isinstance(df, pl.DataFrame)

    def test_returns_polars_dataframe_for_parquet(self, tmp_path: Path) -> None:
        """Return type is always pl.DataFrame regardless of source format."""
        path = _make_parquet(tmp_path)
        df = load_file(path)
        assert isinstance(df, pl.DataFrame)
