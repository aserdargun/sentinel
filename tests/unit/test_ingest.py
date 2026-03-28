"""Unit tests for sentinel.data.ingest."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from sentinel.core.exceptions import ValidationError
from sentinel.data.ingest import ingest_file

# ---------------------------------------------------------------------------
# CSV / Parquet helpers
# ---------------------------------------------------------------------------


def _write_valid_csv(path: Path) -> Path:
    """Write a minimal valid multivariate CSV and return its path."""
    content = (
        "timestamp,feature_1,feature_2\n"
        "2024-01-01T00:00:00,1.0,10.0\n"
        "2024-01-02T00:00:00,2.0,20.0\n"
        "2024-01-03T00:00:00,3.0,30.0\n"
    )
    path.write_text(content)
    return path


def _write_valid_parquet(path: Path) -> Path:
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
    df.write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# TestIngestFileHappyPath
# ---------------------------------------------------------------------------


class TestIngestFileHappyPath:
    """Full happy-path ingest pipeline for CSV and Parquet inputs.

    Verifies that a valid file is validated, assigned a UUID dataset_id,
    saved as Parquet, and registered in datasets.json.
    """

    def test_ingest_csv_returns_dict(self, tmp_path: Path) -> None:
        """ingest_file returns a dict for a valid CSV."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert isinstance(result, dict)

    def test_ingest_parquet_returns_dict(self, tmp_path: Path) -> None:
        """ingest_file returns a dict for a valid Parquet file."""
        pq_file = _write_valid_parquet(tmp_path / "data.parquet")
        result = ingest_file(pq_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert isinstance(result, dict)

    def test_result_contains_dataset_id(self, tmp_path: Path) -> None:
        """Return dict includes 'dataset_id' key."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert "dataset_id" in result

    def test_dataset_id_is_valid_uuid(self, tmp_path: Path) -> None:
        """dataset_id string is a parseable UUID v4."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        parsed = uuid.UUID(result["dataset_id"])
        assert parsed.version == 4

    def test_result_contains_shape(self, tmp_path: Path) -> None:
        """Return dict includes 'shape' as a list [rows, cols]."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert "shape" in result
        assert isinstance(result["shape"], list)
        assert len(result["shape"]) == 2

    def test_shape_matches_actual_dataframe(self, tmp_path: Path) -> None:
        """Reported shape equals (3 rows, 3 cols) for the minimal CSV."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert result["shape"][0] == 3  # rows
        assert result["shape"][1] == 3  # cols: timestamp + feature_1 + feature_2

    def test_result_contains_feature_names(self, tmp_path: Path) -> None:
        """Return dict includes 'feature_names' list."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert "feature_names" in result
        assert isinstance(result["feature_names"], list)

    def test_feature_names_excludes_timestamp(self, tmp_path: Path) -> None:
        """'timestamp' is not listed as a feature."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert "timestamp" not in result["feature_names"]

    def test_feature_names_correct(self, tmp_path: Path) -> None:
        """Feature names match the CSV column names minus timestamp."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert set(result["feature_names"]) == {"feature_1", "feature_2"}

    def test_result_contains_time_range(self, tmp_path: Path) -> None:
        """Return dict includes 'time_range' with 'start' and 'end' keys."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert "time_range" in result
        assert "start" in result["time_range"]
        assert "end" in result["time_range"]

    def test_each_ingest_produces_unique_dataset_id(self, tmp_path: Path) -> None:
        """Two separate ingests of the same file produce different dataset_ids."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        raw = tmp_path / "raw"
        r1 = ingest_file(csv_file, raw, meta)
        r2 = ingest_file(csv_file, raw, meta)
        assert r1["dataset_id"] != r2["dataset_id"]


# ---------------------------------------------------------------------------
# TestIngestFileParquetOutput
# ---------------------------------------------------------------------------


class TestIngestFileParquetOutput:
    """Verifies that the Parquet file is written and readable."""

    def test_parquet_file_is_created(self, tmp_path: Path) -> None:
        """A .parquet file named after the dataset_id is created in data_dir."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        expected = raw_dir / f"{result['dataset_id']}.parquet"
        assert expected.exists()

    def test_parquet_round_trip_values(self, tmp_path: Path) -> None:
        """Values in the stored Parquet match those from the source CSV."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        stored = pl.read_parquet(raw_dir / f"{result['dataset_id']}.parquet")
        assert stored.get_column("feature_1").to_list() == pytest.approx(
            [1.0, 2.0, 3.0]
        )

    def test_parquet_round_trip_row_count(self, tmp_path: Path) -> None:
        """Stored Parquet contains the same number of rows as the source."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        stored = pl.read_parquet(raw_dir / f"{result['dataset_id']}.parquet")
        assert stored.height == 3

    def test_parquet_has_timestamp_column(self, tmp_path: Path) -> None:
        """Stored Parquet retains the timestamp column."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        stored = pl.read_parquet(raw_dir / f"{result['dataset_id']}.parquet")
        assert "timestamp" in stored.columns

    def test_data_dir_is_created_if_missing(self, tmp_path: Path) -> None:
        """ingest_file creates the data_dir if it does not already exist."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "nested" / "deep" / "raw"
        assert not raw_dir.exists()
        ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        assert raw_dir.exists()

    def test_no_tmp_file_remains_after_success(self, tmp_path: Path) -> None:
        """The temporary .parquet.tmp file is removed after a successful write."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        tmp_file = raw_dir / f"{result['dataset_id']}.parquet.tmp"
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# TestIngestMetadata
# ---------------------------------------------------------------------------


class TestIngestMetadata:
    """Verifies datasets.json is written and contains correct metadata."""

    def test_metadata_file_is_created(self, tmp_path: Path) -> None:
        """datasets.json is created after first ingest."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        assert not meta.exists()
        ingest_file(csv_file, tmp_path / "raw", meta)
        assert meta.exists()

    def test_metadata_is_valid_json(self, tmp_path: Path) -> None:
        """datasets.json contains valid JSON after ingest."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        ingest_file(csv_file, tmp_path / "raw", meta)
        parsed = json.loads(meta.read_text())
        assert isinstance(parsed, dict)

    def test_metadata_contains_dataset_id_key(self, tmp_path: Path) -> None:
        """datasets.json has a top-level key equal to the assigned dataset_id."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        result = ingest_file(csv_file, tmp_path / "raw", meta)
        parsed = json.loads(meta.read_text())
        assert result["dataset_id"] in parsed

    def test_metadata_entry_has_original_name(self, tmp_path: Path) -> None:
        """Metadata entry records the original filename."""
        csv_file = _write_valid_csv(tmp_path / "my_data.csv")
        meta = tmp_path / "datasets.json"
        result = ingest_file(csv_file, tmp_path / "raw", meta)
        entry = json.loads(meta.read_text())[result["dataset_id"]]
        assert entry["original_name"] == "my_data.csv"

    def test_metadata_entry_has_source_file(self, tmp_path: Path) -> None:
        """Metadata entry records source='file' for file-based ingestion."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        result = ingest_file(csv_file, tmp_path / "raw", meta)
        entry = json.loads(meta.read_text())[result["dataset_id"]]
        assert entry["source"] == "file"

    def test_metadata_entry_has_shape(self, tmp_path: Path) -> None:
        """Metadata entry records the shape of the ingested DataFrame."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        result = ingest_file(csv_file, tmp_path / "raw", meta)
        entry = json.loads(meta.read_text())[result["dataset_id"]]
        assert entry["shape"] == [3, 3]

    def test_metadata_entry_has_feature_names(self, tmp_path: Path) -> None:
        """Metadata entry lists the feature columns."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        result = ingest_file(csv_file, tmp_path / "raw", meta)
        entry = json.loads(meta.read_text())[result["dataset_id"]]
        assert set(entry["feature_names"]) == {"feature_1", "feature_2"}

    def test_metadata_entry_has_time_range(self, tmp_path: Path) -> None:
        """Metadata entry records a time_range with start and end."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        result = ingest_file(csv_file, tmp_path / "raw", meta)
        entry = json.loads(meta.read_text())[result["dataset_id"]]
        assert "time_range" in entry
        assert "start" in entry["time_range"]
        assert "end" in entry["time_range"]

    def test_metadata_entry_has_uploaded_at(self, tmp_path: Path) -> None:
        """Metadata entry records an uploaded_at ISO timestamp."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        result = ingest_file(csv_file, tmp_path / "raw", meta)
        entry = json.loads(meta.read_text())[result["dataset_id"]]
        assert "uploaded_at" in entry
        # Must be parseable as ISO 8601
        datetime.fromisoformat(entry["uploaded_at"])

    def test_second_ingest_appends_to_metadata(self, tmp_path: Path) -> None:
        """A second ingest adds a new entry without overwriting the first."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "datasets.json"
        raw = tmp_path / "raw"
        r1 = ingest_file(csv_file, raw, meta)
        r2 = ingest_file(csv_file, raw, meta)
        parsed = json.loads(meta.read_text())
        assert r1["dataset_id"] in parsed
        assert r2["dataset_id"] in parsed

    def test_metadata_parent_dir_created_if_missing(self, tmp_path: Path) -> None:
        """datasets.json parent directory is created if it does not exist."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        meta = tmp_path / "nested" / "meta" / "datasets.json"
        assert not meta.parent.exists()
        ingest_file(csv_file, tmp_path / "raw", meta)
        assert meta.exists()


# ---------------------------------------------------------------------------
# TestIngestRejection
# ---------------------------------------------------------------------------


class TestIngestRejection:
    """Verifies that invalid inputs are rejected before any file is written."""

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """ingest_file raises when the source file does not exist."""
        with pytest.raises((ValidationError, FileNotFoundError)):
            ingest_file(
                tmp_path / "ghost.csv",
                tmp_path / "raw",
                tmp_path / "datasets.json",
            )

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """ingest_file raises for files with an unsupported extension."""
        bad_file = tmp_path / "data.xlsx"
        bad_file.write_bytes(b"not a real xlsx")
        with pytest.raises(ValidationError):
            ingest_file(bad_file, tmp_path / "raw", tmp_path / "datasets.json")

    def test_csv_missing_timestamp_column_raises(self, tmp_path: Path) -> None:
        """CSV without a 'timestamp' first column fails validation."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("time,feature_1\n2024-01-01,1.0\n2024-01-02,2.0\n")
        with pytest.raises(ValidationError):
            ingest_file(bad_csv, tmp_path / "raw", tmp_path / "datasets.json")

    def test_csv_single_row_raises(self, tmp_path: Path) -> None:
        """A CSV with only one data row (below minimum 2) is rejected."""
        bad_csv = tmp_path / "single.csv"
        bad_csv.write_text("timestamp,feature_1\n2024-01-01T00:00:00,1.0\n")
        with pytest.raises(ValidationError):
            ingest_file(bad_csv, tmp_path / "raw", tmp_path / "datasets.json")

    def test_csv_constant_feature_raises(self, tmp_path: Path) -> None:
        """A column with zero standard deviation (constant) is rejected."""
        bad_csv = tmp_path / "const.csv"
        bad_csv.write_text(
            "timestamp,feature_1\n"
            "2024-01-01T00:00:00,5.0\n"
            "2024-01-02T00:00:00,5.0\n"
            "2024-01-03T00:00:00,5.0\n"
        )
        with pytest.raises(ValidationError):
            ingest_file(bad_csv, tmp_path / "raw", tmp_path / "datasets.json")

    def test_csv_duplicate_timestamps_raises(self, tmp_path: Path) -> None:
        """Rows with duplicate timestamps are rejected."""
        bad_csv = tmp_path / "dups.csv"
        bad_csv.write_text(
            "timestamp,feature_1\n"
            "2024-01-01T00:00:00,1.0\n"
            "2024-01-01T00:00:00,2.0\n"
            "2024-01-03T00:00:00,3.0\n"
        )
        with pytest.raises(ValidationError):
            ingest_file(bad_csv, tmp_path / "raw", tmp_path / "datasets.json")

    def test_rejection_leaves_no_parquet_file(self, tmp_path: Path) -> None:
        """On validation failure, no Parquet file is left behind."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("time,feature_1\n2024-01-01,1.0\n2024-01-02,2.0\n")
        raw_dir = tmp_path / "raw"
        try:
            ingest_file(bad_csv, raw_dir, tmp_path / "datasets.json")
        except (ValidationError, Exception):
            pass
        parquet_files = list(raw_dir.glob("*.parquet")) if raw_dir.exists() else []
        assert len(parquet_files) == 0

    def test_rejection_does_not_write_metadata(self, tmp_path: Path) -> None:
        """On validation failure, datasets.json is not created/modified."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("time,feature_1\n2024-01-01,1.0\n2024-01-02,2.0\n")
        meta = tmp_path / "datasets.json"
        try:
            ingest_file(bad_csv, tmp_path / "raw", meta)
        except (ValidationError, Exception):
            pass
        assert not meta.exists()


# ---------------------------------------------------------------------------
# TestIngestAtomicWrite
# ---------------------------------------------------------------------------


class TestIngestAtomicWrite:
    """Verifies atomic write behaviour: no partial files on the happy path."""

    def test_no_tmp_file_after_ingest(self, tmp_path: Path) -> None:
        """Confirm .parquet.tmp is cleaned up after successful ingest."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        tmp_file = raw_dir / f"{result['dataset_id']}.parquet.tmp"
        assert not tmp_file.exists()

    def test_final_parquet_exists_after_ingest(self, tmp_path: Path) -> None:
        """The final .parquet file (renamed from .tmp) exists after ingest."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        final_file = raw_dir / f"{result['dataset_id']}.parquet"
        assert final_file.exists()

    def test_parquet_file_is_valid_after_rename(self, tmp_path: Path) -> None:
        """The renamed Parquet file can be read back without error."""
        csv_file = _write_valid_csv(tmp_path / "data.csv")
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        final_file = raw_dir / f"{result['dataset_id']}.parquet"
        df = pl.read_parquet(final_file)
        assert df.height > 0


# ---------------------------------------------------------------------------
# TestIngestWithLabels
# ---------------------------------------------------------------------------


class TestIngestWithLabels:
    """is_anomaly column is handled correctly: stored but excluded from features."""

    def test_is_anomaly_column_not_in_feature_names(self, tmp_path: Path) -> None:
        """is_anomaly is excluded from the returned feature_names list."""
        csv_file = tmp_path / "labeled.csv"
        csv_file.write_text(
            "timestamp,feature_1,is_anomaly\n"
            "2024-01-01T00:00:00,1.0,0\n"
            "2024-01-02T00:00:00,2.0,0\n"
            "2024-01-03T00:00:00,100.0,1\n"
        )
        result = ingest_file(csv_file, tmp_path / "raw", tmp_path / "datasets.json")
        assert "is_anomaly" not in result["feature_names"]

    def test_is_anomaly_preserved_in_stored_parquet(self, tmp_path: Path) -> None:
        """The stored Parquet still contains the is_anomaly column for later use."""
        csv_file = tmp_path / "labeled.csv"
        csv_file.write_text(
            "timestamp,feature_1,is_anomaly\n"
            "2024-01-01T00:00:00,1.0,0\n"
            "2024-01-02T00:00:00,2.0,0\n"
            "2024-01-03T00:00:00,100.0,1\n"
        )
        raw_dir = tmp_path / "raw"
        result = ingest_file(csv_file, raw_dir, tmp_path / "datasets.json")
        stored = pl.read_parquet(raw_dir / f"{result['dataset_id']}.parquet")
        assert "is_anomaly" in stored.columns
