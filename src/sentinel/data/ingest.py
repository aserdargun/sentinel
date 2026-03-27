"""Ingest pipeline: load file -> validate -> assign dataset_id -> save Parquet."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinel.data.loaders import load_file
from sentinel.data.validators import get_feature_columns, validate_dataframe


def ingest_file(
    file_path: str | Path,
    data_dir: str | Path = "data/raw",
    metadata_file: str | Path = "data/datasets.json",
) -> dict[str, Any]:
    """Ingest a CSV/Parquet file into the Sentinel data store.

    Validates the file, assigns a dataset_id, saves as Parquet,
    and updates the metadata registry.

    Args:
        file_path: Path to the source CSV or Parquet file.
        data_dir: Directory to store ingested Parquet files.
        metadata_file: Path to the datasets.json registry.

    Returns:
        Dict with dataset_id, shape, feature_names, time_range.
    """
    file_path = Path(file_path)
    data_dir = Path(data_dir)
    metadata_file = Path(metadata_file)

    df = load_file(file_path)
    df = validate_dataframe(df)

    dataset_id = str(uuid.uuid4())
    feature_names = get_feature_columns(df)

    ts_col = df.get_column("timestamp")
    time_range = {
        "start": str(ts_col.min()),
        "end": str(ts_col.max()),
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / f"{dataset_id}.parquet"
    tmp_path = data_dir / f"{dataset_id}.parquet.tmp"

    try:
        df.write_parquet(str(tmp_path))
        os.rename(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    metadata = _load_metadata(metadata_file)
    metadata[dataset_id] = {
        "original_name": file_path.name,
        "source": "file",
        "uploaded_at": datetime.now(UTC).isoformat(),
        "shape": [df.height, df.width],
        "feature_names": feature_names,
        "time_range": time_range,
    }
    _save_metadata(metadata_file, metadata)

    return {
        "dataset_id": dataset_id,
        "shape": [df.height, df.width],
        "feature_names": feature_names,
        "time_range": time_range,
    }


def _load_metadata(path: Path) -> dict[str, Any]:
    """Load datasets.json, creating it if it doesn't exist."""
    if path.exists():
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    return {}


def _save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Atomically save datasets.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2))
    os.rename(str(tmp), str(path))
