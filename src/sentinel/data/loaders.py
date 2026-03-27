"""Data loading utilities using Polars."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sentinel.core.exceptions import ValidationError


def load_csv(path: str | Path) -> pl.DataFrame:
    """Load a CSV file into a Polars DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        ValidationError: If the file doesn't exist or can't be read.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")

    try:
        return pl.read_csv(path, try_parse_dates=True)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV {path}: {e}") from e


def load_parquet(path: str | Path) -> pl.DataFrame:
    """Load a Parquet file into a Polars DataFrame.

    Args:
        path: Path to the Parquet file.

    Returns:
        Loaded DataFrame.

    Raises:
        ValidationError: If the file doesn't exist or can't be read.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")

    try:
        return pl.read_parquet(path)
    except Exception as e:
        raise ValidationError(f"Failed to read Parquet {path}: {e}") from e


def load_file(path: str | Path) -> pl.DataFrame:
    """Load a CSV or Parquet file based on extension.

    Args:
        path: Path to the data file.

    Returns:
        Loaded DataFrame.

    Raises:
        ValidationError: If the file type is unsupported.
    """
    path = Path(path)
    if path.suffix == ".csv":
        return load_csv(path)
    elif path.suffix in (".parquet", ".pq"):
        return load_parquet(path)
    else:
        raise ValidationError(f"Unsupported file type: {path.suffix}")
