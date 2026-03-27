"""Low-level async generators for streaming data sources.

Pure data source adapters that yield one row at a time as dictionaries.
No detection logic, no anomaly injection -- that belongs in
``sentinel.streaming.simulator``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import polars as pl
import structlog

from sentinel.core.exceptions import ValidationError

logger = structlog.get_logger(__name__)


async def stream_from_parquet(
    path: str | Path,
    speed: float = 1.0,
) -> AsyncIterator[dict]:
    """Yield rows from a Parquet file as an async stream.

    Each row is emitted as a ``dict`` mapping column names to values.
    The optional *speed* multiplier controls replay pacing: a speed of
    ``2.0`` halves the inter-row delay, ``0.5`` doubles it, and a speed
    of ``0.0`` (or negative) disables sleeping entirely (maximum
    throughput).

    The base delay between rows is derived from the ``timestamp``
    column when available (the real time gap between consecutive rows).
    If no parseable timestamps exist the fallback is 0.1 seconds per
    row before the speed multiplier is applied.

    Args:
        path: Path to the Parquet file.
        speed: Replay speed multiplier.  ``1.0`` means real-time, ``>1``
            is faster, ``<=0`` is as-fast-as-possible.

    Yields:
        One row per iteration as a ``dict[str, Any]``.

    Raises:
        ValidationError: If the file does not exist or cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")

    try:
        df = pl.read_parquet(path)
    except Exception as exc:
        raise ValidationError(f"Failed to read Parquet {path}: {exc}") from exc

    logger.info(
        "stream_from_parquet_start",
        path=str(path),
        rows=df.height,
        speed=speed,
    )

    async for row in _stream_dataframe(df, speed):
        yield row


async def stream_from_dataframe(
    df: pl.DataFrame,
    speed: float = 1.0,
) -> AsyncIterator[dict]:
    """Yield rows from an in-memory Polars DataFrame as an async stream.

    Behaviour is identical to :func:`stream_from_parquet` but operates
    on a ``pl.DataFrame`` that is already loaded.

    Args:
        df: Source DataFrame.
        speed: Replay speed multiplier.

    Yields:
        One row per iteration as a ``dict[str, Any]``.
    """
    logger.info(
        "stream_from_dataframe_start",
        rows=df.height,
        speed=speed,
    )

    async for row in _stream_dataframe(df, speed):
        yield row


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_DEFAULT_DELAY_S: float = 0.1


async def _stream_dataframe(
    df: pl.DataFrame,
    speed: float,
) -> AsyncIterator[dict]:
    """Core streaming loop shared by the public entry points.

    Computes inter-row delays from the ``timestamp`` column when
    possible, otherwise falls back to a fixed delay.

    Args:
        df: Source DataFrame.
        speed: Replay speed multiplier.

    Yields:
        One row at a time as a ``dict[str, Any]``.
    """
    delays = _compute_delays(df)

    rows = df.to_dicts()
    for idx, row in enumerate(rows):
        if speed > 0.0 and idx < len(delays):
            delay = delays[idx] / speed
            if delay > 0.0:
                await asyncio.sleep(delay)

        yield row


def _compute_delays(df: pl.DataFrame) -> list[float]:
    """Derive per-row delays from the timestamp column.

    Returns a list of length ``df.height`` where element *i* is the
    number of seconds to wait *before* emitting row *i*.  The first
    row always has a delay of ``0.0``.

    Falls back to ``_DEFAULT_DELAY_S`` for every row when the
    timestamp column is missing or not a temporal type.

    Args:
        df: Source DataFrame.

    Returns:
        List of delays in seconds.
    """
    if df.height == 0:
        return []

    if "timestamp" not in df.columns:
        return [_DEFAULT_DELAY_S] * df.height

    ts_col = df.get_column("timestamp")

    # Only Datetime / Duration types carry meaningful time gaps.
    if not ts_col.dtype.is_temporal():
        return [_DEFAULT_DELAY_S] * df.height

    try:
        diffs = ts_col.diff().dt.total_microseconds().to_list()
    except Exception:
        return [_DEFAULT_DELAY_S] * df.height

    delays: list[float] = [0.0]  # first row has no preceding gap
    for d in diffs[1:]:
        if d is None or d <= 0:
            delays.append(_DEFAULT_DELAY_S)
        else:
            delays.append(d / 1_000_000.0)  # microseconds -> seconds

    return delays
