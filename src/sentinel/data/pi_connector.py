"""PI System data connector via pipolars.

Wraps the PIClient from pipolars to search tags, fetch multi-tag
timeseries, and retrieve current snapshot values.  All PI imports are
gated behind ``try/except ImportError`` so the module can be imported
on any platform -- methods raise ``ImportError`` with a clear message
when pipolars is unavailable.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from sentinel.core.exceptions import SentinelError

logger = structlog.get_logger(__name__)

# Attempt to import pipolars -- will be None on non-Windows or when
# the ``pi`` dependency group is not installed.
try:
    import pipolars as pip  # type: ignore[import-untyped]

    _HAS_PIPOLARS = True
except ImportError:
    _HAS_PIPOLARS = False

_PI_IMPORT_MSG = (
    "pipolars is not installed. Install it with: "
    "uv add --group pi pipolars  "
    "(requires Windows with PI AF SDK / .NET 4.8)"
)

# Maximum retries for PI server connections.
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0


class PIConnectionError(SentinelError):
    """Raised when PI System connection fails after retries."""


class PIConnector:
    """Wrapper around pipolars PIClient for Sentinel data ingestion.

    Provides three capabilities:

    1. **Tag search** -- search PI points by name pattern.
    2. **Multi-tag fetch** -- bulk interpolated timeseries extraction
       with pivot to Sentinel's canonical schema.
    3. **Snapshot** -- current values for selected tags.

    All methods raise ``ImportError`` if pipolars is not available and
    ``PIConnectionError`` after exhausting retries on server errors.

    Args:
        host: PI Data Archive server hostname.
        port: PI server port.  Defaults to 5450.
        timeout: Connection timeout in seconds.  Defaults to 30.
        cache_dir: Optional directory for pipolars query caching.
            Defaults to ``None`` (no caching).

    Example::

        connector = PIConnector(host="my-pi-server")
        tags = connector.search_tags("Pump*")
        df = connector.fetch_tags(
            tags=["Pump.Flow", "Pump.Pressure"],
            start="*-1d",
            end="*",
            interval="5m",
        )
    """

    def __init__(
        self,
        host: str,
        port: int = 5450,
        timeout: int = 30,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._cache_dir = str(cache_dir) if cache_dir is not None else None
        self._client: Any = None

        logger.info(
            "pi_connector.init",
            host=host,
            port=port,
            timeout=timeout,
            has_pipolars=_HAS_PIPOLARS,
            platform=platform.system(),
        )

    def _ensure_pipolars(self) -> None:
        """Raise ImportError if pipolars is not available."""
        if not _HAS_PIPOLARS:
            raise ImportError(_PI_IMPORT_MSG)

    def _get_client(self) -> Any:
        """Lazily create and return a PIClient instance.

        Retries up to ``_MAX_RETRIES`` times with exponential backoff.

        Returns:
            Connected pipolars PIClient.

        Raises:
            ImportError: If pipolars is unavailable.
            PIConnectionError: If connection fails after retries.
        """
        self._ensure_pipolars()

        if self._client is not None:
            return self._client

        last_error: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = {
                    "server": self._host,
                    "port": self._port,
                    "timeout": self._timeout,
                }
                if self._cache_dir is not None:
                    kwargs["cache_dir"] = self._cache_dir

                self._client = pip.PIClient(**kwargs)  # type: ignore[union-attr]
                logger.info(
                    "pi_connector.connected",
                    host=self._host,
                    attempt=attempt + 1,
                )
                return self._client
            except Exception as exc:
                last_error = exc
                delay = _BACKOFF_BASE_S * (2**attempt)
                logger.warning(
                    "pi_connector.retry",
                    host=self._host,
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=str(exc),
                )
                time.sleep(delay)

        raise PIConnectionError(
            f"Failed to connect to PI server {self._host}:{self._port} "
            f"after {_MAX_RETRIES} attempts: {last_error}"
        )

    def search_tags(self, pattern: str) -> list[dict[str, str]]:
        """Search PI points by name pattern.

        Uses pipolars tag search to find matching PI point names on
        the configured server.

        Args:
            pattern: Glob-style pattern for tag name matching
                (e.g., ``"Pump*"`` or ``"*Flow*"``).

        Returns:
            List of dicts with keys ``name``, ``description``, ``uom``
            for each matching tag.

        Raises:
            ImportError: If pipolars is not installed.
            PIConnectionError: If the server is unreachable.
        """
        self._ensure_pipolars()
        client = self._get_client()

        logger.info("pi_connector.search_tags", pattern=pattern)

        try:
            results = client.search(pattern)
        except Exception as exc:
            raise PIConnectionError(
                f"Tag search failed for pattern '{pattern}': {exc}"
            ) from exc

        tags: list[dict[str, str]] = []
        for tag in results:
            tags.append(
                {
                    "name": getattr(tag, "name", str(tag)),
                    "description": getattr(tag, "description", ""),
                    "uom": getattr(tag, "uom", ""),
                }
            )

        logger.info(
            "pi_connector.search_tags.done",
            pattern=pattern,
            count=len(tags),
        )
        return tags

    def fetch_tags(
        self,
        tags: list[str],
        start: str,
        end: str,
        interval: str,
    ) -> pl.DataFrame:
        """Fetch multi-tag interpolated timeseries from PI.

        Queries the PI server for all specified tags over the given time
        range at the specified interpolation interval.  The result is
        pivoted into Sentinel's canonical schema: ``timestamp`` as the
        first column, one column per tag.

        All timestamps are normalized to UTC.

        Args:
            tags: List of PI point names (tag names).
            start: Start time in PI time syntax (e.g., ``"*-7d"``).
            end: End time in PI time syntax (e.g., ``"*"``).
            interval: Interpolation interval (e.g., ``"1m"``, ``"5m"``).

        Returns:
            Polars DataFrame in canonical schema with UTC timestamps.

        Raises:
            ImportError: If pipolars is not installed.
            PIConnectionError: If fetch fails.
            ValueError: If no tags are provided.
        """
        self._ensure_pipolars()

        if not tags:
            raise ValueError("At least one tag must be specified")

        client = self._get_client()

        logger.info(
            "pi_connector.fetch_tags",
            tags=tags,
            start=start,
            end=end,
            interval=interval,
        )

        try:
            df: pl.DataFrame = (
                client.query(tags)
                .time_range(start, end)
                .interpolated(interval)
                .pivot()
                .to_dataframe()
            )
        except Exception as exc:
            raise PIConnectionError(
                f"Failed to fetch tags {tags} from {self._host}: {exc}"
            ) from exc

        # Ensure canonical schema: first column is "timestamp".
        if "timestamp" not in df.columns:
            # pipolars typically names the time column "Timestamp" or
            # similar -- find and rename it.
            time_col = _find_time_column(df)
            if time_col is not None and time_col != "timestamp":
                df = df.rename({time_col: "timestamp"})
            elif time_col is None:
                raise PIConnectionError(
                    "PI query result has no recognizable timestamp column. "
                    f"Columns: {df.columns}"
                )

        # Ensure timestamp is first column.
        if df.columns[0] != "timestamp":
            cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
            df = df.select(cols)

        # Normalize timestamps to UTC.
        ts_dtype = df.schema["timestamp"]
        if isinstance(ts_dtype, pl.Datetime):
            if ts_dtype.time_zone is None:
                df = df.with_columns(
                    pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp")
                )
            elif ts_dtype.time_zone != "UTC":
                df = df.with_columns(
                    pl.col("timestamp").dt.convert_time_zone("UTC").alias("timestamp")
                )

        # Cast all feature columns to Float64.
        feature_cols = [c for c in df.columns if c != "timestamp"]
        for col_name in feature_cols:
            if df.schema[col_name] != pl.Float64:
                df = df.with_columns(pl.col(col_name).cast(pl.Float64).alias(col_name))

        logger.info(
            "pi_connector.fetch_tags.done",
            rows=df.height,
            columns=df.width,
            features=feature_cols,
        )
        return df

    def snapshot(self, tags: list[str]) -> list[dict[str, Any]]:
        """Get current snapshot values for selected tags.

        Retrieves the most recent recorded value for each tag from the
        PI server.

        Args:
            tags: List of PI point names.

        Returns:
            List of dicts with keys ``name``, ``value``, ``timestamp``,
            ``quality`` for each tag.

        Raises:
            ImportError: If pipolars is not installed.
            PIConnectionError: If snapshot retrieval fails.
        """
        self._ensure_pipolars()

        if not tags:
            raise ValueError("At least one tag must be specified")

        client = self._get_client()

        logger.info("pi_connector.snapshot", tags=tags)

        try:
            snapshots_raw = client.snapshots(tags)
        except Exception as exc:
            raise PIConnectionError(
                f"Snapshot retrieval failed for tags {tags}: {exc}"
            ) from exc

        results: list[dict[str, Any]] = []
        for entry in snapshots_raw:
            results.append(
                {
                    "name": getattr(entry, "name", str(entry)),
                    "value": getattr(entry, "value", None),
                    "timestamp": str(getattr(entry, "timestamp", "")),
                    "quality": getattr(entry, "quality", "unknown"),
                }
            )

        logger.info("pi_connector.snapshot.done", count=len(results))
        return results

    def close(self) -> None:
        """Close the PI server connection if open."""
        if self._client is not None:
            try:
                if hasattr(self._client, "close"):
                    self._client.close()
            except Exception as exc:
                logger.warning("pi_connector.close_error", error=str(exc))
            self._client = None
            logger.info("pi_connector.closed")

    def __enter__(self) -> PIConnector:
        """Enter context manager."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit context manager, closing connection."""
        self.close()


def _find_time_column(df: pl.DataFrame) -> str | None:
    """Locate a datetime column in a DataFrame by type or name.

    Args:
        df: DataFrame to search.

    Returns:
        Column name if found, ``None`` otherwise.
    """
    for col_name in df.columns:
        dtype = df.schema[col_name]
        if isinstance(dtype, pl.Datetime) or dtype == pl.Date:
            return col_name
    for col_name in df.columns:
        if col_name.lower() in ("timestamp", "time", "datetime"):
            return col_name
    return None


def is_pi_available() -> bool:
    """Check whether pipolars is installed and importable.

    Returns:
        ``True`` if pipolars can be imported, ``False`` otherwise.
    """
    return _HAS_PIPOLARS
