"""PI System API routes.

Provides endpoints for fetching timeseries from OSIsoft PI System,
searching PI tags, and retrieving current snapshot values.

All endpoints include a platform check -- PI functionality requires
Windows with the PI AF SDK installed.  On non-Windows platforms (or
when pipolars is not installed), endpoints return HTTP 501.
"""

from __future__ import annotations

import os
import platform
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/data", tags=["pi"])

_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PIFetchRequest(BaseModel):
    """Request body for PI timeseries fetch."""

    server: str = Field(description="PI Data Archive server hostname.")
    tags: list[str] = Field(description="List of PI point names to fetch.")
    start: str = Field(
        default="*-1d",
        description="Start time in PI time syntax (e.g., '*-7d').",
    )
    end: str = Field(
        default="*",
        description="End time in PI time syntax (e.g., '*').",
    )
    interval: str = Field(
        default="5m",
        description="Interpolation interval (e.g., '1m', '5m').",
    )
    port: int = Field(default=5450, description="PI server port.")
    timeout: int = Field(default=30, description="Connection timeout in seconds.")


class PIFetchResponse(BaseModel):
    """Response after a successful PI data fetch."""

    dataset_id: str = Field(description="UUID of the ingested dataset.")
    shape: list[int] = Field(description="[rows, columns].")
    features: list[str] = Field(description="Feature column names (tag names).")
    time_range: dict[str, str] = Field(
        description="Start and end timestamps as ISO strings."
    )


class PISearchRequest(BaseModel):
    """Request body for PI tag search."""

    server: str = Field(description="PI Data Archive server hostname.")
    pattern: str = Field(description="Glob-style pattern for tag name matching.")
    port: int = Field(default=5450, description="PI server port.")
    timeout: int = Field(default=30, description="Connection timeout in seconds.")


class PITagInfo(BaseModel):
    """Information about a single PI tag."""

    name: str = Field(description="PI point name.")
    description: str = Field(default="", description="Tag description.")
    uom: str = Field(default="", description="Unit of measurement.")


class PISearchResponse(BaseModel):
    """Response from PI tag search."""

    tags: list[PITagInfo] = Field(default_factory=list, description="Matching tags.")


class PISnapshotEntry(BaseModel):
    """Current value for a single PI tag."""

    name: str = Field(description="PI point name.")
    value: float | str | None = Field(default=None, description="Current value.")
    timestamp: str = Field(default="", description="Snapshot timestamp (ISO).")
    quality: str = Field(default="unknown", description="Value quality.")


class PISnapshotResponse(BaseModel):
    """Response from PI snapshot retrieval."""

    snapshots: list[PISnapshotEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _check_pi_available() -> None:
    """Raise HTTP 501 if PI System is not available.

    Raises:
        HTTPException: 501 on non-Windows or missing pipolars.
    """
    if not _IS_WINDOWS:
        raise HTTPException(
            status_code=501,
            detail=(
                "PI System connector requires Windows with PI AF SDK. "
                f"Current platform: {platform.system()}"
            ),
        )

    try:
        from sentinel.data.pi_connector import is_pi_available

        if not is_pi_available():
            raise HTTPException(
                status_code=501,
                detail=(
                    "pipolars is not installed. "
                    "Install with: uv add --group pi pipolars"
                ),
            )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="pipolars is not installed.",
        )


def _save_pi_dataset(
    df: Any,
    server: str,
    tags: list[str],
) -> dict[str, Any]:
    """Save a PI-fetched DataFrame to the data store.

    Follows the same atomic-write pattern as the ingest pipeline.

    Args:
        df: Polars DataFrame with PI data.
        server: PI server hostname (for metadata).
        tags: List of fetched tag names (for metadata).

    Returns:
        Dict with dataset_id, shape, features, time_range.
    """
    import json

    data_dir = Path("data/raw")
    metadata_file = Path("data/datasets.json")

    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = str(uuid.uuid4())
    feature_cols = [c for c in df.columns if c != "timestamp"]

    ts_col = df.get_column("timestamp")
    time_range = {
        "start": str(ts_col.min()),
        "end": str(ts_col.max()),
    }

    # Atomic write.
    out_path = data_dir / f"{dataset_id}.parquet"
    tmp_path = data_dir / f"{dataset_id}.parquet.tmp"
    try:
        df.write_parquet(str(tmp_path))
        os.rename(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    # Update metadata registry.
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
    else:
        metadata = {}

    metadata[dataset_id] = {
        "original_name": f"pi_{server}_{'_'.join(tags[:3])}",
        "source": "pi",
        "uploaded_at": datetime.now(UTC).isoformat(),
        "shape": [df.height, df.width],
        "feature_names": feature_cols,
        "time_range": time_range,
    }

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_meta = metadata_file.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(metadata, indent=2))
    os.rename(str(tmp_meta), str(metadata_file))

    return {
        "dataset_id": dataset_id,
        "shape": [df.height, df.width],
        "features": feature_cols,
        "time_range": time_range,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/pi-fetch",
    response_model=PIFetchResponse,
    summary="Fetch timeseries from PI System",
)
async def pi_fetch(request: PIFetchRequest) -> PIFetchResponse:
    """Fetch multi-tag interpolated timeseries from a PI server.

    Connects to the specified PI Data Archive, retrieves interpolated
    data for the requested tags, validates, and stores as Parquet.

    Returns the dataset_id and metadata for the ingested data.
    """
    _check_pi_available()

    from sentinel.data.pi_connector import PIConnectionError, PIConnector
    from sentinel.data.validators import validate_dataframe

    logger.info(
        "api.pi_fetch",
        server=request.server,
        tags=request.tags,
        start=request.start,
        end=request.end,
        interval=request.interval,
    )

    try:
        connector = PIConnector(
            host=request.server,
            port=request.port,
            timeout=request.timeout,
        )
        df = connector.fetch_tags(
            tags=request.tags,
            start=request.start,
            end=request.end,
            interval=request.interval,
        )
        connector.close()
    except PIConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        df = validate_dataframe(df)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Validation failed: {exc}"
        ) from exc

    result = _save_pi_dataset(df, request.server, request.tags)

    logger.info("api.pi_fetch.done", dataset_id=result["dataset_id"])
    return PIFetchResponse(**result)


@router.post(
    "/pi-search",
    response_model=PISearchResponse,
    summary="Search PI tags by pattern",
)
async def pi_search(request: PISearchRequest) -> PISearchResponse:
    """Search PI points on a server by name pattern.

    Returns matching tag names with descriptions and units.
    """
    _check_pi_available()

    from sentinel.data.pi_connector import PIConnectionError, PIConnector

    logger.info(
        "api.pi_search",
        server=request.server,
        pattern=request.pattern,
    )

    try:
        connector = PIConnector(
            host=request.server,
            port=request.port,
            timeout=request.timeout,
        )
        tags = connector.search_tags(request.pattern)
        connector.close()
    except PIConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    tag_infos = [PITagInfo(**t) for t in tags]
    return PISearchResponse(tags=tag_infos)


@router.get(
    "/pi-snapshot",
    response_model=PISnapshotResponse,
    summary="Get current PI tag values",
)
async def pi_snapshot(
    server: str,
    tags: str,
    port: int = 5450,
    timeout: int = 30,
) -> PISnapshotResponse:
    """Get current snapshot values for specified PI tags.

    Tags are provided as a comma-separated string in the query
    parameter.

    Args:
        server: PI Data Archive hostname.
        tags: Comma-separated list of PI point names.
        port: PI server port.
        timeout: Connection timeout.
    """
    _check_pi_available()

    from sentinel.data.pi_connector import PIConnectionError, PIConnector

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if not tag_list:
        raise HTTPException(
            status_code=400, detail="At least one tag must be specified."
        )

    logger.info(
        "api.pi_snapshot",
        server=server,
        tags=tag_list,
    )

    try:
        connector = PIConnector(
            host=server,
            port=port,
            timeout=timeout,
        )
        snapshots = connector.snapshot(tag_list)
        connector.close()
    except PIConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    entries = [PISnapshotEntry(**s) for s in snapshots]
    return PISnapshotResponse(snapshots=entries)
