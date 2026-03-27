"""Data management routes: upload, list, preview, delete datasets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import structlog
from fastapi import APIRouter, HTTPException, Query, UploadFile

from sentinel.api.schemas import (
    DatasetListResponse,
    DatasetSummary,
    DatasetUploadResponse,
    ErrorResponse,
)
from sentinel.data.ingest import ingest_file

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/data", tags=["data"])

_MAX_UPLOAD_SIZE_MB = 100
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_SIZE_MB * 1024 * 1024
_ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/octet-stream",
    "application/vnd.apache.parquet",
}
_DATA_DIR = Path("data/raw")
_METADATA_FILE = Path("data/datasets.json")


def _load_metadata() -> dict[str, Any]:
    """Read the datasets.json registry.

    Returns:
        Dictionary mapping dataset_id to metadata, or empty dict.
    """
    if _METADATA_FILE.exists():
        return json.loads(_METADATA_FILE.read_text())  # type: ignore[no-any-return]
    return {}


def _save_metadata(metadata: dict[str, Any]) -> None:
    """Atomically write the datasets.json registry.

    Args:
        metadata: The full metadata dict to persist.
    """
    _METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _METADATA_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2))
    os.rename(str(tmp), str(_METADATA_FILE))


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=201,
    responses={400: {"model": ErrorResponse}},
)
async def upload_dataset(file: UploadFile) -> DatasetUploadResponse:
    """Upload a CSV or Parquet file, validate, and store.

    The file is validated against the Sentinel multivariate schema.
    On success, a Parquet copy is stored in ``data/raw/`` and the
    dataset is registered in ``datasets.json``.

    Args:
        file: The uploaded file (CSV or Parquet).

    Returns:
        Upload response with dataset_id, shape, features, time_range.
    """
    # Validate content type.
    content_type = file.content_type or ""
    filename = file.filename or "upload"

    if content_type not in _ALLOWED_CONTENT_TYPES:
        # Fall back to extension check for common mislabeled uploads.
        ext = Path(filename).suffix.lower()
        if ext not in (".csv", ".parquet", ".pq"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type} ({ext}). "
                f"Allowed: CSV, Parquet.",
            )

    # Read file contents with size limit.
    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(contents) / 1024 / 1024:.1f}MB. "
            f"Maximum: {_MAX_UPLOAD_SIZE_MB}MB.",
        )

    # Write to a temp file so the ingest pipeline can read it.
    ext = Path(filename).suffix.lower() or ".csv"
    with tempfile.NamedTemporaryFile(
        suffix=ext, delete=False, dir=tempfile.gettempdir()
    ) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = ingest_file(
            file_path=tmp_path,
            data_dir=str(_DATA_DIR),
            metadata_file=str(_METADATA_FILE),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        # Clean up the temporary upload file.
        Path(tmp_path).unlink(missing_ok=True)

    logger.info(
        "data.uploaded",
        dataset_id=result["dataset_id"],
        shape=result["shape"],
    )

    return DatasetUploadResponse(
        dataset_id=result["dataset_id"],
        shape=result["shape"],
        features=result["feature_names"],
        time_range=result["time_range"],
    )


@router.get(
    "",
    response_model=DatasetListResponse,
)
async def list_datasets(
    page: int = Query(default=1, ge=1, description="Page number."),
    limit: int = Query(default=50, ge=1, le=200, description="Items per page."),
    sort: str = Query(default="uploaded_at", description="Field to sort by."),
    order: str = Query(default="desc", description="Sort order: 'asc' or 'desc'."),
) -> DatasetListResponse:
    """List all datasets with pagination.

    Reads the ``datasets.json`` registry and returns a paginated
    list of dataset summaries.

    Args:
        page: Page number (1-based).
        limit: Number of items per page.
        sort: Field to sort by (default ``uploaded_at``).
        order: Sort order: ``asc`` or ``desc``.

    Returns:
        Paginated dataset list.
    """
    metadata = _load_metadata()

    items: list[DatasetSummary] = []
    for dataset_id, info in metadata.items():
        shape = info.get("shape", [0, 0])
        items.append(
            DatasetSummary(
                dataset_id=dataset_id,
                name=info.get("original_name", ""),
                source=info.get("source", "upload"),
                rows=shape[0] if len(shape) > 0 else 0,
                columns=shape[1] if len(shape) > 1 else 0,
                features=info.get("feature_names", []),
                time_range=info.get("time_range", {}),
                uploaded_at=info.get("uploaded_at", ""),
            )
        )

    # Sort items.
    reverse = order.lower() == "desc"
    sort_key = sort if sort in ("uploaded_at", "name", "rows") else "uploaded_at"
    items.sort(key=lambda x: getattr(x, sort_key, ""), reverse=reverse)

    total = len(items)
    start = (page - 1) * limit
    end = start + limit
    page_items = items[start:end]

    return DatasetListResponse(items=page_items, total=total, page=page)


@router.get(
    "/{dataset_id}",
    response_model=DatasetSummary,
    responses={404: {"model": ErrorResponse}},
)
async def get_dataset(dataset_id: str) -> DatasetSummary:
    """Get summary metadata for a single dataset.

    Args:
        dataset_id: The UUID dataset identifier.

    Returns:
        Dataset summary with shape, features, time range.
    """
    metadata = _load_metadata()
    if dataset_id not in metadata:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    info = metadata[dataset_id]
    shape = info.get("shape", [0, 0])

    return DatasetSummary(
        dataset_id=dataset_id,
        name=info.get("original_name", ""),
        source=info.get("source", "upload"),
        rows=shape[0] if len(shape) > 0 else 0,
        columns=shape[1] if len(shape) > 1 else 0,
        features=info.get("feature_names", []),
        time_range=info.get("time_range", {}),
        uploaded_at=info.get("uploaded_at", ""),
    )


@router.get(
    "/{dataset_id}/preview",
    responses={404: {"model": ErrorResponse}},
)
async def preview_dataset(
    dataset_id: str,
    rows: int = Query(default=20, ge=1, le=1000, description="Number of rows."),
) -> list[dict[str, Any]]:
    """Return the first N rows of a dataset as JSON.

    Args:
        dataset_id: The UUID dataset identifier.
        rows: Number of rows to return (default 20, max 1000).

    Returns:
        List of row dicts.
    """
    metadata = _load_metadata()
    if dataset_id not in metadata:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    parquet_path = _DATA_DIR / f"{dataset_id}.parquet"
    if not parquet_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Parquet file missing for dataset: {dataset_id}"
        )

    try:
        df = pl.read_parquet(parquet_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to read dataset: {exc}"
        ) from exc

    preview = df.head(rows)
    return preview.to_dicts()  # type: ignore[return-value]


@router.delete(
    "/{dataset_id}",
    responses={404: {"model": ErrorResponse}},
)
async def delete_dataset(dataset_id: str) -> dict[str, str]:
    """Delete a dataset and its Parquet file.

    Args:
        dataset_id: The UUID dataset identifier.

    Returns:
        Confirmation message.
    """
    metadata = _load_metadata()
    if dataset_id not in metadata:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    # Remove Parquet file.
    parquet_path = _DATA_DIR / f"{dataset_id}.parquet"
    if parquet_path.exists():
        parquet_path.unlink()
        logger.info("data.file_deleted", path=str(parquet_path))

    # Remove from metadata registry.
    del metadata[dataset_id]
    _save_metadata(metadata)

    logger.info("data.deleted", dataset_id=dataset_id)
    return {"message": f"Dataset {dataset_id} deleted."}
