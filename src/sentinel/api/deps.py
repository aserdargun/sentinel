"""Dependency injection functions for FastAPI routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from fastapi import HTTPException

from sentinel.core.registry import list_models
from sentinel.tracking.experiment import LocalTracker

logger = structlog.get_logger(__name__)

_tracker_instance: LocalTracker | None = None

# Project root: sentinel/ directory that contains pyproject.toml, configs/, data/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Allowed base directories for user-supplied file paths.  Resolved at
# import time so that they are absolute and symlink-free.
ALLOWED_PATH_BASES: list[Path] = [
    (_PROJECT_ROOT / "data").resolve(),
    (_PROJECT_ROOT / "configs").resolve(),
]


def resolve_safe_path(
    user_path: str,
    allowed_bases: list[Path] | None = None,
) -> Path:
    """Resolve a user-supplied path and verify it is within allowed directories.

    Prevents path-traversal attacks by resolving the path to an absolute,
    symlink-free form and checking that it falls under one of the allowed
    base directories.

    Args:
        user_path: The raw path string received from the API request.
        allowed_bases: Directories the resolved path must be under.
            Defaults to ``ALLOWED_PATH_BASES`` (``data/`` and ``configs/``).

    Returns:
        The resolved ``Path`` object.

    Raises:
        HTTPException: 400 if the path escapes the allowed directories or
            contains null bytes.
    """
    if allowed_bases is None:
        allowed_bases = ALLOWED_PATH_BASES

    # Reject null bytes (can bypass some path checks in C-level code).
    if "\x00" in user_path:
        logger.warning("path_validation.null_byte", user_path=user_path)
        raise HTTPException(
            status_code=400,
            detail="Path contains invalid characters.",
        )

    resolved = Path(user_path).resolve()

    for base in allowed_bases:
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            continue

    logger.warning(
        "path_validation.rejected",
        user_path=user_path,
        resolved=str(resolved),
        allowed_bases=[str(b) for b in allowed_bases],
    )
    raise HTTPException(
        status_code=400,
        detail=(
            f"Path '{user_path}' is outside the allowed directories. "
            f"Allowed: {', '.join(str(b) for b in allowed_bases)}"
        ),
    )


def get_tracker() -> LocalTracker:
    """Return a singleton LocalTracker instance.

    The tracker is created on first call and reused for subsequent
    requests.  The base directory defaults to ``data/experiments``.

    Returns:
        The shared LocalTracker.
    """
    global _tracker_instance  # noqa: PLW0603
    if _tracker_instance is None:
        _tracker_instance = LocalTracker(base_dir="data/experiments")
        logger.info("deps.tracker_created")
    return _tracker_instance


def get_registry() -> dict[str, Any]:
    """Return the current model registry.

    Calls :func:`sentinel.core.registry.list_models` to retrieve all
    registered model classes keyed by name.

    Returns:
        Dict mapping model names to model classes.
    """
    return list_models()
