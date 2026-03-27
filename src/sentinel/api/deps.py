"""Dependency injection functions for FastAPI routes."""

from __future__ import annotations

from typing import Any

import structlog

from sentinel.core.registry import list_models
from sentinel.tracking.experiment import LocalTracker

logger = structlog.get_logger(__name__)

_tracker_instance: LocalTracker | None = None


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
