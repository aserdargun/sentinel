"""Aggregated API router including all sub-routers.

Each route module is imported lazily so that missing optional
dependencies (e.g. the ``pi`` group) do not prevent the rest of the
API from loading.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter

logger = structlog.get_logger(__name__)

api_router = APIRouter()


def _include_optional(module_name: str, attr: str = "router") -> None:
    """Import a route module and include its router if available.

    Args:
        module_name: Fully-qualified module path under
            ``sentinel.api.routes``.
        attr: Name of the :class:`APIRouter` attribute in the module.
    """
    try:
        import importlib

        mod = importlib.import_module(f"sentinel.api.routes.{module_name}")
        sub_router: APIRouter = getattr(mod, attr)
        api_router.include_router(sub_router)
        logger.debug("routes.included", module=module_name)
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "routes.skip",
            module=module_name,
            reason=str(exc),
        )


# Include all available route modules.  Modules that depend on optional
# dependency groups (pi, mcp) are skipped with a log warning if their
# imports fail.
_ROUTE_MODULES = [
    "data",
    "train",
    "detect",
    "evaluate",
    "models",
    "experiments",
    "visualize",
    "prompt",
    "pi",
]

for _mod in _ROUTE_MODULES:
    _include_optional(_mod)
