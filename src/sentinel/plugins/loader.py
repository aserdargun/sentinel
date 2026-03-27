"""External model plugin discovery via entry points.

Scans the ``sentinel.models`` entry-point group and registers any valid
:class:`~sentinel.core.base_model.BaseAnomalyDetector` subclass found.
Invalid or duplicate entries are logged as warnings and skipped.
"""

from __future__ import annotations

from importlib.metadata import entry_points

import structlog

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.registry import _REGISTRY, register_model

logger = structlog.get_logger(__name__)

ENTRY_POINT_GROUP = "sentinel.models"


def discover_plugins() -> list[str]:
    """Scan entry points and register discovered model plugins.

    Iterates over all ``sentinel.models`` entry points.  For each entry
    point, the referenced object is loaded and validated:

    - It must be a class (not an instance or module).
    - It must be a subclass of
      :class:`~sentinel.core.base_model.BaseAnomalyDetector`.
    - Its entry-point name must not already be registered.

    Plugins that fail validation are skipped with a warning log.

    Returns:
        List of model names that were newly registered during this call.

    Example::

        # In an external package's pyproject.toml:
        # [project.entry-points."sentinel.models"]
        # my_detector = "my_package.detector:MyDetector"

        from sentinel.plugins import discover_plugins
        new_models = discover_plugins()
        # ["my_detector"]
    """
    eps = entry_points()

    # Python 3.12: entry_points() returns a SelectableGroups or dict-like;
    # .select(group=...) returns the matching entries.
    if hasattr(eps, "select"):
        model_eps = eps.select(group=ENTRY_POINT_GROUP)
    elif isinstance(eps, dict):
        model_eps = eps.get(ENTRY_POINT_GROUP, [])
    else:
        model_eps = []

    newly_registered: list[str] = []

    for ep in model_eps:
        name = ep.name
        log = logger.bind(plugin_name=name, entry_point=str(ep))

        # Skip if already registered (e.g. a built-in model with the same name).
        if name in _REGISTRY:
            log.warning(
                "plugin_already_registered",
                message=f"Model '{name}' is already registered, skipping plugin",
            )
            continue

        try:
            obj = ep.load()
        except Exception as exc:
            log.warning(
                "plugin_load_failed",
                error=str(exc),
                message=f"Failed to load entry point '{name}'",
            )
            continue

        # Validate: must be a class.
        if not isinstance(obj, type):
            log.warning(
                "plugin_not_a_class",
                loaded_type=type(obj).__name__,
                message=f"Entry point '{name}' is not a class",
            )
            continue

        # Validate: must be a BaseAnomalyDetector subclass.
        if not issubclass(obj, BaseAnomalyDetector):
            log.warning(
                "plugin_invalid_subclass",
                loaded_class=obj.__name__,
                message=(
                    f"Entry point '{name}' ({obj.__name__}) "
                    f"is not a BaseAnomalyDetector subclass"
                ),
            )
            continue

        # Register the plugin model.
        try:
            register_model(name)(obj)
            newly_registered.append(name)
            log.info(
                "plugin_registered",
                model_class=obj.__name__,
                message=f"Plugin model '{name}' registered successfully",
            )
        except ValueError as exc:
            # register_model raises ValueError on duplicate — should not
            # happen since we checked _REGISTRY above, but handle it
            # defensively.
            log.warning(
                "plugin_registration_failed",
                error=str(exc),
                message=f"Failed to register plugin '{name}'",
            )

    if newly_registered:
        logger.info(
            "plugin_discovery_complete",
            newly_registered=newly_registered,
            count=len(newly_registered),
        )
    else:
        logger.debug("plugin_discovery_complete", count=0)

    return newly_registered
