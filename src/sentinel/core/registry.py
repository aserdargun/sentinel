"""Model registry with decorator-based registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentinel.core.exceptions import ModelNotFoundError

if TYPE_CHECKING:
    from sentinel.core.base_model import BaseAnomalyDetector

_REGISTRY: dict[str, type[BaseAnomalyDetector]] = {}


def register_model(name: str) -> type[BaseAnomalyDetector]:
    """Decorator to register a model class in the registry.

    Args:
        name: Unique string identifier for the model.

    Returns:
        The decorated class, unchanged.

    Raises:
        ValueError: If a model with the same name is already registered.
    """

    def decorator(cls: type[BaseAnomalyDetector]) -> type[BaseAnomalyDetector]:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _REGISTRY[name] = cls
        cls.model_name = name  # type: ignore[attr-defined]
        return cls

    return decorator  # type: ignore[return-value]


def get_model_class(name: str) -> type[BaseAnomalyDetector]:
    """Look up a model class by name.

    Args:
        name: Registered model name.

    Returns:
        The model class.

    Raises:
        ModelNotFoundError: If the name is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ModelNotFoundError(f"Model '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def list_models() -> dict[str, type[BaseAnomalyDetector]]:
    """Return a copy of the model registry.

    Returns:
        Dict mapping model names to their classes.
    """
    return dict(_REGISTRY)
