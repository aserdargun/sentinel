"""Core abstractions for Sentinel."""

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.exceptions import (
    ConfigError,
    ModelNotFoundError,
    SentinelError,
    ValidationError,
)
from sentinel.core.registry import get_model_class, list_models, register_model
from sentinel.core.types import DetectionResult, ModelCategory, TrainResult

__all__ = [
    "BaseAnomalyDetector",
    "ConfigError",
    "DetectionResult",
    "ModelCategory",
    "ModelNotFoundError",
    "SentinelError",
    "TrainResult",
    "ValidationError",
    "get_model_class",
    "list_models",
    "register_model",
]
