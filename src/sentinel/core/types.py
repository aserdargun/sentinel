"""Sentinel type definitions."""

from enum import Enum
from typing import TypedDict

import numpy as np


class ModelCategory(Enum):
    """Categories of anomaly detection models."""

    STATISTICAL = "statistical"
    DEEP = "deep"
    ENSEMBLE = "ensemble"


class DetectionResult(TypedDict):
    """Result from anomaly detection."""

    scores: np.ndarray
    labels: np.ndarray
    threshold: float


class TrainResult(TypedDict):
    """Result from model training."""

    run_id: str
    model_name: str
    metrics: dict[str, float | None]
    duration_s: float
