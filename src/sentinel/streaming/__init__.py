"""Streaming detection, drift simulation, and alert rules."""

from sentinel.streaming.alerts import (
    AlertEngine,
    AlertRule,
    ConsecutiveAnomalyAlert,
    RateOfChangeAlert,
    ThresholdBreachAlert,
)
from sentinel.streaming.drift import ADWINDetector, DriftSimulator
from sentinel.streaming.online_detector import OnlineDetector
from sentinel.streaming.simulator import StreamSimulator

__all__ = [
    "ADWINDetector",
    "AlertEngine",
    "AlertRule",
    "ConsecutiveAnomalyAlert",
    "DriftSimulator",
    "OnlineDetector",
    "RateOfChangeAlert",
    "StreamSimulator",
    "ThresholdBreachAlert",
]
