"""Alert rules for streaming anomaly detection.

Defines a hierarchy of alert rules that fire when specific conditions
are met in a stream of anomaly scores.  An :class:`AlertEngine`
aggregates multiple rules and returns all triggered alerts for each
incoming score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class AlertRule(ABC):
    """Base class for streaming alert rules.

    Subclasses implement :meth:`check` which receives the current
    score, binary label, and timestamp, and returns an alert dict
    when the rule fires, or ``None`` otherwise.

    Args:
        name: Human-readable name for the rule (included in alerts).
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Human-readable rule name."""
        return self._name

    @abstractmethod
    def check(
        self,
        score: float,
        label: int,
        timestamp: str | None,
    ) -> dict[str, Any] | None:
        """Evaluate the rule against a single observation.

        Args:
            score: Anomaly score for the current point.
            label: Binary anomaly label (``1`` = anomaly).
            timestamp: ISO-format timestamp string, or ``None``.

        Returns:
            An alert dict if the rule fires, ``None`` otherwise.
        """

    def reset(self) -> None:
        """Reset internal state.  Override in stateful subclasses."""


class ThresholdBreachAlert(AlertRule):
    """Fires when the anomaly score exceeds a fixed threshold.

    This is the simplest alert rule: each score is compared
    independently against the threshold.

    Args:
        threshold: Score value above which the alert fires.
        name: Optional rule name.  Defaults to ``"threshold_breach"``.

    Example::

        rule = ThresholdBreachAlert(threshold=0.8)
        alert = rule.check(score=0.95, label=1, timestamp="2024-01-01T00:05:00Z")
        # alert == {"rule": "threshold_breach", "score": 0.95, ...}
    """

    def __init__(
        self,
        threshold: float,
        name: str = "threshold_breach",
    ) -> None:
        super().__init__(name)
        self._threshold = threshold

    def check(
        self,
        score: float,
        label: int,
        timestamp: str | None,
    ) -> dict[str, Any] | None:
        """Fire if *score* exceeds the configured threshold.

        Args:
            score: Current anomaly score.
            label: Binary anomaly label.
            timestamp: ISO-format timestamp string or ``None``.

        Returns:
            Alert dict or ``None``.
        """
        if score > self._threshold:
            return {
                "rule": self._name,
                "score": score,
                "threshold": self._threshold,
                "timestamp": timestamp,
            }
        return None


class ConsecutiveAnomalyAlert(AlertRule):
    """Fires after *count* consecutive anomaly labels.

    Maintains a running counter of consecutive anomaly observations
    (``label == 1``).  When the counter reaches the configured
    *count*, an alert is emitted.  The counter resets on a normal
    observation (``label == 0``) or after the alert fires.

    Args:
        count: Number of consecutive anomalies required to fire.
        name: Optional rule name.  Defaults to
            ``"consecutive_anomalies"``.

    Example::

        rule = ConsecutiveAnomalyAlert(count=3)
        rule.check(0.9, 1, "t1")  # None (1 in a row)
        rule.check(0.8, 1, "t2")  # None (2 in a row)
        rule.check(0.7, 1, "t3")  # fires: 3 consecutive
    """

    def __init__(
        self,
        count: int,
        name: str = "consecutive_anomalies",
    ) -> None:
        super().__init__(name)
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        self._required = count
        self._streak: int = 0
        self._start_ts: str | None = None

    def check(
        self,
        score: float,
        label: int,
        timestamp: str | None,
    ) -> dict[str, Any] | None:
        """Fire after *count* consecutive anomaly labels.

        Args:
            score: Current anomaly score.
            label: Binary anomaly label.
            timestamp: ISO-format timestamp string or ``None``.

        Returns:
            Alert dict or ``None``.
        """
        if label == 1:
            if self._streak == 0:
                self._start_ts = timestamp
            self._streak += 1

            if self._streak >= self._required:
                alert: dict[str, Any] = {
                    "rule": self._name,
                    "count": self._streak,
                    "start": self._start_ts,
                    "end": timestamp,
                }
                # Reset after firing so we can detect the next run.
                self._streak = 0
                self._start_ts = None
                return alert
        else:
            self._streak = 0
            self._start_ts = None

        return None

    def reset(self) -> None:
        """Clear the consecutive counter."""
        self._streak = 0
        self._start_ts = None


class RateOfChangeAlert(AlertRule):
    """Fires when the anomaly score changes rapidly.

    Compares the absolute difference between the current score and the
    previous score against a configurable *delta*.  If the change
    exceeds *delta*, the alert fires.

    Args:
        delta: Minimum score change to trigger the alert.
        name: Optional rule name.  Defaults to ``"rate_of_change"``.

    Example::

        rule = RateOfChangeAlert(delta=0.5)
        rule.check(0.2, 0, "t1")  # None (no previous)
        rule.check(0.9, 1, "t2")  # fires: |0.9 - 0.2| = 0.7 > 0.5
    """

    def __init__(
        self,
        delta: float,
        name: str = "rate_of_change",
    ) -> None:
        super().__init__(name)
        if delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta}")
        self._delta = delta
        self._prev_score: float | None = None

    def check(
        self,
        score: float,
        label: int,
        timestamp: str | None,
    ) -> dict[str, Any] | None:
        """Fire when the score change exceeds *delta*.

        Args:
            score: Current anomaly score.
            label: Binary anomaly label.
            timestamp: ISO-format timestamp string or ``None``.

        Returns:
            Alert dict or ``None``.
        """
        alert: dict[str, Any] | None = None

        if self._prev_score is not None:
            change = abs(score - self._prev_score)
            if change > self._delta:
                alert = {
                    "rule": self._name,
                    "delta": change,
                    "threshold_delta": self._delta,
                    "previous_score": self._prev_score,
                    "current_score": score,
                    "timestamp": timestamp,
                }

        self._prev_score = score
        return alert

    def reset(self) -> None:
        """Clear the previous score."""
        self._prev_score = None


class AlertEngine:
    """Aggregates multiple alert rules and evaluates them together.

    On each call to :meth:`check`, every registered rule is evaluated
    and all triggered alerts are returned.

    Args:
        rules: List of :class:`AlertRule` instances.

    Example::

        engine = AlertEngine([
            ThresholdBreachAlert(threshold=0.8),
            ConsecutiveAnomalyAlert(count=5),
            RateOfChangeAlert(delta=0.4),
        ])
        alerts = engine.check(score=0.95, label=1, timestamp="...")
    """

    def __init__(self, rules: list[AlertRule]) -> None:
        self._rules = list(rules)

    def check(
        self,
        score: float,
        label: int,
        timestamp: str | None,
    ) -> list[dict[str, Any]]:
        """Evaluate all rules and return triggered alerts.

        Args:
            score: Current anomaly score.
            label: Binary anomaly label (``1`` = anomaly).
            timestamp: ISO-format timestamp string, or ``None``.

        Returns:
            List of alert dicts from rules that fired.  May be empty.
        """
        alerts: list[dict[str, Any]] = []

        for rule in self._rules:
            try:
                result = rule.check(score, label, timestamp)
                if result is not None:
                    alerts.append(result)
            except Exception:
                logger.exception("alert_rule_error", rule=rule.name)

        if alerts:
            logger.info(
                "alerts_triggered",
                count=len(alerts),
                rules=[a.get("rule") for a in alerts],
            )

        return alerts

    def reset(self) -> None:
        """Reset internal state of all rules."""
        for rule in self._rules:
            rule.reset()

    @property
    def rules(self) -> list[AlertRule]:
        """The registered alert rules."""
        return list(self._rules)
