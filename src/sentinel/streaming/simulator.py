"""Stream simulator: orchestrates data replay with online detection.

Consumes async generators from :mod:`sentinel.data.streaming`, adds
speed control and optional anomaly injection, and feeds each row
through an :class:`~sentinel.streaming.online_detector.OnlineDetector`.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from typing import Any

import structlog

from sentinel.streaming.alerts import AlertEngine
from sentinel.streaming.online_detector import OnlineDetector

logger = structlog.get_logger(__name__)


class StreamSimulator:
    """Orchestrates streaming replay with online anomaly detection.

    Wraps an async data source, optionally injects anomalies for
    testing, and passes each row through an
    :class:`OnlineDetector`.  Each iteration yields a result dict
    containing the original row data plus detection scores and any
    triggered alerts.

    Args:
        source: Async iterator yielding row dicts.
        detector: Configured :class:`OnlineDetector` instance.
        speed: Replay speed multiplier applied to the source.
            Only informational here -- the actual pacing is done by
            the source generator (see :mod:`sentinel.data.streaming`).
        inject_anomalies: If ``True``, randomly perturb a fraction of
            rows to simulate anomalous observations.
        anomaly_ratio: Fraction of rows to perturb when
            ``inject_anomalies`` is ``True``.
        alert_engine: Optional :class:`AlertEngine` to evaluate on
            each scored point.

    Example::

        source = stream_from_parquet("data.parquet", speed=10.0)
        detector = OnlineDetector(model, seq_len=50, threshold=0.75)
        sim = StreamSimulator(source, detector)
        async for result in sim.run():
            print(result)
    """

    def __init__(
        self,
        source: AsyncIterator[dict],
        detector: OnlineDetector,
        speed: float = 1.0,
        inject_anomalies: bool = False,
        anomaly_ratio: float = 0.05,
        alert_engine: AlertEngine | None = None,
    ) -> None:
        self._source = source
        self._detector = detector
        self._speed = speed
        self._inject_anomalies = inject_anomalies
        self._anomaly_ratio = anomaly_ratio
        self._alert_engine = alert_engine

        self._rows_processed: int = 0
        self._anomalies_injected: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> AsyncIterator[dict[str, Any]]:
        """Consume the source stream and yield detection results.

        Each yielded dict contains:

        - ``row``: the (possibly perturbed) original data dict.
        - ``detection``: the result from :meth:`OnlineDetector.update`,
          or ``None`` if the buffer is still filling.
        - ``alerts``: list of alert dicts (empty if no alerts fired).
        - ``injected``: ``True`` if this row was artificially perturbed.

        Yields:
            Result dict for every row consumed from the source.
        """
        logger.info("simulator_start", speed=self._speed)

        async for row in self._source:
            self._rows_processed += 1
            injected = False

            # Optionally inject anomalies by scaling feature values.
            if self._inject_anomalies and random.random() < self._anomaly_ratio:
                row = _inject_anomaly(row)
                injected = True
                self._anomalies_injected += 1

            detection = self._detector.update(row)

            # Evaluate alert rules if a detection result is available.
            alerts: list[dict[str, Any]] = []
            if detection is not None and self._alert_engine is not None:
                alerts = self._alert_engine.check(
                    score=detection["score"],
                    label=detection["label"],
                    timestamp=detection.get("timestamp"),
                )

            yield {
                "row": row,
                "detection": detection,
                "alerts": alerts,
                "injected": injected,
            }

        logger.info(
            "simulator_done",
            rows_processed=self._rows_processed,
            anomalies_injected=self._anomalies_injected,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rows_processed(self) -> int:
        """Total rows consumed so far."""
        return self._rows_processed

    @property
    def anomalies_injected(self) -> int:
        """Number of rows that were artificially perturbed."""
        return self._anomalies_injected


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_EXCLUDED_KEYS = {"timestamp", "is_anomaly"}
_ANOMALY_SCALE_MIN = 3.0
_ANOMALY_SCALE_MAX = 5.0


def _inject_anomaly(row: dict[str, Any]) -> dict[str, Any]:
    """Create an anomalous copy of a row by scaling numeric features.

    Numeric feature values are multiplied by a random factor drawn
    from ``[ANOMALY_SCALE_MIN, ANOMALY_SCALE_MAX]``.  Non-numeric
    values and reserved keys (``timestamp``, ``is_anomaly``) are
    left untouched.

    Args:
        row: Original data row.

    Returns:
        A *new* dict with perturbed feature values.
    """
    perturbed = dict(row)
    scale = random.uniform(_ANOMALY_SCALE_MIN, _ANOMALY_SCALE_MAX)

    for key, val in row.items():
        if key in _EXCLUDED_KEYS:
            continue
        try:
            perturbed[key] = float(val) * scale
        except (TypeError, ValueError):
            continue

    return perturbed
