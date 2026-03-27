"""Training callbacks: early stopping and model checkpointing."""

from __future__ import annotations

import os
from typing import Any

import structlog

from sentinel.core.base_model import BaseAnomalyDetector

logger = structlog.get_logger(__name__)


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Tracks the best value of a metric across epochs.  If the metric has
    not improved by at least ``delta`` for ``patience`` consecutive
    checks, ``check()`` returns ``True`` to signal the caller to stop.

    Args:
        patience: Number of checks with no improvement before stopping.
        delta: Minimum absolute change to qualify as an improvement.
        mode: ``"min"`` expects the metric to decrease (e.g., loss);
              ``"max"`` expects it to increase (e.g., F1).

    Example::

        es = EarlyStopping(patience=10, delta=1e-4, mode="min")
        for epoch in range(max_epochs):
            loss = train_one_epoch(...)
            if es.check(loss):
                print(f"Stopping at epoch {epoch}")
                break
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.delta = abs(delta)
        self.mode = mode

        self._best: float | None = None
        self._counter: int = 0
        self._stopped: bool = False

    def check(self, metric_value: float) -> bool:
        """Check whether training should stop.

        Args:
            metric_value: Current value of the monitored metric.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if self._best is None:
            self._best = metric_value
            self._counter = 0
            logger.debug(
                "early_stopping_init",
                best=self._best,
                mode=self.mode,
            )
            return False

        improved = self._is_improvement(metric_value)

        if improved:
            self._best = metric_value
            self._counter = 0
            logger.debug(
                "early_stopping_improved",
                best=self._best,
                mode=self.mode,
            )
        else:
            self._counter += 1
            logger.debug(
                "early_stopping_no_improvement",
                counter=self._counter,
                patience=self.patience,
                best=self._best,
                current=metric_value,
            )

        if self._counter >= self.patience:
            self._stopped = True
            logger.info(
                "early_stopping_triggered",
                patience=self.patience,
                best=self._best,
            )
            return True

        return False

    def reset(self) -> None:
        """Reset internal state for a new training run."""
        self._best = None
        self._counter = 0
        self._stopped = False

    @property
    def stopped(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._stopped

    @property
    def best(self) -> float | None:
        """Best metric value observed so far."""
        return self._best

    @property
    def counter(self) -> int:
        """Number of checks since last improvement."""
        return self._counter

    def _is_improvement(self, value: float) -> bool:
        """Check if value improves over current best by at least delta."""
        assert self._best is not None
        if self.mode == "min":
            return value < (self._best - self.delta)
        return value > (self._best + self.delta)


class ModelCheckpoint:
    """Saves the model whenever a monitored metric improves.

    Tracks the best value of a metric.  When ``check()`` detects an
    improvement, it saves the model to the given path (atomic write).

    Args:
        patience: Not used for saving (present for consistency with
            EarlyStopping). Reserved for future use.
        delta: Minimum absolute change to qualify as an improvement.
        mode: ``"min"`` expects the metric to decrease;
              ``"max"`` expects it to increase.

    Example::

        ckpt = ModelCheckpoint(delta=1e-4, mode="min")
        for epoch in range(max_epochs):
            loss = train_one_epoch(...)
            ckpt.check(loss, model, save_path)
    """

    def __init__(
        self,
        patience: int = 0,
        delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.delta = abs(delta)
        self.mode = mode

        self._best: float | None = None
        self._save_count: int = 0

    def check(
        self,
        metric_value: float,
        model: BaseAnomalyDetector,
        path: str,
    ) -> bool:
        """Check metric and save model if improved.

        Args:
            metric_value: Current value of the monitored metric.
            model: Model instance to save.
            path: Directory path where the model will be saved.

        Returns:
            ``True`` if the model was saved (metric improved),
            ``False`` otherwise.
        """
        if self._best is None:
            self._best = metric_value
            self._save_model(model, path)
            return True

        improved = self._is_improvement(metric_value)

        if improved:
            self._best = metric_value
            self._save_model(model, path)
            return True

        return False

    def reset(self) -> None:
        """Reset internal state for a new training run."""
        self._best = None
        self._save_count = 0

    @property
    def best(self) -> float | None:
        """Best metric value observed so far."""
        return self._best

    @property
    def save_count(self) -> int:
        """Number of times the model has been saved."""
        return self._save_count

    def _is_improvement(self, value: float) -> bool:
        """Check if value improves over current best by at least delta."""
        assert self._best is not None
        if self.mode == "min":
            return value < (self._best - self.delta)
        return value > (self._best + self.delta)

    def _save_model(self, model: BaseAnomalyDetector, path: str) -> None:
        """Save model and update counter."""
        os.makedirs(path, exist_ok=True)
        model.save(path)
        self._save_count += 1
        logger.info(
            "model_checkpoint_saved",
            path=path,
            best=self._best,
            save_count=self._save_count,
        )


def get_callbacks(config: dict[str, Any]) -> dict[str, Any]:
    """Build callback instances from a config dict.

    Expected keys (all optional):

    - ``early_stopping.patience`` (int)
    - ``early_stopping.delta`` (float)
    - ``early_stopping.mode`` (str)
    - ``checkpoint.delta`` (float)
    - ``checkpoint.mode`` (str)

    Args:
        config: Callback configuration dictionary.

    Returns:
        Dict with ``"early_stopping"`` and/or ``"checkpoint"`` keys
        mapped to their callback instances.
    """
    callbacks: dict[str, Any] = {}

    es_cfg = config.get("early_stopping")
    if es_cfg is not None:
        callbacks["early_stopping"] = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            delta=es_cfg.get("delta", 0.0),
            mode=es_cfg.get("mode", "min"),
        )

    ckpt_cfg = config.get("checkpoint")
    if ckpt_cfg is not None:
        callbacks["checkpoint"] = ModelCheckpoint(
            delta=ckpt_cfg.get("delta", 0.0),
            mode=ckpt_cfg.get("mode", "min"),
        )

    return callbacks
