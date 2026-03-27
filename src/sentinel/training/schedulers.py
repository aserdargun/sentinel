"""Learning rate scheduler wrappers with optional warmup.

Supports ReduceLROnPlateau, CosineAnnealingLR, and StepLR from PyTorch,
plus a linear warmup wrapper.  All torch imports are conditional so this
module can be imported even when torch is not installed.

Configured via :class:`~sentinel.core.config.SchedulerConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from sentinel.core.config import SchedulerConfig

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Sentinel for "torch not available".
_TORCH_AVAILABLE = False
try:
    import torch
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LRScheduler,
        ReduceLROnPlateau,
        StepLR,
    )

    _TORCH_AVAILABLE = True
except ImportError:
    pass


class WarmupWrapper:
    """Linear warmup followed by a base LR scheduler.

    During the first ``warmup_epochs`` epochs the learning rate increases
    linearly from near-zero to the optimizer's initial LR.  After warmup
    completes, all ``step()`` calls are forwarded to the wrapped
    ``base_scheduler``.

    Args:
        optimizer: The PyTorch optimizer whose LR groups to adjust.
        base_scheduler: The scheduler to use after warmup finishes.
        warmup_epochs: Number of epochs for the linear warmup phase.

    Example::

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cosine = CosineAnnealingLR(optimizer, T_max=100)
        scheduler = WarmupWrapper(optimizer, cosine, warmup_epochs=5)
        for epoch in range(105):
            train(...)
            scheduler.step()  # warmup for 5, then cosine for 100
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler: LRScheduler,
        warmup_epochs: int,
    ) -> None:
        self._optimizer = optimizer
        self._base_scheduler = base_scheduler
        self._warmup_epochs = warmup_epochs
        self._current_epoch = 0

        # Store the target LR for each param group (set by optimizer init).
        self._base_lrs: list[float] = [group["lr"] for group in optimizer.param_groups]

        # Set initial LR to near-zero for the warmup start.
        if warmup_epochs > 0:
            for group in optimizer.param_groups:
                group["lr"] = 0.0

    @property
    def warmup_epochs(self) -> int:
        """Number of warmup epochs."""
        return self._warmup_epochs

    @property
    def current_epoch(self) -> int:
        """Current epoch counter."""
        return self._current_epoch

    def step(self, metrics: float | None = None) -> None:
        """Advance the scheduler by one epoch.

        During warmup, linearly interpolates the LR.  After warmup,
        delegates to the base scheduler.

        Args:
            metrics: Optional metric value (required by ReduceLROnPlateau
                after warmup; ignored during warmup).
        """
        self._current_epoch += 1

        if self._current_epoch <= self._warmup_epochs:
            # Linear warmup: LR = base_lr * (epoch / warmup_epochs).
            fraction = self._current_epoch / self._warmup_epochs
            for group, base_lr in zip(self._optimizer.param_groups, self._base_lrs):
                group["lr"] = base_lr * fraction
            logger.debug(
                "warmup_step",
                epoch=self._current_epoch,
                fraction=round(fraction, 4),
            )
        else:
            # Delegate to the base scheduler.
            if isinstance(self._base_scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self._base_scheduler.step(metrics)
                else:
                    logger.warning(
                        "scheduler_missing_metrics",
                        message=(
                            "ReduceLROnPlateau.step() requires a metric "
                            "value but none was provided"
                        ),
                    )
            else:
                self._base_scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Return the last computed LR for each parameter group.

        Returns:
            List of learning rates, one per parameter group.
        """
        return [group["lr"] for group in self._optimizer.param_groups]


def create_scheduler(
    optimizer: Any,
    config: SchedulerConfig,
    total_epochs: int = 100,
) -> Any | None:
    """Create a learning rate scheduler from config.

    Supports four scheduler types:

    - ``"reduce_on_plateau"``: :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`
    - ``"cosine"``: :class:`torch.optim.lr_scheduler.CosineAnnealingLR`
    - ``"step"``: :class:`torch.optim.lr_scheduler.StepLR`
    - ``"none"``: returns ``None`` (no scheduling)

    If ``config.warmup_epochs > 0``, the base scheduler is wrapped in a
    :class:`WarmupWrapper` that linearly increases the LR for the specified
    number of epochs before delegating.

    Args:
        optimizer: A PyTorch optimizer instance.
        config: Scheduler configuration from :class:`SchedulerConfig`.
        total_epochs: Total training epochs (used by CosineAnnealingLR for
            ``T_max``). Defaults to 100.

    Returns:
        A scheduler instance, a :class:`WarmupWrapper`, or ``None`` if the
        scheduler type is ``"none"`` or torch is unavailable.

    Raises:
        ValueError: If the scheduler type is not recognized.

    Example::

        from sentinel.core.config import SchedulerConfig
        config = SchedulerConfig(type="cosine", warmup_epochs=5)
        scheduler = create_scheduler(optimizer, config, total_epochs=50)
        for epoch in range(50):
            train(...)
            scheduler.step()
    """
    if not _TORCH_AVAILABLE:
        logger.warning(
            "scheduler_torch_unavailable",
            message="torch not installed, returning no scheduler",
        )
        return None

    scheduler_type = config.type.lower()

    if scheduler_type == "none":
        logger.debug("scheduler_type_none")
        return None

    # Effective epochs for the base scheduler (excluding warmup).
    effective_epochs = max(total_epochs - config.warmup_epochs, 1)

    base_scheduler: LRScheduler
    if scheduler_type == "reduce_on_plateau":
        base_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.patience,
            factor=config.factor,
            min_lr=config.min_lr,
        )
    elif scheduler_type == "cosine":
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=effective_epochs,
            eta_min=config.min_lr,
        )
    elif scheduler_type == "step":
        # Step every `patience` epochs (reusing the patience field).
        base_scheduler = StepLR(
            optimizer,
            step_size=config.patience,
            gamma=config.factor,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Supported: reduce_on_plateau, cosine, step, none"
        )

    logger.info(
        "scheduler_created",
        type=scheduler_type,
        warmup_epochs=config.warmup_epochs,
        total_epochs=total_epochs,
        effective_epochs=effective_epochs,
    )

    # Wrap with warmup if configured.
    if config.warmup_epochs > 0:
        return WarmupWrapper(
            optimizer=optimizer,
            base_scheduler=base_scheduler,
            warmup_epochs=config.warmup_epochs,
        )

    return base_scheduler
