"""Background job manager for async training via ProcessPoolExecutor."""

from __future__ import annotations

import platform
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

import structlog

from sentinel.core.config import RunConfig
from sentinel.core.types import TrainResult

logger = structlog.get_logger(__name__)


def _run_training(config_dict: dict[str, Any], data_path: str | None) -> TrainResult:
    """Execute training in a worker process.

    This function is the top-level callable submitted to
    :class:`ProcessPoolExecutor`.  It must be importable at the module
    level (not a lambda or closure) so that it can be pickled.

    Args:
        config_dict: Serialized RunConfig fields (produced by
            :func:`_config_to_dict`).
        data_path: Optional data file override.

    Returns:
        TrainResult from the training pipeline.
    """
    from sentinel.core.config import RunConfig
    from sentinel.training.trainer import Trainer

    config = RunConfig._from_dict(config_dict)
    trainer = Trainer(config)
    return trainer.run(data_path=data_path)


def _config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Convert a RunConfig to a plain dict for cross-process pickling.

    Args:
        config: The run configuration.

    Returns:
        Plain dict representation.
    """
    from dataclasses import asdict

    result = asdict(config)
    result.update(result.pop("extra", {}))
    return result


class _JobRecord:
    """Internal bookkeeping for a submitted job."""

    __slots__ = (
        "job_id",
        "model_name",
        "future",
        "submitted_at",
        "data_path",
    )

    def __init__(
        self,
        job_id: str,
        model_name: str,
        future: Future[TrainResult],
        data_path: str | None,
    ) -> None:
        self.job_id = job_id
        self.model_name = model_name
        self.future = future
        self.submitted_at = time.monotonic()
        self.data_path = data_path


class BackgroundJobManager:
    """Manages async training jobs via ProcessPoolExecutor.

    On macOS the pool uses the ``fork`` start method because the default
    ``spawn`` method is incompatible with many C-extension libraries used
    in the training pipeline.

    Args:
        max_workers: Maximum number of concurrent training processes.
            Defaults to 2.

    Example::

        manager = BackgroundJobManager()
        job_id = manager.submit_job(config)
        status = manager.get_job_status(job_id)
    """

    def __init__(self, max_workers: int = 2) -> None:
        if platform.system() == "Darwin":
            ctx = get_context("fork")
        else:
            ctx = get_context("fork")
        self._pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
        )
        self._jobs: dict[str, _JobRecord] = {}
        logger.info(
            "job_manager.init",
            max_workers=max_workers,
            mp_start_method=ctx.get_start_method(),
        )

    def submit_job(
        self,
        config: RunConfig,
        data_path: str | None = None,
    ) -> str:
        """Submit a training job to the process pool.

        The config is serialized to a plain dict before being sent to the
        worker process so that dataclass instances do not need to be
        pickled.

        Args:
            config: Fully resolved run configuration.
            data_path: Optional data path override.

        Returns:
            Unique job_id string.
        """
        job_id = uuid.uuid4().hex[:12]
        config_dict = _config_to_dict(config)

        future = self._pool.submit(_run_training, config_dict, data_path)
        record = _JobRecord(
            job_id=job_id,
            model_name=config.model,
            future=future,
            data_path=data_path,
        )
        self._jobs[job_id] = record

        logger.info(
            "job_manager.submitted",
            job_id=job_id,
            model=config.model,
        )
        return job_id

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Return the current status of a training job.

        Args:
            job_id: The job identifier returned by :meth:`submit_job`.

        Returns:
            Dict with keys: ``job_id``, ``status``, ``model_name``,
            ``progress_pct``, ``metrics``, ``error_message``,
            ``run_id``, ``duration_s``.

        Raises:
            KeyError: If the job_id is not found.
        """
        if job_id not in self._jobs:
            raise KeyError(f"Job not found: {job_id}")

        record = self._jobs[job_id]
        future = record.future
        elapsed = time.monotonic() - record.submitted_at

        result: dict[str, Any] = {
            "job_id": job_id,
            "model_name": record.model_name,
            "progress_pct": None,
            "metrics": None,
            "error_message": None,
            "run_id": None,
            "duration_s": round(elapsed, 3),
        }

        if future.cancelled():
            result["status"] = "cancelled"
        elif future.running():
            result["status"] = "running"
        elif future.done():
            exc = future.exception()
            if exc is not None:
                result["status"] = "failed"
                result["error_message"] = str(exc)
                logger.warning(
                    "job_manager.failed",
                    job_id=job_id,
                    error=str(exc),
                )
            else:
                train_result = future.result()
                result["status"] = "completed"
                result["progress_pct"] = 100.0
                result["metrics"] = train_result["metrics"]
                result["run_id"] = train_result["run_id"]
                result["duration_s"] = train_result["duration_s"]
        else:
            result["status"] = "pending"

        return result

    def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel a queued training job.

        Only jobs that have not yet started can be cancelled.  Already
        running or completed jobs cannot be cancelled.

        Args:
            job_id: The job identifier.

        Returns:
            ``True`` if the job was successfully cancelled, ``False``
            otherwise.

        Raises:
            KeyError: If the job_id is not found.
        """
        if job_id not in self._jobs:
            raise KeyError(f"Job not found: {job_id}")

        record = self._jobs[job_id]
        cancelled = record.future.cancel()

        if cancelled:
            logger.info("job_manager.cancelled", job_id=job_id)
        else:
            logger.info("job_manager.cancel_failed", job_id=job_id)

        return cancelled

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all known jobs with their current status.

        Returns:
            List of job status dicts.
        """
        return [self.get_job_status(jid) for jid in self._jobs]

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the process pool.

        Args:
            wait: If ``True``, block until all running jobs finish.
        """
        self._pool.shutdown(wait=wait)
        logger.info("job_manager.shutdown", wait=wait)
