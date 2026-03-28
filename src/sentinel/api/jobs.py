"""Background job manager for async training via subprocess.

Uses ``subprocess.Popen`` to run training in a completely fresh Python
process, avoiding the macOS ``fork`` deadlock issue with
``ProcessPoolExecutor`` (structlog, uvicorn, and other libraries hold
thread locks that deadlock when copied via ``fork``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import structlog

from sentinel.core.config import RunConfig
from sentinel.core.types import TrainResult

logger = structlog.get_logger(__name__)

# Inline training script executed by the subprocess.
_TRAIN_SCRIPT = """\
import json, sys

config_path = sys.argv[1]
data_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "" else None

from sentinel.core.config import RunConfig
from sentinel.training.trainer import Trainer

config = RunConfig._from_dict(json.loads(open(config_path).read()))
trainer = Trainer(config)
result = trainer.run(data_path=data_path)

# Write result as JSON to stdout.
out = {
    "run_id": result["run_id"],
    "model_name": result["model_name"],
    "duration_s": result["duration_s"],
    "metrics": {},
}
for k, v in result["metrics"].items():
    import numpy as np
    if isinstance(v, (np.floating, float)):
        out["metrics"][k] = float(v)
    elif isinstance(v, (np.integer, int)):
        out["metrics"][k] = int(v)
    elif v is None:
        out["metrics"][k] = None
    else:
        out["metrics"][k] = v

print(json.dumps(out))
"""


class _JobRecord:
    """Internal bookkeeping for a submitted job."""

    __slots__ = (
        "job_id",
        "model_name",
        "process",
        "submitted_at",
        "data_path",
        "training_timeout_s",
        "timed_out",
        "result",
        "error_message",
        "config_tmp",
    )

    def __init__(
        self,
        job_id: str,
        model_name: str,
        process: subprocess.Popen[str],
        data_path: str | None,
        training_timeout_s: int,
        config_tmp: Path,
    ) -> None:
        self.job_id = job_id
        self.model_name = model_name
        self.process = process
        self.submitted_at = time.monotonic()
        self.data_path = data_path
        self.training_timeout_s = training_timeout_s
        self.timed_out = False
        self.result: TrainResult | None = None
        self.error_message: str | None = None
        self.config_tmp = config_tmp


def _config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Convert a RunConfig to a plain dict for JSON serialization.

    Args:
        config: The run configuration.

    Returns:
        Plain dict representation.
    """
    result = asdict(config)
    result.update(result.pop("extra", {}))
    return result


class BackgroundJobManager:
    """Manages async training jobs via subprocesses.

    Each training job runs in a completely fresh Python process via
    ``subprocess.Popen``, avoiding the macOS ``fork`` deadlock that
    occurs with ``ProcessPoolExecutor``.

    Args:
        max_workers: Maximum number of concurrent training processes.
            Defaults to 2.

    Example::

        manager = BackgroundJobManager()
        job_id = manager.submit_job(config)
        status = manager.get_job_status(job_id)
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._max_workers = max_workers
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()
        logger.info("job_manager.init", max_workers=max_workers, method="subprocess")

    def submit_job(
        self,
        config: RunConfig,
        data_path: str | None = None,
    ) -> str:
        """Submit a training job as a subprocess.

        The config is serialized to a temp JSON file which the subprocess
        reads on startup.

        Args:
            config: Fully resolved run configuration.
            data_path: Optional data path override.

        Returns:
            Unique job_id string.
        """
        job_id = uuid.uuid4().hex[:12]
        config_dict = _config_to_dict(config)

        # Write config to a temp file for the subprocess to read.
        config_tmp = Path(tempfile.mktemp(suffix=".json", prefix=f"sentinel_job_{job_id}_"))
        config_tmp.write_text(json.dumps(config_dict, default=str))

        # Launch subprocess.
        cmd = [sys.executable, "-c", _TRAIN_SCRIPT, str(config_tmp)]
        if data_path:
            cmd.append(data_path)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        record = _JobRecord(
            job_id=job_id,
            model_name=config.model,
            process=process,
            data_path=data_path,
            training_timeout_s=config.runtime.training_timeout_s,
            config_tmp=config_tmp,
        )

        with self._lock:
            self._jobs[job_id] = record

        # Start a background thread to wait for the process and collect results.
        thread = threading.Thread(
            target=self._wait_for_job,
            args=(record,),
            daemon=True,
        )
        thread.start()

        logger.info(
            "job_manager.submitted",
            job_id=job_id,
            model=config.model,
        )
        return job_id

    def _wait_for_job(self, record: _JobRecord) -> None:
        """Wait for a subprocess to finish and collect its result.

        Args:
            record: The job record to monitor.
        """
        try:
            stdout, stderr = record.process.communicate(
                timeout=record.training_timeout_s,
            )

            # Clean up temp config file.
            record.config_tmp.unlink(missing_ok=True)

            if record.process.returncode != 0:
                record.error_message = stderr.strip() or f"Process exited with code {record.process.returncode}"
                logger.warning(
                    "job_manager.failed",
                    job_id=record.job_id,
                    error=record.error_message,
                )
                return

            # Parse the JSON result from stdout (last line).
            output_lines = stdout.strip().splitlines()
            if not output_lines:
                record.error_message = "No output from training process"
                return

            result_data = json.loads(output_lines[-1])
            record.result = TrainResult(
                run_id=result_data["run_id"],
                model_name=result_data["model_name"],
                metrics=result_data["metrics"],
                duration_s=result_data["duration_s"],
            )

        except subprocess.TimeoutExpired:
            record.process.kill()
            record.process.wait()
            record.timed_out = True
            record.error_message = (
                f"Training job timed out after "
                f"{record.training_timeout_s}s "
                f"(limit: runtime.training_timeout_s="
                f"{record.training_timeout_s})"
            )
            record.config_tmp.unlink(missing_ok=True)
            logger.warning(
                "job_manager.timeout",
                job_id=record.job_id,
                timeout_s=record.training_timeout_s,
            )
        except Exception as exc:
            record.error_message = str(exc)
            record.config_tmp.unlink(missing_ok=True)
            logger.error(
                "job_manager.error",
                job_id=record.job_id,
                error=str(exc),
            )

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
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job not found: {job_id}")
            record = self._jobs[job_id]

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

        if record.timed_out:
            result["status"] = "failed"
            result["error_message"] = record.error_message
        elif record.error_message is not None:
            result["status"] = "failed"
            result["error_message"] = record.error_message
        elif record.result is not None:
            result["status"] = "completed"
            result["progress_pct"] = 100.0
            result["metrics"] = record.result["metrics"]
            result["run_id"] = record.result["run_id"]
            result["duration_s"] = record.result["duration_s"]
        elif record.process.poll() is not None:
            # Process ended but _wait_for_job hasn't set result yet.
            result["status"] = "running"
        else:
            result["status"] = "running"

        return result

    def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel a training job.

        Terminates the subprocess if it is still running.

        Args:
            job_id: The job identifier.

        Returns:
            ``True`` if the job was successfully cancelled, ``False``
            otherwise.

        Raises:
            KeyError: If the job_id is not found.
        """
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job not found: {job_id}")
            record = self._jobs[job_id]

        if record.result is not None or record.error_message is not None:
            return False

        if record.process.poll() is None:
            record.process.terminate()
            record.error_message = "Cancelled by user"
            logger.info("job_manager.cancelled", job_id=job_id)
            return True

        return False

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all known jobs with their current status.

        Returns:
            List of job status dicts.
        """
        with self._lock:
            job_ids = list(self._jobs.keys())
        return [self.get_job_status(jid) for jid in job_ids]

    def shutdown(self, wait: bool = True) -> None:
        """Terminate all running jobs.

        Args:
            wait: If ``True``, wait for processes to exit after terminating.
        """
        with self._lock:
            for record in self._jobs.values():
                if record.process.poll() is None:
                    record.process.terminate()
            if wait:
                for record in self._jobs.values():
                    record.process.wait()
        logger.info("job_manager.shutdown", wait=wait)
