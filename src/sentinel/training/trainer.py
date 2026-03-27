"""Unified Trainer: config -> registry lookup -> fit -> evaluate -> track."""

from __future__ import annotations

import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import structlog

import sentinel.models  # noqa: F401 — triggers model registration
from sentinel.core.config import RunConfig
from sentinel.core.exceptions import ConfigError, ValidationError
from sentinel.core.registry import get_model_class
from sentinel.core.types import TrainResult
from sentinel.data.loaders import load_file
from sentinel.data.preprocessors import (
    chronological_split,
    fill_nan,
    scale_zscore,
    to_numpy,
)
from sentinel.data.validators import separate_labels, validate_dataframe
from sentinel.training.evaluator import Evaluator

logger = structlog.get_logger(__name__)


class Trainer:
    """Orchestrates the full training pipeline.

    The ``run()`` method performs the following sequence:

    1. Set global random seeds for reproducibility.
    2. Load data from file (CSV or Parquet).
    3. Validate the DataFrame against the canonical schema.
    4. Separate ``is_anomaly`` labels from features.
    5. Fill NaN values (forward-fill then zero-fill).
    6. Z-score scale feature columns.
    7. Split chronologically into train / val / test sets.
    8. Convert Polars DataFrames to numpy arrays.
    9. Look up the model class from the registry and instantiate.
    10. If ``training_mode == "normal_only"`` and labels exist, filter
        out anomalous rows from the training set.
    11. Fit the model on the training data.
    12. Score validation and test sets.
    13. Evaluate: threshold on validation, metrics on test.
    14. Return a :class:`~sentinel.core.types.TrainResult`.

    Args:
        config: Fully resolved run configuration.

    Example::

        config = RunConfig.from_yaml("configs/zscore.yaml")
        trainer = Trainer(config)
        result = trainer.run()
        print(result["metrics"])
    """

    def __init__(self, config: RunConfig) -> None:
        self._config = config

    @property
    def config(self) -> RunConfig:
        """The run configuration."""
        return self._config

    def run(self, data_path: str | None = None) -> TrainResult:
        """Execute the full training pipeline.

        Args:
            data_path: Optional override for the data file or directory.
                If ``None``, uses ``config.data_path``.  If a directory
                is given, the first ``.parquet`` file found is loaded.

        Returns:
            A :class:`TrainResult` with run_id, model_name, metrics, and
            wall-clock duration.

        Raises:
            ConfigError: If the model name is missing or not registered.
            ValidationError: If the data fails validation.
        """
        start_time = time.monotonic()
        run_id = uuid.uuid4().hex[:12]

        log = logger.bind(run_id=run_id, model=self._config.model)
        log.info("trainer_start")

        # ---- 1. Seeds ----
        self._set_seeds(self._config.seed)
        log.debug("seeds_set", seed=self._config.seed)

        # ---- 2. Validate config ----
        if not self._config.model:
            raise ConfigError("config.model must be set")

        # ---- 3. Load data ----
        resolved_path = self._resolve_data_path(data_path)
        log.info("data_loading", path=str(resolved_path))
        df = load_file(resolved_path)
        log.info("data_loaded", rows=df.height, columns=df.width)

        # ---- 4. Validate ----
        df = validate_dataframe(df)

        # ---- 5. Separate labels ----
        df, labels_series = separate_labels(df)
        has_labels = labels_series is not None
        log.info("labels_separated", has_labels=has_labels)

        # ---- 6. Fill NaN ----
        df = fill_nan(df)

        # ---- 7. Scale ----
        df, _scale_stats = scale_zscore(df)

        # ---- 8. Split ----
        train_df, val_df, test_df = chronological_split(
            df,
            train_ratio=self._config.split.train,
            val_ratio=self._config.split.val,
            test_ratio=self._config.split.test,
        )
        log.info(
            "data_split",
            train=train_df.height,
            val=val_df.height,
            test=test_df.height,
        )

        # Also split the labels in the same way if they exist.
        train_labels: np.ndarray | None = None
        val_labels: np.ndarray | None = None
        test_labels: np.ndarray | None = None

        if has_labels:
            assert labels_series is not None
            all_labels = labels_series.to_numpy().astype(np.int32)
            n = len(all_labels)
            train_end = int(n * self._config.split.train)
            val_end = train_end + int(n * self._config.split.val)

            train_labels = all_labels[:train_end]
            val_labels = all_labels[train_end:val_end]
            test_labels = all_labels[val_end:]

        # ---- 9. To numpy ----
        X_train = to_numpy(train_df)
        X_val = to_numpy(val_df)
        X_test = to_numpy(test_df)

        # ---- 10. Instantiate model ----
        model_cls = get_model_class(self._config.model)
        model_params = self._extract_model_params(model_cls)
        model = model_cls(**model_params)
        log.info("model_instantiated", model=self._config.model, params=model_params)

        # ---- 11. Filter normal-only training ----
        X_train_fit = self._apply_training_mode(X_train, train_labels, log)

        # ---- 12. Fit ----
        log.info("training_start")
        model.fit(X_train_fit)
        log.info("training_complete")

        # ---- 13. Score ----
        val_scores = model.score(X_val)
        test_scores = model.score(X_test)
        log.info(
            "scoring_complete",
            val_scores_shape=val_scores.shape,
            test_scores_shape=test_scores.shape,
        )

        # ---- 14. Evaluate ----
        evaluator = Evaluator()
        metrics = evaluator.evaluate(
            scores=test_scores,
            labels=test_labels,
            val_scores=val_scores,
            val_labels=val_labels,
        )
        log.info("evaluation_complete", metrics=metrics)

        duration_s = time.monotonic() - start_time

        result = TrainResult(
            run_id=run_id,
            model_name=self._config.model,
            metrics=metrics,
            duration_s=round(duration_s, 3),
        )

        log.info("trainer_done", duration_s=result["duration_s"])
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_data_path(self, data_path: str | None) -> Path:
        """Resolve the data path from the override or config.

        If the path is a directory, returns the first ``.parquet`` file
        found.  If it is a file, returns it directly.

        Args:
            data_path: Optional override path.

        Returns:
            Resolved file path.

        Raises:
            ValidationError: If no suitable file is found.
        """
        raw = data_path or self._config.data_path
        p = Path(raw)

        if p.is_file():
            return p

        if p.is_dir():
            parquet_files = sorted(p.glob("*.parquet"))
            if parquet_files:
                return parquet_files[0]
            csv_files = sorted(p.glob("*.csv"))
            if csv_files:
                return csv_files[0]
            raise ValidationError(f"No .parquet or .csv files in directory: {p}")

        raise ValidationError(f"Data path does not exist: {p}")

    def _extract_model_params(self, model_cls: type) -> dict[str, Any]:
        """Extract constructor parameters for the model from config.extra.

        Only passes keys that the model's ``__init__`` actually accepts,
        to avoid ``TypeError`` on unexpected keyword arguments.

        Args:
            model_cls: The model class to instantiate.

        Returns:
            Filtered dict of constructor keyword arguments.
        """
        import inspect

        sig = inspect.signature(model_cls.__init__)
        valid_params = {
            name
            for name, param in sig.parameters.items()
            if name != "self"
            and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }

        return {k: v for k, v in self._config.extra.items() if k in valid_params}

    def _apply_training_mode(
        self,
        X_train: np.ndarray,
        train_labels: np.ndarray | None,
        log: Any,
    ) -> np.ndarray:
        """Filter training data based on training_mode setting.

        In ``normal_only`` mode, rows where ``is_anomaly == 1`` are
        removed from the training set.  In ``all_data`` mode the
        training array is returned unchanged.

        Args:
            X_train: Full training data.
            train_labels: Training labels (may be ``None``).
            log: Bound structlog logger for context.

        Returns:
            Filtered (or unfiltered) training data.
        """
        if self._config.training_mode == "normal_only" and train_labels is not None:
            normal_mask = train_labels == 0
            n_before = X_train.shape[0]
            X_train = X_train[normal_mask]
            n_after = X_train.shape[0]
            log.info(
                "normal_only_filter",
                before=n_before,
                after=n_after,
                removed=n_before - n_after,
            )
        return X_train

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Set global random seeds for reproducibility.

        Sets seeds for Python's ``random``, numpy, and (if available)
        PyTorch on both CPU and CUDA.

        Args:
            seed: Integer seed value.
        """
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
