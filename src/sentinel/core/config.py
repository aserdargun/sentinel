"""Configuration system with YAML loading and inheritance."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sentinel.core.exceptions import ConfigError

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"


@dataclass
class SplitConfig:
    """Train/val/test split ratios."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    type: str = "reduce_on_plateau"
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6
    warmup_epochs: int = 0


@dataclass
class RuntimeConfig:
    """Runtime limits."""

    max_upload_size_mb: int = 100
    max_features: int = 500
    training_timeout_s: int = 3600
    api_request_timeout_s: int = 300
    max_dataset_rows_matrix_profile: int = 100000
    shap_max_features: int = 10


@dataclass
class LLMConfig:
    """LLM / Ollama configuration."""

    model: str = "nvidia/nemotron-3-nano-4b"
    ollama_url: str = "http://localhost:11434"
    timeout_s: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"


@dataclass
class RunConfig:
    """Complete run configuration.

    Built incrementally: dataclass defaults -> YAML loading ->
    inherits resolution -> CLI overrides.
    """

    model: str = ""
    data_path: str = "data/raw"
    processed_path: str = "data/processed"
    metadata_file: str = "data/datasets.json"
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 42
    device: str = "auto"
    training_mode: str = "normal_only"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load config from a YAML file with inheritance support.

        Args:
            path: Path to the YAML config file.

        Returns:
            Populated RunConfig instance.

        Raises:
            ConfigError: If the file doesn't exist or is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}") from e

        if "inherits" in raw:
            base_path = path.parent / raw.pop("inherits")
            if not base_path.exists():
                raise ConfigError(f"Base config not found: {base_path}")
            try:
                base_raw = yaml.safe_load(base_path.read_text()) or {}
            except yaml.YAMLError as e:
                raise ConfigError(f"Invalid YAML in {base_path}: {e}") from e
            merged = _deep_merge(base_raw, raw)
        else:
            merged = raw

        return cls._from_dict(merged)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Build RunConfig from a flat/nested dict."""
        config = cls()

        simple_fields = {
            "model",
            "data_path",
            "processed_path",
            "metadata_file",
            "seed",
            "device",
            "training_mode",
        }
        for key in simple_fields:
            if key in data:
                setattr(config, key, data[key])

        if "data" in data:
            d = data["data"]
            if "path" in d:
                config.data_path = d["path"]
            if "processed_path" in d:
                config.processed_path = d["processed_path"]
            if "metadata_file" in d:
                config.metadata_file = d["metadata_file"]

        if "split" in data:
            s = data["split"]
            config.split = SplitConfig(
                train=s.get("train", 0.70),
                val=s.get("val", 0.15),
                test=s.get("test", 0.15),
            )

        if "scheduler" in data:
            sc = data["scheduler"]
            config.scheduler = SchedulerConfig(
                type=sc.get("type", "reduce_on_plateau"),
                patience=sc.get("patience", 5),
                factor=sc.get("factor", 0.5),
                min_lr=sc.get("min_lr", 1e-6),
                warmup_epochs=sc.get("warmup_epochs", 0),
            )

        if "runtime" in data:
            r = data["runtime"]
            config.runtime = RuntimeConfig(
                **{k: r[k] for k in RuntimeConfig.__dataclass_fields__ if k in r}
            )

        if "llm" in data:
            ll = data["llm"]
            config.llm = LLMConfig(
                **{k: ll[k] for k in LLMConfig.__dataclass_fields__ if k in ll}
            )

        if "logging" in data:
            lg = data["logging"]
            config.logging = LoggingConfig(
                **{k: lg[k] for k in LoggingConfig.__dataclass_fields__ if k in lg}
            )

        known_keys = {
            "model",
            "data_path",
            "processed_path",
            "metadata_file",
            "seed",
            "device",
            "training_mode",
            "data",
            "split",
            "scheduler",
            "runtime",
            "llm",
            "logging",
            "inherits",
        }
        config.extra = {k: v for k, v in data.items() if k not in known_keys}

        return config

    def override(self, **kwargs: Any) -> RunConfig:
        """Apply CLI overrides to the config.

        Args:
            **kwargs: Key-value overrides. Nested keys use dot notation
                      (e.g., split.train=0.8).

        Returns:
            New RunConfig with overrides applied.
        """
        config = copy.deepcopy(self)
        for key, value in kwargs.items():
            if value is None:
                continue
            if "." in key:
                parts = key.split(".", 1)
                obj = getattr(config, parts[0], None)
                if obj is not None and hasattr(obj, parts[1]):
                    setattr(obj, parts[1], value)
                else:
                    config.extra[key] = value
            elif hasattr(config, key):
                setattr(config, key, value)
            else:
                config.extra[key] = value
        return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts. Override values take precedence at leaf level.

    Args:
        base: Base dictionary.
        override: Override dictionary.

    Returns:
        Merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
