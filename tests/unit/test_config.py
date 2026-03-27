"""Tests for sentinel.core.config — YAML loading, inheritance, overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.core.config import (
    LLMConfig,
    LoggingConfig,
    RunConfig,
    RuntimeConfig,
    SchedulerConfig,
    SplitConfig,
)
from sentinel.core.exceptions import ConfigError

# Path to the real configs shipped with the project.
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


class TestRunConfigDefaults:
    """Tests for RunConfig dataclass defaults."""

    def test_model_default_is_empty_string(self) -> None:
        config = RunConfig()
        assert config.model == ""

    def test_data_path_default(self) -> None:
        config = RunConfig()
        assert config.data_path == "data/raw"

    def test_processed_path_default(self) -> None:
        config = RunConfig()
        assert config.processed_path == "data/processed"

    def test_metadata_file_default(self) -> None:
        config = RunConfig()
        assert config.metadata_file == "data/datasets.json"

    def test_seed_default(self) -> None:
        config = RunConfig()
        assert config.seed == 42

    def test_device_default(self) -> None:
        config = RunConfig()
        assert config.device == "auto"

    def test_training_mode_default(self) -> None:
        config = RunConfig()
        assert config.training_mode == "normal_only"

    def test_split_default(self) -> None:
        config = RunConfig()
        assert isinstance(config.split, SplitConfig)
        assert config.split.train == pytest.approx(0.70)
        assert config.split.val == pytest.approx(0.15)
        assert config.split.test == pytest.approx(0.15)

    def test_scheduler_default(self) -> None:
        config = RunConfig()
        assert isinstance(config.scheduler, SchedulerConfig)
        assert config.scheduler.type == "reduce_on_plateau"
        assert config.scheduler.patience == 5

    def test_runtime_default(self) -> None:
        config = RunConfig()
        assert isinstance(config.runtime, RuntimeConfig)
        assert config.runtime.max_upload_size_mb == 100
        assert config.runtime.max_features == 500

    def test_llm_default(self) -> None:
        config = RunConfig()
        assert isinstance(config.llm, LLMConfig)
        assert config.llm.model == "nvidia/nemotron-3-nano-4b"
        assert config.llm.ollama_url == "http://localhost:11434"

    def test_logging_default(self) -> None:
        config = RunConfig()
        assert isinstance(config.logging, LoggingConfig)
        assert config.logging.level == "INFO"
        assert config.logging.format == "json"

    def test_extra_default_is_empty_dict(self) -> None:
        config = RunConfig()
        assert config.extra == {}

    def test_split_ratios_sum_to_one(self) -> None:
        config = RunConfig()
        total = config.split.train + config.split.val + config.split.test
        assert total == pytest.approx(1.0)


class TestFromYamlZscore:
    """Tests loading the real configs/zscore.yaml (inherits base.yaml)."""

    def test_loads_without_error(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert config is not None

    def test_model_name_is_zscore(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert config.model == "zscore"

    def test_window_size_in_extra(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert "window_size" in config.extra
        assert config.extra["window_size"] == 30

    def test_threshold_sigma_in_extra(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert "threshold_sigma" in config.extra
        assert config.extra["threshold_sigma"] == pytest.approx(3.0)

    def test_split_inherited_from_base(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert config.split.train == pytest.approx(0.70)
        assert config.split.val == pytest.approx(0.15)
        assert config.split.test == pytest.approx(0.15)

    def test_seed_inherited_from_base(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert config.seed == 42

    def test_device_inherited_from_base(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert config.device == "auto"

    def test_llm_model_inherited_from_base(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert config.llm.model == "nvidia/nemotron-3-nano-4b"

    def test_runtime_max_features_inherited_from_base(self) -> None:
        config = RunConfig.from_yaml(CONFIGS_DIR / "zscore.yaml")
        assert config.runtime.max_features == 500


class TestFromYamlInheritanceMerging:
    """Tests that model-specific values override base.yaml values."""

    def test_model_yaml_overrides_base_scalar(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text("seed: 99\ndevice: cpu\n")

        model_yaml = tmp_path / "model.yaml"
        model_yaml.write_text("inherits: base.yaml\nmodel: test_model\nseed: 7\n")

        config = RunConfig.from_yaml(model_yaml)
        assert config.seed == 7  # model overrides base

    def test_base_value_used_when_not_overridden(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text("seed: 99\ndevice: cpu\n")

        model_yaml = tmp_path / "model.yaml"
        model_yaml.write_text("inherits: base.yaml\nmodel: test_model\n")

        config = RunConfig.from_yaml(model_yaml)
        assert config.seed == 99  # from base
        assert config.device == "cpu"  # from base

    def test_nested_section_deep_merged(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text("split:\n  train: 0.70\n  val: 0.15\n  test: 0.15\nseed: 42\n")

        model_yaml = tmp_path / "model.yaml"
        model_yaml.write_text(
            "inherits: base.yaml\nmodel: test_nested\n"
            "split:\n  train: 0.80\n  val: 0.10\n  test: 0.10\n"
        )

        config = RunConfig.from_yaml(model_yaml)
        assert config.split.train == pytest.approx(0.80)
        assert config.split.val == pytest.approx(0.10)
        assert config.split.test == pytest.approx(0.10)

    def test_extra_keys_from_model_yaml_captured(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text("seed: 42\n")

        model_yaml = tmp_path / "model.yaml"
        model_yaml.write_text("inherits: base.yaml\nmodel: test_extra\nmy_param: 123\n")

        config = RunConfig.from_yaml(model_yaml)
        assert config.extra["my_param"] == 123

    def test_no_inherits_loads_standalone(self, tmp_path: Path) -> None:
        model_yaml = tmp_path / "standalone.yaml"
        model_yaml.write_text("model: standalone_model\nseed: 5\n")

        config = RunConfig.from_yaml(model_yaml)
        assert config.model == "standalone_model"
        assert config.seed == 5


class TestOverride:
    """Tests for RunConfig.override()."""

    def test_override_simple_field(self) -> None:
        config = RunConfig()
        new_config = config.override(seed=123)
        assert new_config.seed == 123

    def test_override_does_not_mutate_original(self) -> None:
        config = RunConfig()
        config.override(seed=999)
        assert config.seed == 42  # original unchanged

    def test_override_returns_new_instance(self) -> None:
        config = RunConfig()
        new_config = config.override(seed=1)
        assert new_config is not config

    def test_override_model_field(self) -> None:
        config = RunConfig()
        new_config = config.override(model="my_model")
        assert new_config.model == "my_model"

    def test_override_device_field(self) -> None:
        config = RunConfig()
        new_config = config.override(device="cpu")
        assert new_config.device == "cpu"

    def test_override_nested_with_dot_notation(self) -> None:
        config = RunConfig()
        new_config = config.override(**{"split.train": 0.80})
        assert new_config.split.train == pytest.approx(0.80)

    def test_override_unknown_key_goes_to_extra(self) -> None:
        config = RunConfig()
        new_config = config.override(unknown_param=42)
        assert new_config.extra["unknown_param"] == 42

    def test_override_none_value_skipped(self) -> None:
        config = RunConfig()
        original_seed = config.seed
        new_config = config.override(seed=None)
        assert new_config.seed == original_seed

    def test_override_multiple_fields(self) -> None:
        config = RunConfig()
        new_config = config.override(seed=7, device="cpu", training_mode="all_data")
        assert new_config.seed == 7
        assert new_config.device == "cpu"
        assert new_config.training_mode == "all_data"

    def test_override_training_mode(self) -> None:
        config = RunConfig()
        new_config = config.override(training_mode="all_data")
        assert new_config.training_mode == "all_data"


class TestFromYamlErrors:
    """Tests for error handling in RunConfig.from_yaml."""

    def test_missing_file_raises_config_error(self) -> None:
        with pytest.raises(ConfigError, match="not found"):
            RunConfig.from_yaml("/tmp/_nonexistent_sentinel_config_xyz.yaml")

    def test_invalid_yaml_raises_config_error(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [\nbad yaml\n")

        with pytest.raises(ConfigError):
            RunConfig.from_yaml(bad_yaml)

    def test_missing_base_config_in_inherits_raises_config_error(
        self, tmp_path: Path
    ) -> None:
        model_yaml = tmp_path / "model.yaml"
        model_yaml.write_text("inherits: nonexistent_base.yaml\nmodel: test\n")

        with pytest.raises(ConfigError, match="not found"):
            RunConfig.from_yaml(model_yaml)

    def test_config_error_is_sentinel_error(self, tmp_path: Path) -> None:
        from sentinel.core.exceptions import SentinelError

        with pytest.raises(SentinelError):
            RunConfig.from_yaml("/tmp/_nonexistent_sentinel_config_xyz.yaml")

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("")

        config = RunConfig.from_yaml(empty_yaml)
        assert config.model == ""
        assert config.seed == 42
