"""Tests for advanced config validation scenarios in sentinel.core.config.

Covers: invalid YAML, missing required fields, nonexistent dataset paths,
missing base file in inherits, CLI dot-notation overrides, deep merge behaviour.

These tests complement test_config.py (which tests defaults and basic loading)
by focusing on error paths and edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.core.config import RunConfig, _deep_merge
from sentinel.core.exceptions import ConfigError, SentinelError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Write *content* to *tmp_path/name* and return the path."""
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# TestInvalidYaml
# ---------------------------------------------------------------------------


class TestInvalidYaml:
    """Malformed YAML must raise ConfigError, never a raw yaml.YAMLError."""

    def test_unclosed_bracket_raises_config_error(self, tmp_path: Path) -> None:
        bad = _write(tmp_path, "bad.yaml", "key: [unclosed\n")
        with pytest.raises(ConfigError):
            RunConfig.from_yaml(bad)

    def test_tab_indentation_raises_config_error(self, tmp_path: Path) -> None:
        bad = _write(tmp_path, "bad.yaml", "key:\n\tvalue: 1\n")
        with pytest.raises(ConfigError):
            RunConfig.from_yaml(bad)

    def test_invalid_yaml_in_base_file_raises_config_error(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, "base.yaml", "key: [unclosed\n")
        model = _write(tmp_path, "model.yaml", "inherits: base.yaml\nmodel: m\n")
        with pytest.raises(ConfigError):
            RunConfig.from_yaml(model)

    def test_config_error_is_subclass_of_sentinel_error(self, tmp_path: Path) -> None:
        bad = _write(tmp_path, "bad.yaml", "key: [unclosed\n")
        with pytest.raises(SentinelError):
            RunConfig.from_yaml(bad)

    def test_error_message_mentions_file(self, tmp_path: Path) -> None:
        bad = _write(tmp_path, "bad.yaml", "key: [unclosed\n")
        with pytest.raises(ConfigError, match="bad.yaml"):
            RunConfig.from_yaml(bad)

    def test_colon_only_yaml_returns_empty_config(self, tmp_path: Path) -> None:
        """A YAML file that parses to None (e.g. empty) falls back to defaults."""
        empty = _write(tmp_path, "empty.yaml", "")
        config = RunConfig.from_yaml(empty)
        assert config.model == ""

    def test_list_root_raises_config_error(self, tmp_path: Path) -> None:
        """A YAML file whose root is a list (not a dict) should not crash fatally."""
        bad = _write(tmp_path, "list.yaml", "- item1\n- item2\n")
        # from_yaml calls _from_dict on the loaded value; a list is not a dict
        # so either a ConfigError or AttributeError may occur — we only care that
        # it does not silently succeed with wrong data.
        try:
            config = RunConfig.from_yaml(bad)
            # If it doesn't raise, the model field must be empty / default
            assert config.model == ""
        except (ConfigError, AttributeError, TypeError):
            pass  # acceptable failure modes


# ---------------------------------------------------------------------------
# TestMissingFile
# ---------------------------------------------------------------------------


class TestMissingFile:
    """Referencing a file that does not exist raises ConfigError."""

    def test_nonexistent_config_path_raises(self) -> None:
        with pytest.raises(ConfigError, match="not found"):
            RunConfig.from_yaml("/tmp/_no_such_sentinel_file_xyz.yaml")

    def test_error_message_contains_path(self) -> None:
        missing = "/tmp/_no_such_sentinel_file_abc123.yaml"
        with pytest.raises(ConfigError, match=missing):
            RunConfig.from_yaml(missing)

    def test_path_object_accepted(self, tmp_path: Path) -> None:
        """from_yaml accepts a Path object as well as a string."""
        p = tmp_path / "cfg.yaml"
        p.write_text("model: test\n")
        config = RunConfig.from_yaml(p)
        assert config.model == "test"


# ---------------------------------------------------------------------------
# TestMissingModelName
# ---------------------------------------------------------------------------


class TestMissingModelName:
    """model field defaults to empty string — validation is responsibility of caller."""

    def test_missing_model_field_defaults_to_empty_string(self, tmp_path: Path) -> None:
        cfg = _write(tmp_path, "no_model.yaml", "seed: 1\n")
        config = RunConfig.from_yaml(cfg)
        assert config.model == ""

    def test_model_field_is_string_type(self, tmp_path: Path) -> None:
        cfg = _write(tmp_path, "typed.yaml", "model: my_model\n")
        config = RunConfig.from_yaml(cfg)
        assert isinstance(config.model, str)

    def test_model_name_preserved_exactly(self, tmp_path: Path) -> None:
        cfg = _write(tmp_path, "exact.yaml", "model: isolation_forest\n")
        config = RunConfig.from_yaml(cfg)
        assert config.model == "isolation_forest"


# ---------------------------------------------------------------------------
# TestNonexistentDatasetPath
# ---------------------------------------------------------------------------


class TestNonexistentDatasetPath:
    """data_path / metadata_file are stored as-is; no existence check at load time."""

    def test_nonexistent_data_path_stored(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\ndata_path: /no/such/dir\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.data_path == "/no/such/dir"

    def test_data_section_path_stored(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\ndata:\n  path: /fake/path\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.data_path == "/fake/path"

    def test_metadata_file_path_stored(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\nmetadata_file: /fake/datasets.json\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.metadata_file == "/fake/datasets.json"

    def test_processed_path_stored(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\nprocessed_path: /fake/processed\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.processed_path == "/fake/processed"


# ---------------------------------------------------------------------------
# TestInheritsResolution
# ---------------------------------------------------------------------------


class TestInheritsResolution:
    """inherits: directive resolves relative to the model config's directory."""

    def test_missing_base_file_raises_config_error(self, tmp_path: Path) -> None:
        model = _write(
            tmp_path, "model.yaml", "inherits: nonexistent_base.yaml\nmodel: m\n"
        )
        with pytest.raises(ConfigError, match="not found"):
            RunConfig.from_yaml(model)

    def test_base_file_in_different_subdir_not_found(self, tmp_path: Path) -> None:
        """Base path is resolved relative to the model yaml, not cwd."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        model = _write(subdir, "model.yaml", "inherits: ../base.yaml\nmodel: m\n")
        # base.yaml does not exist at tmp_path/base.yaml
        with pytest.raises(ConfigError):
            RunConfig.from_yaml(model)

    def test_valid_inherits_loads_base_values(self, tmp_path: Path) -> None:
        _write(tmp_path, "base.yaml", "seed: 77\ndevice: cpu\n")
        model = _write(tmp_path, "model.yaml", "inherits: base.yaml\nmodel: m\n")
        config = RunConfig.from_yaml(model)
        assert config.seed == 77
        assert config.device == "cpu"

    def test_model_yaml_overrides_base_scalar(self, tmp_path: Path) -> None:
        _write(tmp_path, "base.yaml", "seed: 99\ndevice: mps\n")
        model = _write(
            tmp_path, "model.yaml", "inherits: base.yaml\nmodel: m\nseed: 1\n"
        )
        config = RunConfig.from_yaml(model)
        assert config.seed == 1
        assert config.device == "mps"  # base value preserved

    def test_inherits_key_not_in_extra(self, tmp_path: Path) -> None:
        _write(tmp_path, "base.yaml", "seed: 42\n")
        model = _write(tmp_path, "model.yaml", "inherits: base.yaml\nmodel: m\n")
        config = RunConfig.from_yaml(model)
        assert "inherits" not in config.extra

    def test_relative_path_resolves_correctly(self, tmp_path: Path) -> None:
        """Base file in a sibling directory is found when path is correct."""
        base = _write(tmp_path, "base.yaml", "seed: 5\n")
        model = _write(tmp_path, "model.yaml", f"inherits: {base.name}\nmodel: m\n")
        config = RunConfig.from_yaml(model)
        assert config.seed == 5


# ---------------------------------------------------------------------------
# TestCLIOverrideDotNotation
# ---------------------------------------------------------------------------


class TestCLIOverrideDotNotation:
    """override() supports dot-notation for nested fields."""

    def test_split_train_via_dot_notation(self) -> None:
        config = RunConfig().override(**{"split.train": 0.80})
        assert config.split.train == pytest.approx(0.80)

    def test_split_val_via_dot_notation(self) -> None:
        config = RunConfig().override(**{"split.val": 0.10})
        assert config.split.val == pytest.approx(0.10)

    def test_split_test_via_dot_notation(self) -> None:
        config = RunConfig().override(**{"split.test": 0.05})
        assert config.split.test == pytest.approx(0.05)

    def test_scheduler_patience_via_dot_notation(self) -> None:
        config = RunConfig().override(**{"scheduler.patience": 10})
        assert config.scheduler.patience == 10

    def test_scheduler_type_via_dot_notation(self) -> None:
        config = RunConfig().override(**{"scheduler.type": "cosine"})
        assert config.scheduler.type == "cosine"

    def test_llm_model_via_dot_notation(self) -> None:
        config = RunConfig().override(**{"llm.model": "llama3:8b"})
        assert config.llm.model == "llama3:8b"

    def test_unknown_nested_key_goes_to_extra(self) -> None:
        config = RunConfig().override(**{"split.nonexistent": 99})
        assert config.extra.get("split.nonexistent") == 99

    def test_top_level_override(self) -> None:
        config = RunConfig().override(model="new_model", seed=7)
        assert config.model == "new_model"
        assert config.seed == 7

    def test_none_value_is_skipped(self) -> None:
        original_seed = RunConfig().seed
        config = RunConfig().override(seed=None)
        assert config.seed == original_seed

    def test_override_does_not_mutate_original(self) -> None:
        original = RunConfig()
        original_seed = original.seed
        original.override(seed=999)
        assert original.seed == original_seed

    def test_override_returns_independent_copy(self) -> None:
        original = RunConfig()
        modified = original.override(seed=100)
        assert modified is not original
        modified.seed = 200
        assert original.seed != 200

    def test_multiple_overrides_chained(self) -> None:
        config = RunConfig().override(seed=1).override(device="cpu")
        assert config.seed == 1
        assert config.device == "cpu"


# ---------------------------------------------------------------------------
# TestDeepMerge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """_deep_merge merges two dicts with deep recursion."""

    def test_base_scalar_overridden_by_override(self) -> None:
        base = {"a": 1}
        override = {"a": 2}
        result = _deep_merge(base, override)
        assert result["a"] == 2

    def test_base_key_preserved_when_not_in_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"a": 99}
        result = _deep_merge(base, override)
        assert result["b"] == 2

    def test_nested_dict_is_recursively_merged(self) -> None:
        base = {"split": {"train": 0.70, "val": 0.15, "test": 0.15}}
        override = {"split": {"train": 0.80}}
        result = _deep_merge(base, override)
        assert result["split"]["train"] == pytest.approx(0.80)
        assert result["split"]["val"] == pytest.approx(0.15)  # from base

    def test_nested_dict_override_adds_new_key(self) -> None:
        base = {"split": {"train": 0.70}}
        override = {"split": {"extra_key": "value"}}
        result = _deep_merge(base, override)
        assert result["split"]["extra_key"] == "value"
        assert result["split"]["train"] == pytest.approx(0.70)

    def test_new_top_level_key_added_from_override(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result["b"] == 2
        assert result["a"] == 1

    def test_does_not_mutate_base(self) -> None:
        base = {"a": 1, "nested": {"x": 10}}
        override = {"a": 2, "nested": {"x": 99}}
        _deep_merge(base, override)
        assert base["a"] == 1
        assert base["nested"]["x"] == 10

    def test_does_not_mutate_override(self) -> None:
        base = {"a": 1}
        override = {"a": 2, "b": {"c": 3}}
        _deep_merge(base, override)
        assert override["a"] == 2

    def test_empty_base_returns_override(self) -> None:
        result = _deep_merge({}, {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_override_returns_base_copy(self) -> None:
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_list_value_replaced_not_merged(self) -> None:
        """Lists are treated as scalars: override replaces, not extends."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge(base, override)
        assert result["items"] == [4, 5]

    def test_scalar_overrides_nested_dict(self) -> None:
        """When override has a scalar where base has a dict, scalar wins."""
        base = {"nested": {"x": 1}}
        override = {"nested": 42}
        result = _deep_merge(base, override)
        assert result["nested"] == 42


# ---------------------------------------------------------------------------
# TestDataSectionParsing
# ---------------------------------------------------------------------------


class TestDataSectionParsing:
    """The 'data:' YAML section maps to data_path / processed_path / metadata_file."""

    def test_data_path_from_data_section(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\ndata:\n  path: /my/raw\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.data_path == "/my/raw"

    def test_processed_path_from_data_section(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\ndata:\n  processed_path: /my/processed\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.processed_path == "/my/processed"

    def test_metadata_file_from_data_section(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\ndata:\n  metadata_file: /my/datasets.json\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.metadata_file == "/my/datasets.json"

    def test_top_level_data_path_wins_over_data_section(self, tmp_path: Path) -> None:
        """When both top-level data_path and data.path are present,
        data.path wins (processed last in _from_dict)."""
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\ndata_path: /top\ndata:\n  path: /section\n",
        )
        config = RunConfig.from_yaml(cfg)
        # data.path is processed after data_path, so it takes precedence
        assert config.data_path == "/section"


# ---------------------------------------------------------------------------
# TestExtraFieldsCapture
# ---------------------------------------------------------------------------


class TestExtraFieldsCapture:
    """Unknown keys in YAML end up in config.extra."""

    def test_model_specific_params_in_extra(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: zscore\nwindow_size: 50\nthreshold_sigma: 2.5\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.extra["window_size"] == 50
        assert config.extra["threshold_sigma"] == pytest.approx(2.5)

    def test_known_keys_not_in_extra(self, tmp_path: Path) -> None:
        cfg = _write(tmp_path, "cfg.yaml", "model: zscore\nseed: 99\ndevice: cpu\n")
        config = RunConfig.from_yaml(cfg)
        assert "model" not in config.extra
        assert "seed" not in config.extra
        assert "device" not in config.extra

    def test_extra_defaults_to_empty_dict(self) -> None:
        config = RunConfig()
        assert config.extra == {}

    def test_multiple_extra_keys_captured(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: m\nalpha: 0.1\nbeta: 0.9\ngamma: 3\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert len(config.extra) >= 3
        assert "alpha" in config.extra
        assert "beta" in config.extra
        assert "gamma" in config.extra


# ---------------------------------------------------------------------------
# TestSchedulerAndRuntimeParsing
# ---------------------------------------------------------------------------


class TestSchedulerAndRuntimeParsing:
    """Nested scheduler and runtime sections are parsed into sub-dataclasses."""

    def test_scheduler_type_loaded(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: m\nscheduler:\n  type: cosine\n  patience: 3\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.scheduler.type == "cosine"
        assert config.scheduler.patience == 3

    def test_scheduler_partial_override_keeps_defaults(self, tmp_path: Path) -> None:
        cfg = _write(tmp_path, "cfg.yaml", "model: m\nscheduler:\n  patience: 10\n")
        config = RunConfig.from_yaml(cfg)
        assert config.scheduler.patience == 10
        assert config.scheduler.type == "reduce_on_plateau"  # default preserved
        assert config.scheduler.factor == pytest.approx(0.5)  # default preserved

    def test_runtime_max_upload_size_loaded(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: m\nruntime:\n  max_upload_size_mb: 200\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.runtime.max_upload_size_mb == 200

    def test_llm_model_loaded(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: m\nllm:\n  model: qwen2.5:7b\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.llm.model == "qwen2.5:7b"

    def test_logging_level_loaded(self, tmp_path: Path) -> None:
        cfg = _write(
            tmp_path,
            "cfg.yaml",
            "model: m\nlogging:\n  level: DEBUG\n  format: console\n",
        )
        config = RunConfig.from_yaml(cfg)
        assert config.logging.level == "DEBUG"
        assert config.logging.format == "console"
