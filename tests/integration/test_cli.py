"""Integration tests for the Sentinel CLI using Typer's test runner.

Each test invokes a real CLI sub-command via CliRunner (no subprocess fork)
and verifies exit code and console output.  All file I/O uses tmp_path so
nothing lands in data/.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from sentinel.cli.app import app
from sentinel.data.synthetic import generate_synthetic

runner = CliRunner()

# Path to the real YAML configs shipped with the project.
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet(tmp_path: Path, n_features: int = 3, length: int = 200) -> Path:
    """Generate synthetic data and write to tmp_path/data.parquet."""
    df = generate_synthetic(n_features=n_features, length=length, seed=42)
    out = tmp_path / "data.parquet"
    df.write_parquet(str(out))
    return out


# ---------------------------------------------------------------------------
# `sentinel train`
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainCommand:
    """Tests for `sentinel train --config <yaml> --data-path <path>`."""

    def test_train_exits_zero_on_success(self, tmp_path: Path) -> None:
        """Train with zscore on valid synthetic data exits 0."""
        parquet = _write_parquet(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(CONFIGS_DIR / "zscore.yaml"),
                "--data-path",
                str(parquet),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_train_output_contains_training_complete(self, tmp_path: Path) -> None:
        parquet = _write_parquet(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(CONFIGS_DIR / "zscore.yaml"),
                "--data-path",
                str(parquet),
            ],
        )
        assert "Training complete" in result.output

    def test_train_output_contains_run_id(self, tmp_path: Path) -> None:
        parquet = _write_parquet(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(CONFIGS_DIR / "zscore.yaml"),
                "--data-path",
                str(parquet),
            ],
        )
        assert "Run ID" in result.output

    def test_train_output_contains_model_name(self, tmp_path: Path) -> None:
        parquet = _write_parquet(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(CONFIGS_DIR / "zscore.yaml"),
                "--data-path",
                str(parquet),
            ],
        )
        assert "zscore" in result.output

    def test_train_missing_config_exits_one(self, tmp_path: Path) -> None:
        """Non-existent config file causes exit code 1."""
        parquet = _write_parquet(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                "/nonexistent/config.yaml",
                "--data-path",
                str(parquet),
            ],
        )
        assert result.exit_code == 1

    def test_train_missing_data_path_exits_one(self, tmp_path: Path) -> None:
        """Pointing at a non-existent data file causes exit code 1."""
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(CONFIGS_DIR / "zscore.yaml"),
                "--data-path",
                str(tmp_path / "nonexistent.parquet"),
            ],
        )
        assert result.exit_code == 1

    def test_train_with_isolation_forest_config(self, tmp_path: Path) -> None:
        """Train using isolation_forest config exits 0."""
        parquet = _write_parquet(tmp_path, length=300)
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(CONFIGS_DIR / "isolation_forest.yaml"),
                "--data-path",
                str(parquet),
            ],
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# `sentinel generate`
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenerateCommand:
    """Tests for `sentinel generate --features N --length M --output <path>`."""

    def test_generate_exits_zero(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        result = runner.invoke(
            app,
            [
                "generate",
                "--features",
                "3",
                "--length",
                "100",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_generate_creates_parquet_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        runner.invoke(
            app,
            [
                "generate",
                "--features",
                "3",
                "--length",
                "100",
                "--output",
                str(out),
            ],
        )
        assert out.exists()

    def test_generated_parquet_has_correct_row_count(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        runner.invoke(
            app,
            [
                "generate",
                "--features",
                "3",
                "--length",
                "150",
                "--output",
                str(out),
            ],
        )
        df = pl.read_parquet(str(out))
        assert df.height == 150

    def test_generated_parquet_has_correct_feature_count(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        runner.invoke(
            app,
            [
                "generate",
                "--features",
                "5",
                "--length",
                "100",
                "--output",
                str(out),
            ],
        )
        df = pl.read_parquet(str(out))
        # Columns: timestamp + 5 features + is_anomaly = 7
        assert "feature_1" in df.columns
        assert "feature_5" in df.columns

    def test_generated_parquet_has_is_anomaly_column(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        runner.invoke(
            app,
            [
                "generate",
                "--features",
                "2",
                "--length",
                "50",
                "--output",
                str(out),
            ],
        )
        df = pl.read_parquet(str(out))
        assert "is_anomaly" in df.columns

    def test_generate_output_contains_generated_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        result = runner.invoke(
            app,
            [
                "generate",
                "--features",
                "2",
                "--length",
                "200",
                "--output",
                str(out),
            ],
        )
        assert "200" in result.output or "Generated" in result.output

    def test_generate_seed_produces_deterministic_output(self, tmp_path: Path) -> None:
        out1 = tmp_path / "out1.parquet"
        out2 = tmp_path / "out2.parquet"
        for out in (out1, out2):
            runner.invoke(
                app,
                [
                    "generate",
                    "--features",
                    "3",
                    "--length",
                    "100",
                    "--seed",
                    "99",
                    "--output",
                    str(out),
                ],
            )
        df1 = pl.read_parquet(str(out1))
        df2 = pl.read_parquet(str(out2))
        assert df1.equals(df2)

    def test_generate_invalid_features_exits_one(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        result = runner.invoke(
            app,
            [
                "generate",
                "--features",
                "0",
                "--length",
                "100",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 1

    def test_generate_invalid_length_exits_one(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        result = runner.invoke(
            app,
            [
                "generate",
                "--features",
                "3",
                "--length",
                "1",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 1

    def test_generate_invalid_anomaly_ratio_exits_one(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        result = runner.invoke(
            app,
            [
                "generate",
                "--features",
                "3",
                "--length",
                "100",
                "--anomaly-ratio",
                "1.5",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# `sentinel validate-config`
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestValidateConfigCommand:
    """Tests for `sentinel validate-config --config <yaml>`."""

    def test_valid_config_exits_zero(self) -> None:
        result = runner.invoke(
            app,
            ["validate-config", "--config", str(CONFIGS_DIR / "zscore.yaml")],
        )
        assert result.exit_code == 0, result.output

    def test_valid_config_output_contains_valid(self) -> None:
        result = runner.invoke(
            app,
            ["validate-config", "--config", str(CONFIGS_DIR / "zscore.yaml")],
        )
        assert "valid" in result.output.lower()

    def test_valid_config_shows_model_name(self) -> None:
        result = runner.invoke(
            app,
            ["validate-config", "--config", str(CONFIGS_DIR / "zscore.yaml")],
        )
        assert "zscore" in result.output

    def test_nonexistent_config_exits_one(self) -> None:
        result = runner.invoke(
            app,
            ["validate-config", "--config", "/nonexistent/config_xyz.yaml"],
        )
        assert result.exit_code == 1

    def test_nonexistent_config_output_mentions_not_found(self) -> None:
        result = runner.invoke(
            app,
            ["validate-config", "--config", "/nonexistent/config_xyz.yaml"],
        )
        assert "not found" in result.output.lower() or "Config" in result.output

    def test_invalid_yaml_exits_one(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [\nbad yaml content\n")
        result = runner.invoke(
            app,
            ["validate-config", "--config", str(bad_yaml)],
        )
        assert result.exit_code == 1

    def test_config_with_unknown_model_exits_one(self, tmp_path: Path) -> None:
        """A config that names an unregistered model fails validation."""
        cfg = tmp_path / "unknown_model.yaml"
        cfg.write_text("model: totally_nonexistent_model_xyz\n")
        result = runner.invoke(
            app,
            ["validate-config", "--config", str(cfg)],
        )
        assert result.exit_code == 1

    def test_isolation_forest_config_exits_zero(self) -> None:
        result = runner.invoke(
            app,
            [
                "validate-config",
                "--config",
                str(CONFIGS_DIR / "isolation_forest.yaml"),
            ],
        )
        assert result.exit_code == 0, result.output
