"""Unit tests for API path traversal prevention.

Tests verify that ``resolve_safe_path`` rejects directory traversal
attempts, absolute paths outside allowed bases, and null-byte
injection, while allowing legitimate paths within the project's
``data/`` and ``configs/`` directories.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from sentinel.api.deps import resolve_safe_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The project root is derived the same way as in deps.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _data_dir() -> Path:
    return (_PROJECT_ROOT / "data").resolve()


def _configs_dir() -> Path:
    return (_PROJECT_ROOT / "configs").resolve()


# ---------------------------------------------------------------------------
# Positive cases — paths that should be accepted
# ---------------------------------------------------------------------------


class TestResolveAccepts:
    """Paths that resolve inside allowed directories should be accepted."""

    def test_relative_data_path(self) -> None:
        """A relative path under data/ should resolve successfully."""
        result = resolve_safe_path("data/raw/test.parquet")
        assert result == (_PROJECT_ROOT / "data/raw/test.parquet").resolve()

    def test_relative_configs_path(self) -> None:
        """A relative path under configs/ should resolve successfully."""
        result = resolve_safe_path("configs/zscore.yaml")
        assert result == (_PROJECT_ROOT / "configs/zscore.yaml").resolve()

    def test_absolute_data_path(self) -> None:
        """An absolute path inside data/ should be accepted."""
        abs_path = str(_data_dir() / "raw" / "test.csv")
        result = resolve_safe_path(abs_path)
        assert result == Path(abs_path).resolve()

    def test_absolute_configs_path(self) -> None:
        """An absolute path inside configs/ should be accepted."""
        abs_path = str(_configs_dir() / "base.yaml")
        result = resolve_safe_path(abs_path)
        assert result == Path(abs_path).resolve()

    def test_data_experiments_subdir(self) -> None:
        """Paths under data/experiments/ should be accepted."""
        result = resolve_safe_path("data/experiments/run-123/model")
        expected = (_PROJECT_ROOT / "data/experiments/run-123/model").resolve()
        assert result == expected

    def test_custom_allowed_bases(self, tmp_path: Path) -> None:
        """A custom allowed_bases list should override the defaults."""
        custom_file = tmp_path / "custom.txt"
        result = resolve_safe_path(str(custom_file), allowed_bases=[tmp_path.resolve()])
        assert result == custom_file.resolve()


# ---------------------------------------------------------------------------
# Negative cases — paths that should be rejected
# ---------------------------------------------------------------------------


class TestResolveRejects:
    """Paths that escape allowed directories must be rejected with HTTP 400."""

    def test_parent_traversal(self) -> None:
        """../../etc/passwd must be rejected."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("../../etc/passwd")
        assert exc_info.value.status_code == 400

    def test_data_parent_traversal(self) -> None:
        """data/../../../etc/shadow must be rejected."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("data/../../../etc/shadow")
        assert exc_info.value.status_code == 400

    def test_absolute_outside_project(self) -> None:
        """/etc/passwd must be rejected."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("/etc/passwd")
        assert exc_info.value.status_code == 400

    def test_absolute_root(self) -> None:
        """/ must be rejected."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("/")
        assert exc_info.value.status_code == 400

    def test_home_directory(self) -> None:
        """~/.ssh/id_rsa must be rejected."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("~/.ssh/id_rsa")
        assert exc_info.value.status_code == 400

    def test_null_byte_injection(self) -> None:
        """Paths with null bytes must be rejected."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("data/raw/test.csv\x00.txt")
        assert exc_info.value.status_code == 400

    def test_configs_parent_escape(self) -> None:
        """configs/../../etc/hosts must be rejected."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("configs/../../etc/hosts")
        assert exc_info.value.status_code == 400

    def test_sibling_directory_src(self) -> None:
        """src/sentinel/core/config.py is not under data/ or configs/."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("src/sentinel/core/config.py")
        assert exc_info.value.status_code == 400

    def test_empty_custom_bases_rejects_everything(self) -> None:
        """An empty allowed_bases list should reject all paths."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("data/raw/test.csv", allowed_bases=[])
        assert exc_info.value.status_code == 400

    def test_error_detail_mentions_path(self) -> None:
        """The error detail should include the offending path."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_safe_path("/etc/passwd")
        assert "/etc/passwd" in exc_info.value.detail
