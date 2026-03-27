"""Tests for sentinel.core.registry — register/lookup/list operations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sentinel.core.base_model import BaseAnomalyDetector
from sentinel.core.exceptions import ModelNotFoundError
from sentinel.core.registry import (
    _REGISTRY,
    get_model_class,
    list_models,
    register_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub(name: str) -> type[BaseAnomalyDetector]:
    """Create a minimal concrete BaseAnomalyDetector subclass for testing.

    The returned class is NOT registered — callers register it themselves so
    tests have full control over when/how registration happens.
    """

    class _Stub(BaseAnomalyDetector):
        def fit(self, X: np.ndarray, **kwargs: Any) -> None:
            pass

        def score(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X))

        def save(self, path: str) -> None:
            pass

        def load(self, path: str) -> None:
            pass

        def get_params(self) -> dict[str, Any]:
            return {}

    _Stub.__name__ = name
    _Stub.__qualname__ = name
    return _Stub


def _cleanup(*names: str) -> None:
    """Remove test model names from the global registry after each test."""
    for n in names:
        _REGISTRY.pop(n, None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterModel:
    """Tests for the register_model decorator."""

    def test_register_adds_to_registry(self) -> None:
        stub = _make_stub("_test_register_adds")
        try:
            register_model("_test_register_adds")(stub)
            assert "_test_register_adds" in _REGISTRY
        finally:
            _cleanup("_test_register_adds")

    def test_register_sets_model_name_attribute(self) -> None:
        stub = _make_stub("_test_name_attr")
        try:
            registered = register_model("_test_name_attr")(stub)
            assert registered.model_name == "_test_name_attr"
        finally:
            _cleanup("_test_name_attr")

    def test_register_returns_class_unchanged(self) -> None:
        stub = _make_stub("_test_returns_class")
        try:
            result = register_model("_test_returns_class")(stub)
            assert result is stub
        finally:
            _cleanup("_test_returns_class")

    def test_duplicate_name_raises_value_error(self) -> None:
        stub_a = _make_stub("_test_dup_a")
        stub_b = _make_stub("_test_dup_b")
        try:
            register_model("_test_dup")(stub_a)
            with pytest.raises(ValueError, match="already registered"):
                register_model("_test_dup")(stub_b)
        finally:
            _cleanup("_test_dup")

    def test_can_be_used_as_decorator(self) -> None:
        try:

            @register_model("_test_decorator_usage")
            class _DecoratorStub(BaseAnomalyDetector):
                def fit(self, X: np.ndarray, **kwargs: Any) -> None:
                    pass

                def score(self, X: np.ndarray) -> np.ndarray:
                    return np.zeros(len(X))

                def save(self, path: str) -> None:
                    pass

                def load(self, path: str) -> None:
                    pass

                def get_params(self) -> dict[str, Any]:
                    return {}

            assert "_test_decorator_usage" in _REGISTRY
        finally:
            _cleanup("_test_decorator_usage")


class TestGetModelClass:
    """Tests for get_model_class lookup."""

    def test_lookup_returns_registered_class(self) -> None:
        stub = _make_stub("_test_lookup")
        try:
            register_model("_test_lookup")(stub)
            assert get_model_class("_test_lookup") is stub
        finally:
            _cleanup("_test_lookup")

    def test_missing_name_raises_model_not_found_error(self) -> None:
        with pytest.raises(ModelNotFoundError):
            get_model_class("_nonexistent_model_xyz")

    def test_error_message_includes_model_name(self) -> None:
        with pytest.raises(ModelNotFoundError, match="_nonexistent_model_xyz"):
            get_model_class("_nonexistent_model_xyz")

    def test_error_message_includes_available_models(self) -> None:
        stub = _make_stub("_test_avail")
        try:
            register_model("_test_avail")(stub)
            with pytest.raises(ModelNotFoundError, match="_test_avail"):
                get_model_class("_nonexistent_for_avail_check")
        finally:
            _cleanup("_test_avail")

    def test_model_not_found_is_sentinel_error(self) -> None:
        from sentinel.core.exceptions import SentinelError

        with pytest.raises(SentinelError):
            get_model_class("_does_not_exist")


class TestListModels:
    """Tests for list_models."""

    def test_returns_dict(self) -> None:
        result = list_models()
        assert isinstance(result, dict)

    def test_returns_copy_not_reference(self) -> None:
        result = list_models()
        result["_injected"] = None  # type: ignore[assignment]
        assert "_injected" not in _REGISTRY

    def test_registered_model_appears_in_list(self) -> None:
        stub = _make_stub("_test_list_appears")
        try:
            register_model("_test_list_appears")(stub)
            assert "_test_list_appears" in list_models()
        finally:
            _cleanup("_test_list_appears")

    def test_values_are_classes(self) -> None:
        stub = _make_stub("_test_list_classes")
        try:
            register_model("_test_list_classes")(stub)
            models = list_models()
            assert isinstance(models["_test_list_classes"], type)
        finally:
            _cleanup("_test_list_classes")
