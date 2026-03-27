"""Tests for sentinel.core.types — enums and TypedDicts."""

import numpy as np
import pytest

from sentinel.core.types import DetectionResult, ModelCategory, TrainResult


class TestModelCategory:
    """Tests for the ModelCategory enum."""

    def test_statistical_value(self) -> None:
        assert ModelCategory.STATISTICAL.value == "statistical"

    def test_deep_value(self) -> None:
        assert ModelCategory.DEEP.value == "deep"

    def test_ensemble_value(self) -> None:
        assert ModelCategory.ENSEMBLE.value == "ensemble"

    def test_all_categories_present(self) -> None:
        names = {m.name for m in ModelCategory}
        assert names == {"STATISTICAL", "DEEP", "ENSEMBLE"}

    def test_enum_lookup_by_value(self) -> None:
        assert ModelCategory("statistical") is ModelCategory.STATISTICAL
        assert ModelCategory("deep") is ModelCategory.DEEP
        assert ModelCategory("ensemble") is ModelCategory.ENSEMBLE

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ModelCategory("unknown")


class TestDetectionResult:
    """Tests for the DetectionResult TypedDict shape."""

    def test_construction_with_required_keys(self) -> None:
        scores = np.array([0.1, 0.9, 0.5])
        labels = np.array([0, 1, 0])
        result: DetectionResult = DetectionResult(
            scores=scores, labels=labels, threshold=0.7
        )
        assert "scores" in result
        assert "labels" in result
        assert "threshold" in result

    def test_scores_is_ndarray(self) -> None:
        scores = np.array([0.2, 0.8])
        labels = np.array([0, 1])
        result: DetectionResult = DetectionResult(
            scores=scores, labels=labels, threshold=0.5
        )
        assert isinstance(result["scores"], np.ndarray)

    def test_labels_is_ndarray(self) -> None:
        scores = np.array([0.2, 0.8])
        labels = np.array([0, 1])
        result: DetectionResult = DetectionResult(
            scores=scores, labels=labels, threshold=0.5
        )
        assert isinstance(result["labels"], np.ndarray)

    def test_threshold_is_float(self) -> None:
        scores = np.array([0.2, 0.8])
        labels = np.array([0, 1])
        result: DetectionResult = DetectionResult(
            scores=scores, labels=labels, threshold=0.5
        )
        assert isinstance(result["threshold"], float)

    def test_scores_and_labels_shape_match(self) -> None:
        n = 10
        scores = np.random.rand(n)
        labels = (scores > 0.5).astype(int)
        result: DetectionResult = DetectionResult(
            scores=scores, labels=labels, threshold=0.5
        )
        assert result["scores"].shape == result["labels"].shape

    def test_key_count(self) -> None:
        result: DetectionResult = DetectionResult(
            scores=np.array([0.1]),
            labels=np.array([0]),
            threshold=0.5,
        )
        assert len(result) == 3


class TestTrainResult:
    """Tests for the TrainResult TypedDict shape."""

    def test_construction_with_required_keys(self) -> None:
        result: TrainResult = TrainResult(
            run_id="abc123",
            model_name="zscore",
            metrics={"f1": 0.9, "precision": 0.85},
            duration_s=2.5,
        )
        assert "run_id" in result
        assert "model_name" in result
        assert "metrics" in result
        assert "duration_s" in result

    def test_run_id_is_str(self) -> None:
        result: TrainResult = TrainResult(
            run_id="run-001",
            model_name="zscore",
            metrics={},
            duration_s=1.0,
        )
        assert isinstance(result["run_id"], str)

    def test_model_name_is_str(self) -> None:
        result: TrainResult = TrainResult(
            run_id="run-001",
            model_name="isolation_forest",
            metrics={},
            duration_s=1.0,
        )
        assert isinstance(result["model_name"], str)

    def test_metrics_is_dict(self) -> None:
        result: TrainResult = TrainResult(
            run_id="run-001",
            model_name="zscore",
            metrics={"f1": 0.9, "auc_roc": None},
            duration_s=1.0,
        )
        assert isinstance(result["metrics"], dict)

    def test_metrics_allows_none_values(self) -> None:
        result: TrainResult = TrainResult(
            run_id="run-001",
            model_name="zscore",
            metrics={"f1": None, "precision": None},
            duration_s=1.0,
        )
        assert result["metrics"]["f1"] is None

    def test_duration_s_is_float(self) -> None:
        result: TrainResult = TrainResult(
            run_id="run-001",
            model_name="zscore",
            metrics={},
            duration_s=3.14,
        )
        assert isinstance(result["duration_s"], float)

    def test_key_count(self) -> None:
        result: TrainResult = TrainResult(
            run_id="r", model_name="m", metrics={}, duration_s=0.0
        )
        assert len(result) == 4
