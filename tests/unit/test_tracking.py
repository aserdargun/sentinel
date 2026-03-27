"""Tests for sentinel.tracking — LocalTracker, artifacts, and compare_runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sentinel.tracking.experiment import LocalTracker

# ---------------------------------------------------------------------------
# LocalTracker — create_run
# ---------------------------------------------------------------------------


class TestLocalTrackerCreateRun:
    """create_run() generates a run directory and meta.json."""

    def test_create_run_returns_string(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_create_run_creates_directory(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        run_dir = tmp_experiment_dir / run_id
        assert run_dir.is_dir()

    def test_create_run_writes_meta_json(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        meta_path = tmp_experiment_dir / run_id / "meta.json"
        assert meta_path.exists()

    def test_meta_json_contains_run_id(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        meta = json.loads((tmp_experiment_dir / run_id / "meta.json").read_text())
        assert meta["run_id"] == run_id

    def test_meta_json_contains_model_name(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("isolation_forest")
        meta = json.loads((tmp_experiment_dir / run_id / "meta.json").read_text())
        assert meta["model_name"] == "isolation_forest"

    def test_meta_json_contains_created_at(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        meta = json.loads((tmp_experiment_dir / run_id / "meta.json").read_text())
        assert "created_at" in meta
        assert len(meta["created_at"]) > 0

    def test_two_runs_have_distinct_ids(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        id1 = tracker.create_run("zscore")
        id2 = tracker.create_run("zscore")
        assert id1 != id2

    def test_base_dir_created_if_missing(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "experiments" / "nested"
        LocalTracker(base_dir=str(new_dir))
        assert new_dir.is_dir()


# ---------------------------------------------------------------------------
# LocalTracker — log_config
# ---------------------------------------------------------------------------


class TestLocalTrackerLogConfig:
    """log_config() writes config.json atomically."""

    def test_log_config_creates_file(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_config(run_id, {"window_size": 30})
        assert (tmp_experiment_dir / run_id / "config.json").exists()

    def test_log_config_content_is_valid_json(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        cfg = {"window_size": 30, "threshold_sigma": 3.0}
        tracker.log_config(run_id, cfg)
        loaded = json.loads((tmp_experiment_dir / run_id / "config.json").read_text())
        assert loaded == cfg

    def test_log_config_missing_run_raises(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        with pytest.raises(FileNotFoundError):
            tracker.log_config("nonexistent_run_id", {"x": 1})


# ---------------------------------------------------------------------------
# LocalTracker — log_metrics
# ---------------------------------------------------------------------------


class TestLocalTrackerLogMetrics:
    """log_metrics() writes metrics.json atomically."""

    def test_log_metrics_creates_file(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_metrics(run_id, {"f1": 0.85})
        assert (tmp_experiment_dir / run_id / "metrics.json").exists()

    def test_log_metrics_content_matches(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        metrics = {"f1": 0.85, "auc_roc": 0.92, "threshold": 2.1}
        tracker.log_metrics(run_id, metrics)
        loaded = json.loads((tmp_experiment_dir / run_id / "metrics.json").read_text())
        assert loaded["f1"] == pytest.approx(0.85)
        assert loaded["auc_roc"] == pytest.approx(0.92)

    def test_log_metrics_serializes_none_values(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_metrics(run_id, {"f1": None, "auc_roc": None})
        loaded = json.loads((tmp_experiment_dir / run_id / "metrics.json").read_text())
        assert loaded["f1"] is None

    def test_log_metrics_serializes_numpy_floats(
        self, tmp_experiment_dir: Path
    ) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        metrics = {"score": np.float64(3.14)}
        tracker.log_metrics(run_id, metrics)
        loaded = json.loads((tmp_experiment_dir / run_id / "metrics.json").read_text())
        assert loaded["score"] == pytest.approx(3.14)

    def test_log_metrics_missing_run_raises(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        with pytest.raises(FileNotFoundError):
            tracker.log_metrics("nonexistent_run_id", {"f1": 0.5})


# ---------------------------------------------------------------------------
# LocalTracker — get_run
# ---------------------------------------------------------------------------


class TestLocalTrackerGetRun:
    """get_run() returns combined run data."""

    def test_get_run_contains_run_id(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        run_data = tracker.get_run(run_id)
        assert run_data["run_id"] == run_id

    def test_get_run_contains_model_name(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("isolation_forest")
        run_data = tracker.get_run(run_id)
        assert run_data["model_name"] == "isolation_forest"

    def test_get_run_contains_config_after_log(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_config(run_id, {"window_size": 30})
        run_data = tracker.get_run(run_id)
        assert run_data["config"]["window_size"] == 30

    def test_get_run_contains_metrics_after_log(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_metrics(run_id, {"f1": 0.75})
        run_data = tracker.get_run(run_id)
        assert run_data["metrics"]["f1"] == pytest.approx(0.75)

    def test_get_run_config_empty_when_not_logged(
        self, tmp_experiment_dir: Path
    ) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        run_data = tracker.get_run(run_id)
        assert run_data["config"] == {}

    def test_get_run_metrics_empty_when_not_logged(
        self, tmp_experiment_dir: Path
    ) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        run_data = tracker.get_run(run_id)
        assert run_data["metrics"] == {}

    def test_get_run_missing_id_raises_file_not_found(
        self, tmp_experiment_dir: Path
    ) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        with pytest.raises(FileNotFoundError):
            tracker.get_run("nonexistent_run_id")


# ---------------------------------------------------------------------------
# LocalTracker — list_runs
# ---------------------------------------------------------------------------


class TestLocalTrackerListRuns:
    """list_runs() returns metadata for all runs."""

    def test_empty_base_dir_returns_empty_list(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        runs = tracker.list_runs()
        assert runs == []

    def test_single_run_listed(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        runs = tracker.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id

    def test_multiple_runs_all_listed(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        ids = {tracker.create_run("zscore") for _ in range(3)}
        runs = tracker.list_runs()
        assert len(runs) == 3
        listed_ids = {r["run_id"] for r in runs}
        assert ids == listed_ids

    def test_list_runs_includes_metrics_when_logged(
        self, tmp_experiment_dir: Path
    ) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_metrics(run_id, {"f1": 0.9})
        runs = tracker.list_runs()
        assert runs[0]["metrics"]["f1"] == pytest.approx(0.9)

    def test_list_runs_model_name_present(self, tmp_experiment_dir: Path) -> None:
        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        tracker.create_run("matrix_profile")
        runs = tracker.list_runs()
        assert runs[0]["model_name"] == "matrix_profile"


# ---------------------------------------------------------------------------
# save_predictions / load_predictions — round-trip
# ---------------------------------------------------------------------------


class TestPredictionsRoundTrip:
    """save_predictions + load_predictions preserve arrays exactly."""

    def test_scores_round_trip(self, tmp_experiment_dir: Path) -> None:
        from sentinel.tracking.artifacts import load_predictions, save_predictions

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")

        scores = np.array([0.1, 0.5, 0.9, 1.2, 0.3], dtype=np.float64)
        labels = np.array([0, 0, 1, 1, 0], dtype=np.int32)

        save_predictions(run_id, scores, labels, base_dir=str(tmp_experiment_dir))
        loaded = load_predictions(run_id, base_dir=str(tmp_experiment_dir))

        np.testing.assert_array_almost_equal(loaded["scores"], scores)

    def test_labels_round_trip(self, tmp_experiment_dir: Path) -> None:
        from sentinel.tracking.artifacts import load_predictions, save_predictions

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")

        scores = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        labels = np.array([0, 1, 1], dtype=np.int32)

        save_predictions(run_id, scores, labels, base_dir=str(tmp_experiment_dir))
        loaded = load_predictions(run_id, base_dir=str(tmp_experiment_dir))

        np.testing.assert_array_equal(loaded["labels"], labels)

    def test_save_predictions_file_created(self, tmp_experiment_dir: Path) -> None:
        from sentinel.tracking.artifacts import save_predictions

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")

        save_predictions(
            run_id,
            np.zeros(5),
            np.zeros(5, dtype=np.int32),
            base_dir=str(tmp_experiment_dir),
        )
        pred_file = tmp_experiment_dir / run_id / "predictions.npz"
        assert pred_file.exists()

    def test_load_predictions_missing_run_raises(
        self, tmp_experiment_dir: Path
    ) -> None:
        from sentinel.tracking.artifacts import load_predictions

        with pytest.raises(FileNotFoundError):
            load_predictions("nonexistent_run", base_dir=str(tmp_experiment_dir))

    def test_load_predictions_missing_file_raises(
        self, tmp_experiment_dir: Path
    ) -> None:
        from sentinel.tracking.artifacts import load_predictions

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        # No save_predictions() call — file absent.
        with pytest.raises(FileNotFoundError):
            load_predictions(run_id, base_dir=str(tmp_experiment_dir))


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    """compare_runs() assembles a Polars DataFrame from multiple runs."""

    def test_returns_polars_dataframe(self, tmp_experiment_dir: Path) -> None:
        from sentinel.tracking.comparison import compare_runs

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_metrics(run_id, {"f1": 0.8})

        df = compare_runs([run_id], base_dir=str(tmp_experiment_dir))
        assert isinstance(df, pl.DataFrame)

    def test_contains_run_id_column(self, tmp_experiment_dir: Path) -> None:
        from sentinel.tracking.comparison import compare_runs

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_metrics(run_id, {"f1": 0.8})

        df = compare_runs([run_id], base_dir=str(tmp_experiment_dir))
        assert "run_id" in df.columns

    def test_contains_model_name_column(self, tmp_experiment_dir: Path) -> None:
        from sentinel.tracking.comparison import compare_runs

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_metrics(run_id, {"f1": 0.8})

        df = compare_runs([run_id], base_dir=str(tmp_experiment_dir))
        assert "model_name" in df.columns

    def test_contains_metric_columns(self, tmp_experiment_dir: Path) -> None:
        from sentinel.tracking.comparison import compare_runs

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        run_id = tracker.create_run("zscore")
        tracker.log_metrics(run_id, {"f1": 0.8, "auc_roc": 0.9})

        df = compare_runs([run_id], base_dir=str(tmp_experiment_dir))
        assert "f1" in df.columns
        assert "auc_roc" in df.columns

    def test_multiple_runs_produce_multiple_rows(
        self, tmp_experiment_dir: Path
    ) -> None:
        from sentinel.tracking.comparison import compare_runs

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        ids = []
        for score in (0.6, 0.8, 0.9):
            run_id = tracker.create_run("zscore")
            tracker.log_metrics(run_id, {"f1": score})
            ids.append(run_id)

        df = compare_runs(ids, base_dir=str(tmp_experiment_dir))
        assert df.height == 3

    def test_sorted_by_f1_descending(self, tmp_experiment_dir: Path) -> None:
        from sentinel.tracking.comparison import compare_runs

        tracker = LocalTracker(base_dir=str(tmp_experiment_dir))
        ids = []
        for score in (0.6, 0.9, 0.75):
            run_id = tracker.create_run("zscore")
            tracker.log_metrics(run_id, {"f1": score})
            ids.append(run_id)

        df = compare_runs(ids, base_dir=str(tmp_experiment_dir))
        f1_values = df.get_column("f1").to_list()
        assert f1_values == sorted(f1_values, reverse=True)

    def test_empty_run_list_returns_empty_dataframe(
        self, tmp_experiment_dir: Path
    ) -> None:
        from sentinel.tracking.comparison import compare_runs

        df = compare_runs([], base_dir=str(tmp_experiment_dir))
        assert isinstance(df, pl.DataFrame)
        assert df.height == 0
