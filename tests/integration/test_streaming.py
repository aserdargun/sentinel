"""Integration tests for the streaming subsystem.

Coverage:
- StreamSimulator consuming async generators from data/streaming.py
- OnlineDetector with sliding window buffer
- ADWIN drift detection
- Alert rules engine (ThresholdBreachAlert, ConsecutiveAnomalyAlert, RateOfChangeAlert)
- Full pipeline: synthetic data -> stream -> detect -> alert
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from datetime import UTC
from typing import Any

import numpy as np
import polars as pl
import pytest

import sentinel.models  # noqa: F401 — trigger registration
from sentinel.data.streaming import stream_from_dataframe, stream_from_parquet
from sentinel.models.statistical.zscore import ZScoreDetector
from sentinel.streaming.alerts import (
    AlertEngine,
    ConsecutiveAnomalyAlert,
    RateOfChangeAlert,
    ThresholdBreachAlert,
)
from sentinel.streaming.drift import ADWINDetector, DriftSimulator
from sentinel.streaming.online_detector import OnlineDetector
from sentinel.streaming.simulator import StreamSimulator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ROWS = 60
N_FEATURES = 2
SEQ_LEN = 10
THRESHOLD = 2.0  # z-score: values above this are labelled anomalous


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_df() -> pl.DataFrame:
    """60-row, 2-feature DataFrame with no anomalies."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N_ROWS, N_FEATURES))
    return pl.DataFrame(
        {
            "feature_1": data[:, 0].tolist(),
            "feature_2": data[:, 1].tolist(),
        }
    )


@pytest.fixture
def anomaly_df() -> pl.DataFrame:
    """60-row DataFrame with a clear spike at index 40."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((N_ROWS, N_FEATURES))
    data[40] += 20.0  # obvious anomaly
    return pl.DataFrame(
        {
            "feature_1": data[:, 0].tolist(),
            "feature_2": data[:, 1].tolist(),
        }
    )


@pytest.fixture
def fitted_zscore(normal_df: pl.DataFrame) -> ZScoreDetector:
    """ZScoreDetector fitted on the normal_df array."""
    model = ZScoreDetector(window_size=SEQ_LEN, threshold_sigma=THRESHOLD)
    X = normal_df.to_numpy()
    model.fit(X)
    return model


@pytest.fixture
def online_detector(fitted_zscore: ZScoreDetector) -> OnlineDetector:
    """OnlineDetector wrapping a fitted ZScoreDetector."""
    return OnlineDetector(
        model=fitted_zscore,
        seq_len=SEQ_LEN,
        threshold=THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Helper async generator
# ---------------------------------------------------------------------------


async def _rows_from_df(df: pl.DataFrame) -> AsyncIterator[dict[str, Any]]:
    """Yield rows as dicts at maximum speed (no sleeping)."""
    async for row in stream_from_dataframe(df, speed=0.0):
        yield row


# ---------------------------------------------------------------------------
# TestOnlineDetector
# ---------------------------------------------------------------------------


class TestOnlineDetector:
    """Unit-style tests for OnlineDetector."""

    def test_buffer_empty_on_init(self, online_detector: OnlineDetector) -> None:
        """Buffer starts empty."""
        assert online_detector.buffer_size == 0

    def test_not_ready_before_seq_len(self, online_detector: OnlineDetector) -> None:
        """is_ready is False until seq_len rows have been ingested."""
        row = {"feature_1": 0.5, "feature_2": -0.3}
        for _ in range(SEQ_LEN - 1):
            result = online_detector.update(row)
            assert result is None

        assert not online_detector.is_ready

    def test_ready_at_seq_len(self, online_detector: OnlineDetector) -> None:
        """is_ready becomes True exactly when seq_len rows have been ingested."""
        row = {"feature_1": 0.5, "feature_2": -0.3}
        for _ in range(SEQ_LEN):
            online_detector.update(row)

        assert online_detector.is_ready

    def test_returns_none_while_filling(self, online_detector: OnlineDetector) -> None:
        """update() returns None for the first seq_len - 1 calls."""
        row = {"feature_1": 0.1, "feature_2": 0.2}
        results = [online_detector.update(row) for _ in range(SEQ_LEN - 1)]
        assert all(r is None for r in results)

    def test_returns_dict_when_ready(self, online_detector: OnlineDetector) -> None:
        """update() returns a dict once the buffer is full."""
        row = {"feature_1": 0.1, "feature_2": 0.2}
        result = None
        for _ in range(SEQ_LEN):
            result = online_detector.update(row)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, online_detector: OnlineDetector) -> None:
        """Detection result contains score, label, and threshold."""
        row = {"feature_1": 0.1, "feature_2": 0.2}
        for _ in range(SEQ_LEN):
            result = online_detector.update(row)

        assert result is not None
        assert "score" in result
        assert "label" in result
        assert "threshold" in result

    def test_timestamp_propagated(self, online_detector: OnlineDetector) -> None:
        """timestamp key in the row dict is included in the result."""
        ts = "2024-01-01T00:00:00Z"
        row = {"timestamp": ts, "feature_1": 0.1, "feature_2": 0.2}
        result = None
        for _ in range(SEQ_LEN):
            result = online_detector.update(row)
        assert result is not None
        assert result.get("timestamp") == ts

    def test_is_anomaly_excluded_from_features(
        self, online_detector: OnlineDetector
    ) -> None:
        """is_anomaly column is not treated as a feature."""
        row = {"feature_1": 0.1, "feature_2": 0.2, "is_anomaly": 1}
        result = None
        for _ in range(SEQ_LEN):
            result = online_detector.update(row)
        assert result is not None

    def test_label_is_zero_for_normal_input(
        self, online_detector: OnlineDetector
    ) -> None:
        """Normal values (z-score well below threshold) are labelled 0."""
        row = {"feature_1": 0.1, "feature_2": 0.1}
        result = None
        for _ in range(SEQ_LEN):
            result = online_detector.update(row)
        assert result is not None
        assert result["label"] == 0

    def test_label_is_one_for_extreme_input(
        self,
        fitted_zscore: ZScoreDetector,
    ) -> None:
        """Extreme values (spike >> threshold sigmas) are labelled 1."""
        detector = OnlineDetector(
            model=fitted_zscore, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        # Fill buffer with normal values first
        for _ in range(SEQ_LEN - 1):
            detector.update({"feature_1": 0.1, "feature_2": 0.1})
        # Insert a very large anomaly as the final point
        result = detector.update({"feature_1": 1000.0, "feature_2": 1000.0})
        assert result is not None
        assert result["label"] == 1

    def test_reset_clears_buffer(self, online_detector: OnlineDetector) -> None:
        """reset() empties the buffer so is_ready becomes False again."""
        row = {"feature_1": 0.1, "feature_2": 0.2}
        for _ in range(SEQ_LEN):
            online_detector.update(row)
        assert online_detector.is_ready

        online_detector.reset()
        assert online_detector.buffer_size == 0
        assert not online_detector.is_ready

    def test_invalid_seq_len_raises(self, fitted_zscore: ZScoreDetector) -> None:
        """seq_len < 1 raises ValueError on construction."""
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            OnlineDetector(model=fitted_zscore, seq_len=0, threshold=THRESHOLD)

    def test_threshold_setter(self, online_detector: OnlineDetector) -> None:
        """threshold property can be updated after construction."""
        online_detector.threshold = 5.0
        assert online_detector.threshold == pytest.approx(5.0)

    def test_score_is_finite(self, online_detector: OnlineDetector) -> None:
        """Detection score is a finite float."""
        row = {"feature_1": 0.5, "feature_2": -0.5}
        result = None
        for _ in range(SEQ_LEN):
            result = online_detector.update(row)
        assert result is not None
        assert np.isfinite(result["score"])


# ---------------------------------------------------------------------------
# TestBasicStreaming
# ---------------------------------------------------------------------------


class TestBasicStreaming:
    """Full pipeline: stream_from_dataframe -> OnlineDetector -> scores."""

    async def test_scores_produced_after_buffer_fills(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """After SEQ_LEN rows are consumed, detections are produced."""
        detections: list[dict[str, Any]] = []

        async for row in stream_from_dataframe(normal_df, speed=0.0):
            result = online_detector.update(row)
            if result is not None:
                detections.append(result)

        expected = N_ROWS - SEQ_LEN + 1
        assert len(detections) == expected

    async def test_all_scores_are_finite(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """All scores produced from normal data are finite."""
        async for row in stream_from_dataframe(normal_df, speed=0.0):
            result = online_detector.update(row)
            if result is not None:
                assert np.isfinite(result["score"])

    async def test_scores_are_non_negative(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """Z-score anomaly scores are >= 0."""
        async for row in stream_from_dataframe(normal_df, speed=0.0):
            result = online_detector.update(row)
            if result is not None:
                assert result["score"] >= 0.0

    async def test_anomaly_detected_at_spike(
        self,
        anomaly_df: pl.DataFrame,
        fitted_zscore: ZScoreDetector,
    ) -> None:
        """The spike at index 40 receives a label of 1."""
        detector = OnlineDetector(
            model=fitted_zscore,
            seq_len=SEQ_LEN,
            threshold=THRESHOLD,
        )
        labelled_anomaly = False

        async for i, row in _indexed_stream(anomaly_df, speed=0.0):
            result = detector.update(row)
            if result is not None and i == 40 and result["label"] == 1:
                labelled_anomaly = True

        assert labelled_anomaly

    async def test_simulator_yields_result_per_row(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """StreamSimulator yields exactly N_ROWS result dicts."""
        source = stream_from_dataframe(normal_df, speed=0.0)
        sim = StreamSimulator(source=source, detector=online_detector)
        results: list[dict[str, Any]] = []
        async for r in sim.run():
            results.append(r)

        assert len(results) == N_ROWS

    async def test_simulator_result_structure(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """Every result dict from the simulator has the expected keys."""
        source = stream_from_dataframe(normal_df, speed=0.0)
        sim = StreamSimulator(source=source, detector=online_detector)
        async for r in sim.run():
            assert "row" in r
            assert "detection" in r
            assert "alerts" in r
            assert "injected" in r
            break  # check first result only

    async def test_simulator_rows_processed_counter(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """rows_processed counter equals N_ROWS after exhausting the source."""
        source = stream_from_dataframe(normal_df, speed=0.0)
        sim = StreamSimulator(source=source, detector=online_detector)
        async for _ in sim.run():
            pass

        assert sim.rows_processed == N_ROWS

    async def test_simulator_stream_from_parquet(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
        tmp_path,
    ) -> None:
        """Simulator works with stream_from_parquet as the source."""
        parquet_path = tmp_path / "stream_test.parquet"
        normal_df.write_parquet(parquet_path)

        source = stream_from_parquet(parquet_path, speed=0.0)
        sim = StreamSimulator(source=source, detector=online_detector)
        results: list[dict[str, Any]] = []
        async for r in sim.run():
            results.append(r)

        assert len(results) == N_ROWS


# ---------------------------------------------------------------------------
# TestAnomalyInjection
# ---------------------------------------------------------------------------


class TestAnomalyInjection:
    """Tests for inject_anomalies=True in StreamSimulator."""

    async def test_injection_flag_increases_injected_count(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """With anomaly_ratio=1.0, every row should be injected."""
        source = stream_from_dataframe(normal_df, speed=0.0)
        sim = StreamSimulator(
            source=source,
            detector=online_detector,
            inject_anomalies=True,
            anomaly_ratio=1.0,
        )
        async for _ in sim.run():
            pass

        assert sim.anomalies_injected == N_ROWS

    async def test_no_injection_by_default(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """By default inject_anomalies=False produces zero injections."""
        source = stream_from_dataframe(normal_df, speed=0.0)
        sim = StreamSimulator(source=source, detector=online_detector)
        async for _ in sim.run():
            pass

        assert sim.anomalies_injected == 0

    async def test_injected_flag_per_row(
        self,
        normal_df: pl.DataFrame,
        online_detector: OnlineDetector,
    ) -> None:
        """With anomaly_ratio=1.0, every result dict has injected=True."""
        source = stream_from_dataframe(normal_df, speed=0.0)
        sim = StreamSimulator(
            source=source,
            detector=online_detector,
            inject_anomalies=True,
            anomaly_ratio=1.0,
        )
        async for r in sim.run():
            assert r["injected"] is True


# ---------------------------------------------------------------------------
# TestStreamingDriftDetection
# ---------------------------------------------------------------------------


class TestStreamingDriftDetection:
    """Integration: ADWIN drift detection on a streamed score sequence."""

    def test_abrupt_drift_detected(self) -> None:
        """ADWIN detects a large mean shift injected mid-stream."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(200)
        shifted = DriftSimulator.abrupt_drift(data, change_point=100, magnitude=5.0)

        detector = ADWINDetector(delta=0.002)
        drift_detected = False
        for val in shifted:
            if detector.update(float(val)):
                drift_detected = True
                break

        assert drift_detected

    def test_gradual_drift_detected(self) -> None:
        """ADWIN detects a gradual ramp-up shift."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal(500)
        shifted = DriftSimulator.gradual_drift(data, change_point=250, magnitude=10.0)

        detector = ADWINDetector(delta=0.002)
        drift_detected = False
        for val in shifted:
            if detector.update(float(val)):
                drift_detected = True
                break

        assert drift_detected

    def test_drifted_data_triggers_more_events_than_stationary(self) -> None:
        """Drifted data triggers more ADWIN events than stationary data.

        ADWIN can produce false positives on i.i.d. data, so we test the
        relative property: a clearly shifted stream should produce more
        drift detections than a stationary stream of the same length.
        """
        rng = np.random.default_rng(99)
        stationary = rng.standard_normal(500)
        drifted = DriftSimulator.abrupt_drift(
            stationary.copy(), change_point=250, magnitude=10.0
        )

        def count_drift(arr: np.ndarray) -> int:
            det = ADWINDetector(delta=0.002)
            return sum(1 for v in arr if det.update(float(v)))

        stationary_events = count_drift(stationary)
        drifted_events = count_drift(drifted)
        assert drifted_events > stationary_events

    def test_adwin_window_shrinks_after_drift(self) -> None:
        """ADWIN window width decreases after a drift event."""
        rng = np.random.default_rng(11)
        data = rng.standard_normal(200)
        shifted = DriftSimulator.abrupt_drift(data, change_point=100, magnitude=8.0)

        detector = ADWINDetector(delta=0.002)
        width_before = 0
        for val in shifted:
            if detector.update(float(val)):
                break
            width_before = detector.width

        assert detector.width <= width_before

    def test_adwin_drift_count_increments(self) -> None:
        """drift_count increments each time drift is detected."""
        rng = np.random.default_rng(55)
        data = rng.standard_normal(100)
        shifted = DriftSimulator.abrupt_drift(data, change_point=50, magnitude=10.0)

        detector = ADWINDetector(delta=0.002)
        for val in shifted:
            detector.update(float(val))

        assert detector.drift_count >= 1

    def test_adwin_reset_clears_state(self) -> None:
        """reset() empties the window and zeroes the drift count."""
        rng = np.random.default_rng(22)
        data = rng.standard_normal(100)
        shifted = DriftSimulator.abrupt_drift(data, change_point=50, magnitude=10.0)

        detector = ADWINDetector(delta=0.002)
        for val in shifted:
            detector.update(float(val))

        detector.reset()
        assert detector.width == 0
        assert detector.mean == pytest.approx(0.0)

    def test_drift_detected_via_online_detector_scores(
        self,
        fitted_zscore: ZScoreDetector,
    ) -> None:
        """ADWIN applied to scores from OnlineDetector detects distribution shift."""
        rng = np.random.default_rng(33)
        pre_shift = rng.standard_normal((50, N_FEATURES))
        post_shift = rng.standard_normal((50, N_FEATURES)) + 10.0
        combined = np.vstack([pre_shift, post_shift])

        detector = OnlineDetector(
            model=fitted_zscore, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        adwin = ADWINDetector(delta=0.002)
        drift_detected = False

        for row_arr in combined:
            row = {f"feature_{i + 1}": float(row_arr[i]) for i in range(N_FEATURES)}
            result = detector.update(row)
            if result is not None:
                if adwin.update(result["score"]):
                    drift_detected = True
                    break

        assert drift_detected


# ---------------------------------------------------------------------------
# TestAlertRules
# ---------------------------------------------------------------------------


class TestThresholdBreachAlert:
    """Tests for ThresholdBreachAlert."""

    def test_fires_above_threshold(self) -> None:
        """Alert fires when score exceeds threshold."""
        rule = ThresholdBreachAlert(threshold=0.8)
        alert = rule.check(score=0.95, label=1, timestamp="t1")
        assert alert is not None
        assert alert["rule"] == "threshold_breach"

    def test_does_not_fire_at_threshold(self) -> None:
        """Alert does not fire when score equals threshold (strict >)."""
        rule = ThresholdBreachAlert(threshold=0.8)
        alert = rule.check(score=0.8, label=1, timestamp="t1")
        assert alert is None

    def test_does_not_fire_below_threshold(self) -> None:
        """Alert does not fire when score is below threshold."""
        rule = ThresholdBreachAlert(threshold=0.8)
        alert = rule.check(score=0.5, label=0, timestamp="t1")
        assert alert is None

    def test_alert_contains_score(self) -> None:
        """Alert dict includes the triggering score."""
        rule = ThresholdBreachAlert(threshold=0.5)
        alert = rule.check(score=0.9, label=1, timestamp="t2")
        assert alert is not None
        assert alert["score"] == pytest.approx(0.9)

    def test_alert_contains_threshold(self) -> None:
        """Alert dict includes the threshold value."""
        rule = ThresholdBreachAlert(threshold=0.5)
        alert = rule.check(score=0.9, label=1, timestamp="t2")
        assert alert is not None
        assert alert["threshold"] == pytest.approx(0.5)

    def test_custom_rule_name(self) -> None:
        """Custom name is reflected in the alert dict."""
        rule = ThresholdBreachAlert(threshold=0.5, name="my_breach")
        alert = rule.check(score=0.9, label=1, timestamp=None)
        assert alert is not None
        assert alert["rule"] == "my_breach"

    def test_timestamp_preserved(self) -> None:
        """Timestamp is included in the alert dict."""
        ts = "2024-06-01T12:00:00Z"
        rule = ThresholdBreachAlert(threshold=0.5)
        alert = rule.check(score=0.9, label=1, timestamp=ts)
        assert alert is not None
        assert alert["timestamp"] == ts

    def test_none_timestamp_preserved(self) -> None:
        """None timestamp is stored as None, not omitted."""
        rule = ThresholdBreachAlert(threshold=0.5)
        alert = rule.check(score=0.9, label=1, timestamp=None)
        assert alert is not None
        assert alert["timestamp"] is None


class TestConsecutiveAnomalyAlert:
    """Tests for ConsecutiveAnomalyAlert."""

    def test_fires_after_required_count(self) -> None:
        """Alert fires exactly when the streak reaches the required count."""
        rule = ConsecutiveAnomalyAlert(count=3)
        rule.check(0.9, 1, "t1")
        rule.check(0.8, 1, "t2")
        alert = rule.check(0.7, 1, "t3")
        assert alert is not None
        assert alert["rule"] == "consecutive_anomalies"

    def test_does_not_fire_before_count(self) -> None:
        """Alert does not fire until the count threshold is reached."""
        rule = ConsecutiveAnomalyAlert(count=3)
        assert rule.check(0.9, 1, "t1") is None
        assert rule.check(0.8, 1, "t2") is None

    def test_resets_on_normal_label(self) -> None:
        """Streak resets when a normal (label=0) observation arrives."""
        rule = ConsecutiveAnomalyAlert(count=3)
        rule.check(0.9, 1, "t1")
        rule.check(0.8, 1, "t2")
        rule.check(0.1, 0, "t3")  # breaks streak
        # Should not fire for the next two anomalies
        assert rule.check(0.9, 1, "t4") is None
        assert rule.check(0.8, 1, "t5") is None

    def test_alert_count_field(self) -> None:
        """Alert dict includes the streak length."""
        rule = ConsecutiveAnomalyAlert(count=2)
        rule.check(0.9, 1, "t1")
        alert = rule.check(0.8, 1, "t2")
        assert alert is not None
        assert alert["count"] == 2

    def test_alert_start_and_end_timestamps(self) -> None:
        """Alert includes the start and end timestamps of the streak."""
        rule = ConsecutiveAnomalyAlert(count=2)
        rule.check(0.9, 1, "t1")
        alert = rule.check(0.8, 1, "t2")
        assert alert is not None
        assert alert["start"] == "t1"
        assert alert["end"] == "t2"

    def test_resets_after_firing(self) -> None:
        """Streak counter resets after the alert fires so the next run can trigger."""
        rule = ConsecutiveAnomalyAlert(count=2)
        rule.check(0.9, 1, "t1")
        rule.check(0.8, 1, "t2")  # fires
        # New streak of 2 should fire again
        rule.check(0.9, 1, "t3")
        alert = rule.check(0.8, 1, "t4")
        assert alert is not None

    def test_invalid_count_raises(self) -> None:
        """count < 1 raises ValueError."""
        with pytest.raises(ValueError, match="count must be >= 1"):
            ConsecutiveAnomalyAlert(count=0)

    def test_reset_clears_streak(self) -> None:
        """reset() zeroes the streak counter."""
        rule = ConsecutiveAnomalyAlert(count=3)
        rule.check(0.9, 1, "t1")
        rule.check(0.8, 1, "t2")
        rule.reset()
        # After reset, need count again to fire
        assert rule.check(0.9, 1, "t3") is None


class TestRateOfChangeAlert:
    """Tests for RateOfChangeAlert."""

    def test_no_alert_on_first_point(self) -> None:
        """No alert on the very first call (no previous score to compare)."""
        rule = RateOfChangeAlert(delta=0.3)
        alert = rule.check(0.5, 0, "t1")
        assert alert is None

    def test_fires_on_large_change(self) -> None:
        """Alert fires when |current - previous| exceeds delta."""
        rule = RateOfChangeAlert(delta=0.3)
        rule.check(0.1, 0, "t1")
        alert = rule.check(0.9, 1, "t2")
        assert alert is not None
        assert alert["rule"] == "rate_of_change"

    def test_does_not_fire_on_small_change(self) -> None:
        """No alert when the change is below delta."""
        rule = RateOfChangeAlert(delta=0.5)
        rule.check(0.4, 0, "t1")
        alert = rule.check(0.5, 0, "t2")
        assert alert is None

    def test_alert_contains_delta(self) -> None:
        """Alert dict includes the observed change magnitude."""
        rule = RateOfChangeAlert(delta=0.3)
        rule.check(0.1, 0, "t1")
        alert = rule.check(0.9, 1, "t2")
        assert alert is not None
        assert alert["delta"] == pytest.approx(0.8)

    def test_invalid_delta_raises(self) -> None:
        """delta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="delta must be > 0"):
            RateOfChangeAlert(delta=0.0)

    def test_reset_clears_previous_score(self) -> None:
        """reset() clears previous score so next call does not fire."""
        rule = RateOfChangeAlert(delta=0.3)
        rule.check(0.1, 0, "t1")
        rule.reset()
        # After reset, there is no previous score
        alert = rule.check(0.9, 1, "t2")
        assert alert is None


# ---------------------------------------------------------------------------
# TestAlertEngine
# ---------------------------------------------------------------------------


class TestAlertEngine:
    """Tests for AlertEngine aggregation."""

    def test_returns_empty_list_when_no_rules_fire(self) -> None:
        """AlertEngine returns [] when no rules are triggered."""
        engine = AlertEngine(
            rules=[
                ThresholdBreachAlert(threshold=1.0),
                ConsecutiveAnomalyAlert(count=5),
            ]
        )
        alerts = engine.check(score=0.3, label=0, timestamp="t1")
        assert alerts == []

    def test_returns_all_triggered_alerts(self) -> None:
        """AlertEngine returns one alert per triggered rule."""
        engine = AlertEngine(
            rules=[
                ThresholdBreachAlert(threshold=0.5),
                ConsecutiveAnomalyAlert(count=1),
            ]
        )
        alerts = engine.check(score=0.9, label=1, timestamp="t1")
        assert len(alerts) == 2

    def test_rules_are_evaluated_independently(self) -> None:
        """Each rule fires or not based on its own state."""
        thresh_rule = ThresholdBreachAlert(threshold=0.5)
        consec_rule = ConsecutiveAnomalyAlert(count=3)
        engine = AlertEngine(rules=[thresh_rule, consec_rule])

        # Only threshold fires here (streak only at 1)
        alerts = engine.check(score=0.9, label=1, timestamp="t1")
        assert len(alerts) == 1
        assert alerts[0]["rule"] == "threshold_breach"

    def test_reset_propagates_to_all_rules(self) -> None:
        """engine.reset() calls reset() on every registered rule."""
        consec_rule = ConsecutiveAnomalyAlert(count=2)
        rate_rule = RateOfChangeAlert(delta=0.3)
        engine = AlertEngine(rules=[consec_rule, rate_rule])

        # Build some state
        engine.check(score=0.9, label=1, timestamp="t1")
        engine.check(score=0.5, label=0, timestamp="t2")

        engine.reset()

        # After reset, consecutive streak is zero
        assert consec_rule._streak == 0
        # After reset, rate of change has no previous score
        assert rate_rule._prev_score is None

    def test_rules_property_returns_copy(self) -> None:
        """rules property returns a list (copy, not internal reference)."""
        engine = AlertEngine(rules=[ThresholdBreachAlert(threshold=0.5)])
        rules_copy = engine.rules
        assert isinstance(rules_copy, list)
        assert len(rules_copy) == 1


# ---------------------------------------------------------------------------
# TestAlertPipelineIntegration
# ---------------------------------------------------------------------------


class TestAlertPipelineIntegration:
    """Integration: alerts fire as the simulator streams anomalous data."""

    async def test_threshold_alerts_fire_during_stream(
        self,
        anomaly_df: pl.DataFrame,
        fitted_zscore: ZScoreDetector,
    ) -> None:
        """ThresholdBreachAlert fires at least once when streaming anomaly data."""
        threshold = THRESHOLD
        alert_engine = AlertEngine(rules=[ThresholdBreachAlert(threshold=threshold)])
        detector = OnlineDetector(
            model=fitted_zscore,
            seq_len=SEQ_LEN,
            threshold=threshold,
        )
        source = stream_from_dataframe(anomaly_df, speed=0.0)
        sim = StreamSimulator(
            source=source,
            detector=detector,
            alert_engine=alert_engine,
        )

        fired: list[dict[str, Any]] = []
        async for result in sim.run():
            fired.extend(result["alerts"])

        assert len(fired) > 0
        assert all(a["rule"] == "threshold_breach" for a in fired)

    async def test_consecutive_alert_fires_with_repeated_anomalies(self) -> None:
        """ConsecutiveAnomalyAlert fires when a run of anomalies is injected.

        Uses a mock model that always returns a high score so that every
        row in the anomalous block receives label=1.  This isolates the
        ConsecutiveAnomalyAlert behaviour from z-score windowing effects.
        """

        # Mock model: always returns score = 10.0 (well above any threshold)
        class _HighScoreModel(ZScoreDetector):
            def score(self, X: np.ndarray) -> np.ndarray:
                return np.full(X.shape[0], 10.0)

        rng = np.random.default_rng(5)
        train_data = rng.standard_normal((60, N_FEATURES))
        mock_model = _HighScoreModel(window_size=SEQ_LEN, threshold_sigma=THRESHOLD)
        mock_model.fit(train_data)

        data = rng.standard_normal((40, N_FEATURES))
        df = pl.DataFrame(
            {
                "feature_1": data[:, 0].tolist(),
                "feature_2": data[:, 1].tolist(),
            }
        )

        alert_engine = AlertEngine(rules=[ConsecutiveAnomalyAlert(count=3)])
        detector = OnlineDetector(
            model=mock_model,
            seq_len=SEQ_LEN,
            threshold=THRESHOLD,
        )
        source = stream_from_dataframe(df, speed=0.0)
        sim = StreamSimulator(
            source=source,
            detector=detector,
            alert_engine=alert_engine,
        )

        fired: list[dict[str, Any]] = []
        async for result in sim.run():
            fired.extend(result["alerts"])

        assert any(a["rule"] == "consecutive_anomalies" for a in fired)

    async def test_no_alerts_on_normal_data(
        self,
        normal_df: pl.DataFrame,
        fitted_zscore: ZScoreDetector,
    ) -> None:
        """No threshold alerts fire on clean normally-distributed data."""
        # Use a high threshold to ensure normal data never crosses it
        high_threshold = 20.0
        alert_engine = AlertEngine(
            rules=[ThresholdBreachAlert(threshold=high_threshold)]
        )
        detector = OnlineDetector(
            model=fitted_zscore,
            seq_len=SEQ_LEN,
            threshold=high_threshold,
        )
        source = stream_from_dataframe(normal_df, speed=0.0)
        sim = StreamSimulator(
            source=source,
            detector=detector,
            alert_engine=alert_engine,
        )

        fired: list[dict[str, Any]] = []
        async for result in sim.run():
            fired.extend(result["alerts"])

        assert len(fired) == 0


# ---------------------------------------------------------------------------
# TestSpeedControl
# ---------------------------------------------------------------------------


class TestSpeedControl:
    """Tests that speed multiplier affects streaming timing."""

    @pytest.mark.slow
    async def test_high_speed_faster_than_real_time(self, tmp_path) -> None:
        """Speed=0 (no sleep) finishes much faster than speed=1.0 would at real-time."""
        # Build a small DataFrame with second-level timestamps so the base delay
        # would be 1 second per row at speed=1.0.  At speed=0.0 there is no sleep.
        from datetime import datetime

        n = 5
        timestamps = [datetime(2024, 1, 1, 0, 0, i, tzinfo=UTC) for i in range(n)]
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(timestamps).dt.cast_time_unit("us"),
                "feature_1": [float(i) for i in range(n)],
            }
        )
        parquet_path = tmp_path / "speed_test.parquet"
        df.write_parquet(parquet_path)

        start = time.monotonic()
        async for _ in stream_from_parquet(parquet_path, speed=0.0):
            pass
        elapsed = time.monotonic() - start

        # At real-time (speed=1.0) this would take ~4 seconds.
        # At speed=0.0 it should finish in well under 1 second.
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the streaming pipeline."""

    async def test_empty_dataframe_produces_no_results(
        self,
        fitted_zscore: ZScoreDetector,
    ) -> None:
        """An empty DataFrame source yields no results from the simulator."""
        empty_df = pl.DataFrame({"feature_1": pl.Series([], dtype=pl.Float64)})
        detector = OnlineDetector(
            model=fitted_zscore, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        source = stream_from_dataframe(empty_df, speed=0.0)
        sim = StreamSimulator(source=source, detector=detector)

        results: list[dict[str, Any]] = []
        async for r in sim.run():
            results.append(r)

        assert results == []

    async def test_stream_shorter_than_seq_len(
        self,
        fitted_zscore: ZScoreDetector,
    ) -> None:
        """When fewer rows than seq_len are streamed, no detections are produced."""
        short_df = pl.DataFrame(
            {
                "feature_1": [0.1, 0.2, 0.3],
                "feature_2": [0.4, 0.5, 0.6],
            }
        )
        # seq_len=10 > 3 rows
        detector = OnlineDetector(
            model=fitted_zscore, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        source = stream_from_dataframe(short_df, speed=0.0)
        sim = StreamSimulator(source=source, detector=detector)

        detections: list[Any] = []
        async for r in sim.run():
            if r["detection"] is not None:
                detections.append(r["detection"])

        assert detections == []

    async def test_single_row_stream(
        self,
        fitted_zscore: ZScoreDetector,
    ) -> None:
        """A single-row DataFrame produces no detection (buffer not full)."""
        single_df = pl.DataFrame({"feature_1": [1.0], "feature_2": [2.0]})
        detector = OnlineDetector(
            model=fitted_zscore, seq_len=SEQ_LEN, threshold=THRESHOLD
        )
        source = stream_from_dataframe(single_df, speed=0.0)
        sim = StreamSimulator(source=source, detector=detector)

        async for r in sim.run():
            assert r["detection"] is None

    async def test_seq_len_one_produces_detections_immediately(
        self,
        fitted_zscore: ZScoreDetector,
        normal_df: pl.DataFrame,
    ) -> None:
        """With seq_len=1, every row produces a detection result."""
        detector = OnlineDetector(model=fitted_zscore, seq_len=1, threshold=THRESHOLD)
        source = stream_from_dataframe(normal_df, speed=0.0)
        sim = StreamSimulator(source=source, detector=detector)

        detected_count = 0
        async for r in sim.run():
            if r["detection"] is not None:
                detected_count += 1

        assert detected_count == N_ROWS

    async def test_parquet_not_found_raises(
        self,
        tmp_path,
    ) -> None:
        """stream_from_parquet raises ValidationError when file is missing."""
        from sentinel.core.exceptions import ValidationError

        missing_path = tmp_path / "does_not_exist.parquet"
        with pytest.raises(ValidationError):
            async for _ in stream_from_parquet(missing_path, speed=0.0):
                pass


# ---------------------------------------------------------------------------
# TestDriftSimulator
# ---------------------------------------------------------------------------


class TestDriftSimulator:
    """Tests for DriftSimulator helper methods."""

    def test_abrupt_drift_shape_preserved(self) -> None:
        """Output array has the same shape as input."""
        data = np.ones(100)
        result = DriftSimulator.abrupt_drift(data, change_point=50, magnitude=3.0)
        assert result.shape == data.shape

    def test_abrupt_drift_does_not_mutate_input(self) -> None:
        """abrupt_drift returns a new array; input is unchanged."""
        data = np.zeros(100)
        DriftSimulator.abrupt_drift(data, change_point=50, magnitude=3.0)
        assert np.all(data == 0.0)

    def test_abrupt_drift_magnitude_applied_after_change_point(self) -> None:
        """Values from change_point onward are increased by magnitude."""
        data = np.zeros(100)
        result = DriftSimulator.abrupt_drift(data, change_point=50, magnitude=5.0)
        assert np.all(result[50:] == pytest.approx(5.0))
        assert np.all(result[:50] == pytest.approx(0.0))

    def test_gradual_drift_ramp_shape(self) -> None:
        """Gradual drift ramps from 0 to magnitude linearly."""
        data = np.zeros(100)
        result = DriftSimulator.gradual_drift(data, change_point=0, magnitude=10.0)
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(10.0)

    def test_recurring_drift_alternates(self) -> None:
        """Recurring drift toggles shift on/off every period samples."""
        data = np.zeros(200)
        result = DriftSimulator.recurring_drift(data, period=50, magnitude=5.0)
        # Segment 0 (0..49): not shifted
        assert np.all(result[:50] == pytest.approx(0.0))
        # Segment 1 (50..99): shifted
        assert np.all(result[50:100] == pytest.approx(5.0))
        # Segment 2 (100..149): not shifted
        assert np.all(result[100:150] == pytest.approx(0.0))

    def test_invalid_change_point_raises(self) -> None:
        """change_point >= len(data) raises ValueError."""
        data = np.zeros(50)
        with pytest.raises(ValueError):
            DriftSimulator.abrupt_drift(data, change_point=50, magnitude=1.0)

    def test_negative_change_point_raises(self) -> None:
        """Negative change_point raises ValueError."""
        data = np.zeros(50)
        with pytest.raises(ValueError):
            DriftSimulator.gradual_drift(data, change_point=-1, magnitude=1.0)

    def test_invalid_period_raises(self) -> None:
        """period <= 0 raises ValueError."""
        data = np.zeros(50)
        with pytest.raises(ValueError):
            DriftSimulator.recurring_drift(data, period=0, magnitude=1.0)

    def test_invalid_delta_adwin_raises(self) -> None:
        """ADWINDetector rejects delta outside (0, 1)."""
        with pytest.raises(ValueError, match="delta must be in"):
            ADWINDetector(delta=0.0)

        with pytest.raises(ValueError, match="delta must be in"):
            ADWINDetector(delta=1.0)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


async def _indexed_stream(
    df: pl.DataFrame, speed: float = 0.0
) -> AsyncIterator[tuple[int, dict[str, Any]]]:
    """Yield (index, row_dict) pairs from a DataFrame."""
    i = 0
    async for row in stream_from_dataframe(df, speed=speed):
        yield i, row
        i += 1
