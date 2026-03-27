"""Synthetic multivariate time series data generator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl


def generate_synthetic(
    n_features: int = 5,
    length: int = 10000,
    anomaly_ratio: float = 0.05,
    seed: int = 42,
    start_time: datetime | None = None,
    interval_seconds: int = 60,
) -> pl.DataFrame:
    """Generate synthetic multivariate time series with anomalies.

    Creates N feature channels with different patterns (sinusoidal, trend,
    noise) and injects point, contextual, and collective anomalies.

    Args:
        n_features: Number of feature columns.
        length: Number of rows/timesteps.
        anomaly_ratio: Fraction of points to mark as anomalous.
        seed: Random seed for reproducibility.
        start_time: Starting timestamp (defaults to 2024-01-01 UTC).
        interval_seconds: Seconds between timestamps.

    Returns:
        DataFrame with timestamp, feature_1..N, and is_anomaly columns.
    """
    rng = np.random.RandomState(seed)

    if start_time is None:
        start_time = datetime(2024, 1, 1, tzinfo=UTC)

    timestamps = [
        start_time + timedelta(seconds=i * interval_seconds) for i in range(length)
    ]

    t = np.arange(length, dtype=np.float64)
    features: dict[str, np.ndarray] = {}

    for i in range(n_features):
        freq = 0.01 + rng.random() * 0.05
        phase = rng.random() * 2 * np.pi
        amplitude = 1.0 + rng.random() * 4.0
        trend = rng.random() * 0.001 * (1 if rng.random() > 0.5 else -1)
        noise_level = 0.1 + rng.random() * 0.5

        signal = (
            amplitude * np.sin(2 * np.pi * freq * t + phase)
            + trend * t
            + noise_level * rng.randn(length)
        )
        features[f"feature_{i + 1}"] = signal

    is_anomaly = np.zeros(length, dtype=np.int64)
    n_anomalies = max(1, int(length * anomaly_ratio))

    n_point = n_anomalies // 3
    n_contextual = n_anomalies // 3
    n_collective = n_anomalies - n_point - n_contextual

    available = list(range(length))
    rng.shuffle(available)
    used: set[int] = set()

    # Point anomalies: large spikes
    for _ in range(n_point):
        if not available:
            break
        idx = available.pop()
        used.add(idx)
        is_anomaly[idx] = 1
        feat_idx = rng.randint(0, n_features)
        col = f"feature_{feat_idx + 1}"
        features[col][idx] += rng.choice([-1, 1]) * (5 + rng.random() * 10)

    # Contextual anomalies: value normal globally but wrong for local context
    for _ in range(n_contextual):
        if not available:
            break
        idx = available.pop()
        used.add(idx)
        is_anomaly[idx] = 1
        feat_idx = rng.randint(0, n_features)
        col = f"feature_{feat_idx + 1}"
        local_start = max(0, idx - 10)
        local_end = min(length, idx + 10)
        local_mean = features[col][local_start:local_end].mean()
        features[col][idx] = -local_mean * (2 + rng.random())

    # Collective anomalies: contiguous runs of unusual behavior
    remaining = n_collective
    while remaining > 0 and available:
        run_len = min(remaining, rng.randint(3, 8))
        start_idx = available.pop()
        if start_idx + run_len > length:
            continue
        feat_idx = rng.randint(0, n_features)
        col = f"feature_{feat_idx + 1}"
        for j in range(run_len):
            idx = start_idx + j
            if idx not in used:
                is_anomaly[idx] = 1
                features[col][idx] += rng.random() * 3
                used.add(idx)
                remaining -= 1

    data: dict[str, list[datetime] | np.ndarray] = {"timestamp": timestamps}
    data.update(features)
    data["is_anomaly"] = is_anomaly

    df = pl.DataFrame(data)
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
    return df
