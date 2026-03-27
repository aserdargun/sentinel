"""CLI command: run anomaly detection in batch or streaming mode."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

import sentinel.models  # noqa: F401 -- triggers model registration
from sentinel.core.exceptions import SentinelError, ValidationError

console = Console()

detect_app = typer.Typer(help="Run anomaly detection on data.")


@detect_app.callback(invoke_without_command=True)
def detect(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to a saved model directory.",
    ),
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to a CSV or Parquet data file.",
    ),
    mode: str = typer.Option(
        "batch",
        "--mode",
        help="Detection mode: batch or streaming.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Anomaly score threshold.",
    ),
    seq_len: int = typer.Option(
        50,
        "--seq-len",
        help="Sliding window length for streaming mode.",
    ),
    speed: float = typer.Option(
        0.0,
        "--speed",
        help="Replay speed for streaming mode (0 = max throughput).",
    ),
) -> None:
    """Detect anomalies using a trained model.

    In **batch** mode the entire dataset is loaded, scored, and a
    summary table is printed.

    In **streaming** mode the data is replayed row-by-row through an
    :class:`~sentinel.streaming.online_detector.OnlineDetector` and
    live results are printed to the console.

    Args:
        model: Path to the saved model directory.
        data: Path to the input data file.
        mode: ``"batch"`` or ``"streaming"``.
        threshold: Score threshold for labelling anomalies.
        seq_len: Window size for the online detector (streaming only).
        speed: Replay speed multiplier (streaming only).
    """
    model_path = Path(model)
    data_path = Path(data)

    if not model_path.exists():
        console.print(f"[red]Model path not found:[/red] {model_path}")
        raise typer.Exit(code=1)

    if not data_path.exists():
        console.print(f"[red]Data path not found:[/red] {data_path}")
        raise typer.Exit(code=1)

    if mode not in ("batch", "streaming"):
        console.print(f"[red]Invalid mode:[/red] {mode}. Use 'batch' or 'streaming'.")
        raise typer.Exit(code=1)

    try:
        if mode == "batch":
            _run_batch(model_path, data_path, threshold)
        else:
            asyncio.run(
                _run_streaming(model_path, data_path, threshold, seq_len, speed)
            )
    except SentinelError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


# ------------------------------------------------------------------
# Batch mode
# ------------------------------------------------------------------


def _run_batch(
    model_path: Path,
    data_path: Path,
    threshold: float,
) -> None:
    """Load model and data, score the full dataset, print results.

    Args:
        model_path: Directory containing the saved model.
        data_path: Path to the data file.
        threshold: Anomaly score threshold.
    """
    import numpy as np

    from sentinel.data.loaders import load_file
    from sentinel.data.preprocessors import fill_nan, scale_zscore, to_numpy
    from sentinel.data.validators import separate_labels, validate_dataframe

    console.print("[bold]Mode:[/bold] batch")
    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print(f"[bold]Data:[/bold] {data_path}")
    console.print()

    # Load and preprocess data.
    df = load_file(data_path)
    df = validate_dataframe(df)
    df, _labels = separate_labels(df)
    df = fill_nan(df)
    df, _stats = scale_zscore(df)
    X = to_numpy(df)

    # Load model.
    detector = _load_model(model_path)
    console.print(f"[green]Model loaded:[/green] {detector.model_name}")

    # Score.
    scores = detector.score(X)
    labels = (scores > threshold).astype(np.int32)

    n_anomalies = int(labels.sum())
    console.print()
    console.print(f"[bold]Samples scored:[/bold] {len(scores)}")
    console.print(f"[bold]Anomalies detected:[/bold] {n_anomalies}")
    console.print(f"[bold]Threshold:[/bold] {threshold:.4f}")
    console.print()

    # Summary statistics.
    table = Table(title="Score Statistics")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="magenta", justify="right")

    table.add_row("Mean", f"{float(scores.mean()):.6f}")
    table.add_row("Std", f"{float(scores.std()):.6f}")
    table.add_row("Min", f"{float(scores.min()):.6f}")
    table.add_row("Max", f"{float(scores.max()):.6f}")
    table.add_row("P50", f"{float(np.percentile(scores, 50)):.6f}")
    table.add_row("P95", f"{float(np.percentile(scores, 95)):.6f}")
    table.add_row("P99", f"{float(np.percentile(scores, 99)):.6f}")

    console.print(table)


# ------------------------------------------------------------------
# Streaming mode
# ------------------------------------------------------------------


async def _run_streaming(
    model_path: Path,
    data_path: Path,
    threshold: float,
    seq_len: int,
    speed: float,
) -> None:
    """Replay data through the online detector and print live results.

    Args:
        model_path: Directory containing the saved model.
        data_path: Path to the data file.
        threshold: Anomaly score threshold.
        seq_len: Sliding window length for the online detector.
        speed: Replay speed multiplier.
    """
    from sentinel.data.streaming import stream_from_parquet
    from sentinel.streaming.alerts import (
        AlertEngine,
        ConsecutiveAnomalyAlert,
        ThresholdBreachAlert,
    )
    from sentinel.streaming.online_detector import OnlineDetector
    from sentinel.streaming.simulator import StreamSimulator

    console.print("[bold]Mode:[/bold] streaming")
    console.print(f"[bold]Model:[/bold] {model_path}")
    console.print(f"[bold]Data:[/bold] {data_path}")
    console.print(f"[bold]Seq len:[/bold] {seq_len}")
    console.print(f"[bold]Speed:[/bold] {speed}")
    console.print()

    # Load model.
    detector_model = _load_model(model_path)
    console.print(f"[green]Model loaded:[/green] {detector_model.model_name}")

    # Set up streaming pipeline.
    source = stream_from_parquet(data_path, speed=speed)
    online = OnlineDetector(detector_model, seq_len=seq_len, threshold=threshold)
    alert_engine = AlertEngine(
        [
            ThresholdBreachAlert(threshold=threshold),
            ConsecutiveAnomalyAlert(count=5),
        ]
    )
    sim = StreamSimulator(
        source=source,
        detector=online,
        alert_engine=alert_engine,
    )

    n_scored = 0
    n_anomalies = 0

    async for result in sim.run():
        det = result.get("detection")
        if det is None:
            continue

        n_scored += 1
        if det["label"] == 1:
            n_anomalies += 1

        ts = det.get("timestamp", "?")
        score_str = f"{det['score']:.4f}"
        if det["label"] == 1:
            label_str = "[red]ANOMALY[/red]"
        else:
            label_str = "[green]normal[/green]"

        console.print(f"  {ts}  score={score_str}  {label_str}")

        for alert in result.get("alerts", []):
            console.print(f"    [yellow]ALERT:[/yellow] {alert.get('rule', '?')}")

    console.print()
    console.print(f"[bold]Total scored:[/bold] {n_scored}")
    console.print(f"[bold]Anomalies:[/bold] {n_anomalies}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _load_model(model_path: Path) -> Any:
    """Load a model from a directory by inspecting its config.

    Reads the ``config.json`` in the model directory to determine the
    model class, then calls ``load()`` on a fresh instance.

    Args:
        model_path: Directory containing model artifacts.

    Returns:
        A loaded :class:`BaseAnomalyDetector` instance.

    Raises:
        ValidationError: If the model cannot be loaded.
    """
    import json

    from sentinel.core.registry import get_model_class

    config_file = model_path / "config.json"
    if not config_file.exists():
        raise ValidationError(f"No config.json found in model directory: {model_path}")

    with open(config_file) as f:
        config = json.load(f)

    model_name = config.get("model_name", "")
    if not model_name:
        raise ValidationError(f"config.json missing 'model_name': {config_file}")

    model_cls = get_model_class(model_name)
    instance = model_cls()
    instance.load(str(model_path))
    return instance
