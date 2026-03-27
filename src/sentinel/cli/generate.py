"""CLI command: generate synthetic multivariate time series data."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

from sentinel.data.synthetic import generate_synthetic

console = Console()

generate_app = typer.Typer(help="Generate synthetic anomaly data.")


@generate_app.callback(invoke_without_command=True)
def generate(
    features: int = typer.Option(
        5,
        "--features",
        "-f",
        help="Number of feature columns.",
    ),
    length: int = typer.Option(
        10000,
        "--length",
        "-l",
        help="Number of rows/timesteps.",
    ),
    anomaly_ratio: float = typer.Option(
        0.05,
        "--anomaly-ratio",
        "-a",
        help="Fraction of points to mark as anomalous.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility.",
    ),
    output: str = typer.Option(
        "data/raw/synthetic.parquet",
        "--output",
        "-o",
        help="Output file path (Parquet format).",
    ),
) -> None:
    """Generate synthetic multivariate time series with anomaly injection.

    Creates N feature channels with sinusoidal, trend, and noise patterns
    and injects point, contextual, and collective anomalies.

    Args:
        features: Number of feature columns.
        length: Number of rows to generate.
        anomaly_ratio: Fraction of anomalous points.
        seed: Random seed.
        output: Output Parquet file path.
    """
    if features < 1:
        console.print("[red]--features must be at least 1.[/red]")
        raise typer.Exit(code=1)

    if length < 2:
        console.print("[red]--length must be at least 2.[/red]")
        raise typer.Exit(code=1)

    if not (0.0 <= anomaly_ratio <= 1.0):
        console.print("[red]--anomaly-ratio must be between 0.0 and 1.0.[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"Generating synthetic data: "
        f"{length} rows, {features} features, "
        f"{anomaly_ratio:.1%} anomaly ratio, seed={seed}"
    )

    df = generate_synthetic(
        n_features=features,
        length=length,
        anomaly_ratio=anomaly_ratio,
        seed=seed,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: temp file then rename.
    tmp_path = out_path.with_suffix(".parquet.tmp")
    df.write_parquet(str(tmp_path))
    os.rename(str(tmp_path), str(out_path))

    n_anomalies = int(df.get_column("is_anomaly").sum())

    console.print(f"[green]Generated {df.height} rows to {out_path}[/green]")
    console.print(f"  Features:  {features}")
    console.print(f"  Anomalies: {n_anomalies} ({n_anomalies / df.height:.1%})")
