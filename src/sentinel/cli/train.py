"""CLI command: train a model from a YAML config."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sentinel.core.config import RunConfig
from sentinel.core.exceptions import ConfigError, SentinelError, ValidationError
from sentinel.training.trainer import Trainer

console = Console()

train_app = typer.Typer(help="Train an anomaly detection model.")


@train_app.callback(invoke_without_command=True)
def train(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
    data_path: str | None = typer.Option(
        None,
        "--data-path",
        "-d",
        help="Override data path from config.",
    ),
) -> None:
    """Train a model using the provided YAML configuration.

    Loads the config, instantiates a Trainer, runs the full pipeline
    (load, validate, preprocess, fit, evaluate), and prints a metrics
    table.

    Args:
        config: Path to the YAML config file.
        data_path: Optional override for the data path.
    """
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Config file not found:[/red] {config_path}")
        raise typer.Exit(code=1)

    try:
        run_config = RunConfig.from_yaml(config_path)
    except ConfigError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Training model:[/bold] {run_config.model or '(not set)'}")

    try:
        trainer = Trainer(run_config)
        result = trainer.run(data_path=data_path)
    except (ConfigError, ValidationError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except SentinelError as exc:
        console.print(f"[red]Sentinel error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Print summary.
    console.print()
    console.print("[green]Training complete.[/green]")
    console.print(f"  Run ID:   {result['run_id']}")
    console.print(f"  Model:    {result['model_name']}")
    console.print(f"  Duration: {result['duration_s']:.3f}s")
    console.print()

    # Metrics table.
    metrics = result.get("metrics", {})
    if metrics:
        table = Table(title="Evaluation Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta", justify="right")

        for name, value in sorted(metrics.items()):
            if value is None:
                display = "N/A"
            elif isinstance(value, float):
                display = f"{value:.6f}"
            else:
                display = str(value)
            table.add_row(name, display)

        console.print(table)
