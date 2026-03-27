"""CLI command: evaluate a completed experiment run."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from sentinel.tracking.experiment import LocalTracker

console = Console()

evaluate_app = typer.Typer(help="Evaluate experiment runs.")


@evaluate_app.callback(invoke_without_command=True)
def evaluate(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="Experiment run ID to evaluate.",
    ),
    experiments_dir: str = typer.Option(
        "data/experiments",
        "--experiments-dir",
        help="Root directory for experiment artifacts.",
    ),
) -> None:
    """Load a completed run and display its evaluation metrics.

    Reads metrics from the experiment tracker and prints them as a
    Rich table.  If the run has no recorded metrics, a warning is shown.

    Args:
        run_id: The experiment run identifier.
        experiments_dir: Root directory for experiment storage.
    """
    try:
        tracker = LocalTracker(base_dir=experiments_dir)
        run_data = tracker.get_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(code=1)

    model_name = run_data.get("model_name", "unknown")
    created_at = run_data.get("created_at", "unknown")

    console.print(f"[bold]Run:[/bold]     {run_id}")
    console.print(f"[bold]Model:[/bold]   {model_name}")
    console.print(f"[bold]Created:[/bold] {created_at}")
    console.print()

    metrics = run_data.get("metrics", {})
    if not metrics:
        console.print("[yellow]No metrics recorded for this run.[/yellow]")
        raise typer.Exit(code=0)

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
