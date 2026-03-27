"""CLI command: generate visualizations for experiment runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from sentinel.core.exceptions import SentinelError
from sentinel.tracking.artifacts import load_predictions
from sentinel.tracking.experiment import LocalTracker

console = Console()

visualize_app = typer.Typer(help="Generate visualizations for experiment runs.")


@visualize_app.callback(invoke_without_command=True)
def visualize(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="Experiment run ID to visualize.",
    ),
    viz_type: str = typer.Option(
        "timeseries",
        "--type",
        "-t",
        help="Visualization type: timeseries, reconstruction, or latent.",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (e.g. plot.png). Defaults to <run_id>_<type>.png.",
    ),
    experiments_dir: str = typer.Option(
        "data/experiments",
        "--experiments-dir",
        help="Root directory for experiment artifacts.",
    ),
) -> None:
    """Generate a visualization for a completed experiment run.

    Loads the run metadata, model, and predictions from the experiment
    tracker, then generates the requested plot type.

    Supported types:

    * **timeseries** -- time series with anomaly regions shaded red
    * **reconstruction** -- original vs reconstructed overlay + error heatmap
    * **latent** -- t-SNE projection of latent embeddings colored by score

    Args:
        run_id: The experiment run identifier.
        viz_type: Type of visualization to generate.
        output: Output file path.  If not provided, defaults to
            ``<run_id>_<type>.png`` in the current directory.
        experiments_dir: Root directory for experiment storage.
    """
    valid_types = {"timeseries", "reconstruction", "latent"}
    if viz_type not in valid_types:
        console.print(
            f"[red]Invalid visualization type:[/red] '{viz_type}'. "
            f"Choose from: {', '.join(sorted(valid_types))}"
        )
        raise typer.Exit(code=1)

    # Resolve output path.
    if output is None:
        output = f"{run_id}_{viz_type}.png"
    output_path = Path(output)

    # Load run data.
    try:
        tracker = LocalTracker(base_dir=experiments_dir)
        run_data = tracker.get_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(code=1)

    model_name = run_data.get("model_name", "unknown")
    metrics = run_data.get("metrics", {})
    threshold = metrics.get("threshold")

    console.print(f"[bold]Run:[/bold]   {run_id}")
    console.print(f"[bold]Model:[/bold] {model_name}")
    console.print(f"[bold]Type:[/bold]  {viz_type}")

    try:
        if viz_type == "timeseries":
            _visualize_timeseries(
                run_id=run_id,
                model_name=model_name,
                threshold=threshold,
                output_path=output_path,
                experiments_dir=experiments_dir,
            )
        elif viz_type == "reconstruction":
            _visualize_reconstruction(
                run_id=run_id,
                model_name=model_name,
                output_path=output_path,
                experiments_dir=experiments_dir,
            )
        elif viz_type == "latent":
            _visualize_latent(
                run_id=run_id,
                output_path=output_path,
                experiments_dir=experiments_dir,
            )
    except (FileNotFoundError, SentinelError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]Saved:[/green] {output_path}")


def _visualize_timeseries(
    run_id: str,
    model_name: str,
    threshold: float | None,
    output_path: Path,
    experiments_dir: str,
) -> None:
    """Generate a time series plot with anomaly scores.

    Args:
        run_id: Experiment run identifier.
        model_name: Name of the model.
        threshold: Anomaly threshold from evaluation metrics.
        output_path: Where to save the plot.
        experiments_dir: Root experiments directory.
    """
    from sentinel.viz.timeseries import plot_timeseries

    predictions = load_predictions(run_id, base_dir=experiments_dir)
    scores = predictions["scores"]
    labels = predictions["labels"]

    timestamps = np.arange(len(scores))

    # Create a simple single-feature view using the scores as proxy.
    plot_timeseries(
        timestamps=timestamps,
        values=scores.reshape(-1, 1),
        scores=scores,
        labels=labels,
        threshold=threshold,
        title=f"{model_name} - Anomaly Scores (Run: {run_id})",
        output_path=output_path,
    )


def _visualize_reconstruction(
    run_id: str,
    model_name: str,
    output_path: Path,
    experiments_dir: str,
) -> None:
    """Generate a reconstruction plot.

    Loads the model, scores the test data, and overlays original vs
    reconstructed.  Falls back to a simple error plot if reconstruction
    arrays are not available.

    Args:
        run_id: Experiment run identifier.
        model_name: Name of the model.
        output_path: Where to save the plot.
        experiments_dir: Root experiments directory.
    """
    from sentinel.viz.reconstruction import plot_reconstruction

    run_dir = Path(experiments_dir) / run_id
    recon_path = run_dir / "reconstruction.npz"

    if recon_path.exists():
        data = np.load(str(recon_path))
        original = data["original"]
        reconstructed = data["reconstructed"]
    else:
        # Fall back: use predictions scores to create a simple error view.
        predictions = load_predictions(run_id, base_dir=experiments_dir)
        scores = predictions["scores"]
        original = scores.reshape(-1, 1)
        reconstructed = np.zeros_like(original)
        console.print(
            "[yellow]No reconstruction data found; "
            "showing score vs zero baseline.[/yellow]"
        )

    plot_reconstruction(
        original=original,
        reconstructed=reconstructed,
        title=f"{model_name} - Reconstruction (Run: {run_id})",
        output_path=output_path,
    )


def _visualize_latent(
    run_id: str,
    output_path: Path,
    experiments_dir: str,
) -> None:
    """Generate a latent space projection plot.

    Loads embeddings from the run directory.  Falls back to using
    prediction scores projected into 2-D if embeddings are not found.

    Args:
        run_id: Experiment run identifier.
        output_path: Where to save the plot.
        experiments_dir: Root experiments directory.
    """
    from sentinel.viz.latent import plot_latent

    run_dir = Path(experiments_dir) / run_id
    embed_path = run_dir / "embeddings.npz"

    if embed_path.exists():
        data = np.load(str(embed_path))
        embeddings = data["embeddings"]
        scores = data.get("scores")
    else:
        # Fall back: use predictions as a 1-D "embedding".
        predictions = load_predictions(run_id, base_dir=experiments_dir)
        scores = predictions["scores"]
        # Create a pseudo-embedding with score + cumulative score.
        embeddings = np.column_stack(
            [scores, np.cumsum(scores) / (np.arange(len(scores)) + 1)]
        )
        console.print(
            "[yellow]No embeddings found; projecting prediction scores.[/yellow]"
        )

    plot_latent(
        embeddings=embeddings,
        scores=scores,
        output_path=output_path,
        title=f"Latent Space (Run: {run_id})",
    )
