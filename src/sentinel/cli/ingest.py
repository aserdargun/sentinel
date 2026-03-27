"""CLI command: ingest a CSV/Parquet file into the data store."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sentinel.core.exceptions import SentinelError, ValidationError
from sentinel.data.ingest import ingest_file

console = Console()

ingest_app = typer.Typer(help="Ingest data files into Sentinel.")


@ingest_app.callback(invoke_without_command=True)
def ingest(
    file: str = typer.Option(
        ...,
        "--file",
        "-f",
        help="Path to CSV or Parquet file to ingest.",
    ),
    data_dir: str = typer.Option(
        "data/raw",
        "--data-dir",
        help="Directory to store ingested Parquet files.",
    ),
    metadata_file: str = typer.Option(
        "data/datasets.json",
        "--metadata-file",
        help="Path to the datasets.json registry.",
    ),
) -> None:
    """Validate and ingest a CSV or Parquet file.

    Validates the schema, assigns a dataset_id, saves as Parquet to the
    data store, and updates the metadata registry.

    Args:
        file: Path to the source file.
        data_dir: Target directory for Parquet storage.
        metadata_file: Path to datasets.json.
    """
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[red]File not found:[/red] {file_path}")
        raise typer.Exit(code=1)

    console.print(f"Ingesting [bold]{file_path.name}[/bold]...")

    try:
        result = ingest_file(
            file_path=file_path,
            data_dir=data_dir,
            metadata_file=metadata_file,
        )
    except ValidationError as exc:
        console.print(f"[red]Validation error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except SentinelError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[green]Ingestion complete.[/green]")
    console.print()

    table = Table(title="Dataset Info")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Dataset ID", result["dataset_id"])
    table.add_row("Shape", f"{result['shape'][0]} rows x {result['shape'][1]} cols")
    table.add_row("Features", ", ".join(result["feature_names"]))

    time_range = result.get("time_range", {})
    table.add_row("Start", time_range.get("start", "N/A"))
    table.add_row("End", time_range.get("end", "N/A"))

    console.print(table)
