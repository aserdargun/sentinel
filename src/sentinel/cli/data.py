"""CLI commands: manage ingested datasets (list, info, delete)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()

data_app = typer.Typer(help="Manage ingested datasets.")


def _load_metadata(metadata_file: str) -> dict[str, Any]:
    """Load the datasets.json registry.

    Args:
        metadata_file: Path to datasets.json.

    Returns:
        Parsed metadata dictionary.
    """
    path = Path(metadata_file)
    if not path.exists():
        return {}
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _save_metadata(metadata_file: str, metadata: dict[str, Any]) -> None:
    """Atomically save the datasets.json registry.

    Args:
        metadata_file: Path to datasets.json.
        metadata: Updated metadata dictionary.
    """
    path = Path(metadata_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, indent=2))
    os.rename(str(tmp), str(path))


@data_app.command("list")
def data_list(
    metadata_file: str = typer.Option(
        "data/datasets.json",
        "--metadata-file",
        help="Path to the datasets.json registry.",
    ),
) -> None:
    """List all ingested datasets.

    Reads the datasets.json registry and prints a summary table with
    dataset IDs, names, shapes, and upload times.

    Args:
        metadata_file: Path to datasets.json.
    """
    metadata = _load_metadata(metadata_file)

    if not metadata:
        console.print("[yellow]No datasets found.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title=f"Datasets ({len(metadata)} total)")
    table.add_column("ID", style="cyan", max_width=36)
    table.add_column("Name", style="white")
    table.add_column("Shape", style="magenta")
    table.add_column("Source", style="blue")
    table.add_column("Uploaded", style="green")

    for dataset_id, info in metadata.items():
        shape = info.get("shape", [0, 0])
        table.add_row(
            dataset_id[:12] + "...",
            info.get("original_name", "unknown"),
            f"{shape[0]}x{shape[1]}",
            info.get("source", "unknown"),
            info.get("uploaded_at", "unknown")[:19],
        )

    console.print(table)


@data_app.command("info")
def data_info(
    id: str = typer.Option(
        ...,
        "--id",
        help="Dataset ID to inspect.",
    ),
    metadata_file: str = typer.Option(
        "data/datasets.json",
        "--metadata-file",
        help="Path to the datasets.json registry.",
    ),
) -> None:
    """Show detailed information about a specific dataset.

    Looks up the dataset by ID in the metadata registry and prints
    all available metadata fields.

    Args:
        id: Dataset identifier.
        metadata_file: Path to datasets.json.
    """
    metadata = _load_metadata(metadata_file)

    if id not in metadata:
        console.print(f"[red]Dataset not found:[/red] {id}")
        raise typer.Exit(code=1)

    info = metadata[id]

    table = Table(title=f"Dataset: {id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Dataset ID", id)
    table.add_row("Original Name", info.get("original_name", "N/A"))
    table.add_row("Source", info.get("source", "N/A"))
    table.add_row("Uploaded At", info.get("uploaded_at", "N/A"))

    shape = info.get("shape", [0, 0])
    table.add_row("Shape", f"{shape[0]} rows x {shape[1]} cols")

    feature_names = info.get("feature_names", [])
    table.add_row("Features", ", ".join(feature_names) if feature_names else "N/A")

    time_range = info.get("time_range", {})
    table.add_row("Time Start", time_range.get("start", "N/A"))
    table.add_row("Time End", time_range.get("end", "N/A"))

    console.print(table)


@data_app.command("delete")
def data_delete(
    id: str = typer.Option(
        ...,
        "--id",
        help="Dataset ID to delete.",
    ),
    metadata_file: str = typer.Option(
        "data/datasets.json",
        "--metadata-file",
        help="Path to the datasets.json registry.",
    ),
    data_dir: str = typer.Option(
        "data/raw",
        "--data-dir",
        help="Directory containing ingested Parquet files.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-y",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Delete a dataset and its Parquet file.

    Removes the dataset entry from the metadata registry and deletes
    the corresponding Parquet file from the data directory.

    Args:
        id: Dataset identifier to delete.
        metadata_file: Path to datasets.json.
        data_dir: Directory containing Parquet files.
        force: Skip confirmation prompt.
    """
    metadata = _load_metadata(metadata_file)

    if id not in metadata:
        console.print(f"[red]Dataset not found:[/red] {id}")
        raise typer.Exit(code=1)

    info = metadata[id]
    name = info.get("original_name", "unknown")

    if not force:
        confirm = typer.confirm(f"Delete dataset '{name}' ({id})?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(code=0)

    # Remove Parquet file.
    parquet_path = Path(data_dir) / f"{id}.parquet"
    if parquet_path.exists():
        parquet_path.unlink()
        console.print(f"Deleted file: {parquet_path}")
    else:
        console.print(f"[yellow]Parquet file not found:[/yellow] {parquet_path}")

    # Remove from metadata.
    del metadata[id]
    _save_metadata(metadata_file, metadata)

    console.print(f"[green]Dataset '{name}' deleted.[/green]")
