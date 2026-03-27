"""CLI commands for PI System data extraction.

Provides ``sentinel pi search``, ``sentinel pi fetch``, and
``sentinel pi snapshot`` sub-commands.  All commands include a
platform check and exit with a clear error on non-Windows systems
or when pipolars is not installed.
"""

from __future__ import annotations

import platform

import typer
from rich.console import Console
from rich.table import Table

console = Console()

pi_app = typer.Typer(
    help="PI System data extraction (Windows only).",
)


def _check_platform() -> None:
    """Exit with error if not running on Windows or pipolars is missing."""
    if platform.system() != "Windows":
        console.print(
            "[red]Error:[/red] PI System connector requires Windows "
            f"with PI AF SDK (.NET 4.8). Current platform: "
            f"{platform.system()}"
        )
        raise typer.Exit(code=1)

    try:
        from sentinel.data.pi_connector import is_pi_available

        if not is_pi_available():
            console.print(
                "[red]Error:[/red] pipolars is not installed. "
                "Install with: uv add --group pi pipolars"
            )
            raise typer.Exit(code=1)
    except ImportError:
        console.print(
            "[red]Error:[/red] pipolars is not installed. "
            "Install with: uv add --group pi pipolars"
        )
        raise typer.Exit(code=1)


@pi_app.command("search")
def pi_search(
    server: str = typer.Option(
        ...,
        "--server",
        "-s",
        help="PI Data Archive server hostname.",
    ),
    pattern: str = typer.Option(
        ...,
        "--pattern",
        "-p",
        help="Glob-style pattern for tag name matching (e.g., 'Pump*').",
    ),
    port: int = typer.Option(
        5450,
        "--port",
        help="PI server port.",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        help="Connection timeout in seconds.",
    ),
) -> None:
    """Search PI points by name pattern.

    Connects to the specified PI Data Archive server and searches
    for tags matching the given pattern.

    Args:
        server: PI server hostname.
        pattern: Tag name pattern.
        port: PI server port.
        timeout: Connection timeout.
    """
    _check_platform()

    from sentinel.data.pi_connector import PIConnectionError, PIConnector

    console.print(
        f"Searching tags on [bold]{server}[/bold] "
        f"with pattern [cyan]{pattern}[/cyan]..."
    )

    try:
        connector = PIConnector(host=server, port=port, timeout=timeout)
        tags = connector.search_tags(pattern)
        connector.close()
    except PIConnectionError as exc:
        console.print(f"[red]Connection error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not tags:
        console.print("[yellow]No matching tags found.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title=f"PI Tags ({len(tags)} found)")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("UOM", style="magenta")

    for tag in tags:
        table.add_row(
            tag["name"],
            tag["description"],
            tag["uom"],
        )

    console.print(table)


@pi_app.command("fetch")
def pi_fetch(
    server: str = typer.Option(
        ...,
        "--server",
        "-s",
        help="PI Data Archive server hostname.",
    ),
    tags: str = typer.Option(
        ...,
        "--tags",
        "-t",
        help="Comma-separated list of PI point names.",
    ),
    start: str = typer.Option(
        "*-7d",
        "--start",
        help="Start time in PI time syntax (e.g., '*-7d').",
    ),
    end: str = typer.Option(
        "*",
        "--end",
        help="End time in PI time syntax (e.g., '*').",
    ),
    interval: str = typer.Option(
        "1m",
        "--interval",
        "-i",
        help="Interpolation interval (e.g., '1m', '5m').",
    ),
    port: int = typer.Option(
        5450,
        "--port",
        help="PI server port.",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        help="Connection timeout in seconds.",
    ),
) -> None:
    """Fetch multi-tag timeseries from PI System.

    Connects to the PI server, extracts interpolated timeseries for
    the specified tags, validates the data, and stores it as Parquet
    in the Sentinel data store.

    Args:
        server: PI server hostname.
        tags: Comma-separated tag names.
        start: Start time in PI syntax.
        end: End time in PI syntax.
        interval: Interpolation interval.
        port: PI server port.
        timeout: Connection timeout.
    """
    _check_platform()

    from sentinel.data.ingest import _load_metadata, _save_metadata
    from sentinel.data.pi_connector import PIConnectionError, PIConnector
    from sentinel.data.validators import validate_dataframe

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if not tag_list:
        console.print("[red]Error:[/red] At least one tag must be specified.")
        raise typer.Exit(code=1)

    console.print(
        f"Fetching [bold]{len(tag_list)}[/bold] tags from [bold]{server}[/bold]..."
    )
    console.print(f"  Tags: [cyan]{', '.join(tag_list)}[/cyan]")
    console.print(f"  Range: {start} to {end}, interval={interval}")

    try:
        connector = PIConnector(host=server, port=port, timeout=timeout)
        df = connector.fetch_tags(
            tags=tag_list,
            start=start,
            end=end,
            interval=interval,
        )
        connector.close()
    except PIConnectionError as exc:
        console.print(f"[red]Connection error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"Received {df.height} rows, validating...")

    try:
        df = validate_dataframe(df)
    except Exception as exc:
        console.print(f"[red]Validation error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Store using the ingest pipeline pattern.
    import os
    import uuid
    from datetime import UTC, datetime
    from pathlib import Path

    data_dir = Path("data/raw")
    metadata_file = Path("data/datasets.json")
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = str(uuid.uuid4())
    feature_cols = [c for c in df.columns if c != "timestamp"]

    ts_col = df.get_column("timestamp")
    time_range = {
        "start": str(ts_col.min()),
        "end": str(ts_col.max()),
    }

    out_path = data_dir / f"{dataset_id}.parquet"
    tmp_path = data_dir / f"{dataset_id}.parquet.tmp"
    try:
        df.write_parquet(str(tmp_path))
        os.rename(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    metadata = _load_metadata(metadata_file)
    metadata[dataset_id] = {
        "original_name": f"pi_{server}_{'_'.join(tag_list[:3])}",
        "source": "pi",
        "uploaded_at": datetime.now(UTC).isoformat(),
        "shape": [df.height, df.width],
        "feature_names": feature_cols,
        "time_range": time_range,
    }
    _save_metadata(metadata_file, metadata)

    console.print("[green]PI data ingested successfully.[/green]")
    console.print()

    table = Table(title="Dataset Info")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Dataset ID", dataset_id)
    table.add_row("Shape", f"{df.height} rows x {df.width} cols")
    table.add_row("Features", ", ".join(feature_cols))
    table.add_row("Start", time_range.get("start", "N/A"))
    table.add_row("End", time_range.get("end", "N/A"))

    console.print(table)


@pi_app.command("snapshot")
def pi_snapshot(
    server: str = typer.Option(
        ...,
        "--server",
        "-s",
        help="PI Data Archive server hostname.",
    ),
    tags: str = typer.Option(
        ...,
        "--tags",
        "-t",
        help="Comma-separated list of PI point names.",
    ),
    port: int = typer.Option(
        5450,
        "--port",
        help="PI server port.",
    ),
    timeout: int = typer.Option(
        30,
        "--timeout",
        help="Connection timeout in seconds.",
    ),
) -> None:
    """Get current snapshot values for PI tags.

    Retrieves the most recent recorded value for each specified tag
    from the PI server.

    Args:
        server: PI server hostname.
        tags: Comma-separated tag names.
        port: PI server port.
        timeout: Connection timeout.
    """
    _check_platform()

    from sentinel.data.pi_connector import PIConnectionError, PIConnector

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if not tag_list:
        console.print("[red]Error:[/red] At least one tag must be specified.")
        raise typer.Exit(code=1)

    console.print(
        f"Fetching snapshots for [bold]{len(tag_list)}[/bold] tags "
        f"from [bold]{server}[/bold]..."
    )

    try:
        connector = PIConnector(host=server, port=port, timeout=timeout)
        snapshots = connector.snapshot(tag_list)
        connector.close()
    except PIConnectionError as exc:
        console.print(f"[red]Connection error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not snapshots:
        console.print("[yellow]No snapshot values returned.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title="PI Tag Snapshots")
    table.add_column("Name", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Timestamp", style="green")
    table.add_column("Quality", style="white")

    for snap in snapshots:
        table.add_row(
            snap["name"],
            str(snap.get("value", "N/A")),
            snap.get("timestamp", "N/A"),
            snap.get("quality", "unknown"),
        )

    console.print(table)
