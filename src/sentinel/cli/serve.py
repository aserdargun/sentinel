"""CLI command: launch the Sentinel FastAPI server with uvicorn."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()

serve_app = typer.Typer(help="Launch the Sentinel API server.")


@serve_app.callback(invoke_without_command=True)
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Bind address.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Bind port.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of worker processes.",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Uvicorn log level.",
    ),
) -> None:
    """Start the Sentinel FastAPI server.

    Launches the API server using uvicorn.  The dashboard is served
    at ``/ui`` and the API at ``/api``.

    Args:
        host: Network interface to bind to.
        port: TCP port to listen on.
        reload: Enable hot-reload (development only).
        workers: Number of uvicorn worker processes.
        log_level: Logging verbosity for uvicorn.
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Error:[/red] uvicorn is not installed. "
            "Install with: uv add --group api uvicorn[standard]"
        )
        raise typer.Exit(code=1)

    console.print(f"Starting Sentinel API server at [bold]http://{host}:{port}[/bold]")
    console.print(f"  Dashboard: [cyan]http://{host}:{port}/ui[/cyan]")
    console.print(f"  API docs:  [cyan]http://{host}:{port}/docs[/cyan]")
    console.print(f"  Health:    [cyan]http://{host}:{port}/health[/cyan]")
    console.print()

    uvicorn.run(
        "sentinel.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        factory=True,
    )
