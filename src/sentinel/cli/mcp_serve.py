"""CLI command: ``sentinel mcp-serve``.

Starts the Sentinel MCP server with configurable transport (stdio or
SSE) and optional host/port for SSE mode.
"""

from __future__ import annotations

import structlog
import typer
from rich.console import Console

console = Console()
logger = structlog.get_logger(__name__)

mcp_serve_app = typer.Typer(
    help="Start the Sentinel MCP server.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@mcp_serve_app.callback(invoke_without_command=True)
def mcp_serve(
    transport: str = typer.Option(
        "stdio",
        help="Transport protocol: 'stdio' or 'sse'.",
    ),
    host: str = typer.Option(
        "localhost",
        help="Host to bind SSE server to (only for 'sse' transport).",
    ),
    port: int = typer.Option(
        3000,
        help="Port for SSE server (only for 'sse' transport).",
    ),
    ollama_url: str = typer.Option(
        "http://localhost:11434",
        help="Ollama server URL.",
    ),
    ollama_model: str = typer.Option(
        "nemotron-3-nano:4b",
        help="Ollama model name for LLM features.",
    ),
) -> None:
    """Start the Sentinel MCP server.

    The MCP server exposes Sentinel tools (train, detect, upload, etc.)
    and resources (experiments, models, datasets) to AI assistants.

    Examples::

        # stdio transport (default, for Claude Code)
        sentinel mcp-serve

        # SSE transport with custom port
        sentinel mcp-serve --transport sse --host 0.0.0.0 --port 3000
    """
    if transport not in ("stdio", "sse"):
        console.print(
            f"[red]Invalid transport '{transport}'. Use 'stdio' or 'sse'.[/red]"
        )
        raise typer.Exit(code=1)

    try:
        from sentinel.mcp.server import create_mcp_server
    except ImportError as exc:
        console.print(
            f"[red]MCP dependencies not available: {exc}[/red]\n"
            "Install with: uv sync --group mcp"
        )
        raise typer.Exit(code=1) from exc

    server = create_mcp_server(
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )

    console.print(f"[green]Starting MCP server (transport={transport})[/green]")
    logger.info(
        "mcp_serve.start",
        transport=transport,
        host=host,
        port=port,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )

    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="sse", host=host, port=port)
