"""CLI command: validate a YAML config without training."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sentinel.core.config import RunConfig
from sentinel.core.exceptions import ConfigError
from sentinel.core.registry import get_model_class

console = Console()

validate_config_app = typer.Typer(help="Validate configuration files.")


@validate_config_app.callback(invoke_without_command=True)
def validate_config(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the YAML config file to validate.",
    ),
) -> None:
    """Validate a YAML configuration file without training.

    Checks that the file exists, parses as valid YAML, resolves
    inheritance, and verifies the model name is registered.

    Args:
        config: Path to the YAML config file.
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

    # Check model is registered.
    errors: list[str] = []
    warnings: list[str] = []

    if not run_config.model:
        errors.append("'model' field is missing or empty.")
    else:
        # Import models to trigger registration before lookup.
        try:
            import sentinel.models  # noqa: F401

            get_model_class(run_config.model)
        except Exception as exc:
            errors.append(f"Model '{run_config.model}' is not registered: {exc}")

    # Validate split ratios.
    total = run_config.split.train + run_config.split.val + run_config.split.test
    if abs(total - 1.0) > 0.01:
        errors.append(
            f"Split ratios sum to {total:.2f}, expected ~1.0 "
            f"(train={run_config.split.train}, val={run_config.split.val}, "
            f"test={run_config.split.test})."
        )

    # Validate seed.
    if run_config.seed < 0:
        warnings.append(f"Seed is negative: {run_config.seed}")

    # Validate training mode.
    valid_modes = {"normal_only", "all_data"}
    if run_config.training_mode not in valid_modes:
        errors.append(
            f"Invalid training_mode: '{run_config.training_mode}'. "
            f"Must be one of: {valid_modes}"
        )

    # Report results.
    if errors:
        console.print("[red]Validation failed:[/red]")
        for err in errors:
            console.print(f"  [red]x[/red] {err}")
        if warnings:
            for warn in warnings:
                console.print(f"  [yellow]![/yellow] {warn}")
        raise typer.Exit(code=1)

    console.print("[green]Config is valid.[/green]")
    console.print()

    # Show parsed config summary.
    table = Table(title=f"Config: {config_path.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Model", run_config.model)
    table.add_row("Data Path", run_config.data_path)
    table.add_row("Seed", str(run_config.seed))
    table.add_row("Device", run_config.device)
    table.add_row("Training Mode", run_config.training_mode)
    table.add_row(
        "Split",
        f"train={run_config.split.train}, "
        f"val={run_config.split.val}, "
        f"test={run_config.split.test}",
    )

    if run_config.extra:
        for key, value in sorted(run_config.extra.items()):
            table.add_row(f"  {key}", str(value))

    console.print(table)

    if warnings:
        console.print()
        for warn in warnings:
            console.print(f"[yellow]Warning:[/yellow] {warn}")
