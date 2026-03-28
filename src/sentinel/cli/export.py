"""CLI command: export a trained model from an experiment run.

Supports two export formats:

* **native** -- copies the model artifacts (joblib for statistical models,
  state_dict + config.json for PyTorch models) to the output path.
* **onnx** -- exports PyTorch deep models via ``torch.onnx.export()``.
  Statistical models do not support ONNX export.
"""

from __future__ import annotations

import json
import os
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
import typer
from rich.console import Console
from rich.table import Table

from sentinel.core.exceptions import SentinelError
from sentinel.tracking.artifacts import load_model_artifact
from sentinel.tracking.experiment import LocalTracker

logger = structlog.get_logger(__name__)
console = Console()

# Deep model names that use PyTorch state_dict serialization.
_DEEP_MODEL_NAMES = frozenset(
    {
        "autoencoder",
        "rnn",
        "lstm",
        "gru",
        "lstm_ae",
        "tcn",
        "vae",
        "gan",
        "tadgan",
        "tranad",
        "deepar",
        "diffusion",
    }
)


class ExportFormat(StrEnum):
    """Supported model export formats."""

    native = "native"
    onnx = "onnx"


export_app = typer.Typer(help="Export a trained model.")


def _get_total_size(path: Path) -> int:
    """Compute total size of a file or directory in bytes.

    Args:
        path: Path to a file or directory.

    Returns:
        Total size in bytes.
    """
    if path.is_file():
        return path.stat().st_size

    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted size string (e.g. ``"1.23 MB"``).
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _export_native(
    run_id: str,
    model_name: str,
    output_path: Path,
    experiments_dir: str,
) -> int:
    """Export model in native format by copying artifact directory.

    Args:
        run_id: Experiment run identifier.
        model_name: Registered model name.
        output_path: Destination path for the exported artifacts.
        experiments_dir: Root directory for experiment storage.

    Returns:
        Total size in bytes of the exported artifacts.

    Raises:
        FileNotFoundError: If model artifacts do not exist.
    """
    model_dir = Path(experiments_dir) / run_id / "model"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model artifacts not found for run '{run_id}': {model_dir}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic copy: write to a temporary location, then rename.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        if output_path.suffix == "" or output_path.is_dir():
            # Copy as directory.
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            shutil.copytree(model_dir, tmp_path)
        else:
            # Single file — package model dir into the target.
            if tmp_path.exists():
                shutil.rmtree(tmp_path)
            shutil.copytree(model_dir, tmp_path)

        # Atomic rename.
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        os.rename(tmp_path, output_path)
    except OSError:
        # Clean up temp on failure.
        if tmp_path.exists():
            if tmp_path.is_dir():
                shutil.rmtree(tmp_path)
            else:
                tmp_path.unlink()
        raise

    logger.info(
        "export.native",
        run_id=run_id,
        model_name=model_name,
        output=str(output_path),
    )
    return _get_total_size(output_path)


def _export_onnx(
    run_id: str,
    model_name: str,
    output_path: Path,
    experiments_dir: str,
) -> int:
    """Export a PyTorch model to ONNX format.

    Loads the model from experiment artifacts, creates a dummy input
    matching the model's expected input shape, and calls
    ``torch.onnx.export()``.

    Args:
        run_id: Experiment run identifier.
        model_name: Registered model name.
        output_path: Destination path for the ONNX file.
        experiments_dir: Root directory for experiment storage.

    Returns:
        Size in bytes of the exported ONNX file.

    Raises:
        SentinelError: If the model is not a deep model, torch is
            unavailable, or export fails.
    """
    if model_name not in _DEEP_MODEL_NAMES:
        raise SentinelError(
            f"ONNX export is only supported for PyTorch models. "
            f"Model '{model_name}' is a statistical/ensemble model. "
            f"Use --format native instead."
        )

    try:
        import torch
    except ImportError as exc:
        raise SentinelError(
            "ONNX export requires PyTorch. Install with: uv sync --group deep"
        ) from exc

    # Load the model config to determine n_features.
    model_config_path = Path(experiments_dir) / run_id / "model" / "config.json"
    if not model_config_path.exists():
        raise FileNotFoundError(
            f"Model config not found for run '{run_id}': {model_config_path}"
        )

    with open(model_config_path) as f:
        config: dict[str, Any] = json.load(f)

    n_features = config.get("n_features")
    if n_features is None:
        raise SentinelError(
            f"Cannot determine n_features from model config for run '{run_id}'."
        )

    # Load the trained model.
    model = load_model_artifact(
        run_id=run_id,
        model_name=model_name,
        base_dir=experiments_dir,
    )

    # Access the internal PyTorch module.
    pytorch_module = getattr(model, "_model", None)
    if pytorch_module is None:
        raise SentinelError(
            f"Model '{model_name}' does not expose a PyTorch module "
            f"(_model attribute). ONNX export is not supported."
        )

    # Determine input shape from model config.
    seq_len = config.get("seq_len")
    if seq_len is not None:
        # Sequence model: (batch, seq_len, n_features).
        dummy_input = torch.randn(1, seq_len, n_features)
    else:
        # Feedforward model: (batch, n_features).
        dummy_input = torch.randn(1, n_features)

    device = getattr(model, "_device", None)
    if device is not None:
        dummy_input = dummy_input.to(device)

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .onnx extension.
    if output_path.suffix != ".onnx":
        output_path = output_path.with_suffix(".onnx")

    # Atomic write: export to temp, then rename.
    tmp_path = output_path.with_suffix(".onnx.tmp")
    try:
        pytorch_module.eval()
        torch.onnx.export(
            pytorch_module,
            dummy_input,
            str(tmp_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=17,
        )
        os.rename(tmp_path, output_path)
    except Exception as exc:
        # Clean up temp on failure.
        if tmp_path.exists():
            tmp_path.unlink()
        raise SentinelError(
            f"ONNX export failed for model '{model_name}': {exc}"
        ) from exc

    logger.info(
        "export.onnx",
        run_id=run_id,
        model_name=model_name,
        output=str(output_path),
    )
    return output_path.stat().st_size


@export_app.callback(invoke_without_command=True)
def export(
    run_id: str = typer.Option(
        ...,
        "--run-id",
        "-r",
        help="Experiment run ID to export.",
    ),
    fmt: ExportFormat = typer.Option(
        ExportFormat.native,
        "--format",
        "-f",
        help="Export format: native (model artifacts) or onnx (PyTorch only).",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file or directory path.",
    ),
    experiments_dir: str = typer.Option(
        "data/experiments",
        "--experiments-dir",
        help="Root directory for experiment artifacts.",
    ),
) -> None:
    """Export a trained model from an experiment run.

    Native export copies the original model artifacts (joblib for
    statistical models, state_dict + config.json for PyTorch models).
    ONNX export converts PyTorch models to the ONNX interchange format.

    Args:
        run_id: The experiment run identifier.
        fmt: Export format choice.
        output: Destination path for exported model.
        experiments_dir: Root directory for experiment storage.
    """
    # Validate run exists and retrieve metadata.
    try:
        tracker = LocalTracker(base_dir=experiments_dir)
        run_data = tracker.get_run(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(code=1)

    model_name = run_data.get("model_name", "")
    if not model_name:
        console.print(f"[red]No model name recorded for run:[/red] {run_id}")
        raise typer.Exit(code=1)

    output_path = Path(output)

    console.print(f"[bold]Exporting model:[/bold] {model_name}")
    console.print(f"[bold]Run ID:[/bold]         {run_id}")
    console.print(f"[bold]Format:[/bold]          {fmt.value}")
    console.print()

    try:
        if fmt == ExportFormat.native:
            size_bytes = _export_native(
                run_id=run_id,
                model_name=model_name,
                output_path=output_path,
                experiments_dir=experiments_dir,
            )
        else:
            size_bytes = _export_onnx(
                run_id=run_id,
                model_name=model_name,
                output_path=output_path,
                experiments_dir=experiments_dir,
            )
    except FileNotFoundError as exc:
        console.print(f"[red]File not found:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except SentinelError as exc:
        console.print(f"[red]Export error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Print export summary.
    table = Table(title="Export Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Run ID", run_id)
    table.add_row("Model", model_name)
    table.add_row("Format", fmt.value)
    table.add_row("Output", str(output_path.resolve()))
    table.add_row("Size", _format_size(size_bytes))

    console.print(table)
    console.print("[green]Export complete.[/green]")
