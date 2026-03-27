"""Sentinel CLI: Typer-based command-line interface.

Entry point: ``sentinel = "sentinel.cli.app:app"``

Registers sub-commands for training, evaluation, data management,
synthetic generation, and configuration validation.
"""

from __future__ import annotations

import typer

from sentinel.cli.data import data_app
from sentinel.cli.evaluate import evaluate_app
from sentinel.cli.generate import generate_app
from sentinel.cli.ingest import ingest_app
from sentinel.cli.train import train_app
from sentinel.cli.validate_config import validate_config_app

app = typer.Typer(
    name="sentinel",
    help="Sentinel: Anomaly Detection Platform",
    no_args_is_help=True,
)

app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(ingest_app, name="ingest")
app.add_typer(generate_app, name="generate")
app.add_typer(data_app, name="data")
app.add_typer(validate_config_app, name="validate-config")

if __name__ == "__main__":
    app()
