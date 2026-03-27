"""Sentinel CLI: Typer-based command-line interface.

Entry point: ``sentinel = "sentinel.cli.app:app"``

Registers sub-commands for training, evaluation, detection, data
management, synthetic generation, configuration validation,
visualization, PI System data extraction, and API serving.
"""

from __future__ import annotations

import typer

from sentinel.cli.data import data_app
from sentinel.cli.detect import detect_app
from sentinel.cli.evaluate import evaluate_app
from sentinel.cli.generate import generate_app
from sentinel.cli.ingest import ingest_app
from sentinel.cli.pi import pi_app
from sentinel.cli.serve import serve_app
from sentinel.cli.train import train_app
from sentinel.cli.validate_config import validate_config_app
from sentinel.cli.visualize import visualize_app

app = typer.Typer(
    name="sentinel",
    help="Sentinel: Anomaly Detection Platform",
    no_args_is_help=True,
)

app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(detect_app, name="detect")
app.add_typer(ingest_app, name="ingest")
app.add_typer(generate_app, name="generate")
app.add_typer(data_app, name="data")
app.add_typer(validate_config_app, name="validate-config")
app.add_typer(visualize_app, name="visualize")
app.add_typer(pi_app, name="pi")
app.add_typer(serve_app, name="serve")

if __name__ == "__main__":
    app()
