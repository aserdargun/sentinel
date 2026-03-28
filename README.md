# Sentinel

Anomaly detection platform with 16 ML models, a FastAPI REST API, an MCP server backed by a local LLM, and a web dashboard.

Built with Python 3.12, Polars, PyTorch, and managed entirely with `uv`.

## Quickstart

```bash
# Install all dependency groups (requires Python 3.12 and uv)
uv sync --all-groups

# Generate synthetic data and train a model
uv run sentinel generate --features 5 --length 10000 --output data/raw/demo.parquet
uv run sentinel train --config configs/zscore.yaml --data-path data/raw/demo.parquet

# Launch the dashboard
uv run sentinel serve --host 0.0.0.0 --port 8000
# Open http://localhost:8000/ui
```

## Model Zoo

16 algorithms spanning statistical, deep learning, generative, and ensemble approaches. All models implement a shared `BaseAnomalyDetector` interface and are registered via `@register_model("name")`.

| # | Model | Category | Config | Key Idea |
|---|-------|----------|--------|----------|
| 1 | Z-Score | Statistical | `zscore.yaml` | Rolling z-score, flags points > k sigma |
| 2 | Isolation Forest | Statistical | `isolation_forest.yaml` | Random-split tree path length scoring |
| 3 | Matrix Profile | Statistical | `matrix_profile.yaml` | STOMP subsequence distance via stumpy |
| 4 | Autoencoder | Deep | `autoencoder.yaml` | Feedforward encoder-decoder, MSE reconstruction |
| 5 | RNN | Deep | `rnn.yaml` | Elman RNN sequence reconstruction |
| 6 | LSTM | Deep | `lstm.yaml` | LSTM next-step prediction, forecast error |
| 7 | GRU | Deep | `gru.yaml` | GRU next-step prediction |
| 8 | LSTM-AE | Deep | `lstm_ae.yaml` | LSTM encoder-decoder with latent bottleneck |
| 9 | TCN | Deep | `tcn.yaml` | Dilated causal convolutions |
| 10 | VAE | Generative | `vae.yaml` | Variational autoencoder, ELBO loss |
| 11 | GAN | Generative | `gan.yaml` | GANomaly encoder-decoder-encoder |
| 12 | TadGAN | Generative | `tadgan.yaml` | Wasserstein critic + cycle-consistency |
| 13 | TranAD | Generative | `tranad.yaml` | Transformer with adversarial decoders |
| 14 | DeepAR | Generative | `deepar.yaml` | Autoregressive LSTM, Gaussian NLL |
| 15 | Diffusion | Generative | `diffusion.yaml` | DDPM denoising, MSE at fixed noise level |
| 16 | Hybrid Ensemble | Ensemble | `ensemble.yaml` | Weighted average, majority voting, or stacking |

## CLI

```
sentinel generate         Generate synthetic multivariate time series
sentinel ingest           Ingest a CSV/Parquet file into the data store
sentinel data             List, inspect, or delete ingested datasets
sentinel train            Train a model from a YAML config
sentinel evaluate         Print metrics for a completed run
sentinel detect           Batch or streaming anomaly detection
sentinel visualize        Generate time series / reconstruction / latent plots
sentinel validate-config  Validate a YAML config without training
sentinel export           Export a trained model (native or ONNX)
sentinel pi               PI System data extraction (Windows only)
sentinel serve            Launch the FastAPI server + dashboard
sentinel mcp-serve        Start the MCP server (stdio or SSE transport)
```

## REST API

Runs at `http://localhost:8000`. Full OpenAPI docs available at `/docs`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (API + Ollama status) |
| POST | `/api/data/upload` | Upload CSV/Parquet dataset |
| GET | `/api/data` | List datasets (paginated) |
| GET | `/api/data/{id}/preview` | Preview first N rows |
| POST | `/api/data/pi-fetch` | Fetch time series from PI System |
| POST | `/api/train` | Submit async training job |
| GET | `/api/train/{job_id}` | Poll training status |
| POST | `/api/detect` | Batch anomaly detection |
| WS | `/api/detect/stream` | Real-time streaming detection |
| GET | `/api/models` | List registered models |
| GET | `/api/experiments` | List experiment runs (paginated) |
| POST | `/api/prompt` | Natural language query via LLM |
| GET | `/ui` | Web dashboard |

## MCP Server

Sentinel exposes an MCP server for AI assistant integration, backed by a local Ollama LLM (default: `nvidia/nemotron-3-nano-4b`).

```bash
# Start the MCP server
uv run sentinel mcp-serve --transport stdio
```

Available tools: `sentinel_train`, `sentinel_detect`, `sentinel_explain`, `sentinel_analyze`, `sentinel_recommend_model`, `sentinel_upload`, `sentinel_list_datasets`, `sentinel_compare_runs`, `sentinel_prompt`, and more.

Resources: `experiments://list`, `models://registry`, `data://datasets`, `data://datasets/{id}`.

## Configuration

YAML configs with single-level inheritance. Precedence: `base.yaml` < `model.yaml` < CLI overrides.

```yaml
# configs/lstm.yaml
inherits: base.yaml
model: lstm
hidden_dim: 64
num_layers: 2
seq_len: 50
learning_rate: 0.001
epochs: 100
batch_size: 32
dropout: 0.1
```

`configs/base.yaml` sets shared defaults: split ratios (70/15/15), seed, device, runtime limits, LLM settings.

## Project Structure

```
src/sentinel/
  core/         Base model ABC, registry, config, types, exceptions, device
  data/         Loaders, validators, preprocessors, synthetic generator, PI connector
  models/       statistical/ deep/ ensemble/ -- all 16 models
  training/     Trainer, evaluator, thresholds, callbacks, schedulers
  tracking/     JSON-backed experiment tracking, artifact management
  viz/          Matplotlib plots (time series, reconstruction, latent)
  explain/      SHAP explainer, reconstruction error decomposition
  streaming/    Online detector, stream simulator, drift detection, alerts
  api/          FastAPI app, routes, schemas, dashboard (HTML/CSS/JS)
  mcp/          FastMCP server, tools, resources, Ollama LLM client
  cli/          Typer CLI entry points
  plugins/      Entry-point plugin discovery
configs/        YAML configuration files
tests/          unit/ integration/ smoke/
```

## Development

```bash
uv run pytest tests/ -v              # Run tests
uv run ruff check src/ tests/        # Lint
uv run ruff format src/ tests/       # Format
uv run mypy src/sentinel/            # Type check
uv run pytest tests/ --cov=sentinel  # Coverage
```

## Dependency Groups

| Group | Packages | Purpose |
|-------|----------|---------|
| core | polars, numpy, scikit-learn, scipy, stumpy, pydantic, typer, rich, matplotlib, structlog, joblib | Always installed |
| `deep` | torch, einops | PyTorch models |
| `explain` | shap, umap-learn | Interpretability |
| `pi` | pipolars | PI System connector (Windows only) |
| `api` | fastapi, uvicorn, httpx, jinja2, python-multipart | REST API + dashboard |
| `mcp` | fastmcp, httpx | MCP server |
| `dev` | pytest, ruff, mypy, pytest-cov, pytest-asyncio | Development tools |

## Design Decisions

- **Polars only** -- no pandas anywhere. DataFrames convert to numpy at model boundary via `.to_numpy()`.
- **Plugin registry** -- models registered via `@register_model("name")`, looked up by string. Never imported directly.
- **PyTorch isolation** -- only `models/deep/` imports torch. Deep models skip registration if torch is absent.
- **Chronological splits** -- no shuffling. Train 70%, validation 15%, test 15%.
- **Threshold on validation only** -- thresholds computed on validation set, metrics reported on test set.
- **Atomic writes** -- all file writes go to `.tmp` then `os.rename()`.
- **Process-based training** -- `ProcessPoolExecutor` for training jobs, no threads.
- **Config inheritance** -- single-level deep merge with CLI overrides.
- **Graceful degradation** -- optional deps (torch, pipolars, Ollama) fail gracefully when unavailable.
