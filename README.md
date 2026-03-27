# Sentinel

Anomaly detection platform with 16 ML models, a FastAPI REST API, an MCP server backed by a local LLM, and a web dashboard.

## Quickstart

```bash
# Install (requires Python 3.12 and uv)
uv sync --all-groups

# Generate synthetic data
uv run sentinel generate --features 5 --length 10000 --output data/raw/demo.parquet

# Train a model
uv run sentinel train --config configs/zscore.yaml --data-path data/raw/demo.parquet

# Validate a config without training
uv run sentinel validate-config --config configs/lstm.yaml

# Launch the API server + dashboard
uv run sentinel serve --host 0.0.0.0 --port 8000
# Open http://localhost:8000/ui

# Start the MCP server (for AI assistant integration)
uv run sentinel mcp-serve --transport stdio
```

## Model Zoo (16 algorithms)

### Statistical
| Model | Config | Description |
|-------|--------|-------------|
| Z-Score | `zscore.yaml` | Rolling z-score, flags points > k sigma |
| Isolation Forest | `isolation_forest.yaml` | Random-split tree path length scoring |
| Matrix Profile | `matrix_profile.yaml` | STOMP subsequence distance via stumpy |

### Deep Learning
| Model | Config | Description |
|-------|--------|-------------|
| Autoencoder | `autoencoder.yaml` | Feedforward encoder-decoder, MSE reconstruction |
| RNN | `rnn.yaml` | Elman RNN sequence reconstruction |
| LSTM | `lstm.yaml` | LSTM next-step prediction, forecast error |
| GRU | `gru.yaml` | GRU next-step prediction |
| LSTM-AE | `lstm_ae.yaml` | LSTM encoder-decoder with latent bottleneck |
| TCN | `tcn.yaml` | Temporal convolutional network, dilated causal convolutions |
| VAE | `vae.yaml` | Variational autoencoder, ELBO loss |
| GAN | `gan.yaml` | GANomaly encoder-decoder-encoder |
| TadGAN | `tadgan.yaml` | Wasserstein critic + cycle-consistency |
| TranAD | `tranad.yaml` | Transformer with adversarial decoders |
| DeepAR | `deepar.yaml` | Autoregressive LSTM, Gaussian NLL |
| Diffusion | `diffusion.yaml` | DDPM denoising, MSE at fixed noise level |

### Ensemble
| Model | Config | Description |
|-------|--------|-------------|
| Hybrid Ensemble | `ensemble.yaml` | Weighted average, majority voting, or stacking |

## CLI Commands

```
sentinel train            Train a model from a YAML config
sentinel evaluate         Print metrics for a completed run
sentinel detect           Batch or streaming anomaly detection
sentinel ingest           Ingest a CSV/Parquet file into the data store
sentinel generate         Generate synthetic multivariate time series
sentinel data             List, inspect, or delete ingested datasets
sentinel validate-config  Validate a YAML config without training
sentinel visualize        Generate time series / reconstruction / latent plots
sentinel pi               PI System data extraction (Windows only)
sentinel serve            Launch the FastAPI server + dashboard
sentinel mcp-serve        Start the MCP server (stdio or SSE transport)
```

## API

The REST API runs at `http://localhost:8000` with these key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (API + Ollama status) |
| POST | `/api/data/upload` | Upload CSV/Parquet dataset |
| GET | `/api/data` | List datasets (paginated) |
| GET | `/api/data/{id}/preview` | Preview first N rows |
| POST | `/api/train` | Submit async training job |
| GET | `/api/train/{job_id}` | Poll training status |
| POST | `/api/detect` | Batch anomaly detection |
| GET | `/api/models` | List registered models |
| GET | `/api/experiments` | List experiment runs |
| POST | `/api/prompt` | Natural language query via LLM |
| GET | `/ui` | Web dashboard |

Full OpenAPI docs at `/docs` when the server is running.

## Configuration

Configs use YAML with single-level inheritance:

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

Base config (`configs/base.yaml`) sets shared defaults: split ratios, seed, device, runtime limits, LLM settings.

## Project Structure

```
src/sentinel/
  core/         Base model ABC, registry, config, types, exceptions
  data/         Loaders, validators, preprocessors, synthetic generator, PI connector
  models/       statistical/ deep/ ensemble/ — all 16 models
  training/     Trainer, evaluator, thresholds, callbacks, schedulers
  tracking/     JSON-backed experiment tracking, artifact management
  viz/          Matplotlib plots (time series, reconstruction, latent)
  explain/      SHAP explainer, reconstruction error decomposition
  streaming/    Online detector, stream simulator, drift detection, alerts
  api/          FastAPI app, routes, schemas, dashboard (HTML/CSS/JS)
  mcp/          FastMCP server, tools, resources, Ollama LLM client
  cli/          Typer CLI commands
  plugins/      Entry-point plugin discovery
configs/        YAML configuration files
tests/          unit/ integration/ smoke/
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/sentinel/
```

## Dependencies

Managed with `uv`. Dependency groups:

| Group | Packages | Purpose |
|-------|----------|---------|
| (core) | polars, numpy, scikit-learn, scipy, stumpy, pydantic, typer, rich, matplotlib, structlog, joblib | Always installed |
| `deep` | torch, einops | PyTorch models |
| `explain` | shap, umap-learn | Interpretability |
| `api` | fastapi, uvicorn, httpx, jinja2 | REST API |
| `mcp` | fastmcp, httpx | MCP server |
| `dev` | pytest, ruff, mypy, pytest-cov, pytest-asyncio | Development tools |

```bash
# Install everything
uv sync --all-groups
```

## Key Design Decisions

- **Polars everywhere** — no pandas. DataFrames convert to numpy at model boundary.
- **Plugin registry** — models registered via `@register_model("name")`, looked up by string.
- **PyTorch isolation** — only `models/deep/` imports torch; deep models skip registration if torch is absent.
- **Chronological splits** — no shuffling. Train 70%, validation 15%, test 15%.
- **Threshold on validation only** — never compute thresholds on test data.
- **Atomic writes** — all file writes go to `.tmp` then `os.rename()`.
- **Config inheritance** — `base.yaml` < `model.yaml` < CLI overrides.
