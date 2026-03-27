# Sentinel: Anomaly Detection Platform - Implementation Plan

## Context

Build a production-grade anomaly detection platform from scratch. The system supports 16 detection algorithms spanning statistical, matrix-profile, classical deep learning, and modern generative approaches. Managed with `uv` on Python 3.12. Uses **Polars** as the sole DataFrame library — no pandas. Designed around a plugin registry so new models slot in without touching existing code. Supports **live data extraction from OSIsoft PI System** via **pipolars**. Exposes a **FastAPI** REST/WebSocket API with a web dashboard, and a **custom MCP server** backed by a local **Ollama** LLM (configurable, default `nvidia/nemotron-3-nano-4b`) for AI-powered anomaly analysis.

**Platform note:** The core platform runs on macOS, Linux, and Windows. The PI System connector (`pipolars`) requires **Windows only** (PI AF SDK = .NET 4.8). On non-Windows platforms, PI commands/routes report that the PI dependency group is unavailable.

---

## Toolchain

| Tool | Version | Purpose |
| ---- | ------- | ------- |
| uv | 0.9.21 | Project manager, venv, dependency resolution, script runner |
| Python | 3.12 | Runtime — chosen for maximum dependency compatibility (stumpy, shap, torch all have stable wheels) |
| Polars | ~1.x (pin in pyproject.toml) | DataFrame library for all data manipulation (replaces pandas) |
| FastAPI | latest | REST/WebSocket API + static dashboard serving |
| Ollama | 0.18.2+ | Local LLM server (configurable model, default nvidia/nemotron-3-nano-4b) |
| PIPolars | latest | PI System data connector — extract timeseries from OSIsoft/AVEVA PI via PI AF SDK |
| FastMCP | latest | MCP server framework for exposing Sentinel tools + resources |

All commands use `uv` — no raw `pip` or `python -m venv`.

```bash
# Project init (Phase 1, first command)
uv init --python 3.12
uv add numpy "polars>=1.0,<2.0" scikit-learn scipy pydantic pyyaml typer rich matplotlib structlog stumpy joblib
uv add --group pi pipolars
uv add --group deep torch einops
uv add --group explain shap umap-learn
uv add --group api fastapi uvicorn[standard] httpx jinja2 python-multipart
uv add --group mcp fastmcp httpx
uv add --group dev pytest pytest-cov pytest-asyncio ruff mypy httpx

# Install everything (no "all" group needed — uv handles it):
uv sync --all-groups
```

---

## Data Format

All datasets follow a single canonical schema: **multivariate time series** where the first column is a `timestamp` index and all remaining columns are numeric feature channels.

```text
timestamp,            cpu_usage, memory_mb, disk_io, network_bytes
2024-01-01T00:00:00,  45.2,      2048,      120,     98432
2024-01-01T00:01:00,  47.8,      2051,      135,     102301
...
```

**Rules enforced by `data/validators.py` (native Polars expressions, no patito):**
- First column must be `timestamp` — parsed as `pl.Datetime` via `pl.col("timestamp").str.to_datetime()`
- All timestamps must be UTC — timezone-aware `pl.Datetime("us", "UTC")` or timezone-naive (assumed UTC). Datetime precision (us/ms/ns) is coerced to microseconds on ingest.
- Validator rejects mixed timezone-aware and timezone-naive columns within a single dataset
- All other columns must be numeric (`pl.Float64` or `pl.Int64`) — these are the feature channels
- No duplicate timestamps — validator **rejects** with duplicate count + first 5 offending indices (no silent deduplication)
- Sorted chronologically (enforced via `.is_sorted()`)
- **Minimum 2 rows** required (single-row datasets cannot form windows or sequences)
- **No all-NaN feature columns** — columns where every value is null/NaN are rejected. Sparse NaN values are allowed but must be handled by preprocessors (forward-fill then zero-fill).
- **No constant feature columns** — columns where `std == 0` are rejected (causes division-by-zero in normalization/Z-Score)
- Feature count is dynamic — models adapt to `n_features` at fit time
- **`is_anomaly` is a reserved column name** — if present, it is **separated** from the feature matrix before model input and stored as ground-truth labels for evaluation. It is NOT treated as a feature channel. The validator recognizes it by name and excludes it from the feature count.
- **Model-specific minimum row requirements:** validated at training time — e.g., sequence models require `len(data) >= seq_len * 2`, Matrix Profile requires `len(data) >= subsequence_length * 3`

**Timezone handling detail:** CSV uploads without timezone info are **assumed UTC** (passive — no conversion). PI connector **actively normalizes** to UTC via pipolars timezone config. These are functionally equivalent for downstream processing, but the mechanism differs. All stored Parquet files use `pl.Datetime("us", "UTC")`.

**Train/val/test splitting:** Splits are always **chronological** (no shuffle). The first 70% of rows form the training set, the next 15% validation, the final 15% test. This preserves temporal ordering and prevents data leakage. Implemented via positional `.slice()` on sorted data.

**Data ingestion paths:**
- **CLI:** `sentinel ingest --file data.csv` — validates, caches as Parquet in `data/raw/`
- **Web app:** POST `/api/data/upload` — accepts CSV/Parquet file upload, validates, stores, returns dataset_id
- **PI System:** `sentinel pi fetch --server <host> --tags TAG1 TAG2 TAG3 --start "*-7d" --end "*" --interval 1m` — connects to PI via pipolars, extracts multi-tag interpolated timeseries, pivots tags into feature columns, validates, stores as Parquet. Also available via `POST /api/data/pi-fetch` and `/ui/pi.html`
- **Synthetic:** `sentinel generate --features 5 --length 10000 --anomaly-ratio 0.05 --seed 42 --output data/raw/synthetic.parquet` — generates N-feature synthetic data with configurable anomaly injection (point, contextual, collective types). Deterministic via `--seed`. Saves to `data/raw/` by default.

**Internal flow:** All ingestion paths converge on the same pipeline: validate schema -> assign `dataset_id` (UUID) -> store as `data/raw/{dataset_id}.parquet` -> return metadata (id, shape, feature names, time range, source). PI System data goes through the same validation — pipolars returns Polars DataFrames natively, so no conversion is needed. If Parquet write fails mid-ingest, the partial file is cleaned up and the dataset_id is not registered (atomic: write to temp file, then rename).

---

## Data Flow

```text
CSV/Parquet Upload ──┐
PI System Fetch ─────┤──> Validate ──> Parquet Cache ──> Preprocess ──> Model.fit() / .score()
Synthetic Generate ──┘                                                        │
                                                                              v
                                                              Scores ──> Threshold ──> Labels
                                                                              │
                                                                              v
                                                              Track ──> Visualize ──> Explain
```

---

## Model Zoo (16 algorithms)

### Statistical / Distance-based

| # | Model | Module | Key Idea |
| - | ----- | ------ | -------- |
| 1 | Z-Score | `statistical/zscore.py` | Flag points > k std devs from rolling mean |
| 2 | Isolation Forest | `statistical/isolation_forest.py` | Random-split tree path length as anomaly score |
| 3 | Matrix Profile | `statistical/matrix_profile.py` | Subsequence distance profile via STOMP (stumpy) |

### Classical Deep Learning (reconstruction-error based)

| # | Model | Module | Key Idea |
| - | ----- | ------ | -------- |
| 4 | Autoencoder | `deep/autoencoder.py` | Vanilla feedforward encoder-decoder, reconstruction MSE |
| 5 | RNN | `deep/rnn.py` | Elman RNN sequence-to-sequence reconstruction |
| 6 | LSTM | `deep/lstm.py` | LSTM-based sequence prediction, forecast error as score |
| 7 | GRU | `deep/gru.py` | GRU-based sequence prediction, lighter than LSTM |
| 8 | LSTM Autoencoder | `deep/lstm_ae.py` | LSTM encoder compresses sequence, LSTM decoder reconstructs |
| 9 | TCN | `deep/tcn.py` | Temporal Convolutional Network, dilated causal convolutions |

### Generative / Advanced Deep Learning

| # | Model | Module | Key Idea |
| - | ----- | ------ | -------- |
| 10 | VAE | `deep/vae.py` | Variational Autoencoder, ELBO loss, latent sampling |
| 11 | GAN | `deep/gan.py` | GANomaly-style: encoder-decoder-encoder, anomaly = reconstruction error + latent-space distance (not raw discriminator score) |
| 12 | TadGAN | `deep/tadgan.py` | Encoder-generator-critic architecture with cycle-consistency loss. Anomaly score = weighted sum of critic score + reconstruction error. Config: `cycle_weight`, `critic_iterations` |
| 13 | TranAD | `deep/tranad.py` | Transformer encoder-decoder, single-phase self-conditioned adversarial training (no labels needed). Two decoders compete: one reconstructs, one adversarially perturbs. Score = reconstruction error from the focus decoder |
| 14 | DeepAR | `deep/deepar.py` | Autoregressive RNN, outputs Gaussian N(μ,σ) per step per feature. Anomaly score = sum of per-feature NLL over a sliding window. Single-step rolling forecast at inference. |
| 15 | Diffusion | `deep/diffusion.py` | DDPM forward/reverse process. Anomaly score = MSE between original input and denoised reconstruction at a fixed noise level t (not full likelihood — too expensive). Trained on normal data only. |

### Ensemble

| # | Model | Module | Key Idea |
| - | ----- | ------ | -------- |
| 16 | Hybrid Ensemble | `ensemble/hybrid.py` | Weighted combination of statistical + deep scores. All sub-model scores are **min-max normalized to [0,1]** before combination. Strategies: weighted average, majority voting, stacking (logistic regression meta-learner). If a sub-model fails mid-score, it is excluded and weights are renormalized. |

### Model Loss Functions & References

Exact loss formulations for generative/advanced models to prevent implementation ambiguity.

**VAE** (Kingma & Welling, 2014):
```
L = MSE(x, x̂) + β * KL(q(z|x) || p(z))
  where q(z|x) = N(μ_enc, σ_enc²), p(z) = N(0, I)
  KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
Anomaly score = MSE(x, x̂)  (reconstruction error only, no KL term)
```

**GAN / GANomaly** (Akcay et al., 2018):
```
Architecture: Encoder₁ → Decoder → Encoder₂ (no separate discriminator)
L = λ_recon * MSE(x, G(x)) + λ_latent * MSE(z₁, E₂(G(x)))
  where z₁ = E₁(x), G(x) = D(z₁), z₂ = E₂(G(x))
  Defaults: λ_recon = 1.0, λ_latent = 1.0
Anomaly score = MSE(x, G(x)) + MSE(z₁, z₂)
```

**TadGAN** (Geiger et al., 2020):
```
Architecture: Encoder E, Generator G, Critic C (Wasserstein)
L_G = MSE(x, G(E(x))) + λ_cycle * MSE(E(x), E(G(E(x))))
L_C = E[C(x)] - E[C(G(E(x)))] + λ_gp * gradient_penalty
  Defaults: λ_cycle = 10.0, λ_gp = 10.0, critic_iterations = 5
Anomaly score = α * (1 - critic_score_normalized) + (1-α) * reconstruction_error
  Default: α = 0.5
```

**TranAD** (Tuli et al., 2022):
```
Architecture: Transformer Encoder E, two Decoders D₁ D₂
Phase (single-phase, self-conditioned adversarial):
  O₁ = D₁(E(x))                    # focus decoder
  O₂ = D₂(E(x), O₁)               # adversarial decoder (conditioned on O₁)
  L = MSE(x, O₁) - λ_adv * MSE(x, O₂)  # D₁ minimizes, D₂ maximizes
  Default: λ_adv = 1.0
Anomaly score = MSE(x, O₁)
```

**DeepAR** (Salinas et al., 2020):
```
Architecture: Autoregressive LSTM, outputs μ_t, σ_t per step per feature
L = -Σ_t Σ_f log N(x_{t,f} | μ_{t,f}, σ_{t,f}²)
  where σ = softplus(linear(h_t)) to ensure positivity
Anomaly score = Σ_f -log N(x_t | μ_t, σ_t²)  (per-step NLL, summed over features)
Inference: single-step rolling forecast (no sampling)
```

**Diffusion / DDPM** (Ho et al., 2020):
```
Forward:  q(x_t | x_0) = N(√ᾱ_t * x_0, (1 - ᾱ_t) * I)
  where ᾱ_t = Π_{s=1}^{t} (1 - β_s), β linear schedule [β_start=1e-4, β_end=0.02]
Reverse:  ε_θ(x_t, t) predicts noise added at step t
L = MSE(ε, ε_θ(x_t, t))  (simplified denoising objective)
Anomaly score = MSE(x_0, x̂_0)  where x̂_0 = denoise(x_t, t) at fixed t
  Default: t = T//2 (mid-schedule — balances noise level vs reconstruction fidelity)
Trained on normal data only.
```

---

## Project Structure

Every line is commented — no orphan entries.

```text
sentinel/                               # Project root
├── pyproject.toml                      # uv-managed: deps, build config, entry points, tool settings
├── uv.lock                             # uv lock file — pinned dependency resolution
├── README.md                           # Project overview, quickstart, architecture summary
├── Makefile                            # Dev shortcuts: lint, test, train, serve, format, typecheck
├── .gitignore                          # Ignore data/, .venv/, __pycache__/, *.egg-info, experiments/
├── .python-version                     # Pins python 3.12 for uv
│
├── configs/                            # YAML configuration files — model hyperparams, connection settings, experiment defaults
│   ├── base.yaml                       # Shared defaults: data_path, train/val/test split (70/15/15), runtime limits, llm.model
│   ├── zscore.yaml                     # Z-Score specific: window_size, threshold_sigma
│   ├── isolation_forest.yaml           # IF specific: n_estimators, contamination
│   ├── matrix_profile.yaml            # Matrix Profile specific: subsequence_length
│   ├── autoencoder.yaml               # Vanilla AE: hidden_dims, learning_rate, epochs
│   ├── rnn.yaml                        # RNN: hidden_dim, num_layers, dropout
│   ├── lstm.yaml                       # LSTM predictor: hidden_dim, num_layers, seq_len
│   ├── gru.yaml                        # GRU predictor: hidden_dim, num_layers, seq_len
│   ├── lstm_ae.yaml                    # LSTM-AE: encoder_dim, decoder_dim, latent_dim
│   ├── tcn.yaml                        # TCN: num_channels, kernel_size, dropout
│   ├── vae.yaml                        # VAE: latent_dim, kl_weight, hidden_dims
│   ├── gan.yaml                        # GANomaly: encoder_dim, decoder_dim, latent_dim (encoder-decoder-encoder, no separate discriminator)
│   ├── tadgan.yaml                     # TadGAN: cycle_weight, critic_iterations
│   ├── tranad.yaml                     # TranAD: d_model, nhead, num_layers, adversarial_weight
│   ├── deepar.yaml                     # DeepAR: hidden_dim, num_layers, num_samples
│   ├── diffusion.yaml                  # Diffusion: timesteps, noise_schedule, hidden_dim
│   ├── ensemble.yaml                   # Ensemble: sub-model list, weights, combination strategy
│   └── pi_connection.yaml              # PI System connection config (see PI Connection Config section below)
│
├── src/                                # Source root (uv uses src-layout)
│   └── sentinel/                       # Main package namespace
│       ├── __init__.py                 # Package version string, top-level convenience imports
│       │
│       ├── core/                       # Foundational abstractions — zero external model deps
│       │   ├── __init__.py             # Re-exports: BaseAnomalyDetector, registry functions
│       │   ├── base_model.py           # BaseAnomalyDetector ABC: fit(), score(), save(), load(), get_params() abstract; detect() concrete (score + threshold)
│       │   ├── registry.py             # @register_model decorator + entry-point plugin discovery
│       │   ├── config.py               # Pydantic RunConfig: loads YAML, validates, merges with defaults
│       │   ├── types.py                # ModelCategory enum, DetectionResult/TrainResult TypedDicts
│       │   ├── exceptions.py           # SentinelError, ModelNotFoundError, ValidationError, ConfigError
│       │   └── device.py              # resolve_device("auto"|"cpu"|"cuda"|"mps") → torch.device, auto-detects best available hardware
│       │
│       ├── data/                       # Data pipeline — all operations use Polars DataFrames/LazyFrames
│       │   ├── __init__.py             # Re-exports: load_csv, generate_synthetic, create_windows, fetch_pi_data, separate_labels
│       │   ├── loaders.py              # pl.read_csv / pl.scan_csv / pl.read_parquet, returns pl.DataFrame
│       │   ├── ingest.py              # Ingest pipeline: load file -> validate -> assign dataset_id -> save Parquet to data/raw/
│       │   ├── pi_connector.py         # PI System adaptor via pipolars: connect, search tags, fetch multi-tag timeseries, pivot to canonical schema
│       │   ├── validators.py           # Native Polars validation: timestamp as pl.Datetime, numeric cols, sorted, unique ts, no all-NaN/constant cols, min 2 rows
│       │   ├── preprocessors.py        # Polars expressions for scaling (z-score, min-max), NaN handling (forward-fill then zero-fill), chronological train/val/test split via .slice(), `create_windows(data, seq_len, stride)` → 3D array (n_windows, seq_len, n_features) for sequence models
│       │   ├── features.py             # Polars .shift() lags, .rolling_mean/std/min/max, FFT seasonality, dt accessors
│       │   ├── streaming.py            # Low-level async generators: convert static data (Parquet, PI snapshots) into async iterables of single rows. Pure data source adapters — no detection logic, no anomaly injection. PI polling lives here (raw generator); streaming/simulator.py wraps it.
│       │   └── synthetic.py            # Generate N-feature multivariate pl.DataFrame, inject point/contextual/collective anomalies
│       │
│       ├── models/                     # Model zoo — all 16 detectors
│       │   ├── __init__.py             # Triggers registration of all built-in models via subpackage imports
│       │   │
│       │   ├── statistical/            # CPU-only, no PyTorch dependency
│       │   │   ├── __init__.py         # Imports zscore, isolation_forest, matrix_profile to trigger registration
│       │   │   ├── zscore.py           # Rolling Z-Score detector: configurable window + sigma threshold
│       │   │   ├── isolation_forest.py # Sklearn IsolationForest wrapper: n_estimators, contamination
│       │   │   └── matrix_profile.py   # STOMP-based matrix profile via stumpy: subsequence distance scoring. Max recommended dataset: 100k rows (O(n²) memory). Longer series are chunked with overlap.
│       │   │
│       │   ├── deep/                   # PyTorch models — optional dependency group "deep"
│       │   │   ├── __init__.py         # Conditional import: registers models only if torch is available
│       │   │   ├── autoencoder.py      # Vanilla feedforward AE: symmetric encoder-decoder, MSE reconstruction
│       │   │   ├── rnn.py              # Elman RNN: sequence reconstruction, hidden state anomaly scoring
│       │   │   ├── lstm.py             # LSTM predictor: next-step forecasting, prediction error as anomaly score
│       │   │   ├── gru.py              # GRU predictor: lighter LSTM alternative, same forecast-error approach
│       │   │   ├── lstm_ae.py          # LSTM Autoencoder: LSTM encoder -> bottleneck -> LSTM decoder
│       │   │   ├── tcn.py              # Temporal Convolutional Network: dilated causal conv stack, residual blocks
│       │   │   ├── vae.py              # Variational Autoencoder: reparameterization trick, ELBO loss
│       │   │   ├── gan.py              # GANomaly-style: encoder-decoder-encoder, anomaly = reconstruction error + latent-space distance
│       │   │   ├── tadgan.py           # TadGAN: encoder-generator-critic with cycle consistency loss
│       │   │   ├── tranad.py           # TranAD: transformer encoder-decoder, single-phase self-conditioned adversarial training
│       │   │   ├── deepar.py           # DeepAR: autoregressive RNN, Gaussian likelihood, NLL-based scoring
│       │   │   └── diffusion.py        # Diffusion-based: DDPM forward/reverse process, MSE at fixed noise level t (trained on normal data only)
│       │   │
│       │   └── ensemble/              # Combination strategies
│       │       ├── __init__.py         # Imports hybrid to trigger registration
│       │       └── hybrid.py           # HybridEnsemble: weighted avg / voting / stacking of sub-model scores
│       │
│       ├── training/                   # Training orchestration and evaluation
│       │   ├── __init__.py             # Re-exports: Trainer, Evaluator
│       │   ├── trainer.py              # Unified Trainer: config -> registry lookup -> fit -> evaluate -> track
│       │   ├── evaluator.py            # Two modes: **supervised** (is_anomaly present → precision, recall, F1, AUC-ROC, AUC-PR, best-F1 threshold) and **unsupervised** (no labels → score distribution stats, reconstruction error stats, threshold-only metrics). Returns null for classification metrics when labels absent.
│       │   ├── thresholds.py           # Percentile, best-F1 search, POT (extreme value theory), ADWIN adaptive
│       │   ├── callbacks.py            # EarlyStopping (patience + delta), ModelCheckpoint (save best by metric)
│       │   └── schedulers.py           # LR scheduler wrappers: ReduceLROnPlateau, CosineAnnealingLR (with warmup), StepLR. Configured via base.yaml `scheduler` section.
│       │
│       ├── tracking/                   # Experiment logging — lightweight, JSON-backed
│       │   ├── __init__.py             # Re-exports: LocalTracker, compare_runs
│       │   ├── experiment.py           # LocalTracker: creates run_id dir, writes config.json + metrics.json
│       │   ├── artifacts.py            # Save/load model checkpoints, plots, predictions to run dir
│       │   └── comparison.py           # Load multiple runs, produce comparison pl.DataFrame, rank by metric
│       │
│       ├── viz/                        # Visualization — matplotlib-based, saveable to PNG/PDF
│       │   ├── __init__.py             # Re-exports: plot_timeseries, plot_reconstruction, plot_latent
│       │   ├── timeseries.py           # Time series line plot with anomaly regions shaded red
│       │   ├── reconstruction.py       # Original vs reconstructed overlay + per-feature error heatmap
│       │   └── latent.py               # t-SNE / UMAP projection of latent embeddings, colored by score
│       │
│       ├── streaming/                  # Real-time simulation and online detection
│       │   ├── __init__.py             # Re-exports: StreamSimulator, OnlineDetector
│       │   ├── simulator.py            # Stream orchestrator: consumes async generators from data/streaming.py, adds speed control, optionally injects anomalies, feeds rows to OnlineDetector
│       │   ├── online_detector.py      # Sliding-window buffer of size seq_len; on each new point, shifts window and calls model.score(window) on the full buffer (no score_single — uses standard batch score() API)
│       │   ├── drift.py                # Concept drift: gradual/abrupt/recurring simulation + ADWIN detection
│       │   └── alerts.py               # Alert rules: threshold breach, consecutive anomalies, rate-of-change
│       │
│       ├── explain/                    # Interpretability — explain why an anomaly was flagged
│       │   ├── __init__.py             # Re-exports: SHAPExplainer, ReconstructionExplainer
│       │   ├── shap_explainer.py       # SHAP KernelExplainer wrapping any model's score() function. Limited to top-k features (default k=10) to avoid O(2^m) explosion. For tree-based models (IF), uses TreeExplainer instead.
│       │   └── reconstruction.py       # Per-feature reconstruction error decomposition with ranking
│       │
│       ├── plugins/                    # External model plugin system
│       │   ├── __init__.py             # Re-exports: discover_plugins
│       │   └── loader.py              # Scan entry_points("sentinel.models"), validate BaseAnomalyDetector subclass
│       │
│       ├── api/                        # FastAPI REST/WebSocket server — optional dep group "api"
│       │   ├── __init__.py             # Re-exports: create_app
│       │   ├── app.py                  # FastAPI app factory: lifespan, CORS, mount dashboard static files
│       │   ├── deps.py                 # Dependency injection: get_registry, get_tracker, get_ollama_client
│       │   ├── schemas.py              # Pydantic v2 request/response models for all endpoints
│       │   ├── jobs.py                 # Background task manager: async training jobs with status polling
│       │   ├── routes/                 # API route modules — one per domain
│       │   │   ├── __init__.py         # Router aggregation: includes all sub-routers
│       │   │   ├── data.py              # POST /api/data/upload — CSV/Parquet upload, validate, store; GET /api/data — list datasets; GET /api/data/{id} — dataset summary; GET /api/data/{id}/preview — first N rows as JSON; GET /api/data/{id}/plot — interactive time series chart data
│       │   │   ├── pi.py               # POST /api/data/pi-fetch — fetch multi-tag PI timeseries; POST /api/data/pi-search — search tags; GET /api/data/pi-snapshot — current values
│       │   │   ├── train.py            # POST /api/train — submit async training job, GET /api/train/{job_id} — poll status
│       │   │   ├── detect.py           # POST /api/detect — batch detection, WS /api/detect/stream — real-time WebSocket
│       │   │   ├── evaluate.py         # GET /api/evaluate/{run_id} — return metrics as JSON
│       │   │   ├── models.py           # GET /api/models — list registered models from registry
│       │   │   ├── experiments.py      # GET /api/experiments — list runs, GET /api/experiments/compare — compare metrics
│       │   │   ├── prompt.py           # POST /api/prompt — user sends natural language, Nemotron selects MCP tools, returns result
│       │   │   └── visualize.py        # GET /api/visualize/{run_id}?type=timeseries — return PNG/SVG plot
│       │   └── dashboard/             # Static frontend served at /ui
│       │       ├── index.html          # Main dashboard: upload, dataset list, model list, training status, prompt bar
│       │       ├── upload.html         # Drag-and-drop CSV/Parquet upload page with validation feedback
│       │       ├── explore.html        # Dataset explorer: interactive multi-feature time series chart after upload
│       │       ├── pi.html             # PI System connector: server config, tag search/select, time range picker, fetch button
│       │       ├── prompt.html         # Natural language prompt page: type question, LLM picks tools, shows result
│       │       ├── style.css           # Dashboard styling
│       │       └── app.js              # Frontend JS: file upload, Chart.js multi-axis time series, prompt chat, WebSocket stream
│       │
│       ├── mcp/                        # MCP server — exposes Sentinel tools + resources to AI assistants
│       │   ├── __init__.py             # Re-exports: create_mcp_server
│       │   ├── server.py              # FastMCP server: registers tools, resources, lifespan with Ollama health check
│       │   ├── tools.py                # MCP tool definitions: train, detect, explain, analyze, recommend_model, upload, pi_fetch, pi_search
│       │   ├── resources.py            # MCP resources: experiments://list, models://registry, data://datasets, data://datasets/{id}
│       │   ├── router.py              # Prompt router: Nemotron parses user prompt -> selects tool(s) -> executes -> assembles reply
│       │   ├── llm_client.py           # Ollama HTTP client: POST /api/generate + /api/chat with nemotron-3-nano-4b
│       │   └── prompts.py              # Prompt templates: anomaly_report, model_recommendation, tool_selection, data_summary
│       │
│       └── cli/                        # Typer CLI — `sentinel <command>`
│           ├── __init__.py             # Empty — app.py is the entry point
│           ├── app.py                  # Typer root: registers train/evaluate/detect/visualize/serve/data/pi sub-commands
│           ├── ingest.py               # `sentinel ingest --file data.csv` — validate multivariate CSV, store as Parquet
│           ├── data.py                 # `sentinel data list`, `sentinel data info --id <id>`, `sentinel data delete --id <id>`
│           ├── pi.py                   # `sentinel pi fetch/search/snapshot` — PI System data extraction commands (platform-checked: errors on non-Windows)
│           ├── generate.py            # `sentinel generate --features 5 --length 10000 --anomaly-ratio 0.05 --seed 42 --output <path>`
│           ├── train.py                # `sentinel train --config <yaml>` — validates config first, then runs Trainer end-to-end
│           ├── evaluate.py             # `sentinel evaluate --run-id <id>` — loads run, prints metrics table
│           ├── detect.py               # `sentinel detect --model <path> --data <path> [--mode streaming] [--source file|pi]` — `--data` and `--source pi` are mutually exclusive
│           ├── export.py               # `sentinel export --run-id <id> --format native|onnx --output <path>` — export trained model
│           ├── validate_config.py      # `sentinel validate-config --config <yaml>` — validate config without training
│           ├── visualize.py            # `sentinel visualize --run-id <id> --type <timeseries|recon|latent>`
│           ├── serve.py                # `sentinel serve --host 0.0.0.0 --port 8000` — launch FastAPI server with uvicorn
│           └── mcp_serve.py            # `sentinel mcp-serve --transport stdio|sse [--host localhost --port 3000]`
│
├── tests/                              # Test suite — three tiers
│   ├── conftest.py                     # Shared pytest fixtures: synthetic data arrays, tmp experiment dirs
│   ├── unit/                           # Fast tests: pure functions, synthetic data, no disk I/O
│   │   ├── test_types.py              # Verify enum values, TypedDict shapes
│   │   ├── test_registry.py           # Register/lookup/duplicate-name error
│   │   ├── test_config.py             # YAML loading, validation, default merging
│   │   ├── test_synthetic.py          # Generated data shape, anomaly label count
│   │   ├── test_loaders.py            # CSV round-trip via pl.read_csv, column types, pl.DataFrame output
│   │   ├── test_validators.py         # Polars validation pass/fail: valid data, empty df, single-row, all-NaN col, constant col, duplicate timestamps, unsorted, mixed timezones
│   │   ├── test_preprocessors.py      # Polars scaling expressions, window shapes, split ratios
│   │   ├── test_features.py           # Polars .shift() lag correctness, rolling stat values
│   │   ├── test_zscore.py             # Detect injected spikes in synthetic data
│   │   ├── test_isolation_forest.py   # Detect injected anomalies, score ordering
│   │   ├── test_matrix_profile.py     # Discord detection in synthetic subsequences
│   │   ├── test_evaluator.py          # Known-label metric computation accuracy
│   │   ├── test_thresholds.py         # Percentile correctness, best-F1 search
│   │   ├── test_ingest.py             # Ingest pipeline: validate multivariate schema, reject bad formats, assign dataset_id
│   │   ├── test_pi_connector.py       # PI connector: mock PIClient, verify multi-tag fetch, pivot, schema compliance
│   │   ├── test_tracking.py           # LocalTracker creates dirs, writes valid JSON
│   │   └── test_config_validation.py  # Invalid YAML, missing model name, out-of-range hyperparams, nonexistent dataset path
│   ├── integration/                    # Pipeline tests: data -> model -> eval -> track
│   │   ├── test_train_pipeline.py     # Full train run with Z-Score, verify artifacts
│   │   ├── test_deep_pipeline.py      # LSTM-AE train on tiny data, verify metrics
│   │   ├── test_streaming.py          # Stream simulator + online detector + ADWIN drift detection integration
│   │   ├── test_ensemble.py           # Ensemble combining statistical + deep models (heterogeneous score ranges)
│   │   ├── test_cli.py                # CLI subprocess invocations, exit codes, output files
│   │   ├── test_api.py                # FastAPI TestClient: all routes, WebSocket stream, pagination, file upload rejection (too large, wrong type)
│   │   ├── test_api_errors.py         # Error paths: invalid config POST, nonexistent model, malformed CSV upload, oversized upload
│   │   ├── test_dashboard.py          # Smoke: dashboard HTML files served correctly at /ui, contain expected elements
│   │   └── test_mcp.py               # MCP tool invocation tests, resource read tests, tool error responses
│   └── smoke/                          # End-to-end acceptance tests (slow, marked @pytest.mark.slow)
│       └── test_end_to_end.py          # Full pipeline: generate data -> train 3 models -> ensemble -> visualize
│
├── data/                               # Git-ignored runtime data directory
│   ├── datasets.json                  # Dataset registry: maps dataset_id → {original_name, source, uploaded_at, shape, feature_names, time_range}. Written atomically on ingest.
│   ├── raw/                            # Original ingested datasets (CSV, Parquet — loaded via pl.read_*)
│   ├── processed/                      # Preprocessed data caches (Parquet format via pl.DataFrame.write_parquet)
│   └── experiments/                    # Experiment artifacts: {run_id}/config.json, metrics.json, model.*
│
└── docs/                               # Project documentation
    ├── quickstart.md                   # 5-minute getting started guide with uv
    ├── architecture.md                 # System design: registry, plugin, training loop diagrams
    ├── models.md                       # Per-model explanation: algorithm, hyperparams, when to use, recommended data characteristics
    ├── plugins.md                      # How to write and register an external model plugin
    ├── api.md                          # FastAPI endpoints, request/response examples, WebSocket protocol
    ├── mcp.md                          # MCP server setup, tool descriptions, resource URIs, Ollama config
    └── troubleshooting.md             # FAQ: Ollama down, PI auth failure, CUDA OOM, dependency conflicts, Python version fallback
```

---

## PI Connection Config

```yaml
# configs/pi_connection.yaml
server:
  host: my-pi-server
  port: 5450
  timeout: 30
default_tags: []              # No defaults — user must specify
time_range:
  start: "*-1d"
  end: "*"
interval: "5m"
cache:
  backend: sqlite             # sqlite | arrow_ipc | none
  directory: "data/.pi_cache" # Cache storage location (git-ignored)
  ttl_hours: 24
  max_size_mb: 500            # Evict oldest entries when exceeded
timezone: "UTC"               # All PI data normalized to this timezone
```

Loaded via Pydantic `PIConnectionConfig` in `core/config.py`.

---

## Base Config Defaults

```yaml
# configs/base.yaml
data:
  path: "data/raw"
  processed_path: "data/processed"
  metadata_file: "data/datasets.json"  # Dataset registry: maps dataset_id → {name, source, uploaded_at, shape, features, time_range}
split:
  train: 0.70
  val: 0.15
  test: 0.15
  # Splits are ALWAYS chronological (no shuffle) — see Data Format section
seed: 42                     # Global random seed for reproducibility (numpy, torch, python random)
device: "auto"               # auto | cpu | cuda | mps — auto-detects best available hardware
training_mode: "normal_only" # normal_only | all_data — semi-supervised models (AE, VAE, GAN, Diffusion) filter out is_anomaly==1 rows before training. all_data uses everything.
scheduler:
  type: "reduce_on_plateau"  # reduce_on_plateau | cosine | step | none
  patience: 5                # Epochs to wait before reducing LR (reduce_on_plateau)
  factor: 0.5                # LR reduction factor
  min_lr: 1.0e-6
  warmup_epochs: 0           # Linear warmup (useful for TranAD — set to 5)
runtime:
  max_upload_size_mb: 100
  max_features: 500
  training_timeout_s: 3600
  api_request_timeout_s: 300
  max_dataset_rows_matrix_profile: 100000
  shap_max_features: 10
llm:
  model: "nvidia/nemotron-3-nano-4b"
  ollama_url: "http://localhost:11434"
  timeout_s: 30
logging:
  level: "INFO"              # DEBUG | INFO | WARNING | ERROR
  format: "json"             # json | console
```

### Example Model Config Defaults

```yaml
# configs/zscore.yaml
inherits: base.yaml
model: zscore
window_size: 30
threshold_sigma: 3.0

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

# configs/ensemble.yaml
inherits: base.yaml
model: hybrid_ensemble
sub_models: [zscore, isolation_forest, lstm]
weights: [0.3, 0.3, 0.4]           # Renormalized if a sub-model fails
strategy: weighted_average           # weighted_average | majority_voting | stacking
```

---

## Architecture Decisions

**Polars everywhere.** All tabular data flows through `polars.DataFrame` and `polars.LazyFrame`. No pandas anywhere. Models receive `numpy.ndarray` via `.to_numpy()` at the boundary — Polars handles everything before that handoff and everything after (result collection, metric aggregation). Cached intermediate data is persisted as Parquet via `df.write_parquet()`.

**Package manager: uv.** `uv` for everything — init, add, sync, run, lock. No pip, no venv manually. `uv run sentinel train` to execute.

**Dependency groups.** Core deps always installed. `--group deep` adds torch/einops. `--group explain` adds shap/umap-learn. `--group pi` adds pipolars (PI System data extraction — Windows only, requires PI AF SDK .NET 4.8). `--group dev` adds pytest/ruff/mypy. Install everything: `uv sync --all-groups`.

**Plugin system.** `BaseAnomalyDetector` ABC + `@register_model("name")` decorator + `importlib.metadata.entry_points("sentinel.models")` for external plugins. Models are never imported directly — looked up by string name from registry.

**PyTorch isolation.** Only `models/deep/*.py` imports torch. Everything else (core, training, CLI) interacts through the base class interface. Deep models register conditionally — if torch is missing, they silently skip. Device selection (`cpu`/`cuda`/`mps`) is configured via `device: "auto"` in base config. A `core/device.py` utility resolves `"auto"` to the best available hardware. All deep model `fit()` methods call `self.model.to(device)` and ensure training tensors are on the same device.

**Training data modes.** Configured via `training_mode` in base config:
- `normal_only` (default): semi-supervised — the Trainer filters out rows where `is_anomaly == 1` before calling `model.fit()`. Used by AE, VAE, GAN, Diffusion, and any reconstruction-based model. For real-world data without labels, ALL data is assumed normal.
- `all_data`: the full dataset is used for training. Used by Isolation Forest, Matrix Profile, and other models that handle anomalies internally.

**Experiment tracking.** Lightweight JSON-backed tracker (zero extra deps). Stores config, metrics, and artifact paths in `data/experiments/{run_id}/`.

**Config-driven.** Every run is reproducible from its YAML config file + the global `seed`. The Trainer sets `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, and `torch.cuda.manual_seed_all()` at the start of each run. Note: full determinism on GPU requires `torch.use_deterministic_algorithms(True)` which may reduce performance — enabled only when `seed` is explicitly set.

**Config inheritance.** Model configs use `inherits: base.yaml`. Resolution: **deep merge** — base values are loaded first, then the model config is merged recursively. At leaf level, model config values override base values. Single-level inheritance only (no chaining). Precedence order: `base.yaml < model.yaml < CLI overrides`.

**CLI.** Typer with subcommands: `ingest`, `generate`, `data`, `train`, `evaluate`, `detect`, `export`, `visualize`, `validate-config`, `serve`, `mcp-serve`, `pi`. Entry point: `sentinel = "sentinel.cli.app:app"`.

**PI System data connection (pipolars).** Live timeseries extraction from OSIsoft/AVEVA PI System via the `pipolars` library (optional dep group `--group pi`). The `data/pi_connector.py` module wraps `PIClient` with three capabilities:
1. **Tag search** — search available PI points on a server by name pattern, returns matching tag names for selection
2. **Multi-tag fetch** — accepts a list of PI point names (tags), a time range, and an interpolation interval, uses pipolars' bulk API (`client.query(tags).time_range(start, end).interpolated(interval).pivot().to_dataframe()`) to pull all tags in a single call and pivot them into Sentinel's canonical schema (timestamp + N feature columns, one per tag)
3. **Snapshot** — get current values for selected tags

The connector outputs a standard `pl.DataFrame` that feeds directly into the existing validation -> Parquet caching pipeline. Caching is supported via pipolars' built-in SQLite/Arrow IPC cache to avoid redundant PI server calls. Connection config is stored in `configs/pi_connection.yaml` and loaded via Pydantic `PIConnectionConfig`. Falls back gracefully if pipolars is not installed — the PI commands/routes simply report that the PI dependency group is missing.

**PI authentication.** Uses Windows Integrated Authentication by default (automatic via pythonnet/.NET — the logged-in user's Windows identity). No credentials stored in config files. Note: pipolars requires Windows (PI AF SDK = .NET 4.8).

**Ollama integration.** HTTP client calls `POST http://localhost:11434/api/generate`. Model name is **configurable** via `configs/base.yaml` (`llm.model: nvidia/nemotron-3-nano-4b`). Falls back gracefully if Ollama is unreachable — tools return raw structured data without LLM narrative. Model is pulled once during setup: `ollama pull <model-name>`.

**Dashboard.** Minimal static frontend (HTML + vanilla JS + Chart.js) served by FastAPI at `/ui`. No build step, no npm. After upload/PI fetch, the user is redirected to `/ui/explore.html?id={dataset_id}` which renders an interactive multi-axis Chart.js line chart with zoom, pan, and anomaly region overlays.

**MCP-API coupling.** The MCP server and FastAPI server are independent processes. Both import Sentinel modules directly (approach: direct import, not HTTP proxy). They share the filesystem as source of truth (`data/raw/`, `data/experiments/`, `data/datasets.json`). The MCP server discovers runs and datasets by reading from disk, not by maintaining in-memory state. If a model is trained via the API, the MCP server can access it immediately because it reads `data/experiments/` on each tool call.

**Prompt-driven MCP tool routing.** When a user types a natural language prompt (via `/ui/prompt.html` or `POST /api/prompt`), the request is forwarded to the LLM via Ollama. The LLM receives available MCP tool schemas + the user's message, responds with a structured tool-call plan. The router (`mcp/router.py`) parses this, executes the tool(s) sequentially, and assembles the results into a final response. Rate-limited to 10 requests/minute to prevent resource abuse.

**LLM output error handling.** The router handles malformed LLM output:
1. Parse response with `json.loads()` in try/except; on failure, retry once with a corrective prompt ("respond in valid JSON").
2. Validate tool name exists in registry before execution.
3. Validate tool parameters against Pydantic schemas before calling the tool.
4. If all retries fail, return an error to the user with the raw LLM response for debugging.
5. For users needing more reliable tool calling, the LLM model is configurable via `base.yaml` — larger models (e.g., `llama3.1:8b-instruct`, `qwen2.5:7b`) improve structured output quality.

**Model serialization.** Statistical models (Z-Score, IF) use `joblib.dump()`/`joblib.load()`. PyTorch models save `state_dict` + `config.json` (architecture params needed to reconstruct the model). Each saved artifact includes a `version` field matching the Sentinel package version — `load()` warns if versions don't match. No pickle for PyTorch (security + portability).

**Concurrency.** Training jobs run in a `ProcessPoolExecutor` (not threads) to avoid GIL contention and PyTorch/asyncio conflicts. On macOS, explicitly use `mp_context=get_context('fork')` since macOS defaults to `spawn`. Experiment artifact writes use atomic file operations (write to `{path}.tmp`, then `os.rename`). Multiple simultaneous API requests are safe — each training job gets its own `run_id` directory.

**Error handling & resilience.**
- PI System connections retry 3 times with exponential backoff (1s, 2s, 4s) before failing.
- Ensemble partial failure: if a sub-model fails during `.score()`, it is excluded and remaining model weights are renormalized. A warning is logged. If all sub-models fail, the ensemble raises.
- Failed Parquet ingestion: partial files are cleaned up, dataset_id is not registered (atomic write via temp file + rename).
- Invalid configs are caught at CLI/API entry point, not deep inside the trainer.

**Logging.** `structlog` with JSON output. Log levels: training progress at INFO, per-batch at DEBUG, PI connection attempts at INFO, errors at ERROR. Output to stderr by default; configurable via `SENTINEL_LOG_FILE` env var.

**Security.**
- File upload: max 100MB, allowed MIME types `text/csv` and `application/octet-stream` (Parquet). Path traversal protection via `pathlib.PurePath` validation.
- `/api/prompt` rate-limited to 10 req/min to prevent LLM-triggered resource abuse (e.g., repeatedly triggering `sentinel_train`).
- No authentication — designed for local/internal use. Auth is out of scope for v1 but the FastAPI app factory accepts optional middleware.

**Runtime limits** (configurable via `configs/base.yaml`):
- `max_upload_size_mb: 100`
- `max_features: 500`
- `training_timeout_s: 3600` (1 hour)
- `api_request_timeout_s: 300`
- `max_dataset_rows_matrix_profile: 100000`
- `shap_max_features: 10`

---

## Compatibility Notes

**Python 3.12** is used for maximum dependency compatibility. All core dependencies (stumpy, shap, scipy, torch, polars) have stable wheels for 3.12.

**PyTorch on ARM64 Mac:** `uv add torch` installs the correct MPS-enabled binary for Apple Silicon. No CUDA on macOS. For GPU training on Linux, use `uv add torch --index-url https://download.pytorch.org/whl/cu121`.

**SHAP + numpy:** SHAP pins older numpy versions. If version conflicts arise with torch or polars, pin `shap<0.45` or use the `--group explain` isolation to avoid polluting the core environment.

**Polars:** Pin to a specific 1.x release in `pyproject.toml` (e.g., `polars>=1.0,<1.1`). Polars has frequent breaking changes — "latest" is dangerous for reproducibility.

---

## Polars Usage Map

> **Note:** This map reflects the planned usage. It will go stale as the codebase evolves — consider regenerating from code after Phase 3.

| Module | Polars Usage |
| ------ | ------------ |
| `data/loaders.py` | `pl.read_csv()`, `pl.scan_csv()`, `pl.read_parquet()` — all ingestion returns `pl.DataFrame` |
| `data/validators.py` | Native Polars expressions: `df.schema`, `.is_sorted()`, `.is_unique()`, `.null_count()`, `.std()` for validation — no patito |
| `data/preprocessors.py` | `pl.Expr` for z-score scaling `(col - mean) / std`, `.slice()` for chronological splits, `.to_numpy()` + stride-based `create_windows()` for 3D sequence tensors |
| `data/features.py` | `.shift(n)` for lags, `.rolling_mean()/.rolling_std()` for stats, `.dt.hour()` for temporal features |
| `data/ingest.py` | `pl.read_csv()` -> validate -> `df.write_parquet()` to `data/raw/{dataset_id}.parquet` |
| `data/pi_connector.py` | pipolars returns native `pl.DataFrame`, `.pivot()` for multi-tag columns, feeds into validators + `df.write_parquet()` |
| `data/synthetic.py` | Build N-feature multivariate `pl.DataFrame` with `timestamp`, `feature_1..N`, `is_anomaly` columns |
| `data/streaming.py` | `df.iter_rows()` to yield one row at a time for async replay |
| `training/evaluator.py` | Collect metrics into `pl.DataFrame` for tabular display and Parquet export |
| `tracking/comparison.py` | `pl.concat()` multiple run metrics, `.sort()` / `.group_by()` for ranking |
| `tracking/experiment.py` | Write metrics as `pl.DataFrame.write_parquet()` alongside JSON config |
| `viz/*` | `.to_numpy()` at plot boundary — matplotlib receives arrays, Polars owns the data until then |
| `cli/*` | `pl.DataFrame` printed via `rich` table or `.glimpse()` for CLI output |
| `api/schemas.py` | Pydantic models serialize from `pl.DataFrame.to_dicts()` for JSON responses |
| `api/routes/*` | Routes receive Polars frames from modules, convert to response schemas |
| `api/routes/data.py` | `pl.read_parquet()` for dataset retrieval, `.head(n).to_dicts()` for preview, `.to_dicts()` for plot data |
| `mcp/resources.py` | Resources return `pl.DataFrame.to_dicts()` as structured MCP content |
| `mcp/router.py` | Receives tool execution results as `pl.DataFrame`, serializes for LLM response assembly |

---

## MCP Tools & Resources

### Tools

All tool outputs are JSON-serializable (numpy arrays → lists, float64 → float, DataFrame → list of dicts). On error, tools return `{error: str, code: str}` instead of raising — the router assembles error context into the response.

| Tool | Input | Output | LLM Used? |
| ---- | ----- | ------ | --------- |
| `sentinel_train` | `{config_path: str}` — validates config exists and is valid YAML before starting | `{run_id, metrics: {precision, recall, f1, auc_roc} (or null if unlabeled), duration_s}` | No |
| `sentinel_detect` | `{model_path: str, data_path: str}` — validates model matches data feature count | `{scores: float[], labels: int[], threshold: float}` | No |
| `sentinel_explain` | `{run_id: str, sample_indices: int[]}` — max 50 indices, validates bounds against dataset length | `{shap_values: float[][], feature_ranking: [{name, importance}]}` | No |
| `sentinel_analyze` | `{run_id: str}` | Natural-language anomaly report | Yes — LLM generates narrative from detection results |
| `sentinel_recommend_model` | `{data_path: str}` — analyzes data length, n_features, periodicity, noise level | Ranked model suggestions with rationale (deterministic feature analysis + LLM narrative) | Yes |
| `sentinel_upload` | `{file_path: str}` | `{dataset_id, shape, features[], time_range}` | No |
| `sentinel_list_models` | `{}` | `{models: [{name, category, description}]}` | No |
| `sentinel_list_datasets` | `{}` | `{datasets: [{id, name, shape, features[], uploaded_at}]}` | No |
| `sentinel_compare_runs` | `{run_ids: str[]}` | Comparison table with metrics per run | No |
| `sentinel_delete_run` | `{run_id: str}` | `{deleted: true, artifacts_removed: int}` | No |
| `sentinel_export_model` | `{run_id: str, format: "native"\|"onnx"}` | `{export_path: str, format: str, size_bytes: int}` | No |
| `sentinel_cancel_job` | `{job_id: str}` | `{cancelled: true}` or `{error: "already completed"}` | No |
| `sentinel_pi_fetch` | `{server: str, tags: str[], start: str, end: str, interval: str}` | `{dataset_id, shape, features[], time_range}` | No |
| `sentinel_pi_search` | `{server: str, pattern: str}` | `{tags: [{name, description, uom}]}` | No |
| `sentinel_prompt` | `{message: str}` | LLM selects tools, executes sequentially, returns assembled response. If a tool fails, error is included in context and remaining tools still execute. | Yes |

### Resources

All resources return JSON-serializable structured content (list of dicts from `pl.DataFrame.to_dicts()`).

| URI | Description |
| --- | ----------- |
| `experiments://list` | All experiment runs with run_id, model, timestamp, key metrics |
| `experiments://{run_id}` | Single run details: full config, all metrics, artifact paths |
| `models://registry` | All registered models: name, category, description, config schema |
| `data://datasets` | All uploaded datasets: dataset_id, name, shape, feature names, time range, upload timestamp |
| `data://datasets/{id}` | Single dataset summary: shape, column types, basic stats, null counts (via Polars `.describe()`) |
| `data://datasets/{id}/preview` | First 20 rows of dataset as structured JSON content |

---

## API Reference

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST | `/api/data/upload` | Upload CSV/Parquet file (max 100MB), validate multivariate schema, store, return dataset_id. Rejects non-CSV/Parquet MIME types. |
| GET | `/api/data?page=1&limit=50&sort=uploaded_at&order=desc` | List uploaded datasets with metadata. **Paginated** — returns `{items: [...], total: int, page: int}`. |
| GET | `/api/data/{id}` | Dataset summary: shape, features, time range, basic stats |
| GET | `/api/data/{id}/preview?rows=20` | First N rows as JSON array of objects. Default N=20, max 1000. |
| GET | `/api/data/{id}/plot?start=&end=&features=cpu,mem&max_points=5000` | Time series data for Chart.js. **Server-side downsampling** via LTTB (Largest-Triangle-Three-Buckets) if dataset exceeds `max_points`. Supports time-window and feature selection. |
| DELETE | `/api/data/{id}` | Delete dataset and its Parquet file. Returns 404 if not found. |
| POST | `/api/data/pi-fetch` | Fetch timeseries from PI System: server, tags[], start, end, interval -> validates, stores, returns dataset_id |
| POST | `/api/data/pi-search` | Search PI points by name pattern on a given server -> returns matching tag names |
| GET | `/api/data/pi-snapshot` | Get current snapshot values for specified tags from PI server |
| POST | `/api/prompt` | Natural language prompt -> LLM selects MCP tools -> executes **sequentially** -> returns assembled response. If a tool fails, remaining tools still execute. Rate-limited: 10 req/min. |
| POST | `/api/train` | Submit async training job. Returns `{job_id, model_name, status: "pending", poll_url: "/api/train/{job_id}"}`. Config validated **before** job is queued. |
| GET | `/api/train/{job_id}` | Poll training job status. Returns `{job_id, status, model_name, progress_pct, metrics (if completed), error_message (if failed), duration_s}`. |
| DELETE | `/api/train/{job_id}` | Cancel a running training job. Returns 409 if already completed. |
| POST | `/api/detect` | Batch anomaly detection. Request: multipart file upload or `{data_path, model_path}`. Response: `{scores: float[], labels: int[], threshold: float, model_name: str}`. Max payload 100MB. |
| WS | `/api/detect/stream` | WebSocket streaming detection. See **WebSocket Protocol** section below. |
| GET | `/api/evaluate/{run_id}` | Return evaluation metrics for a run |
| GET | `/api/models` | List all registered models from registry |
| GET | `/api/experiments?page=1&limit=50` | List experiment runs. **Paginated**. |
| GET | `/api/experiments/compare?ids=a,b,c` | Compare metrics across runs |
| GET | `/api/visualize/{run_id}?type=timeseries` | Return plot as PNG/SVG |
| GET | `/ui` | Serve static web dashboard |
| GET | `/health` | Health check. Returns `{status: "healthy"|"degraded", api: "ok", ollama: "ok"|"unreachable", version: str}`. HTTP 200 if API ok, 503 if API itself unhealthy. Ollama being unreachable = "degraded" not unhealthy. |

### WebSocket Protocol (`/api/detect/stream`)

```json
// Client -> Server (input frame)
{"type": "data", "timestamp": "2024-01-01T00:00:00Z", "features": {"cpu": 45.2, "mem": 2048}}

// Server -> Client (score frame)
{"type": "score", "timestamp": "2024-01-01T00:00:00Z", "score": 0.87, "label": 1, "threshold": 0.75}

// Server -> Client (alert frame)
{"type": "alert", "rule": "consecutive_anomalies", "count": 5, "start": "...", "end": "..."}

// Server -> Client (error frame)
{"type": "error", "message": "Model not loaded", "code": "MODEL_NOT_FOUND"}

// Heartbeat: server sends {"type": "ping"} every 30s, client must respond {"type": "pong"} within 10s or connection is closed.
```

---

## Implementation Phases

### Phase 1: Foundation — Scaffold + Core + Statistical Models

**Goal:** Working end-to-end pipeline with statistical models.

**Files:** pyproject.toml, .gitignore, .python-version, Makefile, core/*, data/synthetic.py, data/loaders.py, data/ingest.py, data/validators.py, data/preprocessors.py, models/statistical/*, training/evaluator.py, training/thresholds.py, configs/base.yaml, configs/zscore.yaml, configs/isolation_forest.yaml, configs/matrix_profile.yaml, tests/conftest.py + unit tests

**Claude Code steps:**
1. `uv init --python 3.12` — scaffold project
2. `uv add <all-deps>` — install ALL dependency groups upfront (core, deep, explain, api, mcp, dev) to validate the full dependency graph early. No patito.
3. `uv sync --all-groups` — verify resolution. Run `uv run python -c "import polars, numpy, sklearn, scipy, stumpy, joblib; print('Core OK')"`. Pin Polars tightly after confirming (e.g., `polars>=1.10,<1.12`).
4. Write `.gitignore`, `.python-version`, Makefile. `git init` + initial commit.
5. Write core abstractions in dependency order: exceptions → types → device → registry → base_model → config. Build config system incrementally: (a) Python dataclass defaults, (b) YAML loading, (c) `inherits:` resolution, (d) CLI overrides. Each layer gets its own unit test.
6. Write data pipeline: multivariate validators (native Polars: timestamp + N features, UTC enforcement, no all-NaN/constant cols, min 2 rows, NaN handling), loaders, ingest pipeline (atomic writes), preprocessors (forward-fill + zero-fill for NaN)
7. Write synthetic generator with configurable `--features N`, `--anomaly-ratio`, `--seed` for multivariate data
8. Write statistical models (zscore, isolation_forest, matrix_profile with chunking for >100k rows). **These 3 can be implemented in parallel** via 3 `implement-model` agents.
9. Write evaluator + thresholds. **Critical rule:** threshold selection (best-F1 search) runs on validation set only. Metric reporting runs on test set only. Never compute thresholds on test data.
10. Write Makefile with targets: `lint`, `test`, `train`, `serve`, `format`, `typecheck`
11. Write unit tests (including edge-case validation: empty, single-row, all-NaN, constant, duplicate timestamps)
12. Write one integration test: `synthetic data → validate → preprocess → zscore.fit() → zscore.score() → evaluate` — verifies the full pipeline works end-to-end
13. `uv run pytest tests/ -v` — verify all unit + integration tests pass

### Phase 2: Training Loop + Tracking + CLI

**Goal:** Unified trainer, experiment tracking, usable CLI.

**Files:** training/trainer.py, training/callbacks.py, tracking/*, cli/*, data/features.py

**Note:** PI connector is deferred to Phase 8 (Windows-only, cannot be tested on macOS, belongs with API routes that consume it).

**Claude Code steps:**
1. Write Trainer class (config -> registry lookup -> fit -> evaluate -> log)
2. Write LocalTracker (JSON-backed experiment logging)
3. Write CLI commands with Typer (including `sentinel ingest`, `sentinel generate`, `sentinel data list/info/delete`, `sentinel validate-config`)
4. Write feature engineering module (lags, rolling stats per feature channel)
5. Write integration test: `sentinel train --config configs/zscore.yaml` end-to-end via Trainer class
6. `uv run sentinel train --config configs/zscore.yaml` — verify artifacts created
7. `uv run sentinel evaluate --run-id <id>` — verify metrics printed
8. `uv run sentinel validate-config --config configs/zscore.yaml` — verify config validation without training
9. `uv run sentinel data list` — verify dataset listing

### Phase 3: Classical Deep Learning Models

**Goal:** AE, LSTM, LSTM-AE, TCN working through unified interface. RNN and GRU deferred to Phase 5 (simpler variants of LSTM — validate deep model pipeline with fewer models first).

**Files:** models/deep/autoencoder.py, lstm.py, lstm_ae.py, tcn.py + configs + tests

**Claude Code steps:**
1. Verify torch is available: `uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"` (deps were added in Phase 1)
2. Write conditional registration in `models/deep/__init__.py`
3. Implement 4 models: AE, LSTM, LSTM-AE, TCN. **These can be implemented in parallel** via 4 `implement-model` agents after step 2.
4. Write matching YAML configs
5. Write integration test: LSTM-AE train on tiny synthetic data → verify metrics + saved artifacts
6. `uv run sentinel train --config configs/lstm_ae.yaml` — verify end-to-end
7. `uv run pytest tests/ -v` — all tests pass including new deep model tests

### Phase 4: Core Generative Models

**Goal:** VAE, TranAD, Diffusion — models with well-defined loss functions (see Model Loss Functions section above).

**Files:** models/deep/vae.py, tranad.py, diffusion.py + configs + tests

**Note:** GAN, TadGAN, and DeepAR are deferred to Phase 5. GAN/TadGAN require adversarial training stability work; DeepAR requires probabilistic output handling. Keeping Phase 4 to 3 models reduces risk.

**Claude Code steps:**
1. Implement each model following the exact loss formulations in the Model Loss Functions section:
   - **VAE:** ELBO = MSE + β*KL, anomaly score = reconstruction MSE only
   - **TranAD:** self-conditioned adversarial with two decoders, anomaly score = MSE(x, O₁)
   - **Diffusion:** DDPM denoising, anomaly score = MSE(x₀, x̂₀) at t=T//2
2. **These 3 can be implemented in parallel** via 3 `implement-model` agents
3. Write YAML configs for each with documented default values
4. Write integration test: train VAE + TranAD + Diffusion on same synthetic data, verify valid DetectionResult
5. `uv run sentinel train --config configs/tranad.yaml` — verify each model
6. `uv run pytest tests/ -v` — all tests pass

### Phase 5: Remaining Models + Ensemble + Drift

**Goal:** Complete the model zoo (RNN, GRU, GAN, TadGAN, DeepAR), hybrid ensemble, thresholding, drift.

**Files:** models/deep/rnn.py, gru.py, gan.py, tadgan.py, deepar.py, models/ensemble/hybrid.py, training/thresholds.py (extend), streaming/drift.py, configs/ensemble.yaml

**Claude Code steps:**
1. Implement RNN and GRU (simpler variants of LSTM — reuse LSTM patterns). **Can be implemented in parallel.**
2. Implement GAN, TadGAN, DeepAR following the exact loss formulations in the Model Loss Functions section:
   - **GAN:** GANomaly encoder-decoder-encoder, score = reconstruction + latent distance
   - **TadGAN:** Wasserstein critic with cycle-consistency, score = critic + reconstruction
   - **DeepAR:** Gaussian NLL per step per feature, single-step rolling forecast
3. Implement HybridEnsemble (weighted avg, voting, stacking with logistic regression meta-learner). All sub-model scores min-max normalized to [0,1] before combination. Partial failure handling: exclude failed sub-models, renormalize weights.
4. Add POT and ADWIN threshold strategies
5. Implement concept drift simulation + detection
6. Run comparison: train all 16 models on same synthetic data (10k x 10), compare metrics
7. `uv run sentinel train --config configs/ensemble.yaml` — verify combined scores
8. `uv run pytest tests/ -v` — all tests pass

### Phase 6: Visualization + Interpretability

**Goal:** Rich plots and anomaly explanations.

**Files:** viz/*, explain/*, cli/visualize.py

**Claude Code steps:**
1. Implement time series, reconstruction, and latent space plots
2. Implement SHAP explainer (top-k features, default k=10; TreeExplainer for IF) and reconstruction error breakdown
3. `uv run sentinel visualize --run-id <id> --type timeseries --output plot.png`
4. Inspect generated plots for correctness

### Phase 7: Streaming + Alerts

**Goal:** Real-time simulation with online detection, including PI System as a live source.

**Files:** streaming/*, data/streaming.py

**Claude Code steps:**
1. Implement async stream simulator (replay from Parquet)
2. Implement online detector with sliding window
3. Implement PI polling source: `client.snapshots(tags)` on a configurable interval, yields rows to OnlineDetector
4. Add `--source pi` option to `sentinel detect --mode streaming` — `sentinel detect --mode streaming --source pi --server <host> --tags TAG1 TAG2 --interval 5s`
5. Implement alert rules engine
6. `uv run sentinel detect --mode streaming --model <path> --data <path>`

### Phase 8: FastAPI + Dashboard + PI Connector

**Goal:** REST/WebSocket API with web dashboard, PI System data extraction.

**Files:** api/app.py, api/deps.py, api/schemas.py, api/jobs.py, api/routes/*, api/dashboard/*, cli/serve.py, data/pi_connector.py, cli/pi.py, configs/pi_connection.yaml

**Note:** PI connector is built here (moved from Phase 2) because: (a) Windows-only, untestable on macOS dev, (b) PI API routes consume it, so co-locating reduces integration risk, (c) core pipeline doesn't depend on it.

**Claude Code steps:**
1. Verify API deps available: `uv run python -c "import fastapi, uvicorn; print('API OK')"` (deps added in Phase 1)
2. Write FastAPI app factory with lifespan (init registry, tracker). Add CORS, max upload size (100MB), rate limiting middleware.
3. Write Pydantic request/response schemas with full type annotations (including PIFetchRequest, PISearchRequest, PISearchResponse, paginated response wrapper)
4. Write data routes: `POST /api/data/upload` (multipart file, validate MIME type + size), `GET /api/data` (paginated), `GET /api/data/{id}/plot` (with LTTB downsampling + time window + feature selection params), `DELETE /api/data/{id}`. **API routes can be implemented in parallel** via `build-api-route` agents.
5. Write route modules: train (async job via ProcessPoolExecutor, config validated before queuing, cancel support), detect (batch + WebSocket with protocol spec), evaluate, models, experiments (paginated), visualize
6. Write `POST /api/prompt` route — forwards to LLM for tool selection, executes sequentially, returns assembled response. Rate-limited 10 req/min.
7. Write background job manager for async training (ProcessPoolExecutor, status includes progress_pct + error_message)
8. Write PI System connector (`data/pi_connector.py`): PIClient wrapper with tag search, multi-tag fetch + pivot, snapshot, caching config. All PI imports gated behind `try/except ImportError`.
9. Write `configs/pi_connection.yaml` with server defaults
10. Write PI routes (`api/routes/pi.py`): `POST /api/data/pi-fetch`, `POST /api/data/pi-search`, `GET /api/data/pi-snapshot`
11. Write `cli/pi.py` — `sentinel pi search/fetch/snapshot`. Platform check: error on non-Windows with clear message.
12. Build dashboard pages: index (upload + dataset list), upload.html (drag-drop), explore.html (multi-feature Chart.js time series), pi.html (PI server config, tag search with multi-select, time range picker, interval selector, fetch button), prompt.html (chat interface)
13. Add `sentinel serve --host 0.0.0.0 --port 8000` CLI command
14. Write API integration tests with `TestClient` (including upload, upload rejection, prompt, pagination, WebSocket, error paths)
15. Write PI connector unit tests (mock PIClient, verify multi-tag fetch, pivot, schema compliance). Mark with `@pytest.mark.pi`.
16. Write dashboard smoke tests (HTML files served, contain expected elements)
17. `uv run sentinel serve` — verify: upload CSV at `/ui/upload.html`, see time series at `/ui/explore.html`, try prompt at `/ui/prompt.html`

### Phase 9: MCP Server + Ollama Integration

**Goal:** MCP server exposing Sentinel tools + resources, backed by local Nemotron LLM.

**Files:** mcp/server.py, mcp/tools.py, mcp/resources.py, mcp/router.py, mcp/llm_client.py, mcp/prompts.py, cli/mcp_serve.py

**Claude Code steps:**
1. `uv add --group mcp fastmcp httpx` — add MCP deps
2. `ollama pull nvidia/nemotron-3-nano-4b` — pull model (one-time setup)
3. Write Ollama HTTP client (POST /api/generate + /api/chat, streaming support, graceful fallback)
4. Write prompt templates for anomaly reports, model recommendations, and tool selection
5. Write MCP tools: `sentinel_train`, `sentinel_detect`, `sentinel_explain`, `sentinel_analyze`, `sentinel_recommend_model`, `sentinel_upload`, `sentinel_list_datasets`, `sentinel_compare_runs`, `sentinel_delete_run`, `sentinel_export_model`, `sentinel_cancel_job`, `sentinel_pi_fetch`, `sentinel_pi_search`, `sentinel_prompt`. All outputs JSON-serializable. Error responses return `{error, code}` instead of raising.
6. Write MCP resources: `experiments://list`, `models://registry`, `data://datasets`, `data://datasets/{id}`
7. Write prompt router (`mcp/router.py`): Nemotron receives user prompt + tool schemas -> returns tool-call plan -> router executes tools -> assembles response
8. Write MCP server with FastMCP, register tools + resources
9. Add `sentinel mcp-serve` CLI command (stdio + SSE transport options)
10. Write MCP integration tests (including prompt routing end-to-end)
11. Test from Claude Code: add server to MCP config, type "analyze my dataset for anomalies" and verify tool chain executes

### Phase 10: Polish + Docs + Tests

**Goal:** Production-ready codebase.

**Files:** plugins/loader.py, README.md, docs/*, Makefile, tests/integration/*, tests/smoke/*

**Claude Code steps:**
1. Formalize plugin loader with validation
2. Write documentation (quickstart, architecture, models with selection guidance + hyperparameter tuning tips, plugins, API, MCP, troubleshooting FAQ)
3. Write integration and smoke tests (including API + MCP + error paths)
4. `uv run ruff check src/` — lint
5. `uv run pytest --cov=sentinel tests/ -v` — full test suite with coverage
6. `uv run sentinel --help` — verify all commands documented

---

## Git Strategy

Initialize git in Phase 1 step 4. Commit after each logical unit within a phase, not at the end.

**Commit pattern per phase (Phase 1 example):**
```
1. chore: scaffold project (pyproject.toml, .gitignore, Makefile)
2. feat: core abstractions (base_model, registry, config, types, exceptions, device)
3. feat: data pipeline (loaders, validators, preprocessors, synthetic, ingest)
4. feat: statistical models (zscore, isolation_forest, matrix_profile)
5. feat: evaluator + thresholds
6. test: unit + integration tests for Phase 1
```

Each commit is a rollback point. The `/status` skill uses git history to track progress.

---

## Performance Baselines

Establish after Phase 1, extend after each model phase. Run on synthetic data (10k rows x 10 features) on CPU.

| Model | Train Target | Score Target | Memory Target |
| ----- | ------------ | ------------ | ------------- |
| Z-Score | <1s | <10ms/window | <100MB |
| Isolation Forest | <5s | <50ms/batch | <200MB |
| Matrix Profile (10k) | <30s | <1s | <500MB |
| Deep models (AE, LSTM, etc.) | <60s (10 epochs) | <100ms/batch | <1GB |
| Generative (VAE, TranAD, etc.) | <120s (10 epochs) | <100ms/batch | <1GB |
| Ensemble (3 sub-models) | sum of sub-models | <200ms/batch | sum of sub-models |

Test via `@pytest.mark.benchmark` in `tests/benchmarks/`. Not required to pass CI, but regressions are flagged.

---

## Evaluation Rules

**Threshold selection:** Always computed on the **validation set** only. Never on test data.
**Metric reporting:** Always computed on the **test set** only. The test set is never seen during threshold tuning.
**Unsupervised mode:** When `is_anomaly` labels are absent, report score distribution stats (mean, std, p50, p95, p99) and the selected threshold value. Classification metrics (precision, recall, F1) return null.
**Minimum positive samples:** Supervised metrics require at least 5 positive samples in the test set. Below this, warn and return null for AUC-ROC/AUC-PR (unstable with few positives).

---

## Config System Layering

Build the config system incrementally in Phase 1. Each layer is a separate implementation step with its own tests.

```
Layer 1: Python dataclass defaults (works immediately, no YAML needed)
Layer 2: YAML file loading (overrides dataclass defaults)
Layer 3: inherits: base.yaml resolution (deep merge, single level only)
Layer 4: CLI overrides via Typer (highest precedence)
```

Precedence: `dataclass defaults < base.yaml < model.yaml < CLI overrides`

---

## Parallelization Guide

Modules within the same phase can often be implemented in parallel. Use the `/implement` skill which routes to specialized agents.

**Safe to parallelize (no dependencies between them):**
- Statistical models: zscore, isolation_forest, matrix_profile (after core/ is done)
- Deep models within the same tier: AE/LSTM/LSTM-AE/TCN (after deep/__init__.py)
- API routes: data.py, train.py, detect.py, models.py (after app.py + schemas.py)
- MCP tools: each tool is independent (after server.py skeleton)

**Must be sequential (dependency chain):**
- `core/exceptions → core/types → core/registry → core/base_model → core/config`
- `data/validators → data/loaders → data/ingest`
- `api/app.py → api/schemas.py → api/routes/*`
- Each phase depends on the previous phase's core modules

---

## Verification Strategy

| Phase | Command | Expected Outcome |
| ----- | ------- | ---------------- |
| 1 | `uv sync --all-groups` | All dependency groups resolve without conflict |
| 1 | `uv run pytest tests/ -v` | Unit + integration tests pass, statistical models detect synthetic anomalies |
| 1 | `make lint && make test` | Makefile targets work correctly |
| 2 | `uv run sentinel train --config configs/zscore.yaml` | Experiment artifacts in data/experiments/ |
| 2 | `uv run sentinel validate-config --config configs/zscore.yaml` | Config validates without training |
| 2 | `uv run sentinel data list` | Lists ingested datasets |
| 3 | `uv run sentinel train --config configs/lstm_ae.yaml` | LSTM-AE trains, reconstruction scores generated |
| 4 | `uv run sentinel train --config configs/tranad.yaml` | VAE, TranAD, Diffusion produce valid DetectionResult |
| 5 | `uv run sentinel train --config configs/ensemble.yaml` | All 16 models work, ensemble combines normalized scores |
| 6 | `uv run sentinel visualize --run-id <id> --type timeseries` | PNG plot with anomalies highlighted |
| 7 | `uv run sentinel detect --mode streaming --model <path> --data <path>` | Real-time scores from file replay, alerts fired |
| 8 | `uv run sentinel serve &` then `curl localhost:8000/health` | Health check returns `{status: "healthy"}`, dashboard loads at /ui |
| 8 | `curl localhost:8000/api/data?page=1&limit=10` | Paginated response with `{items, total, page}` |
| 9 | `uv run sentinel mcp-serve --transport stdio` + Claude Code MCP tool call | MCP tools respond with JSON, errors return `{error, code}` |
| 10 | `uv run ruff check src/ && uv run pytest --cov tests/` | Clean lint, all tests pass, 85%+ coverage |
