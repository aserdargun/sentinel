---
name: implement-module
description: Implements non-model Sentinel modules (data pipeline, training, CLI, tracking, viz, streaming, explain)
model: opus
effort: max
maxTurns: 40
permissionMode: acceptEdits
memory: project
skills:
  - phase-context
---

# Implement Sentinel Module

You are implementing a module for the Sentinel anomaly detection platform.

## Before You Start

1. Read `CLAUDE.md` for project rules
2. Read `PLAN.md` for the full specification of this module
3. Use the `phase-context` skill to load the relevant phase details
4. Read existing files in the same package to understand patterns and imports

## Package-Specific Guidelines

### `sentinel.core` (Phase 1)
- Base abstractions: `BaseAnomalyDetector` ABC, model registry, error hierarchy, type definitions
- `device.py`: resolve `"auto"` → best available (cuda > mps > cpu)
- Keep zero dependencies on other sentinel packages
- Everything here is imported by everything else — stability is critical

### `sentinel.data` (Phase 1–2)
- Data loading, validation, preprocessing, splitting (Phase 1)
- Feature engineering, PI connector (Phase 2)
- Polars ONLY — never pandas
- Validate: min 2 rows, no all-NaN cols, no constant cols, no duplicate timestamps
- `is_anomaly` column: separate before returning features
- Splits: chronological only (70/15/15) via positional `.slice()`
- NaN: forward-fill then zero-fill
- `create_windows(data, seq_len, stride)` → 3D numpy array
- Dataset metadata registry: `data/datasets.json`

### `sentinel.training` (Phase 1–2)
- Evaluator + thresholds (Phase 1)
- Trainer, callbacks, schedulers (Phase 2)
- Config inheritance: `base.yaml` < `model.yaml` < CLI overrides
- Use `ProcessPoolExecutor` for parallel training (not threads)
- Atomic file writes for artifacts
- Seed management: set `random.seed()`, `np.random.seed()`, `torch.manual_seed()`
- LR schedulers: `training/schedulers.py`
- Two-mode evaluation: supervised (with labels) and unsupervised (without)

### `sentinel.cli` (Phase 2)
- Typer + Rich for CLI
- Commands: ingest, generate, data, train, evaluate, detect, export, validate-config, visualize, serve, mcp-serve, pi
- All commands use `uv run sentinel <command>`
- Progress bars with Rich for long operations
- Structured logging via structlog

### `sentinel.tracking` (Phase 2)
- Experiment tracking: run metadata, metrics, artifacts
- Store in `data/experiments/{run_id}/`
- Comparison utilities for multiple runs

### `sentinel.viz` (Phase 6)
- Matplotlib-based visualizations (not Plotly)
- Time series with anomaly regions shaded red
- Reconstruction error: original vs reconstructed overlay + heatmap
- Latent space: t-SNE / UMAP projection colored by score

### `sentinel.streaming` (Phase 7)
- `OnlineDetector` with sliding window buffer
- Uses standard `model.score()` (not score_single)
- Stream simulator consumes async generators from `data/streaming.py`
- Alert rules: threshold breach, consecutive anomalies, rate-of-change

### `sentinel.explain` (Phase 6)
- SHAP KernelExplainer (top-k features, default k=10; TreeExplainer for IF)
- Per-feature reconstruction error decomposition with ranking
- Graceful fallback if SHAP unavailable

## Universal Rules

- Type hints on ALL function signatures
- Google-style docstrings on all public functions/classes
- Error hierarchy: `SentinelError` → specific errors
- Atomic file writes: `{path}.tmp` → `os.rename()`
- No bare `except:` — catch specific exceptions
- f-strings for formatting
- Max line length: 88

## After Implementation

1. Run `uv run ruff check --fix` on all created files
2. Run `uv run ruff format` on all created files
3. Run relevant tests if they exist
4. Verify imports work: `uv run python -c "from sentinel.{package} import ..."`
