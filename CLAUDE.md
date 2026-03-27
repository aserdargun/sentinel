# Sentinel — Claude Code Project Instructions

## Project Identity
Sentinel is an anomaly detection platform. 16 ML models, FastAPI API, MCP server, CLI tooling.

## Toolchain — MANDATORY
- Package manager: `uv` (NEVER pip, NEVER python -m venv, NEVER conda)
- Runtime: Python 3.12 (chosen for maximum dependency compatibility)
- DataFrames: Polars ONLY (NEVER pandas, NEVER import pandas)
- Validation: Native Polars expressions (NEVER patito)
- CLI: Typer + Rich
- Logging: structlog with JSON output
- Testing: pytest + pytest-cov + pytest-asyncio
- Linting: ruff
- Type checking: mypy
- Serialization: joblib for statistical models, state_dict + config.json for PyTorch models
- LLM: Ollama (configurable model via `base.yaml` `llm.model`, default `nvidia/nemotron-3-nano-4b`)
- Dependency groups: `deep` (torch, einops), `explain` (shap, umap-learn), `pi` (pipolars, Windows only), `api` (fastapi, uvicorn), `mcp` (fastmcp), `dev` (pytest, ruff, mypy)

## Architecture Rules
1. All code lives under `src/sentinel/` (src-layout)
2. Models implement `BaseAnomalyDetector` ABC from `sentinel.core.base_model`
3. Models are registered via `@register_model("name")` decorator — NEVER import models directly
4. Models are looked up by string name from the registry
5. PyTorch imports ONLY inside `models/deep/*.py` — conditional registration if torch unavailable
6. All data enters as `polars.DataFrame`, converts to `numpy.ndarray` via `.to_numpy()` at model boundary
7. Config inheritance: `base.yaml` < `model.yaml` < CLI overrides (deep merge, single level). Build config incrementally: dataclass defaults → YAML loading → `inherits:` resolution → CLI overrides.
8. Experiment artifacts go to `data/experiments/{run_id}/`
9. Dataset metadata stored in `data/datasets.json`
10. File writes are atomic: write to `{path}.tmp` then `os.rename()`
11. Training jobs use `ProcessPoolExecutor` (not threads). On macOS, explicitly use `fork` start method (`mp_context=get_context('fork')`) since macOS defaults to `spawn`.
12. Seed is set globally: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`
13. Device selection via `core/device.py`: resolves `"auto"` to best available (cuda > mps > cpu)
14. `is_anomaly` column is RESERVED — separated from features before model input
15. Train/val/test splits are ALWAYS chronological (no shuffle)
16. Training modes: `normal_only` (default, filters `is_anomaly==1` rows) for reconstruction models (AE, VAE, GAN, Diffusion); `all_data` for IF, Matrix Profile
17. Evaluator: supervised mode (with `is_anomaly` labels → precision, recall, F1, AUC-ROC, AUC-PR, best-F1 threshold) or unsupervised mode (no labels → score stats, threshold only)
18. Threshold selection on VALIDATION set only. Metric reporting on TEST set only. Never compute thresholds on test data.
19. PI connector is optional and Windows-only — gate all PI imports behind `try/except ImportError`. PI tests use `@pytest.mark.pi` and mock `pipolars`.
20. Each phase must include at least one integration test that verifies the end-to-end pipeline for that phase's modules.

## Coding Conventions
- Type hints on ALL function signatures (params + return)
- Docstrings: Google style on all public functions and classes
- Imports: stdlib, then third-party, then sentinel (isort-compatible, enforced by ruff)
- Error hierarchy: SentinelError base, then ModelNotFoundError, ValidationError, ConfigError
- No bare `except:` — always catch specific exceptions
- f-strings preferred over `.format()` or `%`
- Constants in UPPER_SNAKE_CASE
- Private methods prefixed with `_`
- Max line length: 88 (ruff default)

## Data Format Rules
- First column: `timestamp` parsed as `pl.Datetime("us", "UTC")`
- All timestamps: UTC
- All other columns: numeric (`pl.Float64` or `pl.Int64`)
- `is_anomaly` is RESERVED — separated before model input, used for evaluation only
- Minimum 2 rows required
- No all-NaN feature columns, no constant feature columns (std==0), no duplicate timestamps
- Chronological splits only (70/15/15 train/val/test via positional `.slice()`)
- NaN handling: forward-fill then zero-fill in preprocessors
- Sliding windows for deep models: `create_windows(data, seq_len, stride)` → 3D array `(n_windows, seq_len, n_features)`

## Command Patterns
```bash
# Run anything
uv run sentinel <command>

# Tests
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run pytest tests/ --cov=sentinel -v

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/sentinel/

# Serve
uv run sentinel serve --host 0.0.0.0 --port 8000

# MCP
uv run sentinel mcp-serve --transport stdio
```

## Key File Locations
- Implementation plan: @PLAN.md
- Configs: `configs/*.yaml`
- Core abstractions: `src/sentinel/core/`
- Data pipeline: `src/sentinel/data/`
- Models: `src/sentinel/models/{statistical,deep,ensemble}/`
- Tests: `tests/{unit,integration,smoke}/`
- API: `src/sentinel/api/`
- MCP: `src/sentinel/mcp/`
- CLI: `src/sentinel/cli/`

## Git Strategy
- Commit after each logical unit (core abstractions, data pipeline, models, tests) — not at the end of a phase
- Commit message format: `type: short description` (types: feat, fix, test, chore, refactor)
- Each commit is a rollback point — keep commits atomic and self-contained

## Development Workflow
Use Claude Code skills in this order per module:
1. `/next` — identify what to implement
2. `/scaffold` — create directory structure for current phase
3. `/implement <module>` — routes to the correct specialized agent
4. `/check` — runs lint + typecheck + tests after each module
5. `/verify` — runs phase verification when a phase is complete
6. `/status` — bird's-eye view anytime

Parallelize sibling modules (e.g., 3 statistical models simultaneously) but implement dependencies sequentially (core → data → models → training).

## What NOT to Do
- NEVER use pandas — use Polars
- NEVER use pip — use uv
- NEVER import models directly — use registry lookup
- NEVER use pickle for PyTorch models — use state_dict + config.json
- NEVER use patito for validation — use native Polars expressions
- NEVER shuffle data splits — chronological only
- NEVER skip type hints or docstrings on public functions
- NEVER put torch imports outside of models/deep/
- NEVER create dependency on models/deep/ from core/, data/, or training/
- NEVER use threading for training jobs — use ProcessPoolExecutor
- NEVER write to data/ directories from tests — use tmp_path fixture
- NEVER compute thresholds on test data — use validation set for threshold selection
- NEVER defer integration tests to the end — each phase gets at least one
