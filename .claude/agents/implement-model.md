---
name: implement-model
description: Implements anomaly detection models following BaseAnomalyDetector ABC with registry pattern
model: opus
effort: max
maxTurns: 40
permissionMode: acceptEdits
memory: project
skills:
  - model-reference
  - phase-context
---

# Implement Anomaly Detection Model

You are implementing an anomaly detection model for the Sentinel platform.

## Before You Start

1. Read `CLAUDE.md` for project rules
2. Read `PLAN.md` for the full model specification (Model Zoo table, Phase details)
3. Use the `model-reference` skill to load BaseAnomalyDetector ABC, registry, types, and existing model examples
4. Use the `phase-context` skill to understand which phase this model belongs to

## What You Must Create

For each model, create these files:

### 1. Model Implementation
**File:** `src/sentinel/models/{category}/{model_name}.py`

Where `{category}` is `statistical`, `deep`, or `ensemble`.

Requirements:
- Import and subclass `BaseAnomalyDetector` from `sentinel.core.base_model`
- Decorate class with `@register_model("{model_name}")`
- Implement ALL abstract methods: `fit()`, `score()`, `save()`, `load()`, `get_params()`
- Type hints on every method signature
- Google-style docstring on the class and all public methods
- For deep models: all torch imports inside this file only, conditional registration:
  ```python
  try:
      import torch
      # ... model implementation ...
      HAS_TORCH = True
  except ImportError:
      HAS_TORCH = False
  ```
- Data enters as `polars.DataFrame`, convert to numpy at model boundary via `.to_numpy()`
- Separate `is_anomaly` column before processing (it's reserved for evaluation)
- For deep models using sequences: use `create_windows(data, seq_len, stride)` for 3D arrays
- Seed management: respect the global seed set in training pipeline
- Device handling: accept `device` param, resolve via `sentinel.core.device.resolve_device()`

### 2. Model Config
**File:** `configs/{model_name}.yaml`

Requirements:
- Start with `inherits: base.yaml`
- Include all model-specific hyperparameters with documented defaults
- Add comments explaining each parameter

### 3. Update `__init__.py`
**File:** `src/sentinel/models/{category}/__init__.py`

Add the import so the module is discoverable:
```python
from sentinel.models.{category}.{model_name} import *  # noqa: F403
```

### 4. Unit Test
**File:** `tests/unit/models/test_{model_name}.py`

Requirements:
- Test `fit()` with synthetic data
- Test `score()` returns correct shape
- Test `save()` / `load()` round-trip
- Test `get_params()` returns expected keys
- Test with edge cases (single feature, minimal rows)
- Use `tmp_path` fixture for file operations
- For deep models: skip if torch unavailable (`pytest.importorskip("torch")`)

## Style Rules
- Max line length: 88
- f-strings for string formatting
- Error hierarchy: raise `SentinelError` subclasses
- Constants in UPPER_SNAKE_CASE
- Private methods prefixed with `_`

## After Implementation
1. Run `uv run ruff check --fix` on all created files
2. Run `uv run ruff format` on all created files
3. Run `uv run pytest tests/unit/models/test_{model_name}.py -v`
4. Fix any failures before finishing
