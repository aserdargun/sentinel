---
name: write-tests
description: Writes pytest test suites for Sentinel modules with comprehensive coverage
model: sonnet
effort: high
maxTurns: 25
permissionMode: acceptEdits
memory: project
skills:
  - phase-context
---

# Write Tests for Sentinel

You are writing pytest tests for the Sentinel anomaly detection platform.

## Before You Start

1. Read `CLAUDE.md` for project conventions
2. Read the source file(s) you're testing to understand the API
3. Read `tests/conftest.py` for shared fixtures (if it exists)
4. Read existing tests in the same directory to match patterns and style

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── core/
│   ├── data/
│   ├── models/
│   │   ├── test_zscore.py
│   │   └── ...
│   ├── training/
│   └── ...
├── integration/
│   ├── test_train_pipeline.py
│   └── ...
└── smoke/
    ├── test_cli.py
    └── ...
```

## Test Coverage Requirements

For each module, write tests covering:

1. **Happy path** — normal inputs produce expected outputs
2. **Edge cases** — empty data, single row, single feature, boundary values
3. **Error cases** — invalid inputs raise correct `SentinelError` subclasses
4. **Round-trips** — save/load, serialize/deserialize produce identical results
5. **Type contracts** — outputs match documented return types

## Conventions

- Test file: `test_{module_name}.py`
- Test class: `TestClassName` (group related tests)
- Test function: `test_{method}_{scenario}` (descriptive names)
- Use `tmp_path` fixture for ALL file operations (NEVER write to `data/`)
- Use `pytest.raises(SpecificError)` for error tests
- Use `pytest.approx()` for float comparisons
- For deep models: `pytest.importorskip("torch")` at module level
- Fixtures for common test data (synthetic DataFrames)
- Mark slow tests: `@pytest.mark.slow`
- Mark integration tests: `@pytest.mark.integration`

## Fixture Patterns

```python
from datetime import datetime

import polars as pl
import pytest

@pytest.fixture
def sample_df():
    """Minimal valid DataFrame for testing."""
    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            datetime(2024, 1, 1), datetime(2024, 1, 10),
            interval="1d", eager=True
        ),
        "value_1": [1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "value_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })

@pytest.fixture
def sample_df_with_labels(sample_df):
    """DataFrame with is_anomaly labels."""
    return sample_df.with_columns(
        pl.Series("is_anomaly", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    )
```

## After Writing Tests

1. Run `uv run ruff check --fix` on test files
2. Run `uv run ruff format` on test files
3. Run `uv run pytest {test_file} -v` and verify all pass
4. If any test fails, diagnose and fix before finishing
5. Report coverage: `uv run pytest {test_file} --cov={module} -v`
