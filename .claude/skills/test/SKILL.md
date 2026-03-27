---
name: test
description: Run pytest with intelligent scope routing
user-invocable: true
argument-hint: "scope (unit, integration, smoke, coverage, models, api, data, or file path)"
allowed-tools: Bash, Read, Glob
context: fork
paths:
  - tests/**
---

# /test $ARGUMENTS

Run Sentinel tests with intelligent scope routing.

## Routing Rules

Parse `$ARGUMENTS` and determine the pytest command:

| Argument | Command |
|----------|---------|
| (empty) | `uv run pytest tests/ -v` |
| `unit` | `uv run pytest tests/unit/ -v` |
| `integration` | `uv run pytest tests/integration/ -v` |
| `smoke` | `uv run pytest tests/smoke/ -v` |
| `coverage` | `uv run pytest tests/ --cov=sentinel --cov-report=term-missing -v` |
| `models` | `uv run pytest tests/unit/models/ -v` |
| `api` | `uv run pytest tests/unit/api/ -v` |
| `data` | `uv run pytest tests/unit/data/ -v` |
| a file path | `uv run pytest {path} -v` |
| a model name | `uv run pytest tests/unit/models/test_{name}.py -v` |

## Instructions

1. Parse `$ARGUMENTS` using the routing table above
2. Verify the test path exists (if specific)
3. Run the pytest command
4. Show results with:
   - Pass/fail count
   - Any failures with tracebacks
   - Coverage summary (if coverage mode)
5. If tests fail, briefly analyze the failure cause
