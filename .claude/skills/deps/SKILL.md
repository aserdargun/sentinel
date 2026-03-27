---
name: deps
description: Check and sync project dependencies
user-invocable: true
argument-hint: "[group] (e.g., deps deep, deps all, deps check)"
allowed-tools: Bash, Read
context: fork
---

# /deps $ARGUMENTS

Manage Sentinel project dependencies.

## Instructions

Parse `$ARGUMENTS` and route:

| Argument | Command |
|----------|---------|
| (empty) | `uv sync` — sync default deps |
| `all` | `uv sync --all-groups` — sync everything including deep, api, mcp, explain, pi, dev |
| `deep` | `uv sync --group deep` — add PyTorch + einops |
| `api` | `uv sync --group api` — add FastAPI + uvicorn |
| `mcp` | `uv sync --group mcp` — add FastMCP |
| `explain` | `uv sync --group explain` — add SHAP + UMAP |
| `dev` | `uv sync --group dev` — add pytest, ruff, mypy |
| `check` | `uv pip list 2>/dev/null \|\| uv run pip list` — show installed packages |
| `outdated` | `uv lock --check` — check if lock file is current |

After running, show:
- Success/failure status
- Number of packages installed
- Any version conflicts or warnings
