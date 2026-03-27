---
name: lint
description: Run ruff linting and format checking
user-invocable: true
allowed-tools: Bash, Glob
context: fork
---

# /lint

## Current Lint State
!`if [ -d src/sentinel ]; then uv run ruff check --statistics src/ tests/ 2>&1 | tail -20; echo "---FORMAT---"; uv run ruff format --check src/ tests/ 2>&1 | tail -5; else echo "No source files to lint yet."; fi`

Run linting and format checking on the Sentinel codebase.

## Instructions

1. If `src/sentinel/` does not exist, report "No source files to lint yet."

2. Using the pre-scanned lint state above, analyze the results.

3. Show results grouped by:
   - **Errors by rule**: group violations by ruff rule code (E501, F401, etc.)
   - **Files with most issues**: top 5 files by violation count
   - **Auto-fixable**: how many issues can be fixed with `--fix`

4. If there are auto-fixable issues, suggest:
   ```
   Run `uv run ruff check --fix src/ tests/` to auto-fix {N} issues
   Run `uv run ruff format src/ tests/` to auto-format
   ```

5. Show summary:
   ```
   Lint: {N} errors, {M} warnings
   Format: {N} files need formatting
   ```
