---
name: check
description: Run all quality checks — lint, typecheck, and tests in one go
user-invocable: true
allowed-tools: Bash, Glob
context: fork
---

# /check

Run all quality checks on the Sentinel codebase in sequence.

## Instructions

1. Check if `src/sentinel/` exists. If not, report "No source files yet."

2. Run all three checks and collect results:

   **Step 1 — Lint:**
   ```bash
   uv run ruff check src/ tests/ 2>&1
   ```

   **Step 2 — Format check:**
   ```bash
   uv run ruff format --check src/ tests/ 2>&1
   ```

   **Step 3 — Type check:**
   ```bash
   uv run mypy src/sentinel/ 2>&1
   ```

   **Step 4 — Tests:**
   ```bash
   uv run pytest tests/ -v --tb=short 2>&1
   ```

3. Show a combined summary:

```
## Quality Check Results

| Check      | Result | Details          |
|------------|--------|------------------|
| Lint       | PASS/FAIL | {N} errors    |
| Format     | PASS/FAIL | {N} files     |
| Types      | PASS/FAIL | {N} errors    |
| Tests      | PASS/FAIL | {passed}/{total} |

{If any failures, show the top 3 most important issues to fix first}
```
