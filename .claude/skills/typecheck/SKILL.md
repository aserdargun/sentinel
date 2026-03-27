---
name: typecheck
description: Run mypy type checking on Sentinel source code
user-invocable: true
allowed-tools: Bash, Read, Glob
context: fork
---

# /typecheck

## Current Type Check State
!`if [ -d src/sentinel ]; then uv run mypy src/sentinel/ 2>&1 | tail -30; else echo "No source files to type check yet."; fi`

Run mypy type checking on the Sentinel codebase.

## Instructions

1. If `src/sentinel/` does not exist, report "No source files to type check yet."

2. Using the pre-scanned state above, analyze the results.

3. Group errors by category:
   - **Missing type hints** — missing return type, missing parameter type
   - **Type mismatches** — incompatible types, incompatible return value
   - **Import errors** — cannot find module, no stubs

4. Show top 10 files with most errors

5. Suggest fixes for the most common error patterns

6. Show summary:
   ```
   Type errors: {N}
   Files with errors: {M}
   Most common: {error_type} ({count} occurrences)
   ```
