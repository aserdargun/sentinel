---
name: lint-fix
description: Auto-fix ruff lint issues and format code
user-invocable: true
allowed-tools: Bash, Glob
context: fork
---

# /lint-fix

Auto-fix ruff lint issues and format the entire Sentinel codebase.

## Instructions

1. If `src/sentinel/` does not exist, report "No source files to fix yet."

2. Run auto-fix:
   ```bash
   uv run ruff check --fix src/ tests/ 2>&1
   ```

3. Run auto-format:
   ```bash
   uv run ruff format src/ tests/ 2>&1
   ```

4. Show what was changed (summary of fixes applied)

5. Run lint check again to show remaining unfixable issues:
   ```bash
   uv run ruff check src/ tests/ 2>&1
   ```

6. Show summary:
   ```
   Fixed: {N} issues auto-fixed
   Formatted: {M} files reformatted
   Remaining: {R} issues require manual fix
   ```
