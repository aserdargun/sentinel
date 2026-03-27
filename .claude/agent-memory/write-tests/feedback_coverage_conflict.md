---
name: pytest-cov numpy reimport conflict
description: Running --cov flags alongside the test suite causes a numpy reimport crash in this environment; skip coverage flags and report pass/fail counts instead
type: feedback
---

`uv run pytest ... --cov=...` triggers `ImportError: cannot load module more than once per process` for numpy when conftest.py is present and imports numpy at module level.

**Why:** pytest-cov instruments the process before conftest imports, causing numpy's C-extension to be loaded twice.

**How to apply:** For coverage reporting, skip `--cov` flags and rely on the 125/N pass count. If the user explicitly needs a coverage number, note the limitation and suggest an alternative (e.g., running coverage separately via `uv run coverage run -m pytest ... && uv run coverage report`).
