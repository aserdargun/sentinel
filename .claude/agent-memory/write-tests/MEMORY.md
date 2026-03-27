# Write-Tests Agent Memory

- [pytest-cov numpy reimport conflict](feedback_coverage_conflict.md) — `--cov` flags crash with numpy reimport; skip them, report pass count instead
- [FastAPI async testing pattern](feedback_api_testing.md) — use `httpx.AsyncClient` + `ASGITransport`, not `TestClient`; `asyncio_mode=auto` in pyproject.toml
