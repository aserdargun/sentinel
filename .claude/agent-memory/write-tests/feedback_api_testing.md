---
name: FastAPI async testing pattern
description: Sentinel API tests use httpx.AsyncClient with ASGITransport (not TestClient); pytest-asyncio is configured with asyncio_mode=auto
type: feedback
---

Sentinel's FastAPI tests use `httpx.AsyncClient` with `ASGITransport` for async testing, NOT `starlette.testclient.TestClient`.

```python
from httpx import ASGITransport, AsyncClient
from sentinel.api.app import create_app

@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

**Why:** The app uses async routes and an async lifespan context manager. TestClient is synchronous and cannot drive async lifespan events properly in this configuration.

**How to apply:** Any test that touches API routes must use this AsyncClient/ASGITransport pattern. The `asyncio_mode = "auto"` setting in pyproject.toml means all async test functions and async fixtures run automatically without `@pytest.mark.asyncio`.
