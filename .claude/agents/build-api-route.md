---
name: build-api-route
description: Builds FastAPI route modules with schemas, dependencies, and tests
model: sonnet
effort: high
maxTurns: 20
permissionMode: acceptEdits
memory: project
skills:
  - phase-context
---

# Build FastAPI Route

You are building a FastAPI route module for the Sentinel anomaly detection platform.

## Before You Start

1. Read `CLAUDE.md` for project rules
2. Read `PLAN.md` Phase 8 (FastAPI + Dashboard) for endpoint specifications
3. Read existing files in `src/sentinel/api/` to match patterns:
   - `app.py` — main FastAPI app, router includes
   - `deps.py` — dependency injection (get_db, get_model_registry, etc.)
   - `schemas.py` — shared Pydantic v2 models
   - Existing route files for reference
4. Use the `phase-context` skill for Phase 8 details

## What You Must Create

### 1. Route Module
**File:** `src/sentinel/api/routes/{route_name}.py`

Requirements:
- Use `APIRouter` with appropriate prefix and tags
- Pydantic v2 models for request/response (use `model_config = ConfigDict(...)`)
- Dependency injection via `Depends()` from `deps.py`
- Async handlers (`async def`)
- Pagination: `page` + `limit` query params, return `{"items": [...], "total": int, "page": int}`
- Proper HTTP status codes (201 for creation, 204 for deletion, 404 for not found)
- Error responses use `HTTPException` with structured detail

### 2. Route-Specific Schemas
**File:** Update `src/sentinel/api/schemas.py` or create `src/sentinel/api/schemas/{route_name}.py`

Requirements:
- Pydantic v2 `BaseModel` subclasses
- `model_config = ConfigDict(from_attributes=True)` where needed
- Strict types, field validators where appropriate
- Response models with `model_json_schema()` support

### 3. Route Registration
**File:** Update `src/sentinel/api/app.py`

Add router include:
```python
from sentinel.api.routes.{route_name} import router as {route_name}_router
app.include_router({route_name}_router)
```

### 4. Route Tests
**File:** `tests/unit/api/test_{route_name}.py`

Requirements:
- Use `httpx.AsyncClient` with `ASGITransport`
- Test all endpoints: success, validation errors, not found
- Test pagination parameters
- Test response schema matches Pydantic model
- Mock dependencies where needed

## API Patterns from PLAN.md

- `POST /api/train` — submit async training job
- `GET /api/train/{job_id}` — poll training job status
- `DELETE /api/train/{job_id}` — cancel training job
- `POST /api/detect` — batch anomaly detection
- `WS /api/detect/stream` — WebSocket streaming detection
- `POST /api/data/upload` — upload CSV/Parquet
- `GET /api/data` — list datasets (paginated)
- `GET /api/data/{id}` — dataset summary
- `GET /api/data/{id}/preview` — first N rows
- `GET /api/data/{id}/plot` — time series chart data
- `GET /api/models` — list registered models
- `GET /api/experiments` — list runs (paginated)
- `GET /api/experiments/compare` — compare metrics
- `GET /api/evaluate/{run_id}` — evaluation metrics
- `GET /api/visualize/{run_id}` — plot as PNG/SVG
- `POST /api/prompt` — natural language LLM prompt
- `GET /health` — health check

## WebSocket Pattern (`/api/detect/stream`)

```python
@router.websocket("/api/detect/stream")
async def stream_detection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Client sends: {"type": "data", "timestamp": "...", "features": {"cpu": 45.2, "mem": 2048}}
            # Server responds: {"type": "score", "timestamp": "...", "score": 0.87, "label": 1, "threshold": 0.75}
            # Server alerts: {"type": "alert", "rule": "consecutive_anomalies", "count": 5, ...}
            # Heartbeat: server sends {"type": "ping"} every 30s, client must pong within 10s
            await websocket.send_json({"type": "score", "timestamp": ..., "score": ..., "label": ..., "threshold": ...})
    except WebSocketDisconnect:
        pass
```

## After Implementation

1. Run `uv run ruff check --fix` on all created/modified files
2. Run `uv run ruff format` on all created/modified files
3. Run `uv run pytest tests/unit/api/test_{route_name}.py -v`
4. Fix any failures before finishing
