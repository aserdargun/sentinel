---
name: verify
description: Run phase verification checks from PLAN.md
user-invocable: true
argument-hint: "phase number (1-10)"
allowed-tools: Bash, Read, Glob, Grep
context: fork
agent: Explore
---

# /verify $ARGUMENTS

Run verification checks for Phase $ARGUMENTS of the Sentinel implementation.

## Instructions

1. Read `PLAN.md` and find the Verification Strategy for Phase $ARGUMENTS

2. Run each verification check. Common patterns per phase:

### Phase 1: Foundation
- `uv run python -c "from sentinel.core.base_model import BaseAnomalyDetector"`
- `uv run python -c "from sentinel.core.registry import register_model, get_model"`
- `uv run python -c "from sentinel.core.types import *"`
- `uv run python -c "from sentinel.core.exceptions import SentinelError"`
- `uv run pytest tests/unit/ -v`
- `make lint && make test`

### Phase 2: Training + Tracking + CLI + PI
- `uv run sentinel train --config configs/zscore.yaml`
- `uv run sentinel validate-config --config configs/zscore.yaml`
- `uv run sentinel data list`
- `uv run pytest tests/unit/ -v`

### Phase 3: Classical Deep Learning
- `uv run python -c "from sentinel.core.registry import get_model; m = get_model('autoencoder')"`
- Repeat for: rnn, lstm, gru, lstm_ae, tcn
- `uv run sentinel train --config configs/lstm_ae.yaml`
- `uv run pytest tests/unit/models/ -v`

### Phase 4: Generative / Advanced
- `uv run python -c "from sentinel.core.registry import get_model; m = get_model('vae')"`
- Repeat for: gan, tadgan, tranad, deepar, diffusion
- `uv run sentinel train --config configs/tranad.yaml`

### Phase 5: Ensemble + Thresholding + Drift
- `uv run sentinel train --config configs/ensemble.yaml`
- `uv run pytest tests/ -v -k "ensemble or threshold or drift"`

### Phase 6: Visualization + Interpretability
- `uv run sentinel visualize --run-id <id> --type timeseries`
- `uv run pytest tests/ -v -k "viz or explain"`

### Phase 7: Streaming + Alerts
- `uv run sentinel detect --mode streaming --model <path> --data <path>`
- `uv run pytest tests/ -v -k "streaming or alert"`

### Phase 8: FastAPI + Dashboard
- `uv run sentinel serve &` then `curl localhost:8000/health`
- `curl localhost:8000/api/data?page=1&limit=10`
- `uv run pytest tests/integration/test_api.py -v`

### Phase 9: MCP Server + Ollama
- `uv run sentinel mcp-serve --transport stdio` (quick test)
- `uv run pytest tests/integration/test_mcp.py -v`

### Phase 10: Polish + Docs + Tests
- `uv run ruff check src/ tests/`
- `uv run mypy src/sentinel/`
- `uv run pytest tests/ --cov=sentinel`
- Full test suite green, 85%+ coverage

3. Report results:
   ```
   ## Phase {N} Verification: {PASS/FAIL}

   ✅ {check 1}: passed
   ❌ {check 2}: failed — {reason}
   ...

   Result: {passed}/{total} checks passed
   ```
