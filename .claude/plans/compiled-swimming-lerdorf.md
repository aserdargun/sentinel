# Sentinel — System Architecture Assessment

## Context
Comprehensive architectural audit of the Sentinel anomaly detection platform by a system architect agent. The assessment evaluates 8 dimensions: architecture quality, code quality, data pipeline, model zoo, API/MCP design, testing, production readiness, and technical debt.

---

## Summary Ratings

| Dimension | Rating |
|-----------|--------|
| Architecture Quality | **Strong** |
| Code Quality & Patterns | **Strong** |
| Data Pipeline Robustness | **Strong** |
| Model Zoo (16 algorithms) | **Strong** |
| API & MCP Design | **Adequate** |
| Testing Strategy | **Adequate** |
| Production Readiness | **Adequate** |
| Technical Debt & Risks | **Adequate** |

---

## 1. Architecture Quality — Strong

- **Clean layering**: `core/` → `data/` → `models/` → `training/` → `api/mcp/cli`. No circular dependencies.
- **Registry pattern well-isolated**: single `_REGISTRY` dict, decorator-based, entry-point plugin discovery.
- **Minor**: `plugins/loader.py` accesses private `_REGISTRY` directly instead of using public API.
- **Minor**: `import sentinel.models` triggers registration at 7 call sites — a single `ensure_models_registered()` would be cleaner.

## 2. Code Quality & Patterns — Strong

- **BaseAnomalyDetector ABC**: well-designed with 5 abstract + 1 concrete method.
- **Config system**: proper inheritance, deep merge, dot-notation overrides.
- **Missing validation**: `SplitConfig` doesn't validate ratios sum to 1.0 — user could set `train: 0.8, val: 0.5, test: 0.5`.
- **Logging inconsistency**: `tadgan.py`, `tranad.py`, `hybrid.py` use stdlib `logging` instead of mandated `structlog`.
- **Repeated boilerplate**: `_check_fitted()`, `save()`, `load()` are nearly identical across all 16 models (~80-100 lines each).

## 3. Data Pipeline — Strong

- **Validation comprehensive**: all PLAN.md checks implemented (min 2 rows, UTC, sorted, unique timestamps, no all-NaN/constant).
- **Atomic writes correct**: `.tmp` + `os.rename()` pattern throughout.
- **PI connector exemplary**: imports gated, retry with exponential backoff, context manager, UTC normalization.
- **Polars→numpy boundary clean**: single `to_numpy()` call after all Polars operations complete.
- **Minor**: `max_features: 500` runtime limit defined but not enforced in validators.

## 4. Model Zoo — Strong

- **All 6 generative model loss functions verified correct** against PLAN.md and original papers (VAE/ELBO, GANomaly, TadGAN/WGAN-GP, TranAD self-conditioned adversarial, DeepAR Gaussian NLL, DDPM).
- **Training modes correct**: `normal_only` filters anomalies for reconstruction models.
- **Serialization safe**: all `torch.load()` calls use `weights_only=True`.
- **Ensemble handles partial failure** with weight renormalization.
- **Bug**: DeepAR stores resolved device string (`"cuda"`) instead of original `"auto"` — breaks cross-device model loading. Other models store `device_str` correctly.

## 5. API & MCP Design — Adequate

- **REST API functionally complete**: 17+ endpoints covering train, detect, upload, evaluate, experiments, prompt.
- **MCP well-structured**: 10 tools, 5 resources, graceful Ollama fallback.
- **Critical: Path traversal vulnerability** — `DetectRequest` and `TrainRequest` accept raw file paths without sanitization. No `resolve()` or base-directory check.
- **CORS wildcard** (`allow_origins=["*"]`) with credentials enabled.
- **Rate limiting only on `/api/prompt`** — training/detection endpoints have none.
- **WebSocket endpoint** for streaming detection specified in PLAN.md but not implemented.

## 6. Testing Strategy — Adequate

- **959 tests** across 35 files (27 PLAN.md-specified + extras).
- **Good isolation**: all tests use `tmp_path`, PI tests marked `@pytest.mark.pi`.
- **Gap**: No individual unit tests for deep learning models (VAE, TranAD, etc.) — only integration coverage.
- **Gap**: No security tests (path traversal, upload limits, rate limiting).
- **Gap**: No concurrent access tests for job manager or atomic writes.

## 7. Production Readiness — Adequate

- **Graceful degradation works**: missing torch, pipolars, Ollama all handled correctly.
- **Bug**: `jobs.py` uses `"fork"` start method on all platforms — should use `"fork"` only on macOS, `"spawn"` on Linux.
- **Training timeout defined (3600s) but never enforced** — no timeout mechanism in Trainer or ProcessPoolExecutor.
- **Job state in-memory only** — lost on process restart.

## 8. Technical Debt & Risks — Adequate

- **`datasets.json` race condition**: read-modify-write without locking under concurrent uploads.
- **ZScore scoring loop**: Python for-loop over samples instead of vectorized numpy.
- **Global singletons**: `job_manager` in `api/app.py`, `_ollama_client` in `mcp/server.py`.
- **In-memory rate limiter**: process-local, won't work behind load balancer.

---

## Prioritized Recommendations

### Critical
1. **Fix path traversal** in API routes (`api/routes/detect.py`, `train.py`) — resolve paths and validate against allowed base directory

### High Priority
2. **Fix DeepAR device management** — store `device_str` not resolved device (`deepar.py:176`)
3. **Fix fork/spawn platform logic** in `BackgroundJobManager` (`jobs.py:106`) — `"fork"` on Darwin only
4. **Add split ratio validation** to `SplitConfig` (`config.py`) — ratios must sum to ~1.0
5. **Implement training timeout** enforcement — `future.result(timeout=...)` in job manager
6. **Add rate limiting** to training/detection endpoints

### Medium Priority
7. Extract common model boilerplate (`_check_fitted`, `save`, `load`) into base class helpers
8. Replace `datasets.json` with file-locking or SQLite for concurrent safety
9. Add deep learning model unit tests (VAE, TranAD, etc.)
10. Standardize logging to `structlog` in `tadgan.py`, `tranad.py`, `hybrid.py`
11. Implement WebSocket endpoint for streaming detection (PLAN.md spec)

### Low Priority
12. Vectorize ZScore scoring with `numpy.lib.stride_tricks.sliding_window_view`
13. Enforce `max_features` limit in validators
14. Add observability (Prometheus metrics, distributed tracing)
15. Restrict CORS origins for production deployment

---

## Critical Files for Fixes

| File | Issue |
|------|-------|
| `src/sentinel/api/routes/detect.py` | Path traversal |
| `src/sentinel/api/routes/train.py` | Path traversal |
| `src/sentinel/api/jobs.py` | Fork/spawn platform + timeout |
| `src/sentinel/models/deep/deepar.py` | Device string storage |
| `src/sentinel/core/config.py` | Split ratio validation |
| `src/sentinel/core/base_model.py` | Extract common helpers |
| `src/sentinel/models/deep/tadgan.py` | stdlib logging → structlog |
| `src/sentinel/models/deep/tranad.py` | stdlib logging → structlog |
| `src/sentinel/models/ensemble/hybrid.py` | stdlib logging → structlog |
