"""FastAPI application factory for the Sentinel API."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sentinel import __version__
from sentinel.api.deps import get_tracker
from sentinel.api.jobs import BackgroundJobManager
from sentinel.api.schemas import ErrorResponse, HealthResponse

logger = structlog.get_logger(__name__)

# Module-level singleton so that routes can import it.
job_manager: BackgroundJobManager | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Performs startup tasks:
    * Import ``sentinel.models`` to trigger model registration.
    * Pre-create the experiment tracker (ensures the data directory exists).
    * Initialise the background job manager.

    On shutdown the job manager's process pool is drained.
    """
    global job_manager  # noqa: PLW0603

    # --- startup ---
    logger.info("api.startup")

    # Trigger model registration by importing the models package.
    try:
        import sentinel.models  # noqa: F401

        logger.info("api.models_registered")
    except Exception as exc:
        logger.error("api.model_registration_failed", error=str(exc))

    # Warm the tracker singleton.
    get_tracker()

    # Start the background job manager.
    job_manager = BackgroundJobManager(max_workers=2)
    app.state.job_manager = job_manager

    yield

    # --- shutdown ---
    if job_manager is not None:
        job_manager.shutdown(wait=False)
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    """Create and configure the Sentinel FastAPI application.

    Returns:
        Fully configured FastAPI instance with CORS, static files,
        health endpoint, and all API routes mounted.

    Example::

        app = create_app()
        # uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    app = FastAPI(
        title="Sentinel",
        description="Anomaly Detection Platform API",
        version=__version__,
        lifespan=_lifespan,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )

    # --- CORS (allow all origins for local development) ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Health endpoint ---
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> dict[str, Any]:
        """Health check endpoint.

        Returns ``"healthy"`` when the API is operational.  Ollama
        reachability is checked asynchronously; if it is unreachable
        the overall status is ``"degraded"`` (not unhealthy).
        """
        ollama_status = await _check_ollama()
        overall = "healthy" if ollama_status == "ok" else "degraded"
        return {
            "status": overall,
            "api": "ok",
            "ollama": ollama_status,
            "version": __version__,
        }

    # --- API routes ---
    from sentinel.api.routes import api_router

    app.include_router(api_router, prefix="/api")

    # --- Static dashboard files ---
    dashboard_dir = Path(__file__).resolve().parent / "dashboard"
    if dashboard_dir.is_dir():
        app.mount(
            "/ui",
            StaticFiles(directory=str(dashboard_dir), html=True),
            name="dashboard",
        )
        logger.info("api.dashboard_mounted", path=str(dashboard_dir))
    else:
        logger.warning("api.dashboard_missing", path=str(dashboard_dir))

    logger.info("api.app_created", version=__version__)
    return app


async def _check_ollama() -> str:
    """Probe the Ollama server for availability.

    Returns:
        ``"ok"`` if reachable, ``"unreachable"`` otherwise.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                return "ok"
    except Exception:
        pass
    return "unreachable"
