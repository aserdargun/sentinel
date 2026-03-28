"""FastMCP server: registers Sentinel tools and resources.

Creates a FastMCP instance with all Sentinel tools and resources
registered.  The server exposes:

* **Tools** -- callable operations (train, detect, upload, etc.)
* **Resources** -- read-only data endpoints (experiments, models, datasets)

The server checks Ollama availability at startup and logs a warning
if it is unreachable (degraded mode -- tools still work, LLM-powered
features fall back gracefully).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastmcp import FastMCP

from sentinel.mcp.llm_client import OllamaClient
from sentinel.mcp.router import PromptRouter

logger = structlog.get_logger(__name__)

# Module-level references populated during lifespan.
_ollama_client: OllamaClient | None = None
_prompt_router: PromptRouter | None = None


def get_ollama_client() -> OllamaClient | None:
    """Return the module-level OllamaClient singleton.

    Returns:
        The :class:`OllamaClient` instance, or ``None`` if the server
        has not been started.
    """
    return _ollama_client


def get_prompt_router() -> PromptRouter | None:
    """Return the module-level PromptRouter singleton.

    Returns:
        The :class:`PromptRouter` instance, or ``None`` if the server
        has not been started.
    """
    return _prompt_router


def create_mcp_server(
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "nemotron-3-nano:4b",
    ollama_timeout: int = 30,
) -> FastMCP:
    """Create and configure the Sentinel MCP server.

    Registers all tools and resources with a :class:`FastMCP` instance.
    The server's lifespan handler initialises the Ollama client and
    checks connectivity.

    Args:
        ollama_url: Base URL for the Ollama server.
        ollama_model: LLM model name for generation.
        ollama_timeout: HTTP timeout in seconds for Ollama requests.

    Returns:
        Configured :class:`FastMCP` server ready to be run.

    Example::

        server = create_mcp_server()
        server.run(transport="stdio")
    """

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncGenerator[None, None]:
        """Server lifespan: initialise Ollama client and check health."""
        global _ollama_client, _prompt_router  # noqa: PLW0603

        logger.info(
            "mcp.startup",
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )

        # Initialise Ollama client
        _ollama_client = OllamaClient(
            base_url=ollama_url,
            model=ollama_model,
            timeout=ollama_timeout,
        )

        # Check Ollama availability
        available = await _ollama_client.is_available()
        if available:
            logger.info("mcp.ollama_connected", model=ollama_model)
        else:
            logger.warning(
                "mcp.ollama_unavailable",
                url=ollama_url,
                message="LLM-powered features will fall back to structured data.",
            )

        # Trigger model registration
        try:
            import sentinel.models  # noqa: F401

            logger.info("mcp.models_registered")
        except Exception as exc:
            logger.error("mcp.model_registration_failed", error=str(exc))

        # Build tool map and router
        tool_map, tool_schemas = _build_tool_map()
        _prompt_router = PromptRouter(
            ollama=_ollama_client,
            tools=tool_map,
            tool_schemas=tool_schemas,
        )

        yield

        logger.info("mcp.shutdown")

    # Create the FastMCP server
    mcp = FastMCP(
        name="sentinel",
        instructions=(
            "Sentinel is an anomaly detection platform with 16 ML models. "
            "Use the available tools to train models, detect anomalies, "
            "manage datasets, and analyse results."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- Register tools ---
    _register_tools(mcp)

    # --- Register resources ---
    _register_resources(mcp)

    return mcp


# ------------------------------------------------------------------
# Tool registration
# ------------------------------------------------------------------


def _register_tools(mcp: FastMCP) -> None:
    """Register all Sentinel tools with the FastMCP server.

    Args:
        mcp: The FastMCP server instance.
    """
    from sentinel.mcp import tools as t

    @mcp.tool(
        name="sentinel_train",
        description="Train an anomaly detection model from a YAML config file.",
    )
    def sentinel_train(config_path: str) -> dict[str, Any]:
        """Train model from config.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Training result with run_id, metrics, duration.
        """
        return t.sentinel_train(config_path)

    @mcp.tool(
        name="sentinel_detect",
        description="Run batch anomaly detection on data with a saved model.",
    )
    def sentinel_detect(model_path: str, data_path: str) -> dict[str, Any]:
        """Detect anomalies in data.

        Args:
            model_path: Path to saved model directory.
            data_path: Path to data file (CSV/Parquet).

        Returns:
            Detection results with scores, labels, threshold.
        """
        return t.sentinel_detect(model_path, data_path)

    @mcp.tool(
        name="sentinel_list_models",
        description="List all registered anomaly detection models.",
    )
    def sentinel_list_models() -> dict[str, Any]:
        """List registered models.

        Returns:
            Dict with models list.
        """
        return t.sentinel_list_models()

    @mcp.tool(
        name="sentinel_list_datasets",
        description="List all uploaded datasets.",
    )
    def sentinel_list_datasets() -> dict[str, Any]:
        """List datasets.

        Returns:
            Dict with datasets list.
        """
        return t.sentinel_list_datasets()

    @mcp.tool(
        name="sentinel_upload",
        description="Ingest a CSV or Parquet file into the Sentinel data store.",
    )
    def sentinel_upload(file_path: str) -> dict[str, Any]:
        """Upload data file.

        Args:
            file_path: Path to CSV/Parquet file.

        Returns:
            Upload result with dataset_id, shape, features.
        """
        return t.sentinel_upload(file_path)

    @mcp.tool(
        name="sentinel_compare_runs",
        description="Compare metrics across multiple experiment runs.",
    )
    def sentinel_compare_runs(run_ids: list[str]) -> dict[str, Any]:
        """Compare experiment runs.

        Args:
            run_ids: List of run identifiers.

        Returns:
            Comparison data with per-run metrics.
        """
        return t.sentinel_compare_runs(run_ids)

    @mcp.tool(
        name="sentinel_analyze",
        description=(
            "Generate a natural-language anomaly analysis report. "
            "Uses LLM if available, falls back to structured data."
        ),
    )
    async def sentinel_analyze(run_id: str) -> dict[str, Any]:
        """Analyse an experiment run.

        Args:
            run_id: Experiment run identifier.

        Returns:
            Analysis report with metrics and narrative.
        """
        return await t.sentinel_analyze(run_id, ollama_client=_ollama_client)

    @mcp.tool(
        name="sentinel_recommend_model",
        description=(
            "Recommend anomaly detection models for a dataset "
            "based on its characteristics."
        ),
    )
    async def sentinel_recommend_model(data_path: str) -> dict[str, Any]:
        """Recommend models for data.

        Args:
            data_path: Path to data file to analyse.

        Returns:
            Recommendations with data summary.
        """
        return await t.sentinel_recommend_model(data_path, ollama_client=_ollama_client)

    @mcp.tool(
        name="sentinel_delete_run",
        description="Delete an experiment run and its artifacts.",
    )
    def sentinel_delete_run(run_id: str) -> dict[str, Any]:
        """Delete experiment run.

        Args:
            run_id: Run identifier to delete.

        Returns:
            Deletion confirmation.
        """
        return t.sentinel_delete_run(run_id)

    @mcp.tool(
        name="sentinel_export_model",
        description="Export a trained model from an experiment run.",
    )
    def sentinel_export_model(run_id: str, format: str = "native") -> dict[str, Any]:
        """Export model.

        Args:
            run_id: Run identifier.
            format: Export format ('native' or 'onnx').

        Returns:
            Export path and metadata.
        """
        return t.sentinel_export_model(run_id, format)

    @mcp.tool(
        name="sentinel_prompt",
        description=(
            "Process a natural language prompt. The LLM selects "
            "and executes appropriate tools automatically."
        ),
    )
    async def sentinel_prompt(message: str) -> dict[str, Any]:
        """Process natural language prompt.

        Args:
            message: User's natural language request.

        Returns:
            Assembled response with tool results.
        """
        if _prompt_router is None:
            return {
                "response": "MCP server not fully initialized.",
                "tools_called": [],
            }
        return await _prompt_router.route(message)


# ------------------------------------------------------------------
# Resource registration
# ------------------------------------------------------------------


def _register_resources(mcp: FastMCP) -> None:
    """Register all Sentinel resources with the FastMCP server.

    Args:
        mcp: The FastMCP server instance.
    """
    from sentinel.mcp import resources as r

    @mcp.resource(
        "experiments://list",
        name="experiments_list",
        description="All experiment runs with metadata and key metrics.",
    )
    def experiments_list() -> str:
        """List all experiment runs.

        Returns:
            JSON string of experiment run summaries.
        """
        data = r.experiments_list()
        return json.dumps(data, default=str)

    @mcp.resource(
        "experiments://{run_id}",
        name="experiment_detail",
        description="Full details for a single experiment run.",
    )
    def experiment_detail(run_id: str) -> str:
        """Get experiment run details.

        Args:
            run_id: Experiment run identifier.

        Returns:
            JSON string of run details.
        """
        data = r.experiment_detail(run_id)
        return json.dumps(data, default=str)

    @mcp.resource(
        "models://registry",
        name="models_registry",
        description="All registered models with metadata and parameter schemas.",
    )
    def models_registry() -> str:
        """List registered models.

        Returns:
            JSON string of model registry.
        """
        data = r.models_registry()
        return json.dumps(data, default=str)

    @mcp.resource(
        "data://datasets",
        name="datasets_list",
        description="All uploaded datasets with metadata.",
    )
    def datasets_list() -> str:
        """List all datasets.

        Returns:
            JSON string of dataset summaries.
        """
        data = r.datasets_list()
        return json.dumps(data, default=str)

    @mcp.resource(
        "data://datasets/{dataset_id}",
        name="dataset_detail",
        description="Detailed information about a single dataset including statistics.",
    )
    def dataset_detail(dataset_id: str) -> str:
        """Get dataset details.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            JSON string of dataset details with stats.
        """
        data = r.dataset_detail(dataset_id)
        return json.dumps(data, default=str)

    @mcp.resource(
        "data://datasets/{dataset_id}/preview",
        name="dataset_preview",
        description="First 20 rows of a dataset as structured JSON.",
    )
    def dataset_preview(dataset_id: str) -> str:
        """Preview dataset rows.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            JSON string with first N rows of the dataset.
        """
        data = r.dataset_preview(dataset_id)
        return json.dumps(data, default=str)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _build_tool_map() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the tool function map and schema list for the prompt router.

    Returns:
        Tuple of (tool_map, tool_schemas) where tool_map maps tool
        names to callables and tool_schemas is a list of tool metadata
        dicts for the LLM prompt.
    """
    from sentinel.mcp.tools import TOOL_DEFINITIONS

    tool_map: dict[str, Any] = {}
    tool_schemas: list[dict[str, Any]] = []

    for defn in TOOL_DEFINITIONS:
        name = defn["name"]
        tool_map[name] = defn["function"]
        tool_schemas.append(
            {
                "name": name,
                "description": defn["description"],
                "parameters": defn["parameters"],
            }
        )

    return tool_map, tool_schemas
