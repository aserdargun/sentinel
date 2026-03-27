"""Prompt routes: LLM-powered natural language processing.

Forwards user prompts to the Ollama LLM via the :class:`PromptRouter`,
which selects and executes MCP tools based on the request.  Falls back
gracefully when Ollama is unavailable.

Rate-limited to 10 requests per minute.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException

from sentinel.api.schemas import PromptRequest, PromptResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/prompt", tags=["prompt"])

# Simple in-memory rate limiter: track timestamps of recent requests.
_RATE_LIMIT = 10  # max requests
_RATE_WINDOW = 60.0  # per N seconds
_request_timestamps: deque[float] = deque()


def _check_rate_limit() -> None:
    """Enforce rate limiting on prompt requests.

    Raises:
        HTTPException: 429 if rate limit is exceeded.
    """
    now = time.monotonic()

    # Evict expired entries
    while _request_timestamps and now - _request_timestamps[0] > _RATE_WINDOW:
        _request_timestamps.popleft()

    if len(_request_timestamps) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded. Maximum {_RATE_LIMIT} prompt requests "
                f"per {int(_RATE_WINDOW)} seconds."
            ),
        )

    _request_timestamps.append(now)


@router.post(
    "",
    response_model=PromptResponse,
)
async def process_prompt(body: PromptRequest) -> PromptResponse:
    """Process a natural language prompt via LLM tool selection.

    Sends the user's message to the Ollama LLM along with available
    Sentinel tool schemas.  The LLM selects appropriate tools which
    are then executed sequentially.  Results are assembled into a
    final response.

    Falls back to a structured message if Ollama is unreachable.

    Args:
        body: Request containing the natural language message.

    Returns:
        Response with assembled text and list of tools called.

    Raises:
        HTTPException: 429 if rate limit is exceeded.
    """
    _check_rate_limit()

    logger.info("prompt.received", message_length=len(body.message))

    # Try to use the MCP router
    router_result = await _route_prompt(body.message)

    return PromptResponse(
        response=router_result.get("response", ""),
        tools_called=router_result.get("tools_called", []),
    )


async def _route_prompt(message: str) -> dict[str, Any]:
    """Route a prompt through the MCP PromptRouter.

    Creates an OllamaClient and PromptRouter on demand.  If MCP
    dependencies or Ollama are unavailable, returns a fallback message.

    Args:
        message: The user's natural language request.

    Returns:
        Dict with ``response`` and ``tools_called``.
    """
    try:
        from sentinel.mcp.llm_client import OllamaClient
        from sentinel.mcp.router import PromptRouter
        from sentinel.mcp.tools import TOOL_DEFINITIONS

        # Build tool map and schemas
        tool_map: dict[str, Any] = {}
        tool_schemas: list[dict[str, Any]] = []
        for defn in TOOL_DEFINITIONS:
            tool_map[defn["name"]] = defn["function"]
            tool_schemas.append(
                {
                    "name": defn["name"],
                    "description": defn["description"],
                    "parameters": defn["parameters"],
                }
            )

        # Use default Ollama settings from config
        ollama = OllamaClient()
        prompt_router = PromptRouter(
            ollama=ollama,
            tools=tool_map,
            tool_schemas=tool_schemas,
        )

        return await prompt_router.route(message)

    except ImportError as exc:
        logger.warning("prompt.mcp_unavailable", error=str(exc))
        return {
            "response": (
                "MCP/LLM dependencies not available. Install with: uv sync --group mcp"
            ),
            "tools_called": [],
        }
    except Exception as exc:
        logger.error("prompt.routing_failed", error=str(exc))
        return {
            "response": f"Prompt processing failed: {exc}",
            "tools_called": [],
        }
