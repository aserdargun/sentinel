"""Prompt routes: LLM-powered natural language processing (placeholder)."""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from sentinel.api.schemas import PromptRequest, PromptResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/prompt", tags=["prompt"])


@router.post(
    "",
    response_model=PromptResponse,
)
async def process_prompt(body: PromptRequest) -> PromptResponse:
    """Process a natural language prompt via LLM tool selection.

    This is a placeholder endpoint.  Full LLM integration with MCP
    tool routing will be implemented in Phase 9.

    Args:
        body: Request containing the natural language message.

    Returns:
        Response indicating that LLM integration is pending.
    """
    logger.info("prompt.received", message_length=len(body.message))

    return PromptResponse(
        response="LLM integration pending. This endpoint will use Ollama "
        "to select and execute MCP tools based on your prompt.",
        tools_called=[],
    )
