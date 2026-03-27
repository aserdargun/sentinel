"""Prompt router: parses user prompts via LLM and executes MCP tools.

The :class:`PromptRouter` sends the user's natural-language message
(along with available tool schemas) to the configured Ollama LLM.  The
LLM responds with a structured tool-call plan which is parsed,
validated, and executed sequentially.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from sentinel.mcp.llm_client import OllamaClient
from sentinel.mcp.prompts import tool_selection_prompt

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 1


class PromptRouter:
    """Routes natural-language prompts to Sentinel MCP tools via LLM.

    The router:
    1. Sends the user message + tool schemas to the LLM.
    2. Parses the LLM response as JSON.
    3. Validates tool names and parameters.
    4. Executes tools sequentially.
    5. Assembles the results into a final response.

    If the LLM output is malformed, retries once with a corrective
    prompt.  If all retries fail, returns the raw LLM response with an
    error flag.

    Args:
        ollama: :class:`OllamaClient` instance for LLM calls.
        tools: Dict mapping tool name to callable.  Each callable
            accepts keyword arguments and returns a dict.
        tool_schemas: List of tool schema dicts (name, description,
            parameters) for the LLM prompt.

    Example::

        router = PromptRouter(ollama=client, tools=tool_map, tool_schemas=schemas)
        result = await router.route("Train a zscore model on my data")
    """

    def __init__(
        self,
        ollama: OllamaClient,
        tools: dict[str, Any],
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ollama = ollama
        self._tools = tools
        self._tool_schemas = tool_schemas or []

    async def route(self, user_message: str) -> dict[str, Any]:
        """Route a user message through LLM tool selection and execution.

        Args:
            user_message: Natural-language request from the user.

        Returns:
            Dict with ``response`` (text), ``tools_called`` (list of
            tool names), and optionally ``tool_results``.
        """
        logger.info("router.route", message_length=len(user_message))

        # Check LLM availability
        is_available = await self._ollama.is_available()
        if not is_available:
            logger.warning("router.ollama_unavailable")
            return {
                "response": (
                    "LLM is currently unavailable. Please ensure Ollama is "
                    "running and try again. Available tools can be called "
                    "directly."
                ),
                "tools_called": [],
            }

        # Build prompt and call LLM
        prompt = tool_selection_prompt(self._tool_schemas, user_message)
        raw_response = await self._ollama.generate(prompt)

        if raw_response is None:
            return {
                "response": "LLM did not return a response.",
                "tools_called": [],
            }

        # Parse tool calls from LLM response
        tool_calls = await self._parse_tool_calls(raw_response)
        if tool_calls is None:
            return {
                "response": raw_response,
                "tools_called": [],
                "parse_error": "Could not parse LLM response as tool calls.",
            }

        # Execute tools sequentially
        tools_called: list[str] = []
        tool_results: list[dict[str, Any]] = []
        response_parts: list[str] = []

        for call in tool_calls:
            tool_name = call.get("tool", "")
            parameters = call.get("parameters", {})

            result = await self._execute_tool(tool_name, parameters)
            tools_called.append(tool_name)
            tool_results.append({"tool": tool_name, "result": result})

            if "error" in result:
                response_parts.append(f"Tool '{tool_name}' failed: {result['error']}")
            else:
                response_parts.append(f"Tool '{tool_name}' completed successfully.")

        # Assemble response
        response_text = "\n".join(response_parts) if response_parts else raw_response

        return {
            "response": response_text,
            "tools_called": tools_called,
            "tool_results": tool_results,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _parse_tool_calls(
        self,
        raw_response: str,
    ) -> list[dict[str, Any]] | None:
        """Parse tool calls from the LLM response JSON.

        Attempts to extract a JSON object with a ``tool_calls`` array.
        If parsing fails, retries once with a corrective prompt.

        Args:
            raw_response: Raw text from the LLM.

        Returns:
            List of tool-call dicts, or ``None`` if parsing fails
            after retries.
        """
        # Try to extract JSON from the response (may have surrounding text)
        parsed = _extract_json(raw_response)
        if parsed is not None:
            tool_calls = parsed.get("tool_calls")
            if isinstance(tool_calls, list):
                return tool_calls  # type: ignore[return-value]

        # Retry with corrective prompt
        logger.warning("router.parse_retry", raw_length=len(raw_response))
        corrective = (
            "Your previous response was not valid JSON. "
            "Please respond ONLY with valid JSON in this format:\n"
            '{"tool_calls": [{"tool": "<name>", "parameters": {}}]}'
        )
        retry_response = await self._ollama.generate(corrective)
        if retry_response is None:
            return None

        parsed = _extract_json(retry_response)
        if parsed is not None:
            tool_calls = parsed.get("tool_calls")
            if isinstance(tool_calls, list):
                return tool_calls  # type: ignore[return-value]

        logger.warning("router.parse_failed")
        return None

    async def _execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single tool by name with given parameters.

        Validates that the tool exists in the registry before calling.

        Args:
            tool_name: Name of the tool to execute.
            parameters: Keyword arguments for the tool function.

        Returns:
            Tool result dict, or error dict if execution fails.
        """
        if tool_name not in self._tools:
            logger.warning("router.tool_not_found", tool=tool_name)
            return {
                "error": f"Tool '{tool_name}' not found.",
                "code": "TOOL_NOT_FOUND",
            }

        tool_fn = self._tools[tool_name]
        try:
            import asyncio
            import inspect

            # Handle both sync and async tool functions
            if inspect.iscoroutinefunction(tool_fn):
                # For async tools that need ollama_client, inject it
                sig = inspect.signature(tool_fn)
                if "ollama_client" in sig.parameters:
                    parameters["ollama_client"] = self._ollama
                result = await tool_fn(**parameters)
            else:
                # Run sync tools in executor to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: tool_fn(**parameters))

            logger.info("router.tool_executed", tool=tool_name)
            return result  # type: ignore[return-value]

        except TypeError as exc:
            logger.warning(
                "router.tool_param_error",
                tool=tool_name,
                error=str(exc),
            )
            return {
                "error": f"Invalid parameters for '{tool_name}': {exc}",
                "code": "INVALID_PARAMS",
            }
        except Exception as exc:
            logger.error(
                "router.tool_execution_error",
                tool=tool_name,
                error=str(exc),
            )
            return {
                "error": f"Tool '{tool_name}' execution failed: {exc}",
                "code": "TOOL_ERROR",
            }


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from text that may contain surrounding prose.

    Tries ``json.loads`` on the full text first, then searches for the
    first ``{...}`` block.

    Args:
        text: Raw text that may contain embedded JSON.

    Returns:
        Parsed dict, or ``None`` if no valid JSON is found.
    """
    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Search for JSON block within text
    start = text.find("{")
    if start == -1:
        return None

    # Find matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start : i + 1])
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass
                break

    return None
