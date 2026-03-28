"""Ollama HTTP client for LLM-powered analysis and tool routing.

Provides an async client for the Ollama REST API.  All methods return
``None`` when Ollama is unreachable so callers can implement graceful
fallback paths without catching transport errors.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


class OllamaClient:
    """Async HTTP client for the Ollama local LLM server.

    Args:
        base_url: Ollama server base URL (e.g. ``http://localhost:11434``).
        model: Model name to use for generation (e.g. ``nemotron-3-nano:4b``).
        timeout: HTTP request timeout in seconds.

    Example::

        client = OllamaClient()
        if await client.is_available():
            text = await client.generate("Summarise this data")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nemotron-3-nano:4b",
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        """The configured LLM model name."""
        return self._model

    @property
    def base_url(self) -> str:
        """The Ollama server base URL."""
        return self._base_url

    async def is_available(self) -> bool:
        """Check whether the Ollama server is reachable.

        Sends a lightweight ``GET /api/tags`` request.

        Returns:
            ``True`` if the server responds with HTTP 200, ``False`` otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                available = resp.status_code == 200
                logger.debug("ollama.health_check", available=available)
                return available
        except httpx.HTTPError as exc:
            logger.debug("ollama.unreachable", error=str(exc))
            return False

    async def generate(self, prompt: str) -> str | None:
        """Generate text from a prompt via ``POST /api/generate``.

        Args:
            prompt: The text prompt to send to the model.

        Returns:
            The generated response text, or ``None`` if Ollama is
            unreachable or the request fails.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        return await self._post("/api/generate", payload, response_key="response")

    async def chat(self, messages: list[dict[str, str]]) -> str | None:
        """Send a chat conversation via ``POST /api/chat``.

        Args:
            messages: List of message dicts, each with ``"role"`` and
                ``"content"`` keys (e.g. ``[{"role": "user", "content": "..."}]``).

        Returns:
            The assistant's reply text, or ``None`` if Ollama is
            unreachable or the request fails.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        return await self._post("/api/chat", payload, response_key="message")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        response_key: str,
    ) -> str | None:
        """Send a POST request to Ollama and extract the response text.

        Args:
            path: API path (e.g. ``/api/generate``).
            payload: JSON request body.
            response_key: Top-level key in the JSON response that holds
                the generated text (``"response"`` for generate,
                ``"message"`` for chat).

        Returns:
            Extracted text string, or ``None`` on failure.
        """
        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=float(self._timeout)) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                # /api/generate returns {"response": "..."}
                # /api/chat returns {"message": {"role": "...", "content": "..."}}
                value = data.get(response_key)
                if isinstance(value, dict):
                    return value.get("content")  # type: ignore[return-value]
                return value  # type: ignore[return-value]

        except httpx.TimeoutException:
            logger.warning("ollama.timeout", path=path, timeout=self._timeout)
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "ollama.http_error",
                path=path,
                status=exc.response.status_code,
                body=exc.response.text[:200],
            )
            return None
        except httpx.HTTPError as exc:
            logger.warning("ollama.request_failed", path=path, error=str(exc))
            return None
