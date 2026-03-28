"""Smoke tests for the Sentinel web dashboard.

Verifies that all static dashboard pages and assets are served correctly
at /ui by the FastAPI application.

Tests cover:
- GET /ui — serves HTML content (index.html)
- GET /ui/upload.html — upload page
- GET /ui/explore.html — explore/chart page
- GET /ui/pi.html — PI System connector page
- GET /ui/prompt.html — LLM prompt page
- GET /ui/style.css — stylesheet with correct content type
- GET /ui/app.js — JavaScript bundle with correct content type
- Expected HTML elements present in each page
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import sentinel.models  # noqa: F401 — trigger model registration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client() -> AsyncClient:
    """Async httpx client backed by the Sentinel ASGI app."""
    from sentinel.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _assert_html(response_text: str) -> None:
    """Assert that a response body looks like an HTML document."""
    lower = response_text.lower()
    assert "<!doctype html>" in lower or "<html" in lower


# ---------------------------------------------------------------------------
# Dashboard pages — existence and content type
# ---------------------------------------------------------------------------


class TestDashboardPages:
    """Smoke tests: every dashboard page is served with HTTP 200."""

    @pytest.mark.integration
    async def test_index_returns_200(self, client: AsyncClient) -> None:
        """GET /ui/ should return HTTP 200."""
        response = await client.get("/ui/")
        assert response.status_code == 200

    @pytest.mark.integration
    async def test_index_html_returns_200(self, client: AsyncClient) -> None:
        """GET /ui/index.html should return HTTP 200."""
        response = await client.get("/ui/index.html")
        assert response.status_code == 200

    @pytest.mark.integration
    async def test_upload_html_returns_200(self, client: AsyncClient) -> None:
        """GET /ui/upload.html should return HTTP 200."""
        response = await client.get("/ui/upload.html")
        assert response.status_code == 200

    @pytest.mark.integration
    async def test_explore_html_returns_200(self, client: AsyncClient) -> None:
        """GET /ui/explore.html should return HTTP 200."""
        response = await client.get("/ui/explore.html")
        assert response.status_code == 200

    @pytest.mark.integration
    async def test_pi_html_returns_200(self, client: AsyncClient) -> None:
        """GET /ui/pi.html should return HTTP 200."""
        response = await client.get("/ui/pi.html")
        assert response.status_code == 200

    @pytest.mark.integration
    async def test_prompt_html_returns_200(self, client: AsyncClient) -> None:
        """GET /ui/prompt.html should return HTTP 200."""
        response = await client.get("/ui/prompt.html")
        assert response.status_code == 200

    @pytest.mark.integration
    async def test_style_css_returns_200(self, client: AsyncClient) -> None:
        """GET /ui/style.css should return HTTP 200."""
        response = await client.get("/ui/style.css")
        assert response.status_code == 200

    @pytest.mark.integration
    async def test_app_js_returns_200(self, client: AsyncClient) -> None:
        """GET /ui/app.js should return HTTP 200."""
        response = await client.get("/ui/app.js")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Content-type checks
# ---------------------------------------------------------------------------


class TestDashboardContentTypes:
    """Verify that assets are served with the correct Content-Type headers."""

    @pytest.mark.integration
    async def test_index_html_content_type(self, client: AsyncClient) -> None:
        """HTML pages must be served with text/html content type."""
        response = await client.get("/ui/index.html")
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.integration
    async def test_upload_html_content_type(self, client: AsyncClient) -> None:
        response = await client.get("/ui/upload.html")
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.integration
    async def test_explore_html_content_type(self, client: AsyncClient) -> None:
        response = await client.get("/ui/explore.html")
        assert "text/html" in response.headers.get("content-type", "")

    @pytest.mark.integration
    async def test_style_css_content_type(self, client: AsyncClient) -> None:
        """style.css must be served with text/css content type."""
        response = await client.get("/ui/style.css")
        content_type = response.headers.get("content-type", "")
        assert "text/css" in content_type or "css" in content_type

    @pytest.mark.integration
    async def test_app_js_content_type(self, client: AsyncClient) -> None:
        """app.js must be served with a JavaScript content type."""
        response = await client.get("/ui/app.js")
        content_type = response.headers.get("content-type", "")
        assert (
            "javascript" in content_type
            or "application/js" in content_type
            or "text/plain" in content_type  # some static servers use this
        )


# ---------------------------------------------------------------------------
# HTML content checks — expected elements present
# ---------------------------------------------------------------------------


class TestDashboardHtmlContent:
    """Verify that each dashboard page contains expected HTML elements."""

    @pytest.mark.integration
    async def test_index_is_html(self, client: AsyncClient) -> None:
        """Index page should be a valid HTML document."""
        response = await client.get("/ui/index.html")
        _assert_html(response.text)

    @pytest.mark.integration
    async def test_index_has_sentinel_brand(self, client: AsyncClient) -> None:
        """Index page should reference 'Sentinel' as the brand name."""
        response = await client.get("/ui/index.html")
        assert "Sentinel" in response.text

    @pytest.mark.integration
    async def test_index_has_navbar(self, client: AsyncClient) -> None:
        """Index page should contain a navigation bar."""
        response = await client.get("/ui/index.html")
        assert "navbar" in response.text.lower() or "<nav" in response.text.lower()

    @pytest.mark.integration
    async def test_index_links_to_upload(self, client: AsyncClient) -> None:
        """Index page should link to the upload page."""
        response = await client.get("/ui/index.html")
        assert "upload.html" in response.text

    @pytest.mark.integration
    async def test_index_links_to_explore(self, client: AsyncClient) -> None:
        """Index page should link to the explore page."""
        response = await client.get("/ui/index.html")
        assert "explore.html" in response.text

    @pytest.mark.integration
    async def test_index_links_to_pi(self, client: AsyncClient) -> None:
        """Index page should link to the PI System page."""
        response = await client.get("/ui/index.html")
        assert "pi.html" in response.text

    @pytest.mark.integration
    async def test_index_links_to_prompt(self, client: AsyncClient) -> None:
        """Index page should link to the prompt page."""
        response = await client.get("/ui/index.html")
        assert "prompt.html" in response.text

    @pytest.mark.integration
    async def test_upload_page_is_html(self, client: AsyncClient) -> None:
        """Upload page should be a valid HTML document."""
        response = await client.get("/ui/upload.html")
        _assert_html(response.text)

    @pytest.mark.integration
    async def test_upload_page_has_title(self, client: AsyncClient) -> None:
        """Upload page title should mention 'Upload'."""
        response = await client.get("/ui/upload.html")
        assert "Upload" in response.text

    @pytest.mark.integration
    async def test_upload_page_references_stylesheet(self, client: AsyncClient) -> None:
        """Upload page should link to style.css."""
        response = await client.get("/ui/upload.html")
        assert "style.css" in response.text

    @pytest.mark.integration
    async def test_explore_page_is_html(self, client: AsyncClient) -> None:
        """Explore page should be a valid HTML document."""
        response = await client.get("/ui/explore.html")
        _assert_html(response.text)

    @pytest.mark.integration
    async def test_explore_page_has_chart_js(self, client: AsyncClient) -> None:
        """Explore page should reference Chart.js for interactive charts."""
        response = await client.get("/ui/explore.html")
        assert "chart" in response.text.lower()

    @pytest.mark.integration
    async def test_pi_page_is_html(self, client: AsyncClient) -> None:
        """PI System page should be a valid HTML document."""
        response = await client.get("/ui/pi.html")
        _assert_html(response.text)

    @pytest.mark.integration
    async def test_pi_page_has_pi_title(self, client: AsyncClient) -> None:
        """PI System page title should reference 'PI'."""
        response = await client.get("/ui/pi.html")
        assert "PI" in response.text

    @pytest.mark.integration
    async def test_prompt_page_is_html(self, client: AsyncClient) -> None:
        """Prompt page should be a valid HTML document."""
        response = await client.get("/ui/prompt.html")
        _assert_html(response.text)

    @pytest.mark.integration
    async def test_prompt_page_has_title(self, client: AsyncClient) -> None:
        """Prompt page title should mention 'Prompt'."""
        response = await client.get("/ui/prompt.html")
        assert "Prompt" in response.text


# ---------------------------------------------------------------------------
# 404 for missing assets
# ---------------------------------------------------------------------------


class TestDashboardMissingAssets:
    """Verify that nonexistent dashboard assets return 404."""

    @pytest.mark.integration
    async def test_missing_page_returns_404(self, client: AsyncClient) -> None:
        """A nonexistent dashboard file should yield 404."""
        response = await client.get("/ui/nonexistent_page.html")
        assert response.status_code == 404

    @pytest.mark.integration
    async def test_missing_asset_returns_404(self, client: AsyncClient) -> None:
        """A nonexistent static asset should yield 404."""
        response = await client.get("/ui/does_not_exist.js")
        assert response.status_code == 404
