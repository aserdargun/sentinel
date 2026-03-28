"""Integration tests for path traversal prevention in API routes.

Verifies that the detect and train endpoints reject paths that
attempt to escape the allowed project directories.
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
    from sentinel.api.jobs import BackgroundJobManager

    app = create_app()
    app.state.job_manager = BackgroundJobManager(max_workers=1)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.state.job_manager.shutdown()


# ---------------------------------------------------------------------------
# POST /api/detect — path traversal
# ---------------------------------------------------------------------------


class TestDetectPathTraversal:
    """Detect endpoint must reject path traversal attempts."""

    @pytest.mark.integration
    async def test_detect_traversal_data_path(self, client: AsyncClient) -> None:
        """data_path with ../../etc/passwd must yield 400."""
        response = await client.post(
            "/api/detect",
            json={
                "data_path": "../../etc/passwd",
                "model_path": "data/experiments/run-1/model",
            },
        )
        assert response.status_code == 400
        assert "outside" in response.json()["detail"].lower()

    @pytest.mark.integration
    async def test_detect_traversal_model_path(self, client: AsyncClient) -> None:
        """model_path with /etc/shadow must yield 400."""
        response = await client.post(
            "/api/detect",
            json={
                "data_path": "data/raw/test.csv",
                "model_path": "/etc/shadow",
            },
        )
        assert response.status_code == 400

    @pytest.mark.integration
    async def test_detect_absolute_outside_project(self, client: AsyncClient) -> None:
        """Absolute path outside project must yield 400."""
        response = await client.post(
            "/api/detect",
            json={
                "data_path": "/tmp/malicious.csv",
                "model_path": "data/experiments/run-1",
            },
        )
        assert response.status_code == 400

    @pytest.mark.integration
    async def test_detect_null_byte_in_path(self, client: AsyncClient) -> None:
        """Null byte in path must yield 400."""
        response = await client.post(
            "/api/detect",
            json={
                "data_path": "data/raw/test.csv\x00.txt",
                "model_path": "data/experiments/run-1",
            },
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/train — path traversal
# ---------------------------------------------------------------------------


class TestTrainPathTraversal:
    """Train endpoint must reject path traversal attempts."""

    @pytest.mark.integration
    async def test_train_traversal_config_path(self, client: AsyncClient) -> None:
        """config_path with ../../etc/passwd must yield 400."""
        response = await client.post(
            "/api/train",
            json={"config_path": "../../etc/passwd"},
        )
        assert response.status_code == 400
        assert "outside" in response.json()["detail"].lower()

    @pytest.mark.integration
    async def test_train_absolute_config_outside_project(
        self, client: AsyncClient
    ) -> None:
        """Absolute config path outside project must yield 400."""
        response = await client.post(
            "/api/train",
            json={"config_path": "/etc/hosts"},
        )
        assert response.status_code == 400

    @pytest.mark.integration
    async def test_train_traversal_data_path(self, client: AsyncClient) -> None:
        """data_path override with traversal must yield 400."""
        response = await client.post(
            "/api/train",
            json={
                "config_path": "configs/zscore.yaml",
                "data_path": "../../../etc/passwd",
            },
        )
        assert response.status_code == 400

    @pytest.mark.integration
    async def test_train_null_byte_in_config(self, client: AsyncClient) -> None:
        """Null byte in config_path must yield 400."""
        response = await client.post(
            "/api/train",
            json={"config_path": "configs/zscore.yaml\x00.sh"},
        )
        assert response.status_code == 400
