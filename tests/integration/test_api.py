"""Integration tests for the Sentinel FastAPI application.

Tests cover:
- GET /health
- GET /api/models
- GET /api/data (paginated list)
- POST /api/data/upload (valid CSV, invalid file type, oversized)
- GET /api/data/{id}/preview
- DELETE /api/data/{id}
- GET /api/experiments (paginated list)
- POST /api/detect with bad paths -> 400/404
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import sentinel.models  # noqa: F401 — trigger model registration

_KNOWN_MODEL_NAMES = {
    "autoencoder",
    "deepar",
    "diffusion",
    "gan",
    "gru",
    "hybrid_ensemble",
    "isolation_forest",
    "lstm",
    "lstm_ae",
    "matrix_profile",
    "rnn",
    "tadgan",
    "tcn",
    "tranad",
    "vae",
    "zscore",
}

_TOTAL_MODELS = len(_KNOWN_MODEL_NAMES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Async httpx client using ASGITransport (no real network port)."""
    from sentinel.api.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def valid_csv_bytes() -> bytes:
    """Minimal valid multivariate CSV content as bytes."""
    rows = [
        "timestamp,feature_1,feature_2",
    ]
    for i in range(20):
        ts = datetime(2024, 1, 1 + i, tzinfo=UTC).isoformat()
        rows.append(f"{ts},{float(i + 1)},{float((i + 1) * 2)}")
    return "\n".join(rows).encode()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for GET /health."""

    async def test_health_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_health_has_status_field(self, client: AsyncClient) -> None:
        data = (await client.get("/health")).json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    async def test_health_has_api_ok(self, client: AsyncClient) -> None:
        data = (await client.get("/health")).json()
        assert data["api"] == "ok"

    async def test_health_has_version(self, client: AsyncClient) -> None:
        data = (await client.get("/health")).json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    async def test_health_has_ollama_field(self, client: AsyncClient) -> None:
        data = (await client.get("/health")).json()
        assert "ollama" in data


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    """Tests for GET /api/models."""

    async def test_list_models_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/api/models")
        assert response.status_code == 200

    async def test_list_models_has_models_key(self, client: AsyncClient) -> None:
        data = (await client.get("/api/models")).json()
        assert "models" in data
        assert isinstance(data["models"], list)

    async def test_list_models_count(self, client: AsyncClient) -> None:
        data = (await client.get("/api/models")).json()
        assert len(data["models"]) == _TOTAL_MODELS

    async def test_list_models_includes_zscore(self, client: AsyncClient) -> None:
        data = (await client.get("/api/models")).json()
        names = {m["name"] for m in data["models"]}
        assert "zscore" in names

    async def test_list_models_includes_isolation_forest(
        self, client: AsyncClient
    ) -> None:
        data = (await client.get("/api/models")).json()
        names = {m["name"] for m in data["models"]}
        assert "isolation_forest" in names

    async def test_list_models_includes_all_known(self, client: AsyncClient) -> None:
        data = (await client.get("/api/models")).json()
        names = {m["name"] for m in data["models"]}
        assert _KNOWN_MODEL_NAMES.issubset(names)

    async def test_list_models_each_has_name_category_description(
        self, client: AsyncClient
    ) -> None:
        data = (await client.get("/api/models")).json()
        for model in data["models"]:
            assert "name" in model
            assert "category" in model
            assert "description" in model

    async def test_list_models_statistical_category(self, client: AsyncClient) -> None:
        data = (await client.get("/api/models")).json()
        by_name = {m["name"]: m for m in data["models"]}
        assert by_name["zscore"]["category"] == "statistical"
        assert by_name["isolation_forest"]["category"] == "statistical"
        assert by_name["matrix_profile"]["category"] == "statistical"

    async def test_list_models_ensemble_category(self, client: AsyncClient) -> None:
        data = (await client.get("/api/models")).json()
        by_name = {m["name"]: m for m in data["models"]}
        assert by_name["hybrid_ensemble"]["category"] == "ensemble"


# ---------------------------------------------------------------------------
# Datasets (list)
# ---------------------------------------------------------------------------


class TestDatasetList:
    """Tests for GET /api/data."""

    async def test_list_datasets_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/api/data")
        assert response.status_code == 200

    async def test_list_datasets_has_pagination_fields(
        self, client: AsyncClient
    ) -> None:
        data = (await client.get("/api/data")).json()
        assert "items" in data
        assert "total" in data
        assert "page" in data

    async def test_list_datasets_items_is_list(self, client: AsyncClient) -> None:
        data = (await client.get("/api/data")).json()
        assert isinstance(data["items"], list)

    async def test_list_datasets_page_defaults_to_1(self, client: AsyncClient) -> None:
        data = (await client.get("/api/data")).json()
        assert data["page"] == 1

    async def test_list_datasets_pagination_params(self, client: AsyncClient) -> None:
        response = await client.get("/api/data?page=1&limit=10")
        assert response.status_code == 200

    async def test_list_datasets_invalid_page_returns_422(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/api/data?page=0")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Dataset upload
# ---------------------------------------------------------------------------


class TestDatasetUpload:
    """Tests for POST /api/data/upload."""

    async def test_upload_valid_csv_returns_201(
        self, client: AsyncClient, valid_csv_bytes: bytes, tmp_path: Path
    ) -> None:
        response = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        assert response.status_code == 201

    async def test_upload_valid_csv_returns_dataset_id(
        self, client: AsyncClient, valid_csv_bytes: bytes
    ) -> None:
        response = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        data = response.json()
        assert "dataset_id" in data
        assert isinstance(data["dataset_id"], str)
        assert len(data["dataset_id"]) > 0

    async def test_upload_valid_csv_returns_shape(
        self, client: AsyncClient, valid_csv_bytes: bytes
    ) -> None:
        response = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        data = response.json()
        assert "shape" in data
        assert isinstance(data["shape"], list)
        assert len(data["shape"]) == 2

    async def test_upload_valid_csv_returns_features(
        self, client: AsyncClient, valid_csv_bytes: bytes
    ) -> None:
        response = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        data = response.json()
        assert "features" in data
        assert "feature_1" in data["features"]
        assert "feature_2" in data["features"]

    async def test_upload_valid_csv_returns_time_range(
        self, client: AsyncClient, valid_csv_bytes: bytes
    ) -> None:
        response = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        data = response.json()
        assert "time_range" in data
        assert "start" in data["time_range"]
        assert "end" in data["time_range"]

    async def test_upload_unsupported_type_returns_400(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/api/data/upload",
            files={
                "file": (
                    "data.json",
                    io.BytesIO(b'{"key": "value"}'),
                    "application/json",
                )
            },
        )
        assert response.status_code == 400

    async def test_upload_invalid_csv_returns_400(self, client: AsyncClient) -> None:
        """A CSV without a timestamp column should fail validation."""
        bad_csv = b"col_a,col_b\n1.0,2.0\n3.0,4.0\n"
        response = await client.post(
            "/api/data/upload",
            files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")},
        )
        assert response.status_code == 400

    async def test_upload_dataset_appears_in_list(
        self, client: AsyncClient, valid_csv_bytes: bytes
    ) -> None:
        """After upload, the dataset should appear in GET /api/data."""
        upload_resp = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["dataset_id"]

        list_resp = await client.get("/api/data")
        ids = [item["dataset_id"] for item in list_resp.json()["items"]]
        assert dataset_id in ids


# ---------------------------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------------------------


class TestDatasetPreview:
    """Tests for GET /api/data/{id}/preview."""

    @pytest.fixture
    async def uploaded_dataset_id(
        self, client: AsyncClient, valid_csv_bytes: bytes
    ) -> str:
        """Upload a dataset and return its id."""
        resp = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        return resp.json()["dataset_id"]

    async def test_preview_returns_200(
        self, client: AsyncClient, uploaded_dataset_id: str
    ) -> None:
        resp = await client.get(f"/api/data/{uploaded_dataset_id}/preview")
        assert resp.status_code == 200

    async def test_preview_returns_list_of_dicts(
        self, client: AsyncClient, uploaded_dataset_id: str
    ) -> None:
        data = (await client.get(f"/api/data/{uploaded_dataset_id}/preview")).json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert isinstance(data[0], dict)

    async def test_preview_default_rows(
        self, client: AsyncClient, uploaded_dataset_id: str
    ) -> None:
        data = (await client.get(f"/api/data/{uploaded_dataset_id}/preview")).json()
        assert len(data) <= 20

    async def test_preview_custom_rows(
        self, client: AsyncClient, uploaded_dataset_id: str
    ) -> None:
        data = (
            await client.get(f"/api/data/{uploaded_dataset_id}/preview?rows=5")
        ).json()
        assert len(data) <= 5

    async def test_preview_not_found_returns_404(self, client: AsyncClient) -> None:
        resp = await client.get("/api/data/nonexistent-uuid/preview")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Dataset delete
# ---------------------------------------------------------------------------


class TestDatasetDelete:
    """Tests for DELETE /api/data/{id}."""

    async def test_delete_existing_dataset(
        self, client: AsyncClient, valid_csv_bytes: bytes
    ) -> None:
        upload_resp = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        dataset_id = upload_resp.json()["dataset_id"]

        del_resp = await client.delete(f"/api/data/{dataset_id}")
        assert del_resp.status_code == 200

    async def test_delete_removes_from_list(
        self, client: AsyncClient, valid_csv_bytes: bytes
    ) -> None:
        upload_resp = await client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(valid_csv_bytes), "text/csv")},
        )
        dataset_id = upload_resp.json()["dataset_id"]

        await client.delete(f"/api/data/{dataset_id}")

        list_resp = await client.get("/api/data")
        ids = [item["dataset_id"] for item in list_resp.json()["items"]]
        assert dataset_id not in ids

    async def test_delete_nonexistent_returns_404(self, client: AsyncClient) -> None:
        resp = await client.delete("/api/data/nonexistent-uuid")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class TestExperiments:
    """Tests for GET /api/experiments."""

    async def test_list_experiments_returns_200(self, client: AsyncClient) -> None:
        response = await client.get("/api/experiments")
        assert response.status_code == 200

    async def test_list_experiments_has_pagination_fields(
        self, client: AsyncClient
    ) -> None:
        data = (await client.get("/api/experiments")).json()
        assert "items" in data
        assert "total" in data
        assert "page" in data

    async def test_list_experiments_items_is_list(self, client: AsyncClient) -> None:
        data = (await client.get("/api/experiments")).json()
        assert isinstance(data["items"], list)

    async def test_list_experiments_page_defaults_to_1(
        self, client: AsyncClient
    ) -> None:
        data = (await client.get("/api/experiments")).json()
        assert data["page"] == 1

    async def test_list_experiments_pagination_params(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/api/experiments?page=1&limit=25")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Detect (error paths)
# ---------------------------------------------------------------------------


class TestDetectErrors:
    """Tests for POST /api/detect error handling."""

    async def test_detect_path_traversal_returns_400(self, client: AsyncClient) -> None:
        body: dict[str, Any] = {
            "data_path": "../../etc/passwd",
            "model_path": "data/experiments/run-1",
        }
        response = await client.post("/api/detect", json=body)
        assert response.status_code == 400

    async def test_detect_outside_allowed_dir_returns_400(
        self, client: AsyncClient
    ) -> None:
        body: dict[str, Any] = {
            "data_path": "/nonexistent/data.csv",
            "model_path": "/nonexistent/model",
        }
        response = await client.post("/api/detect", json=body)
        assert response.status_code == 400

    async def test_detect_missing_data_file_returns_400(
        self, client: AsyncClient
    ) -> None:
        body: dict[str, Any] = {
            "data_path": "data/raw/nonexistent.csv",
            "model_path": "data/experiments/run-1",
        }
        response = await client.post("/api/detect", json=body)
        assert response.status_code == 400

    async def test_detect_empty_body_returns_422(self, client: AsyncClient) -> None:
        response = await client.post("/api/detect", json={})
        assert response.status_code == 422
