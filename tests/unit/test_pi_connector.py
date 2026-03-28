"""Unit tests for sentinel.data.pi_connector.

All tests are marked with @pytest.mark.pi.
pipolars is Windows-only, so every test mocks the pipolars module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

pytestmark = pytest.mark.pi


# ---------------------------------------------------------------------------
# Helper: build a minimal fake pipolars module so we can inject it.
# ---------------------------------------------------------------------------


def _make_fake_pipolars() -> ModuleType:
    """Return a minimal stub module for pipolars."""
    mod = ModuleType("pipolars")
    mod.PIClient = MagicMock()  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_pipolars() -> ModuleType:
    """Inject a fake pipolars into sys.modules for the duration of the test."""
    mod = _make_fake_pipolars()
    with patch.dict(sys.modules, {"pipolars": mod}):
        yield mod


@pytest.fixture()
def mock_client(fake_pipolars: ModuleType) -> MagicMock:
    """Return the MagicMock that PIClient() will return."""
    instance = MagicMock()
    fake_pipolars.PIClient.return_value = instance  # type: ignore[attr-defined]
    return instance


@pytest.fixture()
def connector(mock_client: MagicMock) -> Any:
    """PIConnector instance wired to the mock client."""
    # Re-import inside the patched context so _HAS_PIPOLARS == True.
    import importlib

    import sentinel.data.pi_connector as pc_mod

    importlib.reload(pc_mod)
    pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
    conn = pc_mod.PIConnector(host="test-server", port=5450, timeout=30)
    # Prime the cached client so _get_client() never calls PIClient().
    conn._client = mock_client
    return conn


@pytest.fixture()
def sample_pi_df() -> pl.DataFrame:
    """Minimal pivoted DataFrame that a pipolars query chain would return."""
    return pl.DataFrame(
        {
            "Timestamp": pl.Series(["2024-01-01T00:00:00", "2024-01-01T00:05:00"]).cast(
                pl.Datetime("us")
            ),
            "Pump.Flow": [10.5, 11.2],
            "Pump.Pressure": [100.0, 102.3],
        }
    )


# ---------------------------------------------------------------------------
# TestPIConnectorInit
# ---------------------------------------------------------------------------


class TestPIConnectorInit:
    """PIConnector stores constructor args and starts with no client."""

    def test_stores_host(self, tmp_path: Path) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        conn = pc_mod.PIConnector(host="my-server")
        assert conn._host == "my-server"

    def test_stores_port(self, tmp_path: Path) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        conn = pc_mod.PIConnector(host="h", port=1234)
        assert conn._port == 1234

    def test_stores_timeout(self, tmp_path: Path) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        conn = pc_mod.PIConnector(host="h", timeout=60)
        assert conn._timeout == 60

    def test_default_port_is_5450(self, tmp_path: Path) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        conn = pc_mod.PIConnector(host="h")
        assert conn._port == 5450

    def test_default_timeout_is_30(self, tmp_path: Path) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        conn = pc_mod.PIConnector(host="h")
        assert conn._timeout == 30

    def test_no_client_on_init(self, tmp_path: Path) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        conn = pc_mod.PIConnector(host="h")
        assert conn._client is None

    def test_cache_dir_stored_as_str(self, tmp_path: Path) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        conn = pc_mod.PIConnector(host="h", cache_dir=tmp_path / "cache")
        assert conn._cache_dir == str(tmp_path / "cache")

    def test_cache_dir_none_stays_none(self, tmp_path: Path) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        conn = pc_mod.PIConnector(host="h", cache_dir=None)
        assert conn._cache_dir is None


# ---------------------------------------------------------------------------
# TestSearchTags
# ---------------------------------------------------------------------------


class TestSearchTags:
    """search_tags() returns a list of dicts from the mock client."""

    def test_returns_list(self, connector: Any, mock_client: MagicMock) -> None:
        tag = MagicMock()
        tag.name = "Pump.Flow"
        tag.description = "Flow rate"
        tag.uom = "GPM"
        mock_client.search.return_value = [tag]

        results = connector.search_tags("Pump*")
        assert isinstance(results, list)

    def test_each_result_has_name_key(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        tag = MagicMock()
        tag.name = "Pump.Flow"
        tag.description = ""
        tag.uom = ""
        mock_client.search.return_value = [tag]

        results = connector.search_tags("Pump*")
        assert all("name" in r for r in results)

    def test_each_result_has_description_key(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        tag = MagicMock()
        tag.name = "T"
        tag.description = "desc"
        tag.uom = ""
        mock_client.search.return_value = [tag]

        results = connector.search_tags("*")
        assert all("description" in r for r in results)

    def test_each_result_has_uom_key(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        tag = MagicMock()
        tag.name = "T"
        tag.description = ""
        tag.uom = "psi"
        mock_client.search.return_value = [tag]

        results = connector.search_tags("*")
        assert all("uom" in r for r in results)

    def test_values_match_mock(self, connector: Any, mock_client: MagicMock) -> None:
        tag = MagicMock()
        tag.name = "Pump.Flow"
        tag.description = "Flow rate"
        tag.uom = "GPM"
        mock_client.search.return_value = [tag]

        results = connector.search_tags("Pump*")
        assert results[0]["name"] == "Pump.Flow"
        assert results[0]["description"] == "Flow rate"
        assert results[0]["uom"] == "GPM"

    def test_empty_result(self, connector: Any, mock_client: MagicMock) -> None:
        mock_client.search.return_value = []
        results = connector.search_tags("NoMatch*")
        assert results == []

    def test_multiple_tags_returned(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        tags = []
        for name in ("A", "B", "C"):
            t = MagicMock()
            t.name = name
            t.description = ""
            t.uom = ""
            tags.append(t)
        mock_client.search.return_value = tags

        results = connector.search_tags("*")
        assert len(results) == 3

    def test_search_error_raises_pi_connection_error(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        import sentinel.data.pi_connector as pc_mod

        mock_client.search.side_effect = RuntimeError("server down")

        with pytest.raises(pc_mod.PIConnectionError, match="Tag search failed"):
            connector.search_tags("Pump*")

    def test_client_search_called_with_pattern(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        mock_client.search.return_value = []
        connector.search_tags("Sensor*")
        mock_client.search.assert_called_once_with("Sensor*")


# ---------------------------------------------------------------------------
# TestFetchTags
# ---------------------------------------------------------------------------


class TestFetchTags:
    """fetch_tags() pivots query result to canonical schema with UTC timestamps."""

    def _setup_chain(self, mock_client: MagicMock, return_df: pl.DataFrame) -> None:
        """Wire the mock client's fluent query chain."""
        chain = MagicMock()
        chain.time_range.return_value = chain
        chain.interpolated.return_value = chain
        chain.pivot.return_value = chain
        chain.to_dataframe.return_value = return_df
        mock_client.query.return_value = chain

    def test_returns_polars_dataframe(
        self,
        connector: Any,
        mock_client: MagicMock,
        sample_pi_df: pl.DataFrame,
    ) -> None:
        self._setup_chain(mock_client, sample_pi_df)
        result = connector.fetch_tags(
            tags=["Pump.Flow", "Pump.Pressure"],
            start="*-1d",
            end="*",
            interval="5m",
        )
        assert isinstance(result, pl.DataFrame)

    def test_timestamp_is_first_column(
        self,
        connector: Any,
        mock_client: MagicMock,
        sample_pi_df: pl.DataFrame,
    ) -> None:
        self._setup_chain(mock_client, sample_pi_df)
        result = connector.fetch_tags(
            tags=["Pump.Flow", "Pump.Pressure"],
            start="*-1d",
            end="*",
            interval="5m",
        )
        assert result.columns[0] == "timestamp"

    def test_timestamps_have_utc_timezone(
        self,
        connector: Any,
        mock_client: MagicMock,
        sample_pi_df: pl.DataFrame,
    ) -> None:
        self._setup_chain(mock_client, sample_pi_df)
        result = connector.fetch_tags(
            tags=["Pump.Flow", "Pump.Pressure"],
            start="*-1d",
            end="*",
            interval="5m",
        )
        ts_dtype = result.schema["timestamp"]
        assert isinstance(ts_dtype, pl.Datetime)
        assert ts_dtype.time_zone == "UTC"

    def test_feature_columns_are_float64(
        self,
        connector: Any,
        mock_client: MagicMock,
        sample_pi_df: pl.DataFrame,
    ) -> None:
        self._setup_chain(mock_client, sample_pi_df)
        result = connector.fetch_tags(
            tags=["Pump.Flow", "Pump.Pressure"],
            start="*-1d",
            end="*",
            interval="5m",
        )
        feature_cols = [c for c in result.columns if c != "timestamp"]
        for col in feature_cols:
            assert result.schema[col] == pl.Float64

    def test_row_count_preserved(
        self,
        connector: Any,
        mock_client: MagicMock,
        sample_pi_df: pl.DataFrame,
    ) -> None:
        self._setup_chain(mock_client, sample_pi_df)
        result = connector.fetch_tags(
            tags=["Pump.Flow", "Pump.Pressure"],
            start="*-1d",
            end="*",
            interval="5m",
        )
        assert result.height == sample_pi_df.height

    def test_feature_columns_present(
        self,
        connector: Any,
        mock_client: MagicMock,
        sample_pi_df: pl.DataFrame,
    ) -> None:
        self._setup_chain(mock_client, sample_pi_df)
        result = connector.fetch_tags(
            tags=["Pump.Flow", "Pump.Pressure"],
            start="*-1d",
            end="*",
            interval="5m",
        )
        assert "Pump.Flow" in result.columns
        assert "Pump.Pressure" in result.columns

    def test_timestamp_column_renamed_from_Timestamp(
        self,
        connector: Any,
        mock_client: MagicMock,
    ) -> None:
        """pipolars may return 'Timestamp' (capital T); connector renames it."""
        df = pl.DataFrame(
            {
                "Timestamp": pl.Series(["2024-01-01T00:00:00"]).cast(pl.Datetime("us")),
                "Tag1": [1.0],
            }
        )
        self._setup_chain(mock_client, df)
        result = connector.fetch_tags(
            tags=["Tag1"], start="*-1d", end="*", interval="1m"
        )
        assert "timestamp" in result.columns
        assert "Timestamp" not in result.columns

    def test_empty_tags_raises_value_error(self, connector: Any) -> None:
        with pytest.raises(ValueError, match="At least one tag"):
            connector.fetch_tags(tags=[], start="*-1d", end="*", interval="5m")

    def test_fetch_error_raises_pi_connection_error(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        import sentinel.data.pi_connector as pc_mod

        mock_client.query.side_effect = RuntimeError("network error")

        with pytest.raises(pc_mod.PIConnectionError, match="Failed to fetch tags"):
            connector.fetch_tags(tags=["Tag1"], start="*-1d", end="*", interval="5m")

    def test_non_utc_timezone_converted_to_utc(
        self,
        connector: Any,
        mock_client: MagicMock,
    ) -> None:
        """Timestamps in a non-UTC timezone must be converted to UTC."""
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(["2024-01-01T00:00:00"])
                .str.to_datetime()
                .cast(pl.Datetime("us", "UTC"))
                .dt.convert_time_zone("America/New_York"),
                "Tag1": [1.0],
            }
        )
        self._setup_chain(mock_client, df)
        result = connector.fetch_tags(
            tags=["Tag1"], start="*-1d", end="*", interval="5m"
        )
        assert result.schema["timestamp"].time_zone == "UTC"

    def test_already_utc_timestamps_unchanged(
        self,
        connector: Any,
        mock_client: MagicMock,
    ) -> None:
        df = pl.DataFrame(
            {
                "timestamp": pl.Series(["2024-01-01T00:00:00"])
                .str.to_datetime()
                .cast(pl.Datetime("us", "UTC")),
                "Tag1": [42.0],
            }
        )
        self._setup_chain(mock_client, df)
        result = connector.fetch_tags(
            tags=["Tag1"], start="*-1d", end="*", interval="5m"
        )
        assert result.schema["timestamp"].time_zone == "UTC"

    def test_query_called_with_tags(
        self,
        connector: Any,
        mock_client: MagicMock,
        sample_pi_df: pl.DataFrame,
    ) -> None:
        self._setup_chain(mock_client, sample_pi_df)
        tags = ["Pump.Flow", "Pump.Pressure"]
        connector.fetch_tags(tags=tags, start="*-1d", end="*", interval="5m")
        mock_client.query.assert_called_once_with(tags)


# ---------------------------------------------------------------------------
# TestSnapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    """snapshot() returns a list of dicts for each tag."""

    def _make_entry(
        self, name: str, value: float, ts: str = "2024-01-01T00:00:00"
    ) -> MagicMock:
        entry = MagicMock()
        entry.name = name
        entry.value = value
        entry.timestamp = ts
        entry.quality = "Good"
        return entry

    def test_returns_list(self, connector: Any, mock_client: MagicMock) -> None:
        mock_client.snapshots.return_value = [self._make_entry("T1", 1.0)]
        result = connector.snapshot(["T1"])
        assert isinstance(result, list)

    def test_each_entry_has_name(self, connector: Any, mock_client: MagicMock) -> None:
        mock_client.snapshots.return_value = [self._make_entry("T1", 1.0)]
        result = connector.snapshot(["T1"])
        assert all("name" in r for r in result)

    def test_each_entry_has_value(self, connector: Any, mock_client: MagicMock) -> None:
        mock_client.snapshots.return_value = [self._make_entry("T1", 99.9)]
        result = connector.snapshot(["T1"])
        assert all("value" in r for r in result)

    def test_each_entry_has_timestamp(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        mock_client.snapshots.return_value = [self._make_entry("T1", 1.0)]
        result = connector.snapshot(["T1"])
        assert all("timestamp" in r for r in result)

    def test_each_entry_has_quality(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        mock_client.snapshots.return_value = [self._make_entry("T1", 1.0)]
        result = connector.snapshot(["T1"])
        assert all("quality" in r for r in result)

    def test_values_match_mock(self, connector: Any, mock_client: MagicMock) -> None:
        mock_client.snapshots.return_value = [
            self._make_entry("Pump.Flow", 55.3, "2024-06-01T12:00:00")
        ]
        result = connector.snapshot(["Pump.Flow"])
        assert result[0]["name"] == "Pump.Flow"
        assert result[0]["value"] == pytest.approx(55.3)
        assert result[0]["quality"] == "Good"

    def test_multiple_tags(self, connector: Any, mock_client: MagicMock) -> None:
        mock_client.snapshots.return_value = [
            self._make_entry("A", 1.0),
            self._make_entry("B", 2.0),
        ]
        result = connector.snapshot(["A", "B"])
        assert len(result) == 2

    def test_empty_tags_raises_value_error(self, connector: Any) -> None:
        with pytest.raises(ValueError, match="At least one tag"):
            connector.snapshot([])

    def test_snapshot_error_raises_pi_connection_error(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        import sentinel.data.pi_connector as pc_mod

        mock_client.snapshots.side_effect = RuntimeError("timeout")

        with pytest.raises(pc_mod.PIConnectionError, match="Snapshot retrieval failed"):
            connector.snapshot(["T1"])

    def test_client_called_with_tags_list(
        self, connector: Any, mock_client: MagicMock
    ) -> None:
        mock_client.snapshots.return_value = []
        connector.snapshot(["T1", "T2"])
        mock_client.snapshots.assert_called_once_with(["T1", "T2"])


# ---------------------------------------------------------------------------
# TestImportErrorWhenPipolarsUnavailable
# ---------------------------------------------------------------------------


class TestImportErrorWhenPipolarsUnavailable:
    """Methods raise ImportError when pipolars is not installed."""

    def _connector_without_pipolars(self) -> Any:
        """Return a PIConnector instance with _HAS_PIPOLARS forced False."""
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = False  # type: ignore[attr-defined]
        return pc_mod.PIConnector(host="h")

    def test_search_tags_raises_import_error(self) -> None:
        conn = self._connector_without_pipolars()
        with pytest.raises(ImportError, match="pipolars is not installed"):
            conn.search_tags("*")

    def test_fetch_tags_raises_import_error(self) -> None:
        conn = self._connector_without_pipolars()
        with pytest.raises(ImportError, match="pipolars is not installed"):
            conn.fetch_tags(tags=["T1"], start="*-1d", end="*", interval="5m")

    def test_snapshot_raises_import_error(self) -> None:
        conn = self._connector_without_pipolars()
        with pytest.raises(ImportError, match="pipolars is not installed"):
            conn.snapshot(["T1"])

    def test_is_pi_available_returns_false(self) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = False  # type: ignore[attr-defined]
        assert pc_mod.is_pi_available() is False

    def test_is_pi_available_returns_true_when_available(
        self, fake_pipolars: ModuleType
    ) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]
        assert pc_mod.is_pi_available() is True


# ---------------------------------------------------------------------------
# TestRetryLogic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """_get_client() retries 3 times then raises PIConnectionError."""

    def test_raises_after_max_retries(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]

        fake_pipolars.PIClient.side_effect = RuntimeError("refused")  # type: ignore[attr-defined]

        conn = pc_mod.PIConnector(host="bad-server")

        with patch("time.sleep"):
            with pytest.raises(pc_mod.PIConnectionError, match="Failed to connect"):
                conn._get_client()

    def test_retries_exactly_three_times(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]

        fake_pipolars.PIClient.side_effect = RuntimeError("refused")  # type: ignore[attr-defined]

        conn = pc_mod.PIConnector(host="bad-server")

        with patch("time.sleep"):
            with pytest.raises(pc_mod.PIConnectionError):
                conn._get_client()

        assert fake_pipolars.PIClient.call_count == pc_mod._MAX_RETRIES  # type: ignore[attr-defined]

    def test_succeeds_on_second_attempt(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        """If first call fails but second succeeds, _get_client returns client."""
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]

        good_client = MagicMock()
        fake_pipolars.PIClient.side_effect = [RuntimeError("fail"), good_client]  # type: ignore[attr-defined]

        conn = pc_mod.PIConnector(host="flaky-server")

        with patch("time.sleep"):
            client = conn._get_client()

        assert client is good_client

    def test_backoff_sleep_called_on_failure(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        """time.sleep is called between retry attempts."""
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]

        fake_pipolars.PIClient.side_effect = RuntimeError("refused")  # type: ignore[attr-defined]

        conn = pc_mod.PIConnector(host="bad-server")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(pc_mod.PIConnectionError):
                conn._get_client()

        # sleep is called once after each failed attempt (including the last)
        assert mock_sleep.call_count == pc_mod._MAX_RETRIES  # type: ignore[attr-defined]

    def test_cached_client_returned_without_reconnect(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        """If _client is already set, _get_client returns it immediately."""
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]

        existing = MagicMock()
        conn = pc_mod.PIConnector(host="h")
        conn._client = existing

        result = conn._get_client()

        assert result is existing
        fake_pipolars.PIClient.assert_not_called()  # type: ignore[attr-defined]

    def test_pi_connection_error_is_sentinel_error(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod
        from sentinel.core.exceptions import SentinelError

        importlib.reload(pc_mod)

        assert issubclass(pc_mod.PIConnectionError, SentinelError)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestContextManager
# ---------------------------------------------------------------------------


class TestContextManager:
    """PIConnector works as a context manager and closes the client on exit."""

    def test_enter_returns_connector(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]

        conn = pc_mod.PIConnector(host="h")
        with conn as ctx:
            assert ctx is conn

    def test_exit_sets_client_to_none(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]

        conn = pc_mod.PIConnector(host="h")
        mock_client = MagicMock()
        conn._client = mock_client

        conn.close()

        assert conn._client is None

    def test_close_calls_client_close(
        self, fake_pipolars: ModuleType, tmp_path: Path
    ) -> None:
        import importlib

        import sentinel.data.pi_connector as pc_mod

        importlib.reload(pc_mod)
        pc_mod._HAS_PIPOLARS = True  # type: ignore[attr-defined]

        conn = pc_mod.PIConnector(host="h")
        mock_client = MagicMock()
        conn._client = mock_client

        conn.close()

        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# TestFindTimeColumn (internal helper)
# ---------------------------------------------------------------------------


class TestFindTimeColumn:
    """_find_time_column locates datetime columns by type or name."""

    def test_finds_datetime_column_by_type(self) -> None:
        from sentinel.data.pi_connector import _find_time_column

        df = pl.DataFrame(
            {
                "Timestamp": pl.Series(["2024-01-01T00:00:00"]).cast(pl.Datetime("us")),
                "val": [1.0],
            }
        )
        assert _find_time_column(df) == "Timestamp"

    def test_finds_by_name_when_type_is_string(self) -> None:
        from sentinel.data.pi_connector import _find_time_column

        df = pl.DataFrame(
            {
                "time": ["2024-01-01"],
                "val": [1.0],
            }
        )
        assert _find_time_column(df) == "time"

    def test_returns_none_when_no_time_column(self) -> None:
        from sentinel.data.pi_connector import _find_time_column

        df = pl.DataFrame({"a": [1.0], "b": [2.0]})
        assert _find_time_column(df) is None

    def test_prefers_datetime_type_over_name_match(self) -> None:
        from sentinel.data.pi_connector import _find_time_column

        df = pl.DataFrame(
            {
                "actual_ts": pl.Series(["2024-01-01T00:00:00"]).cast(pl.Datetime("us")),
                "time": ["not-a-datetime"],
            }
        )
        # datetime type column should be found first
        assert _find_time_column(df) == "actual_ts"
