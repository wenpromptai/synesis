"""Unit tests for provider factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.providers.base import TickerProvider, WatchlistDataProvider
from synesis.providers.factory import create_ticker_provider


@pytest.fixture
def mock_redis() -> AsyncMock:
    return AsyncMock()


class TestCreateTickerProvider:
    """Tests for create_ticker_provider factory."""

    @pytest.mark.asyncio
    async def test_creates_finnhub_provider(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.finnhub_api_key = MagicMock()
            settings.finnhub_api_key.get_secret_value.return_value = "test_key"
            mock_settings.return_value = settings

            provider = await create_ticker_provider(mock_redis)
            assert provider is not None

    @pytest.mark.asyncio
    async def test_no_key_raises(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.finnhub_api_key = None
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="FINNHUB_API_KEY required"):
                await create_ticker_provider(mock_redis)

    @pytest.mark.asyncio
    async def test_explicit_api_key_override(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.finnhub_api_key = None
            mock_settings.return_value = settings

            provider = await create_ticker_provider(mock_redis, api_key="override_key")
            assert provider is not None


class TestTickerProviderProtocol:
    """Tests for TickerProvider protocol."""

    def test_ticker_provider_has_close(self) -> None:
        assert hasattr(TickerProvider, "close")
        assert callable(getattr(TickerProvider, "close", None))


class TestWatchlistDataProviderProtocol:
    """Tests for WatchlistDataProvider protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(WatchlistDataProvider, "__protocol_attrs__") or hasattr(
            WatchlistDataProvider, "__abstractmethods__"
        )

    def test_protocol_methods_exist(self) -> None:
        for method in [
            "resolve_company",
            "get_market_cap",
            "get_fundamentals",
            "get_price",
            "close",
        ]:
            assert hasattr(WatchlistDataProvider, method)

    def test_finnhub_adapter_satisfies_protocol(self) -> None:
        from synesis.providers.finnhub.fundamentals import FinnhubWatchlistAdapter

        adapter = FinnhubWatchlistAdapter(api_key="test", redis=MagicMock())
        assert isinstance(adapter, WatchlistDataProvider)
