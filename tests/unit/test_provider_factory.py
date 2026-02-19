"""Unit tests for provider factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.providers.base import TickerProvider, WatchlistDataProvider
from synesis.providers.factory import (
    create_fundamentals_provider,
    create_ticker_provider,
)


@pytest.fixture
def mock_redis() -> AsyncMock:
    return AsyncMock()


class TestCreateTickerProvider:
    """Tests for create_ticker_provider factory."""

    @pytest.mark.asyncio
    async def test_factset_ticker_provider(self, mock_redis: AsyncMock) -> None:
        with (
            patch("synesis.providers.factory.get_settings") as mock_settings,
            patch("synesis.providers.factset.client.get_factset_client") as mock_client,
        ):
            settings = MagicMock()
            settings.ticker_provider = "factset"
            mock_settings.return_value = settings
            mock_client.return_value = MagicMock()

            provider = await create_ticker_provider(mock_redis)
            assert provider is not None

    @pytest.mark.asyncio
    async def test_finnhub_ticker_provider(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.ticker_provider = "finnhub"
            settings.finnhub_api_key = MagicMock()
            settings.finnhub_api_key.get_secret_value.return_value = "test_key"
            mock_settings.return_value = settings

            provider = await create_ticker_provider(mock_redis)
            assert provider is not None

    @pytest.mark.asyncio
    async def test_finnhub_ticker_no_key_raises(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.ticker_provider = "finnhub"
            settings.finnhub_api_key = None
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="API key required"):
                await create_ticker_provider(mock_redis)

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.ticker_provider = "unsupported"
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="Unsupported ticker provider"):
                await create_ticker_provider(mock_redis)

    @pytest.mark.asyncio
    async def test_finnhub_explicit_api_key_override(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.ticker_provider = "finnhub"
            settings.finnhub_api_key = None
            mock_settings.return_value = settings

            provider = await create_ticker_provider(mock_redis, api_key="override_key")
            assert provider is not None


class TestCreateFundamentalsProvider:
    """Tests for create_fundamentals_provider factory."""

    @pytest.mark.asyncio
    async def test_none_returns_none(self, mock_redis: AsyncMock) -> None:
        """Test that 'none' config returns None."""
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.fundamentals_provider = "none"
            mock_settings.return_value = settings

            result = await create_fundamentals_provider(mock_redis)
            assert result is None

    @pytest.mark.asyncio
    async def test_factset_creates_adapter(self, mock_redis: AsyncMock) -> None:
        """Test that 'factset' config creates a FactSetWatchlistAdapter."""
        with (
            patch("synesis.providers.factory.get_settings") as mock_settings,
            patch("synesis.providers.factset.client.FactSetClient") as mock_client_cls,
        ):
            settings = MagicMock()
            settings.fundamentals_provider = "factset"
            mock_settings.return_value = settings
            mock_client_cls.return_value = MagicMock()

            provider = await create_fundamentals_provider(mock_redis)
            assert provider is not None

            from synesis.providers.factset.provider import FactSetWatchlistAdapter

            assert isinstance(provider, FactSetWatchlistAdapter)

    @pytest.mark.asyncio
    async def test_finnhub_creates_adapter(self, mock_redis: AsyncMock) -> None:
        """Test that 'finnhub' config creates a FinnhubWatchlistAdapter."""
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.fundamentals_provider = "finnhub"
            settings.finnhub_api_key = MagicMock()
            settings.finnhub_api_key.get_secret_value.return_value = "test_key"
            mock_settings.return_value = settings

            provider = await create_fundamentals_provider(mock_redis)
            assert provider is not None

            from synesis.providers.finnhub.fundamentals import FinnhubWatchlistAdapter

            assert isinstance(provider, FinnhubWatchlistAdapter)

    @pytest.mark.asyncio
    async def test_finnhub_no_key_raises(self, mock_redis: AsyncMock) -> None:
        """Test that 'finnhub' without API key raises ValueError."""
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.fundamentals_provider = "finnhub"
            settings.finnhub_api_key = None
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="Finnhub API key required"):
                await create_fundamentals_provider(mock_redis)

    @pytest.mark.asyncio
    async def test_unsupported_raises(self, mock_redis: AsyncMock) -> None:
        """Test that unsupported provider raises ValueError."""
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.fundamentals_provider = "unsupported"
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="Unsupported fundamentals provider"):
                await create_fundamentals_provider(mock_redis)


class TestTickerProviderProtocol:
    """Tests for TickerProvider protocol."""

    def test_ticker_provider_has_close(self) -> None:
        """Verify the TickerProvider protocol includes close()."""
        assert hasattr(TickerProvider, "close")
        # Check it's defined in the protocol's annotations/methods
        assert callable(getattr(TickerProvider, "close", None))


class TestWatchlistDataProviderProtocol:
    """Tests for WatchlistDataProvider protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify the protocol can be used with isinstance()."""
        assert hasattr(WatchlistDataProvider, "__protocol_attrs__") or hasattr(
            WatchlistDataProvider, "__abstractmethods__"
        )

    def test_protocol_methods_exist(self) -> None:
        """Verify all expected methods are defined."""
        for method in [
            "resolve_company",
            "get_market_cap",
            "get_fundamentals",
            "get_price",
            "close",
        ]:
            assert hasattr(WatchlistDataProvider, method)

    def test_factset_adapter_satisfies_protocol(self) -> None:
        """Verify FactSetWatchlistAdapter passes isinstance check."""
        from synesis.providers.factset.provider import FactSetWatchlistAdapter

        adapter = FactSetWatchlistAdapter(provider=MagicMock())
        assert isinstance(adapter, WatchlistDataProvider)

    def test_finnhub_adapter_satisfies_protocol(self) -> None:
        """Verify FinnhubWatchlistAdapter passes isinstance check."""
        from synesis.providers.finnhub.fundamentals import FinnhubWatchlistAdapter

        adapter = FinnhubWatchlistAdapter(api_key="test", redis=MagicMock())
        assert isinstance(adapter, WatchlistDataProvider)
