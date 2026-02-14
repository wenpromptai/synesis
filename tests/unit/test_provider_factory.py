"""Unit tests for provider factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.providers.base import TickerProvider
from synesis.providers.factory import (
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


class TestTickerProviderProtocol:
    """Tests for TickerProvider protocol."""

    def test_ticker_provider_has_close(self) -> None:
        """Verify the TickerProvider protocol includes close()."""
        assert hasattr(TickerProvider, "close")
        # Check it's defined in the protocol's annotations/methods
        assert callable(getattr(TickerProvider, "close", None))
