"""Unit tests for provider factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.providers.factory import (
    create_price_provider,
    create_ticker_provider,
)


@pytest.fixture
def mock_redis() -> AsyncMock:
    return AsyncMock()


class TestCreatePriceProvider:
    """Tests for create_price_provider factory."""

    @pytest.mark.asyncio
    async def test_finnhub_with_key(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.price_provider = "finnhub"
            settings.finnhub_api_key = MagicMock()
            settings.finnhub_api_key.get_secret_value.return_value = "test_key"
            mock_settings.return_value = settings

            provider = await create_price_provider(mock_redis)
            assert provider is not None

    @pytest.mark.asyncio
    async def test_finnhub_no_key_raises(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.price_provider = "finnhub"
            settings.finnhub_api_key = None
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="API key required"):
                await create_price_provider(mock_redis)

    @pytest.mark.asyncio
    async def test_explicit_api_key_override(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.price_provider = "finnhub"
            settings.finnhub_api_key = None
            mock_settings.return_value = settings

            provider = await create_price_provider(mock_redis, api_key="override_key")
            assert provider is not None


class TestCreateTickerProvider:
    """Tests for create_ticker_provider factory."""

    @pytest.mark.asyncio
    async def test_finnhub_ticker_provider(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.ticker_provider = "finnhub"
            settings.finnhub_api_key = MagicMock()
            settings.finnhub_api_key.get_secret_value.return_value = "fh_key"
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
