"""Unit tests for provider factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.providers.factory import (
    FinnhubService,
    create_fundamentals_provider,
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


class TestCreateFundamentalsProvider:
    """Tests for create_fundamentals_provider factory."""

    @pytest.mark.asyncio
    async def test_none_returns_none(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.fundamentals_provider = "none"
            mock_settings.return_value = settings

            provider = await create_fundamentals_provider(mock_redis)
            assert provider is None

    @pytest.mark.asyncio
    async def test_finnhub_with_key(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.fundamentals_provider = "finnhub"
            settings.finnhub_api_key = MagicMock()
            settings.finnhub_api_key.get_secret_value.return_value = "test_key"
            mock_settings.return_value = settings

            provider = await create_fundamentals_provider(mock_redis)
            assert provider is not None

    @pytest.mark.asyncio
    async def test_finnhub_no_key_raises(self, mock_redis: AsyncMock) -> None:
        with patch("synesis.providers.factory.get_settings") as mock_settings:
            settings = MagicMock()
            settings.fundamentals_provider = "finnhub"
            settings.finnhub_api_key = None
            mock_settings.return_value = settings

            with pytest.raises(ValueError, match="API key required"):
                await create_fundamentals_provider(mock_redis)


class TestFinnhubService:
    """Tests for FinnhubService combined class."""

    def test_delegation_to_ticker(self, mock_redis: AsyncMock) -> None:
        service = FinnhubService(api_key="test", redis=mock_redis)
        assert service._ticker is not None
        assert service._fundamentals is not None

    @pytest.mark.asyncio
    async def test_verify_ticker_delegates(self, mock_redis: AsyncMock) -> None:
        service = FinnhubService(api_key="test", redis=mock_redis)
        with patch.object(service._ticker, "verify_ticker", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = (True, "Apple Inc")
            result = await service.verify_ticker("AAPL")
            assert result == (True, "Apple Inc")
            mock_verify.assert_awaited_once_with("AAPL")

    @pytest.mark.asyncio
    async def test_get_basic_financials_delegates(self, mock_redis: AsyncMock) -> None:
        service = FinnhubService(api_key="test", redis=mock_redis)
        with patch.object(
            service._fundamentals, "get_basic_financials", new_callable=AsyncMock
        ) as mock_fin:
            mock_fin.return_value = {"peRatio": 30.0}
            result = await service.get_basic_financials("AAPL")
            assert result == {"peRatio": 30.0}

    @pytest.mark.asyncio
    async def test_close_calls_both(self, mock_redis: AsyncMock) -> None:
        service = FinnhubService(api_key="test", redis=mock_redis)
        with (
            patch.object(service._ticker, "close", new_callable=AsyncMock) as mock_t,
            patch.object(service._fundamentals, "close", new_callable=AsyncMock) as mock_f,
        ):
            await service.close()
            mock_t.assert_awaited_once()
            mock_f.assert_awaited_once()
