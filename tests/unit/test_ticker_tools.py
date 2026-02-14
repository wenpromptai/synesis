"""Unit tests for ticker verification tools."""

from unittest.mock import AsyncMock

import pytest

from synesis.processing.common.ticker_tools import verify_ticker


class TestVerifyTicker:
    """Tests for verify_ticker function."""

    @pytest.mark.asyncio
    async def test_no_provider_fallback(self) -> None:
        result = await verify_ticker("AAPL", ticker_provider=None)
        assert "unavailable" in result.lower()
        assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_valid_ticker(self) -> None:
        provider = AsyncMock()
        provider.verify_ticker.return_value = (True, "AAPL-US", "Apple Inc")
        result = await verify_ticker("aapl", provider)
        assert "VERIFIED" in result
        assert "Apple Inc" in result

    @pytest.mark.asyncio
    async def test_invalid_ticker(self) -> None:
        provider = AsyncMock()
        provider.verify_ticker.return_value = (False, None, None)
        result = await verify_ticker("XYZZY", provider)
        assert "NOT FOUND" in result

    @pytest.mark.asyncio
    async def test_provider_error(self) -> None:
        provider = AsyncMock()
        provider.verify_ticker.side_effect = ConnectionError("timeout")
        result = await verify_ticker("AAPL", provider)
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_uppercases_ticker(self) -> None:
        provider = AsyncMock()
        provider.verify_ticker.return_value = (True, "TSLA-US", "Tesla Inc")
        await verify_ticker("tsla", provider)
        provider.verify_ticker.assert_called_once_with("TSLA")
