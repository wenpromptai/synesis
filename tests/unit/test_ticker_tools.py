"""Unit tests for ticker verification tools."""

import pytest

from synesis.processing.common.ticker_tools import verify_ticker


class TestVerifyTicker:
    """Tests for verify_ticker — checks against data/us_tickers.json."""

    @pytest.mark.asyncio
    async def test_known_ticker(self) -> None:
        """AAPL is in us_tickers.json → VERIFIED."""
        result = await verify_ticker("AAPL")
        assert "VERIFIED" in result
        assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_unknown_ticker(self) -> None:
        """XYZZY is not in us_tickers.json → NOT FOUND."""
        result = await verify_ticker("XYZZY")
        assert "NOT FOUND" in result

    @pytest.mark.asyncio
    async def test_uppercases_ticker(self) -> None:
        """Lowercase input is uppercased before lookup."""
        result = await verify_ticker("aapl")
        assert "VERIFIED" in result
        assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_returns_company_name(self) -> None:
        """VERIFIED result includes company name from file."""
        result = await verify_ticker("MSFT")
        assert "VERIFIED" in result
        assert "MICROSOFT" in result.upper()
