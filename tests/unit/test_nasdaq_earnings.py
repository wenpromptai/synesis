"""Tests for NASDAQ earnings provider."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import orjson
import pytest

from synesis.providers.nasdaq.client import NasdaqClient, _parse_float, _parse_market_cap
from synesis.providers.nasdaq.models import EarningsEvent


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_EARNINGS_RESPONSE = {
    "data": {
        "asOf": "2026-02-13T00:00:00.000",
        "headers": {"symbol": {}, "name": {}, "marketCap": {}, "epsForecast": {}},
        "rows": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "marketCap": "$3,500,000,000,000",
                "epsForecast": "2.35",
                "noOfEsts": "28",
                "time": "time-after-hours",
                "fiscalQuarterEnding": "Dec/2025",
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corporation",
                "marketCap": "$2,800,000,000,000",
                "epsForecast": "3.10",
                "noOfEsts": "32",
                "time": "time-pre-market",
                "fiscalQuarterEnding": "Dec/2025",
            },
            {
                "symbol": "NVDA",
                "name": "NVIDIA Corporation",
                "marketCap": "N/A",
                "epsForecast": "N/A",
                "noOfEsts": "0",
                "time": "time-not-supplied",
                "fiscalQuarterEnding": "Jan/2026",
            },
        ],
    }
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_redis():
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = True
    return redis


@pytest.fixture()
def client(mock_redis):
    return NasdaqClient(redis=mock_redis)


# ---------------------------------------------------------------------------
# Parsing Helpers
# ---------------------------------------------------------------------------


class TestParsingHelpers:
    def test_parse_market_cap_normal(self):
        assert _parse_market_cap("$3,500,000,000,000") == 3_500_000_000_000.0

    def test_parse_market_cap_na(self):
        assert _parse_market_cap("N/A") is None

    def test_parse_market_cap_empty(self):
        assert _parse_market_cap("") is None

    def test_parse_float_normal(self):
        assert _parse_float("2.35") == 2.35

    def test_parse_float_na(self):
        assert _parse_float("N/A") is None

    def test_parse_float_none(self):
        assert _parse_float(None) is None


# ---------------------------------------------------------------------------
# get_earnings_by_date
# ---------------------------------------------------------------------------


class TestGetEarningsByDate:
    async def test_get_earnings(self, client: NasdaqClient, mock_redis):
        """Test fetching earnings from NASDAQ API."""
        mock_response = MagicMock()
        mock_response.content = orjson.dumps(SAMPLE_EARNINGS_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            events = await client.get_earnings_by_date(date(2026, 2, 13))

        assert len(events) == 3

        aapl = events[0]
        assert aapl.ticker == "AAPL"
        assert aapl.company_name == "Apple Inc."
        assert aapl.time == "after-hours"
        assert aapl.eps_forecast == 2.35
        assert aapl.num_estimates == 28
        assert aapl.market_cap == 3_500_000_000_000.0
        assert aapl.fiscal_quarter == "Dec/2025"

        msft = events[1]
        assert msft.ticker == "MSFT"
        assert msft.time == "pre-market"

        nvda = events[2]
        assert nvda.ticker == "NVDA"
        assert nvda.market_cap is None
        assert nvda.eps_forecast is None
        assert nvda.time == "during-market"

    async def test_get_earnings_from_cache(self, client: NasdaqClient, mock_redis):
        """Test earnings loaded from Redis cache."""
        cached = [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "earnings_date": "2026-02-13",
                "time": "after-hours",
                "eps_forecast": 2.35,
                "num_estimates": 28,
                "market_cap": 3500000000000.0,
                "fiscal_quarter": "Dec/2025",
            }
        ]
        mock_redis.get.return_value = orjson.dumps(cached)

        events = await client.get_earnings_by_date(date(2026, 2, 13))
        assert len(events) == 1
        assert events[0].ticker == "AAPL"

    async def test_get_earnings_empty(self, client: NasdaqClient, mock_redis):
        """Test empty earnings response."""
        mock_response = MagicMock()
        mock_response.content = orjson.dumps({"data": {"rows": []}})
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=mock_response)
            events = await client.get_earnings_by_date(date(2026, 3, 1))

        assert events == []

    async def test_get_earnings_http_failure(self, client: NasdaqClient, mock_redis):
        """Test HTTP failure returns empty list."""
        with patch.object(client, "_get_http_client") as mock_http:
            mock_http.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            events = await client.get_earnings_by_date(date(2026, 2, 13))

        assert events == []


class TestGetUpcomingEarnings:
    async def test_upcoming_filters_tickers(self, client: NasdaqClient, mock_redis):
        """Test upcoming earnings filters to requested tickers."""
        with patch.object(client, "get_earnings_by_date") as mock_by_date:
            mock_by_date.return_value = [
                EarningsEvent(
                    ticker="AAPL",
                    company_name="Apple Inc.",
                    earnings_date=date(2026, 2, 13),
                    time="after-hours",
                ),
                EarningsEvent(
                    ticker="MSFT",
                    company_name="Microsoft Corporation",
                    earnings_date=date(2026, 2, 13),
                    time="pre-market",
                ),
                EarningsEvent(
                    ticker="NVDA",
                    company_name="NVIDIA",
                    earnings_date=date(2026, 2, 13),
                    time="during-market",
                ),
            ]

            events = await client.get_upcoming_earnings(["AAPL", "NVDA"], days=1)

        assert len(events) == 2
        tickers = {e.ticker for e in events}
        assert tickers == {"AAPL", "NVDA"}

    async def test_upcoming_no_matches(self, client: NasdaqClient, mock_redis):
        """Test upcoming returns empty when no tickers match."""
        with patch.object(client, "get_earnings_by_date") as mock_by_date:
            mock_by_date.return_value = [
                EarningsEvent(
                    ticker="TSLA",
                    company_name="Tesla",
                    earnings_date=date(2026, 2, 13),
                    time="after-hours",
                ),
            ]

            events = await client.get_upcoming_earnings(["AAPL"], days=1)

        assert events == []
