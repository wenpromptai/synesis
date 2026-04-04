"""Tests for Massive.com provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import orjson
import pytest

from synesis.providers.massive.client import MassiveClient

# ---------------------------------------------------------------------------
# Sample API Responses
# ---------------------------------------------------------------------------

SAMPLE_BARS = {
    "ticker": "AAPL",
    "queryCount": 2,
    "resultsCount": 2,
    "adjusted": True,
    "results": [
        {
            "o": 262.41,
            "h": 266.53,
            "l": 260.2,
            "c": 264.72,
            "v": 41827946.0,
            "vw": 264.4409,
            "t": 1772427600000,
            "n": 705079,
        },
        {
            "o": 263.48,
            "h": 265.56,
            "l": 260.13,
            "c": 263.75,
            "v": 38568921.0,
            "vw": 263.1124,
            "t": 1772514000000,
            "n": 627107,
        },
    ],
    "status": "OK",
}

SAMPLE_PREV_DAY = {
    "ticker": "AAPL",
    "queryCount": 1,
    "resultsCount": 1,
    "adjusted": True,
    "results": [
        {
            "T": "AAPL",
            "o": 254.2,
            "h": 256.13,
            "l": 250.65,
            "c": 255.92,
            "v": 31289369.0,
            "vw": 254.6924,
            "t": 1775160000000,
            "n": 533471,
        }
    ],
    "status": "OK",
}

SAMPLE_DAILY_SUMMARY = {
    "status": "OK",
    "from": "2026-04-02",
    "symbol": "AAPL",
    "open": 254.2,
    "high": 256.13,
    "low": 250.65,
    "close": 255.92,
    "volume": 31289369.0,
    "afterHours": 255.45,
    "preMarket": 252.26,
}

SAMPLE_GROUPED_DAILY = {
    "queryCount": 2,
    "resultsCount": 2,
    "adjusted": True,
    "results": [
        {
            "T": "AAPL",
            "o": 254.2,
            "h": 256.13,
            "l": 250.65,
            "c": 255.92,
            "v": 31289369.0,
            "vw": 254.69,
            "t": 1775160000000,
            "n": 533471,
        },
        {
            "T": "MSFT",
            "o": 380.0,
            "h": 385.0,
            "l": 378.0,
            "c": 383.5,
            "v": 20000000.0,
            "vw": 382.1,
            "t": 1775160000000,
            "n": 300000,
        },
    ],
    "status": "OK",
}

SAMPLE_TICKERS_SEARCH = {
    "results": [
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "market": "stocks",
            "locale": "us",
            "type": "CS",
            "active": True,
            "primary_exchange": "XNAS",
            "currency_name": "usd",
            "cik": "0000320193",
            "composite_figi": "BBG000B9XRY4",
        }
    ],
    "status": "OK",
    "count": 1,
}

SAMPLE_TICKER_OVERVIEW = {
    "request_id": "abc123",
    "status": "OK",
    "results": {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "market": "stocks",
        "locale": "us",
        "primary_exchange": "XNAS",
        "type": "CS",
        "active": True,
        "currency_name": "usd",
        "cik": "0000320193",
        "composite_figi": "BBG000B9XRY4",
        "market_cap": 3757197348800.0,
        "description": "Apple is among the largest companies.",
        "homepage_url": "https://www.apple.com",
        "total_employees": 166000,
        "list_date": "1980-12-12",
        "sic_code": "3571",
        "sic_description": "ELECTRONIC COMPUTERS",
        "weighted_shares_outstanding": 14681140000,
        "phone_number": "1-408-996-1010",
        "address": {
            "address1": "One Apple Park Way",
            "city": "Cupertino",
            "state": "CA",
            "postal_code": "95014",
        },
        "branding": {
            "logo_url": "https://api.polygon.io/v1/reference/company-branding/AAPL/logo.svg",
            "icon_url": "https://api.polygon.io/v1/reference/company-branding/AAPL/icon.png",
        },
        "ticker_root": "AAPL",
        "share_class_figi": "BBG001S5N8V8",
        "share_class_shares_outstanding": 14681140000,
        "round_lot": 100,
    },
}

SAMPLE_TICKER_EVENTS = {
    "results": {
        "name": "Apple Inc.",
        "events": [
            {"type": "ticker_change", "date": "2003-09-10", "ticker_change": {"ticker": "AAPL"}}
        ],
    }
}

SAMPLE_RELATED = {
    "request_id": "abc",
    "results": [{"ticker": "MSFT"}, {"ticker": "AMZN"}, {"ticker": "GOOG"}],
}

SAMPLE_FINANCIALS = {
    "results": [
        {
            "start_date": "2024-12-28",
            "end_date": "2025-12-27",
            "tickers": ["AAPL"],
            "filing_date": "2025-10-31",
            "fiscal_period": "FY",
            "fiscal_year": "2025",
            "timeframe": "annual",
            "financials": {
                "income_statement": {"revenues": {"value": 391035000000}},
                "balance_sheet": {"assets": {"value": 352583000000}},
            },
        }
    ],
    "status": "OK",
}

SAMPLE_DIVIDENDS = {
    "status": "OK",
    "results": [
        {
            "ticker": "AAPL",
            "ex_dividend_date": "2022-11-04",
            "pay_date": "2022-11-10",
            "record_date": "2022-11-07",
            "declaration_date": "2022-10-27",
            "cash_amount": 0.23,
            "currency": "USD",
            "frequency": 4,
            "distribution_type": "recurring",
        }
    ],
}

SAMPLE_SPLITS = {
    "status": "OK",
    "results": [
        {
            "ticker": "AAPL",
            "execution_date": "2020-08-31",
            "split_from": 1.0,
            "split_to": 4.0,
            "adjustment_type": "forward_split",
        }
    ],
}

SAMPLE_SHORT_INTEREST = {
    "status": "OK",
    "results": [
        {
            "ticker": "AAPL",
            "settlement_date": "2017-12-29",
            "short_interest": 45746430,
            "avg_daily_volume": 23901107,
            "days_to_cover": 1.91,
        }
    ],
}

SAMPLE_SHORT_VOLUME = {
    "status": "OK",
    "results": [
        {
            "ticker": "AAPL",
            "date": "2024-02-06",
            "total_volume": 16264662.0,
            "short_volume": 5683713.0,
            "exempt_volume": 67840.0,
        }
    ],
}

SAMPLE_NEWS = {
    "results": [
        {
            "id": "abc123",
            "title": "Apple Earnings Beat",
            "published_utc": "2026-04-03T16:12:00Z",
            "article_url": "https://example.com/article",
            "tickers": ["AAPL"],
            "description": "Apple reported strong earnings.",
            "keywords": ["earnings", "apple"],
            "insights": [
                {
                    "ticker": "AAPL",
                    "sentiment": "positive",
                    "sentiment_reasoning": "Beat estimates.",
                }
            ],
            "author": "Test Author",
            "image_url": "https://example.com/img.jpg",
        }
    ],
    "count": 1,
}

SAMPLE_SMA = {
    "results": {
        "underlying": {"url": "https://api.polygon.io/..."},
        "values": [
            {"timestamp": 1775102400000, "value": 260.363},
            {"timestamp": 1775016000000, "value": 259.874},
        ],
    }
}

SAMPLE_MACD = {
    "results": {
        "underlying": {"url": "https://api.polygon.io/..."},
        "values": [
            {"timestamp": 1775102400000, "value": -2.404, "signal": -3.198, "histogram": 0.795}
        ],
    }
}

SAMPLE_MARKET_STATUS = {
    "afterHours": False,
    "earlyHours": False,
    "market": "closed",
    "serverTime": "2026-04-04T12:00:00-04:00",
    "exchanges": {"nasdaq": "closed", "nyse": "closed", "otc": "closed"},
    "currencies": {"crypto": "open", "fx": "open"},
}

SAMPLE_MARKET_HOLIDAYS = [
    {"date": "2026-04-03", "exchange": "NYSE", "name": "Good Friday", "status": "closed"},
    {"date": "2026-04-03", "exchange": "NASDAQ", "name": "Good Friday", "status": "closed"},
]

SAMPLE_OPTIONS_CONTRACTS = {
    "results": [
        {
            "ticker": "O:AAPL260406C00180000",
            "underlying_ticker": "AAPL",
            "contract_type": "call",
            "exercise_style": "american",
            "expiration_date": "2026-04-06",
            "strike_price": 180.0,
            "shares_per_contract": 100,
            "primary_exchange": "BATO",
            "cfi": "OCASPS",
        }
    ],
    "status": "OK",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(data: dict | list) -> MagicMock:
    resp = MagicMock()
    resp.content = orjson.dumps(data)
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_rate_limiter():
    """Bypass the Massive rate limiter for all unit tests."""
    with patch("synesis.providers.massive.client._rate_limiter") as mock_limiter:
        mock_limiter.acquire = AsyncMock(return_value=None)
        yield mock_limiter


@pytest.fixture()
def mock_redis():
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = True
    return redis


@pytest.fixture()
def client(mock_redis):
    with patch("synesis.providers.massive.client.get_settings") as mock_settings:
        settings = MagicMock()
        settings.massive_api_key.get_secret_value.return_value = "test_key"
        settings.massive_api_url = "https://api.massive.com"
        settings.massive_cache_ttl_bars = 300
        settings.massive_cache_ttl_reference = 21600
        settings.massive_cache_ttl_static = 86400
        settings.massive_cache_ttl_fundamentals = 21600
        settings.massive_cache_ttl_news = 900
        settings.massive_cache_ttl_indicators = 300
        settings.massive_cache_ttl_market_status = 60
        settings.massive_cache_ttl_options = 3600
        mock_settings.return_value = settings
        return MassiveClient(redis=mock_redis)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_missing_api_key_raises(self, mock_redis):
        with patch("synesis.providers.massive.client.get_settings") as mock_settings:
            settings = MagicMock()
            settings.massive_api_key = None
            mock_settings.return_value = settings
            with pytest.raises(ValueError, match="MASSIVE_API_KEY"):
                MassiveClient(redis=mock_redis)

    def test_valid_init(self, client: MassiveClient):
        assert client._api_key == "test_key"
        assert client._base_url == "https://api.massive.com"


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


class TestGetBars:
    async def test_get_bars_cache_miss(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_BARS))
            result = await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-04-01")

        assert result is not None
        assert result.ticker == "AAPL"
        assert len(result.bars) == 2
        assert result.bars[0].open == 262.41
        assert result.bars[0].high == 266.53
        assert result.bars[0].close == 264.72
        assert result.bars[0].vwap == 264.4409
        assert result.bars[0].transactions == 705079
        assert result.adjusted is True

    async def test_get_bars_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_BARS)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-04-01")
            mock_http.assert_not_called()

        assert result is not None
        assert result.ticker == "AAPL"

    async def test_get_bars_empty_response_returns_none(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-04-01")
        assert result is None

    async def test_get_bars_http_error_returns_none(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_http.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=mock_resp)
            )
            result = await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-04-01")
        assert result is None


class TestGetPreviousDay:
    async def test_get_previous_day(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_PREV_DAY))
            result = await client.get_previous_day("AAPL")

        assert result is not None
        assert result.ticker == "AAPL"
        assert len(result.bars) == 1
        assert result.bars[0].close == 255.92
        assert result.bars[0].volume == 31289369.0

    async def test_get_previous_day_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_PREV_DAY)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_previous_day("AAPL")
            mock_http.assert_not_called()
        assert result is not None

    async def test_get_previous_day_empty_returns_none(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_previous_day("AAPL")
        assert result is None


class TestGetDailySummary:
    async def test_get_daily_summary(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_DAILY_SUMMARY)
            )
            result = await client.get_daily_summary("AAPL", "2026-04-02")

        assert result is not None
        assert result.ticker == "AAPL"
        assert result.date == "2026-04-02"
        assert result.open == 254.2
        assert result.after_hours == 255.45
        assert result.pre_market == 252.26

    async def test_get_daily_summary_non_ok_returns_none(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response({"status": "NOT_FOUND"})
            )
            result = await client.get_daily_summary("AAPL", "2026-04-02")
        assert result is None

    async def test_get_daily_summary_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_DAILY_SUMMARY)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_daily_summary("AAPL", "2026-04-02")
            mock_http.assert_not_called()
        assert result is not None


class TestGetGroupedDaily:
    async def test_get_grouped_daily(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_GROUPED_DAILY)
            )
            result = await client.get_grouped_daily("2026-04-02")

        assert result is not None
        assert len(result) == 2
        assert result[0].open == 254.2
        assert result[1].open == 380.0

    async def test_get_grouped_daily_empty_returns_none(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_grouped_daily("2026-04-02")
        assert result is None


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------


class TestSearchTickers:
    async def test_search_tickers(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_TICKERS_SEARCH)
            )
            result = await client.search_tickers("Apple")

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].name == "Apple Inc."
        assert result[0].market == "stocks"
        assert result[0].cik == "0000320193"
        assert result[0].composite_figi == "BBG000B9XRY4"

    async def test_search_tickers_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_TICKERS_SEARCH)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.search_tickers("Apple")
            mock_http.assert_not_called()
        assert len(result) == 1

    async def test_search_tickers_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.search_tickers("NONEXISTENT")
        assert result == []

    async def test_search_tickers_with_filters(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_TICKERS_SEARCH)
            )
            result = await client.search_tickers(
                query="Apple", market="stocks", ticker_type="CS", active=True, limit=10
            )
        assert len(result) == 1


class TestGetTickerOverview:
    async def test_get_ticker_overview(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_TICKER_OVERVIEW)
            )
            result = await client.get_ticker_overview("AAPL")

        assert result is not None
        assert result.ticker == "AAPL"
        assert result.name == "Apple Inc."
        assert result.market_cap == 3757197348800.0
        assert result.total_employees == 166000
        assert result.sic_code == "3571"
        assert result.description == "Apple is among the largest companies."

    async def test_get_ticker_overview_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_TICKER_OVERVIEW)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_ticker_overview("AAPL")
            mock_http.assert_not_called()
        assert result is not None

    async def test_get_ticker_overview_empty_returns_none(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_ticker_overview("AAPL")
        assert result is None

    async def test_get_ticker_overview_no_results_returns_none(
        self, client: MassiveClient, mock_redis
    ):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response({"status": "OK", "results": None})
            )
            result = await client.get_ticker_overview("AAPL")
        assert result is None


class TestGetTickerEvents:
    async def test_get_ticker_events(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_TICKER_EVENTS)
            )
            result = await client.get_ticker_events("AAPL")

        assert len(result) == 1
        assert result[0].type == "ticker_change"
        assert result[0].date == "2003-09-10"
        assert result[0].ticker_change == {"ticker": "AAPL"}

    async def test_get_ticker_events_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_ticker_events("AAPL")
        assert result == []


class TestGetRelatedTickers:
    async def test_get_related_tickers(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_RELATED))
            result = await client.get_related_tickers("AAPL")

        assert result == ["MSFT", "AMZN", "GOOG"]

    async def test_get_related_tickers_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_RELATED)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_related_tickers("AAPL")
            mock_http.assert_not_called()
        assert len(result) == 3

    async def test_get_related_tickers_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_related_tickers("AAPL")
        assert result == []


class TestGetTickerTypes:
    async def test_get_ticker_types(self, client: MassiveClient, mock_redis):
        sample = {
            "results": [
                {"code": "CS", "description": "Common Stock"},
                {"code": "ETF", "description": "Exchange Traded Fund"},
            ]
        }
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(sample))
            result = await client.get_ticker_types()
        assert len(result) == 2
        assert result[0]["code"] == "CS"

    async def test_get_ticker_types_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_ticker_types()
        assert result == []


class TestGetExchanges:
    async def test_get_exchanges(self, client: MassiveClient, mock_redis):
        sample = {"results": [{"mic": "XNAS", "name": "NASDAQ"}]}
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(sample))
            result = await client.get_exchanges()
        assert len(result) == 1
        assert result[0]["mic"] == "XNAS"

    async def test_get_exchanges_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_exchanges()
        assert result == []


class TestGetConditions:
    async def test_get_conditions(self, client: MassiveClient, mock_redis):
        sample = {"results": [{"id": 1, "name": "Regular Sale"}]}
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(sample))
            result = await client.get_conditions()
        assert len(result) == 1

    async def test_get_conditions_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_conditions()
        assert result == []


# ---------------------------------------------------------------------------
# Fundamentals & Corporate Actions
# ---------------------------------------------------------------------------


class TestGetFinancials:
    async def test_get_financials(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_FINANCIALS))
            result = await client.get_financials("AAPL")

        assert len(result) == 1
        assert result[0].tickers == ["AAPL"]
        assert result[0].fiscal_period == "FY"
        assert result[0].fiscal_year == "2025"
        assert result[0].timeframe == "annual"
        assert "income_statement" in result[0].financials

    async def test_get_financials_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_FINANCIALS)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_financials("AAPL")
            mock_http.assert_not_called()
        assert len(result) == 1

    async def test_get_financials_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_financials("AAPL")
        assert result == []

    async def test_get_financials_with_timeframe(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_FINANCIALS))
            result = await client.get_financials("AAPL", timeframe="annual", limit=1)
        assert len(result) == 1


class TestGetDividends:
    async def test_get_dividends(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_DIVIDENDS))
            result = await client.get_dividends("AAPL")

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].ex_dividend_date == "2022-11-04"
        assert result[0].cash_amount == 0.23
        assert result[0].frequency == 4
        assert result[0].distribution_type == "recurring"

    async def test_get_dividends_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_DIVIDENDS)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_dividends("AAPL")
            mock_http.assert_not_called()
        assert len(result) == 1

    async def test_get_dividends_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_dividends("AAPL")
        assert result == []


class TestGetSplits:
    async def test_get_splits(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_SPLITS))
            result = await client.get_splits("AAPL")

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].execution_date == "2020-08-31"
        assert result[0].split_from == 1.0
        assert result[0].split_to == 4.0
        assert result[0].adjustment_type == "forward_split"

    async def test_get_splits_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_SPLITS)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_splits("AAPL")
            mock_http.assert_not_called()
        assert len(result) == 1

    async def test_get_splits_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_splits("AAPL")
        assert result == []


class TestGetShortInterest:
    async def test_get_short_interest(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_SHORT_INTEREST)
            )
            result = await client.get_short_interest("AAPL")

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].settlement_date == "2017-12-29"
        assert result[0].short_interest == 45746430
        assert result[0].avg_daily_volume == 23901107
        assert result[0].days_to_cover == 1.91

    async def test_get_short_interest_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_short_interest("AAPL")
        assert result == []


class TestGetShortVolume:
    async def test_get_short_volume(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_SHORT_VOLUME))
            result = await client.get_short_volume("AAPL")

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].date == "2024-02-06"
        assert result[0].short_volume == 5683713.0
        assert result[0].total_volume == 16264662.0
        assert result[0].exempt_volume == 67840.0

    async def test_get_short_volume_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_short_volume("AAPL")
        assert result == []


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


class TestGetNews:
    async def test_get_news(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_NEWS))
            result = await client.get_news("AAPL")

        assert len(result) == 1
        article = result[0]
        assert article.id == "abc123"
        assert article.title == "Apple Earnings Beat"
        assert article.tickers == ["AAPL"]
        assert article.author == "Test Author"
        assert len(article.insights) == 1
        assert article.insights[0].sentiment == "positive"
        assert article.insights[0].sentiment_reasoning == "Beat estimates."

    async def test_get_news_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_NEWS)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_news("AAPL")
            mock_http.assert_not_called()
        assert len(result) == 1

    async def test_get_news_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_news("AAPL")
        assert result == []

    async def test_get_news_with_date_filter(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_NEWS))
            result = await client.get_news(ticker="AAPL", published_utc_gte="2026-04-01", limit=5)
        assert len(result) == 1

    async def test_get_news_no_ticker(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_NEWS))
            result = await client.get_news()
        assert len(result) == 1

    async def test_get_news_http_error_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_http.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "rate limit", request=MagicMock(), response=mock_resp
                )
            )
            result = await client.get_news("AAPL")
        assert result == []


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------


class TestGetSMA:
    async def test_get_sma(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_SMA))
            result = await client.get_sma("AAPL")

        assert len(result) == 2
        assert result[0].timestamp == 1775102400000
        assert result[0].value == 260.363
        assert result[1].value == 259.874

    async def test_get_sma_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_SMA)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_sma("AAPL")
            mock_http.assert_not_called()
        assert len(result) == 2

    async def test_get_sma_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_sma("AAPL")
        assert result == []

    async def test_get_sma_params(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_SMA))
            result = await client.get_sma("AAPL", timespan="week", window=20, limit=5)
        assert len(result) == 2


class TestGetEMA:
    async def test_get_ema(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_SMA))
            result = await client.get_ema("AAPL")
        assert len(result) == 2
        assert result[0].value == 260.363

    async def test_get_ema_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_ema("AAPL")
        assert result == []


class TestGetRSI:
    async def test_get_rsi(self, client: MassiveClient, mock_redis):
        sample_rsi = {"results": {"values": [{"timestamp": 1775102400000, "value": 58.3}]}}
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(sample_rsi))
            result = await client.get_rsi("AAPL")
        assert len(result) == 1
        assert result[0].value == 58.3

    async def test_get_rsi_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_rsi("AAPL")
        assert result == []


class TestGetMACD:
    async def test_get_macd(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_MACD))
            result = await client.get_macd("AAPL")

        assert len(result) == 1
        assert result[0].timestamp == 1775102400000
        assert result[0].value == -2.404
        assert result[0].signal == -3.198
        assert result[0].histogram == 0.795

    async def test_get_macd_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_MACD)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_macd("AAPL")
            mock_http.assert_not_called()
        assert len(result) == 1

    async def test_get_macd_empty_returns_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_macd("AAPL")
        assert result == []

    async def test_get_macd_custom_windows(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_MACD))
            result = await client.get_macd("AAPL", short_window=8, long_window=21, signal_window=5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Market Operations
# ---------------------------------------------------------------------------


class TestGetMarketStatus:
    async def test_get_market_status(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_MARKET_STATUS)
            )
            result = await client.get_market_status()

        assert result is not None
        assert result.market == "closed"
        assert result.after_hours is False
        assert result.early_hours is False
        assert result.exchanges["nasdaq"] == "closed"
        assert result.currencies["crypto"] == "open"

    async def test_get_market_status_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_MARKET_STATUS)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_market_status()
            mock_http.assert_not_called()
        assert result is not None

    async def test_get_market_status_empty_returns_none(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_market_status()
        assert result is None

    async def test_get_market_status_missing_market_key_returns_none(
        self, client: MassiveClient, mock_redis
    ):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response({"serverTime": "2026-04-04T12:00:00"})
            )
            result = await client.get_market_status()
        assert result is None


class TestGetMarketHolidays:
    async def test_get_market_holidays(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_MARKET_HOLIDAYS)
            )
            result = await client.get_market_holidays()

        assert len(result) == 2
        assert result[0].date == "2026-04-03"
        assert result[0].exchange == "NYSE"
        assert result[0].name == "Good Friday"
        assert result[0].status == "closed"

    async def test_get_market_holidays_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_MARKET_HOLIDAYS)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_market_holidays()
            mock_http.assert_not_called()
        assert len(result) == 2

    async def test_get_market_holidays_non_list_returns_empty(
        self, client: MassiveClient, mock_redis
    ):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_market_holidays()
        assert result == []


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class TestGetOptionsContracts:
    async def test_get_options_contracts(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_OPTIONS_CONTRACTS)
            )
            result = await client.get_options_contracts("AAPL")

        assert len(result) == 1
        contract = result[0]
        assert contract.ticker == "O:AAPL260406C00180000"
        assert contract.underlying_ticker == "AAPL"
        assert contract.contract_type == "call"
        assert contract.exercise_style == "american"
        assert contract.expiration_date == "2026-04-06"
        assert contract.strike_price == 180.0
        assert contract.shares_per_contract == 100
        assert contract.cfi == "OCASPS"

    async def test_get_options_contracts_cache_hit(self, client: MassiveClient, mock_redis):
        mock_redis.get.return_value = orjson.dumps(SAMPLE_OPTIONS_CONTRACTS)
        with patch.object(client, "_get_http") as mock_http:
            result = await client.get_options_contracts("AAPL")
            mock_http.assert_not_called()
        assert len(result) == 1

    async def test_get_options_contracts_empty_returns_list(
        self, client: MassiveClient, mock_redis
    ):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_options_contracts("AAPL")
        assert result == []

    async def test_get_options_contracts_with_filters(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_OPTIONS_CONTRACTS)
            )
            result = await client.get_options_contracts(
                "AAPL",
                contract_type="call",
                expiration_date="2026-04-06",
                strike_price=180.0,
                expired=False,
                limit=50,
            )
        assert len(result) == 1

    async def test_get_options_contracts_http_error_returns_list(
        self, client: MassiveClient, mock_redis
    ):
        with patch.object(client, "_get_http") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 403
            mock_http.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "forbidden", request=MagicMock(), response=mock_resp
                )
            )
            result = await client.get_options_contracts("AAPL")
        assert result == []


# ---------------------------------------------------------------------------
# Cache + HTTP Interaction
# ---------------------------------------------------------------------------


class TestCaching:
    async def test_cache_miss_calls_http_and_caches(self, client: MassiveClient, mock_redis):
        """Verify cache miss triggers HTTP request and stores result in Redis."""
        mock_redis.get.return_value = None
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_BARS))
            await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-04-01")

        mock_redis.get.assert_called_once()
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args.kwargs.get("ex") == 300  # massive_cache_ttl_bars

    async def test_cache_hit_skips_http(self, client: MassiveClient, mock_redis):
        """Verify cache hit skips HTTP request."""
        mock_redis.get.return_value = orjson.dumps(SAMPLE_BARS)
        with patch.object(client, "_get_http") as mock_http:
            await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-04-01")
            mock_http.assert_not_called()

        mock_redis.set.assert_not_called()

    async def test_corrupted_cache_falls_back_to_http(self, client: MassiveClient, mock_redis):
        """Verify corrupted cache bytes don't crash and fall back to HTTP."""
        mock_redis.get.return_value = b"not-valid-json{{{"
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(SAMPLE_BARS))
            result = await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-04-01")

        assert result is not None
        mock_http.return_value.get.assert_called_once()

    async def test_empty_response_not_cached(self, client: MassiveClient, mock_redis):
        """Verify empty API response is not cached (avoids poisoning cache)."""
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            await client.get_bars("AAPL", 1, "day", "2026-01-01", "2026-04-01")

        mock_redis.set.assert_not_called()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_close_active_client(self, client: MassiveClient):
        mock_http = AsyncMock()
        client._http = mock_http
        await client.close()
        mock_http.aclose.assert_called_once()
        assert client._http is None

    async def test_close_no_client(self, client: MassiveClient):
        assert client._http is None
        await client.close()  # Should not raise

    async def test_get_http_lazy_init(self, client: MassiveClient):
        assert client._http is None
        http = client._get_http()
        assert http is not None
        assert isinstance(http, type(http))
        assert client._http is http
        await client.close()

    async def test_get_http_returns_same_instance(self, client: MassiveClient):
        http1 = client._get_http()
        http2 = client._get_http()
        assert http1 is http2
        await client.close()


# ---------------------------------------------------------------------------
# _build_params helper
# ---------------------------------------------------------------------------


class TestBuildParams:
    def test_gte_suffix_maps_to_dot_notation(self, client: MassiveClient):
        params = client._build_params(expiration_date_gte="2024-01-01")
        assert params == {"expiration_date.gte": "2024-01-01"}

    def test_gt_suffix_maps_to_dot_notation(self, client: MassiveClient):
        params = client._build_params(strike_price_gt=100.0)
        assert params == {"strike_price.gt": 100.0}

    def test_lte_suffix_maps_to_dot_notation(self, client: MassiveClient):
        params = client._build_params(filing_date_lte="2025-12-31")
        assert params == {"filing_date.lte": "2025-12-31"}

    def test_lt_suffix_maps_to_dot_notation(self, client: MassiveClient):
        params = client._build_params(settlement_date_lt="2024-06-01")
        assert params == {"settlement_date.lt": "2024-06-01"}

    def test_any_of_suffix_maps_to_dot_notation(self, client: MassiveClient):
        params = client._build_params(ticker_any_of="AAPL,MSFT")
        assert params == {"ticker.any_of": "AAPL,MSFT"}

    def test_none_values_are_skipped(self, client: MassiveClient):
        params = client._build_params(limit=10, ticker=None, sort=None)
        assert params == {"limit": 10}

    def test_plain_params_pass_through_unchanged(self, client: MassiveClient):
        params = client._build_params(limit=50, ticker="AAPL", timespan="day")
        assert params == {"limit": 50, "ticker": "AAPL", "timespan": "day"}

    def test_mixed_plain_and_range_params(self, client: MassiveClient):
        params = client._build_params(
            ticker="AAPL",
            limit=100,
            ex_dividend_date_gte="2024-01-01",
            ex_dividend_date_lte="2024-12-31",
            sort=None,
        )
        assert params == {
            "ticker": "AAPL",
            "limit": 100,
            "ex_dividend_date.gte": "2024-01-01",
            "ex_dividend_date.lte": "2024-12-31",
        }

    def test_empty_kwargs_returns_empty_dict(self, client: MassiveClient):
        params = client._build_params()
        assert params == {}

    def test_all_none_returns_empty_dict(self, client: MassiveClient):
        params = client._build_params(a=None, b=None, c=None)
        assert params == {}


# ---------------------------------------------------------------------------
# _get_all_pages pagination helper
# ---------------------------------------------------------------------------

SAMPLE_PAGE_1 = {
    "results": [{"ticker": "O:AAPL260406C00180000", "underlying_ticker": "AAPL"}],
    "next_url": "https://api.massive.com/v3/reference/options/contracts?cursor=page2",
    "status": "OK",
}

SAMPLE_PAGE_2 = {
    "results": [{"ticker": "O:AAPL260406P00180000", "underlying_ticker": "AAPL"}],
    "status": "OK",
}

SAMPLE_PAGE_2_WITH_NEXT = {
    "results": [{"ticker": "O:AAPL260406P00180000", "underlying_ticker": "AAPL"}],
    "next_url": "https://api.massive.com/v3/reference/options/contracts?cursor=page3",
    "status": "OK",
}

SAMPLE_PAGE_3 = {
    "results": [{"ticker": "O:AAPL260413C00185000", "underlying_ticker": "AAPL"}],
    "status": "OK",
}


class TestGetAllPages:
    async def test_single_page_no_next_url(self, client: MassiveClient, mock_redis):
        """When there is no next_url, only one page is fetched."""
        single_page = {"results": [{"ticker": "O:AAPL260406C00180000"}], "status": "OK"}
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(single_page))
            results = await client._get_all_pages("v3/reference/options/contracts", {})
        assert len(results) == 1
        assert results[0]["ticker"] == "O:AAPL260406C00180000"

    async def test_two_pages_via_next_url(self, client: MassiveClient, mock_redis):
        """Two pages are merged when next_url is present on page 1."""
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                side_effect=[_mock_response(SAMPLE_PAGE_1), _mock_response(SAMPLE_PAGE_2)]
            )
            results = await client._get_all_pages("v3/reference/options/contracts", {})
        assert len(results) == 2
        assert results[0]["ticker"] == "O:AAPL260406C00180000"
        assert results[1]["ticker"] == "O:AAPL260406P00180000"

    async def test_three_pages_chained(self, client: MassiveClient, mock_redis):
        """Three-page chain is fully consumed."""
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                side_effect=[
                    _mock_response(SAMPLE_PAGE_1),
                    _mock_response(SAMPLE_PAGE_2_WITH_NEXT),
                    _mock_response(SAMPLE_PAGE_3),
                ]
            )
            results = await client._get_all_pages("v3/reference/options/contracts", {})
        assert len(results) == 3

    async def test_max_pages_respected(self, client: MassiveClient, mock_redis):
        """Stops after max_pages even if next_url keeps appearing."""
        infinite_page = {
            "results": [{"ticker": "O:AAPL260406C00180000"}],
            "next_url": "https://api.massive.com/v3/next?cursor=loop",
            "status": "OK",
        }
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(infinite_page))
            results = await client._get_all_pages("v3/reference/options/contracts", {}, max_pages=3)
        # 1 initial page + 2 follow-up pages = 3 total
        assert len(results) == 3

    async def test_empty_first_response_returns_empty_list(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            results = await client._get_all_pages("v3/reference/options/contracts", {})
        assert results == []

    async def test_passes_api_key_via_params(self, client: MassiveClient, mock_redis):
        """apiKey is passed via params dict on paginated requests."""
        page1 = {
            "results": [{"ticker": "X"}],
            "next_url": "https://api.massive.com/v3/next?cursor=abc",
            "status": "OK",
        }
        page2 = {"results": [], "status": "OK"}
        captured_kwargs: list[dict] = []

        async def fake_get(url: str, **kwargs: object) -> MagicMock:
            captured_kwargs.append(dict(kwargs))
            if "cursor=abc" in url:
                return _mock_response(page2)
            return _mock_response(page1)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=fake_get)
            await client._get_all_pages("v3/next", {})

        # The paginated request should pass apiKey via params, not in URL
        assert any(kw.get("params", {}).get("apiKey") == "test_key" for kw in captured_kwargs)

    async def test_http_error_on_second_page_breaks_gracefully(
        self, client: MassiveClient, mock_redis
    ):
        """HTTP error on a follow-up page returns results gathered so far."""
        with patch.object(client, "_get_http") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_http.return_value.get = AsyncMock(
                side_effect=[
                    _mock_response(SAMPLE_PAGE_1),
                    httpx.HTTPStatusError("error", request=MagicMock(), response=mock_resp),
                ]
            )
            results = await client._get_all_pages("v3/reference/options/contracts", {})
        # Only the first page was collected before the error
        assert len(results) == 1


# ---------------------------------------------------------------------------
# get_full_options_chain
# ---------------------------------------------------------------------------


SAMPLE_FULL_CHAIN_PAGE_1 = {
    "results": [
        {
            "ticker": "O:AAPL260406C00180000",
            "underlying_ticker": "AAPL",
            "contract_type": "call",
            "exercise_style": "american",
            "expiration_date": "2026-04-06",
            "strike_price": 180.0,
            "shares_per_contract": 100,
            "primary_exchange": "BATO",
            "cfi": "OCASPS",
        }
    ],
    "next_url": "https://api.massive.com/v3/reference/options/contracts?cursor=p2",
    "status": "OK",
}

SAMPLE_FULL_CHAIN_PAGE_2 = {
    "results": [
        {
            "ticker": "O:AAPL260406P00180000",
            "underlying_ticker": "AAPL",
            "contract_type": "put",
            "exercise_style": "american",
            "expiration_date": "2026-04-06",
            "strike_price": 180.0,
            "shares_per_contract": 100,
            "primary_exchange": "BATO",
            "cfi": "OPASPS",
        }
    ],
    "status": "OK",
}


class TestGetFullOptionsChain:
    async def test_single_page_chain(self, client: MassiveClient, mock_redis):
        single = {
            "results": [SAMPLE_FULL_CHAIN_PAGE_1["results"][0]],
            "status": "OK",
        }
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(single))
            result = await client.get_full_options_chain("AAPL")
        assert len(result) == 1
        assert result[0].ticker == "O:AAPL260406C00180000"
        assert result[0].contract_type == "call"

    async def test_multi_page_chain_merged(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                side_effect=[
                    _mock_response(SAMPLE_FULL_CHAIN_PAGE_1),
                    _mock_response(SAMPLE_FULL_CHAIN_PAGE_2),
                ]
            )
            result = await client.get_full_options_chain("AAPL")
        assert len(result) == 2
        types = {c.contract_type for c in result}
        assert types == {"call", "put"}

    async def test_returns_empty_on_empty_response(self, client: MassiveClient, mock_redis):
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response({}))
            result = await client.get_full_options_chain("AAPL")
        assert result == []

    async def test_passes_contract_type_filter(self, client: MassiveClient, mock_redis):
        """contract_type kwarg is forwarded correctly."""
        calls_params: list[dict] = []

        async def capture_get(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response({"results": [], "status": "OK"})

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture_get)
            await client.get_full_options_chain("AAPL", contract_type="call")

        assert any("contract_type" in p and p["contract_type"] == "call" for p in calls_params)

    async def test_passes_expiration_range_params(self, client: MassiveClient, mock_redis):
        """expiration_date_gte and expiration_date_lte are forwarded as dot notation."""
        calls_params: list[dict] = []

        async def capture_get(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response({"results": [], "status": "OK"})

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture_get)
            await client.get_full_options_chain(
                "AAPL",
                expiration_date_gte="2026-04-01",
                expiration_date_lte="2026-04-30",
            )

        assert any(
            "expiration_date.gte" in p and p["expiration_date.gte"] == "2026-04-01"
            for p in calls_params
        )
        assert any(
            "expiration_date.lte" in p and p["expiration_date.lte"] == "2026-04-30"
            for p in calls_params
        )


# ---------------------------------------------------------------------------
# Range operator params passed through to API (per-endpoint group)
# ---------------------------------------------------------------------------


class TestRangeParamPassthrough:
    """Verify range operator kwargs are translated to dot-notation before hitting the API."""

    async def test_get_dividends_ex_dividend_date_gte(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_DIVIDENDS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_dividends("AAPL", ex_dividend_date_gte="2024-01-01")

        assert any(
            "ex_dividend_date.gte" in p and p["ex_dividend_date.gte"] == "2024-01-01"
            for p in calls_params
        )

    async def test_get_splits_execution_date_range(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SPLITS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_splits(
                "AAPL", execution_date_gte="2019-01-01", execution_date_lte="2021-12-31"
            )

        assert any("execution_date.gte" in p for p in calls_params)
        assert any("execution_date.lte" in p for p in calls_params)

    async def test_get_financials_filing_date_range(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_FINANCIALS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_financials(
                "AAPL", filing_date_gte="2024-01-01", filing_date_lte="2025-12-31"
            )

        assert any(
            "filing_date.gte" in p and p["filing_date.gte"] == "2024-01-01" for p in calls_params
        )
        assert any("filing_date.lte" in p for p in calls_params)

    async def test_get_short_interest_settlement_date_range(
        self, client: MassiveClient, mock_redis
    ):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SHORT_INTEREST)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_short_interest("AAPL", settlement_date_gte="2024-01-01")

        assert any("settlement_date.gte" in p for p in calls_params)

    async def test_get_news_published_utc_range(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_NEWS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_news(
                "AAPL", published_utc_gte="2026-04-01", published_utc_lte="2026-04-04"
            )

        assert any("published_utc.gte" in p for p in calls_params)
        assert any("published_utc.lte" in p for p in calls_params)

    async def test_get_short_volume_date_range(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SHORT_VOLUME)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_short_volume("AAPL", date_gte="2024-01-01", date_lte="2024-06-30")

        assert any("date.gte" in p for p in calls_params)
        assert any("date.lte" in p for p in calls_params)

    async def test_search_tickers_ticker_range(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_TICKERS_SEARCH)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.search_tickers(ticker_gte="A", ticker_lte="Z")

        assert any("ticker.gte" in p and p["ticker.gte"] == "A" for p in calls_params)
        assert any("ticker.lte" in p for p in calls_params)


# ---------------------------------------------------------------------------
# get_options_contracts with range filtering params
# ---------------------------------------------------------------------------


class TestGetOptionsContractsRangeParams:
    async def test_strike_price_range_params(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_OPTIONS_CONTRACTS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_options_contracts(
                "AAPL", strike_price_gte=150.0, strike_price_lte=200.0
            )

        assert any("strike_price.gte" in p and p["strike_price.gte"] == 150.0 for p in calls_params)
        assert any("strike_price.lte" in p and p["strike_price.lte"] == 200.0 for p in calls_params)

    async def test_expiration_date_range_params(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_OPTIONS_CONTRACTS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_options_contracts(
                "AAPL",
                expiration_date_gte="2026-04-01",
                expiration_date_lte="2026-04-30",
            )

        assert any(
            "expiration_date.gte" in p and p["expiration_date.gte"] == "2026-04-01"
            for p in calls_params
        )
        assert any("expiration_date.lte" in p for p in calls_params)

    async def test_contract_type_and_expiration_combined(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_OPTIONS_CONTRACTS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_options_contracts(
                "AAPL",
                contract_type="call",
                expiration_date="2026-04-06",
                strike_price_gte=170.0,
                strike_price_lte=190.0,
            )

        assert any(
            "contract_type" in p
            and p["contract_type"] == "call"
            and "expiration_date" in p
            and "strike_price.gte" in p
            for p in calls_params
        )

    async def test_default_limit_is_1000(self, client: MassiveClient, mock_redis):
        """Verify default limit changed from 100 to 1000."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_OPTIONS_CONTRACTS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_options_contracts("AAPL")

        assert any("limit" in p and p["limit"] == 1000 for p in calls_params)


# ---------------------------------------------------------------------------
# New param coverage: adjusted, include_otc, filter params, timestamp, etc.
# ---------------------------------------------------------------------------


class TestGetDailySummaryAdjusted:
    async def test_adjusted_false_passes_param(self, client: MassiveClient, mock_redis):
        """adjusted=False is forwarded to the API."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_DAILY_SUMMARY)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_daily_summary("AAPL", "2026-04-02", adjusted=False)

        assert result is not None
        assert result.ticker == "AAPL"
        assert any("adjusted" in p and p["adjusted"] == "false" for p in calls_params)

    async def test_adjusted_true_is_default(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_DAILY_SUMMARY)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_daily_summary("AAPL", "2026-04-02")

        assert any("adjusted" in p and p["adjusted"] == "true" for p in calls_params)


class TestGetGroupedDailyIncludeOtc:
    async def test_include_otc_true_passes_param(self, client: MassiveClient, mock_redis):
        """include_otc=True is forwarded to the API."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_GROUPED_DAILY)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_grouped_daily("2026-04-02", include_otc=True)

        assert result is not None
        assert len(result) == 2
        assert any("include_otc" in p and p["include_otc"] == "true" for p in calls_params)

    async def test_include_otc_false_is_default(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_GROUPED_DAILY)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_grouped_daily("2026-04-02")

        assert any("include_otc" in p and p["include_otc"] == "false" for p in calls_params)


class TestGetConditionsFilterParams:
    async def test_conditions_with_asset_class_and_data_type(
        self, client: MassiveClient, mock_redis
    ):
        """asset_class and data_type filter params are forwarded."""
        sample = {"results": [{"id": 1, "name": "Regular Sale", "type": "trade"}]}
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(sample)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_conditions(asset_class="stocks", data_type="trade")

        assert len(result) == 1
        assert result[0]["name"] == "Regular Sale"
        assert any(
            "asset_class" in p and p["asset_class"] == "stocks" and "data_type" in p
            for p in calls_params
        )

    async def test_conditions_with_sip_and_id(self, client: MassiveClient, mock_redis):
        sample = {"results": [{"id": 14, "name": "Odd Lot Trade"}]}
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(sample)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_conditions(sip="CTA", id=14)

        assert len(result) == 1
        assert any(
            "sip" in p and p["sip"] == "CTA" and "id" in p and p["id"] == 14 for p in calls_params
        )

    async def test_conditions_with_order_and_limit(self, client: MassiveClient, mock_redis):
        sample = {"results": [{"id": 1, "name": "Regular Sale"}]}
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(sample)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_conditions(order="asc", limit=50, sort="id")

        assert any(
            "order" in p and p["order"] == "asc" and "limit" in p and p["limit"] == 50
            for p in calls_params
        )


class TestGetTickerTypesWithFilters:
    async def test_ticker_types_with_asset_class_and_locale(
        self, client: MassiveClient, mock_redis
    ):
        sample = {
            "results": [
                {
                    "code": "CS",
                    "description": "Common Stock",
                    "asset_class_type": "stocks",
                    "locale": "us",
                }
            ]
        }
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(sample)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_ticker_types(asset_class="stocks", locale="us")

        assert len(result) == 1
        assert result[0]["code"] == "CS"
        assert any(
            "asset_class" in p
            and p["asset_class"] == "stocks"
            and "locale" in p
            and p["locale"] == "us"
            for p in calls_params
        )

    async def test_ticker_types_no_filters(self, client: MassiveClient, mock_redis):
        sample = {"results": [{"code": "CS"}, {"code": "ETF"}]}
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(sample))
            result = await client.get_ticker_types()
        assert len(result) == 2


class TestGetExchangesWithFilters:
    async def test_exchanges_with_asset_class_and_locale(self, client: MassiveClient, mock_redis):
        sample = {"results": [{"mic": "XNAS", "name": "NASDAQ", "locale": "us"}]}
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(sample)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_exchanges(asset_class="stocks", locale="us")

        assert len(result) == 1
        assert result[0]["mic"] == "XNAS"
        assert any(
            "asset_class" in p
            and p["asset_class"] == "stocks"
            and "locale" in p
            and p["locale"] == "us"
            for p in calls_params
        )

    async def test_exchanges_no_filters(self, client: MassiveClient, mock_redis):
        sample = {"results": [{"mic": "XNAS"}, {"mic": "XNYS"}]}
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(sample))
            result = await client.get_exchanges()
        assert len(result) == 2


class TestGetTickerEventsTypesFilter:
    async def test_ticker_events_with_types_filter(self, client: MassiveClient, mock_redis):
        """types filter is forwarded to the API."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_TICKER_EVENTS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_ticker_events("META", types="ticker_change")

        assert len(result) == 1
        assert result[0].type == "ticker_change"
        assert any("types" in p and p["types"] == "ticker_change" for p in calls_params)

    async def test_ticker_events_no_types_filter_omits_param(
        self, client: MassiveClient, mock_redis
    ):
        """When types is None, it is not included in params."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_TICKER_EVENTS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_ticker_events("AAPL")

        assert all("types" not in p for p in calls_params)


class TestGetShortVolumeRatioParams:
    async def test_short_volume_ratio_gte(self, client: MassiveClient, mock_redis):
        """short_volume_ratio_gte is forwarded as dot notation."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SHORT_VOLUME)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_short_volume("AAPL", short_volume_ratio_gte=0.5)

        assert len(result) == 1
        assert any(
            "short_volume_ratio.gte" in p and p["short_volume_ratio.gte"] == 0.5
            for p in calls_params
        )

    async def test_short_volume_ratio_exact(self, client: MassiveClient, mock_redis):
        """Exact short_volume_ratio filter is forwarded."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SHORT_VOLUME)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_short_volume("AAPL", short_volume_ratio=0.35)

        assert any(
            "short_volume_ratio" in p and p["short_volume_ratio"] == 0.35 for p in calls_params
        )

    async def test_short_volume_ratio_range(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SHORT_VOLUME)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_short_volume(
                "AAPL", short_volume_ratio_gte=0.3, short_volume_ratio_lte=0.7
            )

        assert any("short_volume_ratio.gte" in p for p in calls_params)
        assert any("short_volume_ratio.lte" in p for p in calls_params)


class TestTechnicalIndicatorsTimestampParams:
    async def test_sma_with_timestamp_range(self, client: MassiveClient, mock_redis):
        """timestamp_gte and timestamp_lte are forwarded as dot notation."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SMA)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_sma(
                "AAPL", timestamp_gte="2026-01-01", timestamp_lte="2026-03-31"
            )

        assert len(result) == 2
        assert any(
            "timestamp.gte" in p and p["timestamp.gte"] == "2026-01-01" for p in calls_params
        )
        assert any("timestamp.lte" in p for p in calls_params)

    async def test_ema_with_timestamp_params(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SMA)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_ema("AAPL", timestamp_gte="2026-01-01", timestamp_gt="2025-12-31")

        assert any("timestamp.gte" in p for p in calls_params)
        assert any("timestamp.gt" in p for p in calls_params)

    async def test_rsi_with_timestamp_params(self, client: MassiveClient, mock_redis):
        sample_rsi = {"results": {"values": [{"timestamp": 1775102400000, "value": 58.3}]}}
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(sample_rsi)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_rsi("AAPL", timestamp_lte="2026-04-01")

        assert len(result) == 1
        assert any("timestamp.lte" in p for p in calls_params)

    async def test_macd_with_timestamp_range(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_MACD)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_macd("AAPL", timestamp_gte="2026-01-01")

        assert len(result) == 1
        assert any("timestamp.gte" in p for p in calls_params)

    async def test_sma_exact_timestamp(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_SMA)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_sma("AAPL", timestamp="2026-03-28")

        assert any("timestamp" in p and p["timestamp"] == "2026-03-28" for p in calls_params)


class TestGetOptionsContractsOptionTicker:
    async def test_option_ticker_maps_to_ticker_param(self, client: MassiveClient, mock_redis):
        """option_ticker is forwarded to the API as 'ticker' param."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_OPTIONS_CONTRACTS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_options_contracts(
                "AAPL", option_ticker="O:AAPL260406C00180000"
            )

        assert len(result) == 1
        assert any("ticker" in p and p["ticker"] == "O:AAPL260406C00180000" for p in calls_params)

    async def test_option_ticker_none_omits_ticker_param(self, client: MassiveClient, mock_redis):
        """When option_ticker is None, 'ticker' is not set in params (underlying_ticker used instead)."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_OPTIONS_CONTRACTS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_options_contracts("AAPL")

        # 'ticker' as standalone (from option_ticker) should not be set;
        # underlying_ticker is passed separately
        assert all(p.get("ticker") != "O:AAPL260406C00180000" for p in calls_params)


class TestGetFullOptionsChainExplicitParams:
    async def test_option_ticker_forwarded(self, client: MassiveClient, mock_redis):
        """option_ticker explicit param is forwarded as 'ticker' to the API."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response({"results": [], "status": "OK"})

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_full_options_chain("AAPL", option_ticker="O:AAPL260406C00180000")

        assert any("ticker" in p and p["ticker"] == "O:AAPL260406C00180000" for p in calls_params)

    async def test_expired_true_forwarded(self, client: MassiveClient, mock_redis):
        """expired=True is forwarded as 'expired=true' string param."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response({"results": [], "status": "OK"})

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_full_options_chain("AAPL", expired=True)

        assert any("expired" in p and p["expired"] == "true" for p in calls_params)

    async def test_strike_price_range_params(self, client: MassiveClient, mock_redis):
        """strike_price_gte/lte are forwarded with dot notation."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response({"results": [], "status": "OK"})

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_full_options_chain(
                "AAPL", strike_price_gte=150.0, strike_price_lte=200.0
            )

        assert any("strike_price.gte" in p and p["strike_price.gte"] == 150.0 for p in calls_params)
        assert any("strike_price.lte" in p for p in calls_params)

    async def test_as_of_param_forwarded(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response({"results": [], "status": "OK"})

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_full_options_chain("AAPL", as_of="2026-04-01")

        assert any("as_of" in p and p["as_of"] == "2026-04-01" for p in calls_params)


class TestGetTickerOverviewNewFields:
    async def test_overview_new_fields_mapped(self, client: MassiveClient, mock_redis):
        """Verify address, branding, phone_number, ticker_root, etc. are mapped."""
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(
                return_value=_mock_response(SAMPLE_TICKER_OVERVIEW)
            )
            result = await client.get_ticker_overview("AAPL")

        assert result is not None
        assert result.phone_number == "1-408-996-1010"
        assert result.address == {
            "address1": "One Apple Park Way",
            "city": "Cupertino",
            "state": "CA",
            "postal_code": "95014",
        }
        assert result.branding is not None
        assert "logo_url" in result.branding
        assert "icon_url" in result.branding
        assert result.ticker_root == "AAPL"
        assert result.share_class_figi == "BBG001S5N8V8"
        assert result.share_class_shares_outstanding == 14681140000
        assert result.round_lot == 100

    async def test_overview_new_fields_none_when_absent(self, client: MassiveClient, mock_redis):
        """Fields absent from API response default to None."""
        minimal = {
            "status": "OK",
            "results": {
                "ticker": "XYZ",
                "name": "XYZ Corp",
                "market": "stocks",
                "locale": "us",
                "type": "CS",
                "active": True,
                "currency_name": "usd",
            },
        }
        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(return_value=_mock_response(minimal))
            result = await client.get_ticker_overview("XYZ")

        assert result is not None
        assert result.phone_number is None
        assert result.address is None
        assert result.branding is None
        assert result.ticker_root is None
        assert result.share_class_figi is None
        assert result.delisted_utc is None


class TestGetFinancialsCompanyNameSearch:
    async def test_company_name_search_forwarded(self, client: MassiveClient, mock_redis):
        """company_name_search is forwarded as 'company_name.search' param."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_FINANCIALS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_financials("AAPL", company_name_search="apple")

        assert len(result) == 1
        assert any(
            "company_name.search" in p and p["company_name.search"] == "apple" for p in calls_params
        )

    async def test_company_name_search_none_omits_param(self, client: MassiveClient, mock_redis):
        """When company_name_search is None, 'company_name.search' is not sent."""
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_FINANCIALS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            await client.get_financials("AAPL")

        assert all("company_name.search" not in p for p in calls_params)

    async def test_company_name_search_with_timeframe(self, client: MassiveClient, mock_redis):
        calls_params: list[dict] = []

        async def capture(url: str, params: dict, **kwargs: object) -> MagicMock:
            calls_params.append(params)
            return _mock_response(SAMPLE_FINANCIALS)

        with patch.object(client, "_get_http") as mock_http:
            mock_http.return_value.get = AsyncMock(side_effect=capture)
            result = await client.get_financials(
                "AAPL", company_name_search="apple", timeframe="annual"
            )

        assert len(result) == 1
        assert any(
            "company_name.search" in p and "timeframe" in p and p["timeframe"] == "annual"
            for p in calls_params
        )
