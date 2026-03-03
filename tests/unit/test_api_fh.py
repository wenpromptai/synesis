"""Tests for Finnhub API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.routes.fh import router
from synesis.core.dependencies import get_price_provider, get_ticker_provider
from synesis.providers.finnhub import QuoteData


def _make_quote(price: float = 185.50) -> QuoteData:
    return QuoteData(
        current=price,
        change=0.54,
        percent_change=0.20,
        high=price + 2,
        low=price - 5,
        open=price - 1,
        previous_close=price - 0.54,
        timestamp=1772485200,
    )


@pytest.fixture()
def mock_price_service():
    svc = AsyncMock()
    svc._subscribed_tickers = {"AAPL", "TSLA"}
    svc._ws = object()  # truthy = connected
    return svc


@pytest.fixture()
def mock_ticker_provider():
    return AsyncMock()


@pytest.fixture()
def app(mock_price_service, mock_ticker_provider):
    test_app = FastAPI()
    test_app.include_router(router, prefix="/fh")
    test_app.dependency_overrides[get_price_provider] = lambda: mock_price_service
    test_app.dependency_overrides[get_ticker_provider] = lambda: mock_ticker_provider
    return test_app


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


PREFIX = "/fh"


# ===========================================================================
# GET /fh/{ticker}
# ===========================================================================


class TestGetSinglePrice:
    async def test_cached_price(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service.get_quote.return_value = _make_quote(185.50)
        r = await client.get(f"{PREFIX}/aapl")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "AAPL"
        assert body["current"] == 185.50
        assert body["change"] == 0.54
        assert body["percent_change"] == 0.20
        assert "high" in body
        assert "low" in body
        mock_price_service.get_quote.assert_awaited_once_with("AAPL")

    async def test_not_found(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service.get_quote.side_effect = ValueError("No valid price for ZZZZ")
        r = await client.get(f"{PREFIX}/ZZZZ")
        assert r.status_code == 404

    async def test_rest_fallback(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        """get_quote fetches directly from Finnhub REST API."""
        mock_price_service.get_quote.return_value = _make_quote(42.00)
        r = await client.get(f"{PREFIX}/NVDA")
        assert r.status_code == 200
        assert r.json()["current"] == 42.00


# ===========================================================================
# GET /fh?tickers=...
# ===========================================================================


class TestGetBatchPrices:
    async def test_batch(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service.get_quotes.return_value = {
            "AAPL": _make_quote(185.50),
            "TSLA": _make_quote(250.00),
        }
        r = await client.get(f"{PREFIX}", params={"tickers": "AAPL,TSLA"})
        assert r.status_code == 200
        body = r.json()
        assert body["quotes"]["AAPL"]["current"] == 185.50
        assert body["quotes"]["TSLA"]["current"] == 250.00
        assert body["found"] == 2
        assert body["missing"] == []

    async def test_batch_partial(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service.get_quotes.return_value = {"AAPL": _make_quote(185.50)}
        r = await client.get(f"{PREFIX}", params={"tickers": "AAPL,ZZZZ"})
        assert r.status_code == 200
        body = r.json()
        assert body["found"] == 1
        assert "ZZZZ" in body["missing"]

    async def test_batch_empty_param(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}", params={"tickers": ""})
        assert r.status_code == 400


# ===========================================================================
# GET /fh/subscriptions
# ===========================================================================


class TestGetSubscriptions:
    async def test_subscriptions(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        r = await client.get(f"{PREFIX}/subscriptions")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert sorted(body["subscribed_tickers"]) == ["AAPL", "TSLA"]
        assert body["ws_connected"] is True

    async def test_ws_disconnected(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service._ws = None
        r = await client.get(f"{PREFIX}/subscriptions")
        assert r.status_code == 200
        assert r.json()["ws_connected"] is False


# ===========================================================================
# POST /fh/subscribe
# ===========================================================================


class TestSubscribe:
    async def test_subscribe(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        r = await client.post(f"{PREFIX}/subscribe", json={"tickers": ["NVDA", "MSFT"]})
        assert r.status_code == 200
        body = r.json()
        assert sorted(body["subscribed"]) == ["MSFT", "NVDA"]
        mock_price_service.subscribe.assert_awaited_once_with(["NVDA", "MSFT"])

    async def test_subscribe_empty(self, client: httpx.AsyncClient):
        r = await client.post(f"{PREFIX}/subscribe", json={"tickers": []})
        assert r.status_code == 400

    async def test_subscribe_exceeds_limit(
        self, client: httpx.AsyncClient, mock_price_service: AsyncMock
    ):
        # Already at 2, try to add 100 new tickers (limit is 50)
        new_tickers = [f"T{i}" for i in range(100)]
        r = await client.post(f"{PREFIX}/subscribe", json={"tickers": new_tickers})
        assert r.status_code == 400
        assert "limit" in r.json()["detail"].lower()


# ===========================================================================
# POST /fh/unsubscribe
# ===========================================================================


class TestUnsubscribe:
    async def test_unsubscribe(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        r = await client.post(f"{PREFIX}/unsubscribe", json={"tickers": ["AAPL"]})
        assert r.status_code == 200
        mock_price_service.unsubscribe.assert_awaited_once_with(["AAPL"])

    async def test_unsubscribe_empty(self, client: httpx.AsyncClient):
        r = await client.post(f"{PREFIX}/unsubscribe", json={"tickers": []})
        assert r.status_code == 400


# ===========================================================================
# 503 when PriceService not initialized
# ===========================================================================


# ===========================================================================
# GET /fh/ticker/verify/{ticker}
# ===========================================================================


class TestVerifyTicker:
    async def test_valid_ticker(self, client: httpx.AsyncClient, mock_ticker_provider: AsyncMock):
        mock_ticker_provider.verify_ticker.return_value = (True, "APPLE INC")
        r = await client.get(f"{PREFIX}/ticker/verify/AAPL")
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is True
        assert body["company_name"] == "APPLE INC"
        mock_ticker_provider.verify_ticker.assert_awaited_once_with("AAPL")

    async def test_invalid_ticker(self, client: httpx.AsyncClient, mock_ticker_provider: AsyncMock):
        mock_ticker_provider.verify_ticker.return_value = (False, None)
        r = await client.get(f"{PREFIX}/ticker/verify/ZZZZ")
        assert r.status_code == 200
        body = r.json()
        assert body["valid"] is False
        assert body["company_name"] is None


# ===========================================================================
# GET /fh/ticker/search?q=...
# ===========================================================================


class TestSearchTicker:
    async def test_search_results(self, client: httpx.AsyncClient, mock_ticker_provider: AsyncMock):
        mock_ticker_provider.search_symbol.return_value = [
            {"symbol": "AAPL", "description": "APPLE INC", "type": "Common Stock"},
            {"symbol": "APLE", "description": "APPLE HOSPITALITY REIT", "type": "REIT"},
        ]
        r = await client.get(f"{PREFIX}/ticker/search", params={"q": "apple"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert len(body["results"]) == 2
        assert body["results"][0]["symbol"] == "AAPL"
        mock_ticker_provider.search_symbol.assert_awaited_once_with("apple")

    async def test_search_empty_query(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/ticker/search", params={"q": ""})
        assert r.status_code == 400

    async def test_search_no_results(
        self, client: httpx.AsyncClient, mock_ticker_provider: AsyncMock
    ):
        mock_ticker_provider.search_symbol.return_value = []
        r = await client.get(f"{PREFIX}/ticker/search", params={"q": "xyznonexistent"})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0
        assert body["results"] == []


# ===========================================================================
# 503 when PriceService not initialized
# ===========================================================================


class TestServiceUnavailable:
    async def test_503_when_not_initialized(self):
        """get_price_provider catches RuntimeError and returns 503."""
        from synesis.core.dependencies import get_price_provider as real_dep

        test_app = FastAPI()
        test_app.include_router(router, prefix="/fh")
        # Use the real dependency — no global singleton means RuntimeError -> 503
        test_app.dependency_overrides.pop(get_price_provider, None)
        test_app.dependency_overrides[get_price_provider] = real_dep

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/fh/subscriptions")
            assert r.status_code == 503
            assert "not available" in r.json()["detail"].lower()
