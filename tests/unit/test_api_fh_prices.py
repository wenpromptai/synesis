"""Tests for Finnhub price API endpoints."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.routes.fh_prices import router
from synesis.core.dependencies import get_price_provider


@pytest.fixture()
def mock_price_service():
    svc = AsyncMock()
    svc._subscribed_tickers = {"AAPL", "TSLA"}
    svc._ws = object()  # truthy = connected
    return svc


@pytest.fixture()
def app(mock_price_service):
    test_app = FastAPI()
    test_app.include_router(router, prefix="/fh_prices")
    test_app.dependency_overrides[get_price_provider] = lambda: mock_price_service
    return test_app


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


PREFIX = "/fh_prices"


# ===========================================================================
# GET /fh_prices/{ticker}
# ===========================================================================


class TestGetSinglePrice:
    async def test_cached_price(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service.get_price.return_value = Decimal("185.50")
        r = await client.get(f"{PREFIX}/aapl")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "AAPL"
        assert body["price"] == 185.50
        mock_price_service.get_price.assert_awaited_once_with("AAPL")

    async def test_not_found(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service.get_price.return_value = None
        r = await client.get(f"{PREFIX}/ZZZZ")
        assert r.status_code == 404

    async def test_rest_fallback(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        """get_price already has REST fallback built in."""
        mock_price_service.get_price.return_value = Decimal("42.00")
        r = await client.get(f"{PREFIX}/NVDA")
        assert r.status_code == 200
        assert r.json()["price"] == 42.00


# ===========================================================================
# GET /fh_prices?tickers=...
# ===========================================================================


class TestGetBatchPrices:
    async def test_batch(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service.get_prices.return_value = {
            "AAPL": Decimal("185.50"),
            "TSLA": Decimal("250.00"),
        }
        r = await client.get(f"{PREFIX}", params={"tickers": "AAPL,TSLA"})
        assert r.status_code == 200
        body = r.json()
        assert body["prices"]["AAPL"] == 185.50
        assert body["prices"]["TSLA"] == 250.00
        assert body["found"] == 2
        assert body["missing"] == []

    async def test_batch_partial(self, client: httpx.AsyncClient, mock_price_service: AsyncMock):
        mock_price_service.get_prices.return_value = {"AAPL": Decimal("185.50")}
        r = await client.get(f"{PREFIX}", params={"tickers": "AAPL,ZZZZ"})
        assert r.status_code == 200
        body = r.json()
        assert body["found"] == 1
        assert "ZZZZ" in body["missing"]

    async def test_batch_empty_param(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}", params={"tickers": ""})
        assert r.status_code == 400


# ===========================================================================
# GET /fh_prices/subscriptions
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
# POST /fh_prices/subscribe
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
# POST /fh_prices/unsubscribe
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


class TestServiceUnavailable:
    async def test_503_when_not_initialized(self):
        """get_price_provider catches RuntimeError and returns 503."""
        from synesis.core.dependencies import get_price_provider as real_dep

        test_app = FastAPI()
        test_app.include_router(router, prefix="/fh_prices")
        # Use the real dependency — no global singleton means RuntimeError -> 503
        test_app.dependency_overrides.pop(get_price_provider, None)
        test_app.dependency_overrides[get_price_provider] = real_dep

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/fh_prices/subscriptions")
            assert r.status_code == 503
            assert "not available" in r.json()["detail"].lower()
