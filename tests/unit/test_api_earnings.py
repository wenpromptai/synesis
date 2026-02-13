"""Tests for earnings API routes."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.routes.earnings import router
from synesis.core.dependencies import get_nasdaq_client
from synesis.providers.nasdaq.models import EarningsEvent
from synesis.storage.redis import get_redis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_event(**overrides: Any) -> EarningsEvent:
    defaults: dict[str, Any] = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "earnings_date": date(2026, 2, 13),
        "time": "after-hours",
        "eps_forecast": 2.35,
        "num_estimates": 28,
        "market_cap": 3_500_000_000_000.0,
        "fiscal_quarter": "Dec/2025",
    }
    defaults.update(overrides)
    return EarningsEvent(**defaults)


@pytest.fixture()
def mock_nasdaq_client():
    client = AsyncMock()
    client.get_earnings_by_date = AsyncMock(return_value=[])
    client.get_upcoming_earnings = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def mock_redis():
    redis = AsyncMock()
    redis.smembers.return_value = {b"AAPL", b"TSLA"}
    return redis


@pytest.fixture()
def app(mock_nasdaq_client, mock_redis):
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1/earnings")
    test_app.dependency_overrides[get_nasdaq_client] = lambda: mock_nasdaq_client
    test_app.dependency_overrides[get_redis] = lambda: mock_redis
    return test_app


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalendarEndpoint:
    async def test_get_calendar(self, client: httpx.AsyncClient, mock_nasdaq_client):
        mock_nasdaq_client.get_earnings_by_date.return_value = [
            _sample_event(),
            _sample_event(ticker="MSFT", company_name="Microsoft Corporation"),
        ]
        r = await client.get("/api/v1/earnings/calendar?date=2026-02-13")

        assert r.status_code == 200
        data = r.json()
        assert data["date"] == "2026-02-13"
        assert data["count"] == 2

    async def test_get_calendar_default_date(self, client: httpx.AsyncClient, mock_nasdaq_client):
        r = await client.get("/api/v1/earnings/calendar")

        assert r.status_code == 200
        assert r.json()["count"] == 0


class TestUpcomingEndpoint:
    async def test_upcoming_with_watchlist(self, client: httpx.AsyncClient, mock_nasdaq_client):
        mock_nasdaq_client.get_upcoming_earnings.return_value = [_sample_event()]
        r = await client.get("/api/v1/earnings/upcoming?days=7")

        assert r.status_code == 200
        data = r.json()
        assert data["tickers_checked"] == 2  # AAPL, TSLA from mock
        assert data["count"] == 1

    async def test_upcoming_empty_watchlist(self, client: httpx.AsyncClient, mock_redis):
        mock_redis.smembers.return_value = set()
        r = await client.get("/api/v1/earnings/upcoming")

        assert r.status_code == 200
        data = r.json()
        assert data["tickers_checked"] == 0
        assert data["count"] == 0


class TestUpcomingTickerEndpoint:
    async def test_upcoming_for_ticker(self, client: httpx.AsyncClient, mock_nasdaq_client):
        mock_nasdaq_client.get_upcoming_earnings.return_value = [_sample_event()]
        r = await client.get("/api/v1/earnings/upcoming/AAPL")

        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["next_earnings"] is not None
        assert data["next_earnings"]["ticker"] == "AAPL"

    async def test_upcoming_for_ticker_none(self, client: httpx.AsyncClient, mock_nasdaq_client):
        mock_nasdaq_client.get_upcoming_earnings.return_value = []
        r = await client.get("/api/v1/earnings/upcoming/ZZZZ")

        assert r.status_code == 200
        data = r.json()
        assert data["next_earnings"] is None
