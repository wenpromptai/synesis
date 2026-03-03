"""Tests for earnings API routes."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.routes.earnings import router
from synesis.core.dependencies import get_db, get_nasdaq_client
from synesis.providers.nasdaq.models import EarningsEvent


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
def mock_db():
    db = AsyncMock()
    db.get_active_watchlist = AsyncMock(return_value=["AAPL", "TSLA"])
    return db


@pytest.fixture()
def app(mock_nasdaq_client, mock_db):
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1/earnings")
    test_app.dependency_overrides[get_nasdaq_client] = lambda: mock_nasdaq_client
    test_app.dependency_overrides[get_db] = lambda: mock_db
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

    async def test_get_calendar_earnings_fields(
        self, client: httpx.AsyncClient, mock_nasdaq_client
    ):
        mock_nasdaq_client.get_earnings_by_date.return_value = [_sample_event()]
        r = await client.get("/api/v1/earnings/calendar?date=2026-02-13")

        assert r.status_code == 200
        data = r.json()
        assert len(data["earnings"]) == 1
        event = data["earnings"][0]
        assert event["ticker"] == "AAPL"
        assert event["company_name"] == "Apple Inc."


class TestUpcomingEndpoint:
    async def test_upcoming_with_watchlist(self, client: httpx.AsyncClient, mock_nasdaq_client):
        mock_nasdaq_client.get_upcoming_earnings.return_value = [_sample_event()]
        r = await client.get("/api/v1/earnings/upcoming?days=7")

        assert r.status_code == 200
        data = r.json()
        assert data["tickers_checked"] == 2  # AAPL, TSLA from mock
        assert data["count"] == 1

    async def test_upcoming_empty_watchlist(self, client: httpx.AsyncClient, mock_db):
        mock_db.get_active_watchlist.return_value = []
        r = await client.get("/api/v1/earnings/upcoming")

        assert r.status_code == 200
        data = r.json()
        assert data["tickers_checked"] == 0
        assert data["count"] == 0

    async def test_upcoming_default_days(self, client: httpx.AsyncClient, mock_nasdaq_client):
        mock_nasdaq_client.get_upcoming_earnings.return_value = []
        r = await client.get("/api/v1/earnings/upcoming")

        assert r.status_code == 200
        mock_nasdaq_client.get_upcoming_earnings.assert_called_once()
        call_kwargs = mock_nasdaq_client.get_upcoming_earnings.call_args
        assert call_kwargs.kwargs["days"] == 14


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

    async def test_upcoming_for_ticker_all_in_range(
        self, client: httpx.AsyncClient, mock_nasdaq_client
    ):
        mock_nasdaq_client.get_upcoming_earnings.return_value = [
            _sample_event(),
            _sample_event(earnings_date=date(2026, 5, 1)),
        ]
        r = await client.get("/api/v1/earnings/upcoming/AAPL")

        assert r.status_code == 200
        data = r.json()
        assert len(data["all_in_range"]) == 2

    async def test_upcoming_for_ticker_uppercases(
        self, client: httpx.AsyncClient, mock_nasdaq_client
    ):
        mock_nasdaq_client.get_upcoming_earnings.return_value = []
        r = await client.get("/api/v1/earnings/upcoming/aapl")

        assert r.status_code == 200
        assert r.json()["ticker"] == "AAPL"
