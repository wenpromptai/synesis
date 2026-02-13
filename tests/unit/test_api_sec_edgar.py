"""Tests for SEC EDGAR API routes."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.routes.sec_edgar import router
from synesis.core.dependencies import get_crawler, get_sec_edgar_client
from synesis.providers.sec_edgar.models import EarningsRelease, InsiderTransaction, SECFiling


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_filing(**overrides: Any) -> SECFiling:
    defaults: dict[str, Any] = {
        "ticker": "AAPL",
        "form": "8-K",
        "filed_date": date(2026, 2, 10),
        "accepted_datetime": datetime(2026, 2, 10, 16, 30),
        "accession_number": "0000320193-26-000010",
        "primary_document": "doc.htm",
        "items": "2.02",
        "url": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019326000010/doc.htm",
    }
    defaults.update(overrides)
    return SECFiling(**defaults)


def _sample_insider(**overrides: Any) -> InsiderTransaction:
    defaults: dict[str, Any] = {
        "ticker": "AAPL",
        "owner_name": "Tim Cook",
        "owner_relationship": "Officer (CEO)",
        "transaction_date": date(2026, 1, 15),
        "transaction_code": "S",
        "shares": 50000.0,
        "price_per_share": 185.50,
        "shares_after": 200000.0,
        "acquired_or_disposed": "D",
        "filing_date": date(2026, 1, 16),
        "filing_url": "https://www.sec.gov/example",
    }
    defaults.update(overrides)
    return InsiderTransaction(**defaults)


def _sample_release(**overrides: Any) -> EarningsRelease:
    defaults: dict[str, Any] = {
        "ticker": "AAPL",
        "filed_date": date(2026, 2, 10),
        "accepted_datetime": datetime(2026, 2, 10, 16, 30),
        "accession_number": "0000320193-26-000010",
        "url": "https://www.sec.gov/Archives/edgar/data/0000320193/000032019326000010/doc.htm",
        "items": "2.02",
        "content": "# Earnings Press Release",
    }
    defaults.update(overrides)
    return EarningsRelease(**defaults)


@pytest.fixture()
def mock_sec_client():
    client = AsyncMock()
    client.get_filings = AsyncMock(return_value=[])
    client.get_insider_transactions = AsyncMock(return_value=[])
    client.get_insider_sentiment = AsyncMock(return_value=None)
    client.search_filings = AsyncMock(return_value=[])
    client.get_earnings_releases = AsyncMock(return_value=[])
    return client


@pytest.fixture()
def mock_crawler():
    return AsyncMock()


@pytest.fixture()
def app(mock_sec_client, mock_crawler):
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1/sec_edgar")
    test_app.dependency_overrides[get_sec_edgar_client] = lambda: mock_sec_client
    test_app.dependency_overrides[get_crawler] = lambda: mock_crawler
    return test_app


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFilingsEndpoint:
    async def test_get_filings(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_filings.return_value = [
            _sample_filing(),
            _sample_filing(form="10-Q", items=""),
        ]
        r = await client.get("/api/v1/sec_edgar/filings?ticker=AAPL")

        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["count"] == 2
        assert data["filings"][0]["form"] == "8-K"

    async def test_get_filings_with_form_filter(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_filings.return_value = [_sample_filing()]
        r = await client.get("/api/v1/sec_edgar/filings?ticker=AAPL&forms=8-K")

        assert r.status_code == 200
        assert r.json()["count"] == 1
        # Verify forms were parsed and passed
        mock_sec_client.get_filings.assert_called_once()

    async def test_get_filings_missing_ticker(self, client: httpx.AsyncClient):
        r = await client.get("/api/v1/sec_edgar/filings")
        assert r.status_code == 422


class TestInsidersEndpoint:
    async def test_get_insiders(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_insider_transactions.return_value = [_sample_insider()]
        r = await client.get("/api/v1/sec_edgar/insiders?ticker=AAPL")

        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["count"] == 1
        assert data["transactions"][0]["owner_name"] == "Tim Cook"


class TestInsiderSellsEndpoint:
    async def test_get_sells_with_threshold(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_insider_transactions.return_value = [
            _sample_insider(shares=50000, price_per_share=185.50),  # value = 9,275,000
            _sample_insider(
                shares=100, price_per_share=185.50, transaction_code="S"
            ),  # value = 18,550
        ]
        r = await client.get("/api/v1/sec_edgar/insiders/sells?ticker=AAPL&min_value=500000")

        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1  # Only the big sell


class TestSentimentEndpoint:
    async def test_get_sentiment(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_insider_sentiment.return_value = {
            "ticker": "AAPL",
            "mspr": 0.3211,
            "change": 0,
            "buy_count": 1,
            "sell_count": 1,
            "total_buy_value": 180000.0,
            "total_sell_value": 92500.0,
        }
        r = await client.get("/api/v1/sec_edgar/sentiment?ticker=AAPL")

        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["mspr"] == pytest.approx(0.3211)

    async def test_get_sentiment_not_found(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_insider_sentiment.return_value = None
        r = await client.get("/api/v1/sec_edgar/sentiment?ticker=ZZZZ")
        assert r.status_code == 404


class TestSearchEndpoint:
    async def test_search_filings(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.search_filings.return_value = [
            {
                "entity": "Apple Inc.",
                "filed": "2026-02-10",
                "form": "8-K",
                "url": "https://example.com",
                "description": "Current report",
            }
        ]
        r = await client.get("/api/v1/sec_edgar/search?query=Apple+earnings")

        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["query"] == "Apple earnings"

    async def test_search_requires_query(self, client: httpx.AsyncClient):
        r = await client.get("/api/v1/sec_edgar/search")
        assert r.status_code == 422


class TestEarningsEndpoint:
    async def test_get_earnings(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_earnings_releases.return_value = [
            _sample_release(),
            _sample_release(accession_number="0000320193-25-000005"),
        ]
        r = await client.get("/api/v1/sec_edgar/earnings?ticker=AAPL")

        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["count"] == 2
        assert data["releases"][0]["items"] == "2.02"

    async def test_get_earnings_empty(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_earnings_releases.return_value = []
        r = await client.get("/api/v1/sec_edgar/earnings?ticker=AAPL")

        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 0
        assert data["releases"] == []


class TestLatestEarningsEndpoint:
    async def test_get_latest_earnings(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_earnings_releases.return_value = [_sample_release()]
        r = await client.get("/api/v1/sec_edgar/earnings/latest?ticker=AAPL")

        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "AAPL"
        assert data["content"] == "# Earnings Press Release"

    async def test_get_latest_earnings_not_found(self, client: httpx.AsyncClient, mock_sec_client):
        mock_sec_client.get_earnings_releases.return_value = []
        r = await client.get("/api/v1/sec_edgar/earnings/latest?ticker=ZZZZ")
        assert r.status_code == 404


class TestInsiderSellsPriceNone:
    async def test_sells_excludes_none_price(self, client: httpx.AsyncClient, mock_sec_client):
        """Transaction with price_per_share=None should be excluded when min_value > 0."""
        mock_sec_client.get_insider_transactions.return_value = [
            _sample_insider(shares=50000, price_per_share=None),
            _sample_insider(shares=50000, price_per_share=185.50),
        ]
        r = await client.get("/api/v1/sec_edgar/insiders/sells?ticker=AAPL&min_value=100000")

        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1  # Only the one with a price
