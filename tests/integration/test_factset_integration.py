"""Integration tests for FactSet provider and API endpoints.

Hits the real FactSet SQL Server — skipped when SQLSERVER_HOST is not configured.
Run with: pytest -m integration -v tests/integration/test_factset_integration.py
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.router import api_router
from synesis.config import get_settings
from synesis.core.dependencies import get_agent_state, get_db, get_factset_provider
from synesis.providers.factset.client import FactSetClient
from synesis.providers.factset.models import (
    FactSetCorporateAction,
    FactSetFundamentals,
    FactSetPrice,
    FactSetSecurity,
    FactSetSharesOutstanding,
)
from synesis.providers.factset.provider import FactSetProvider
from synesis.storage.redis import get_redis

# ---------------------------------------------------------------------------
# Canonical test ticker — guaranteed to exist with rich data
# ---------------------------------------------------------------------------
TICKER = "AAPL"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def factset_provider():
    """Real FactSetProvider connected to SQL Server. Skip if not reachable."""
    settings = get_settings()
    if not settings.sqlserver_host:
        pytest.skip("SQLSERVER_HOST not configured")
    try:
        client = FactSetClient()
        # Force a connection attempt so we fail fast
        client._get_connection()
    except (ConnectionError, Exception) as exc:
        pytest.skip(f"FactSet DB not reachable: {exc}")
    provider = FactSetProvider(client=client)
    yield provider


@pytest.fixture(scope="module")
def factset_client(factset_provider: FactSetProvider) -> FactSetClient:
    """Expose the underlying FactSetClient for direct health check tests."""
    return factset_provider._client


@pytest.fixture(scope="module")
def app(factset_provider: FactSetProvider) -> FastAPI:
    """FastAPI app wired to the real FactSet provider."""
    test_app = FastAPI()
    test_app.include_router(api_router, prefix="/api/v1")

    # Real FactSet provider
    test_app.dependency_overrides[get_factset_provider] = lambda: factset_provider

    # Mock out unrelated deps (not under test)
    mock_agent = MagicMock()
    mock_agent.agent_task = MagicMock(done=MagicMock(return_value=False))
    test_app.dependency_overrides[get_agent_state] = lambda: mock_agent
    test_app.dependency_overrides[get_redis] = lambda: AsyncMock()
    test_app.dependency_overrides[get_db] = lambda: MagicMock()

    return test_app


@pytest.fixture()
async def client(app: FastAPI):
    """httpx.AsyncClient wired to the real-provider FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ===========================================================================
# Provider-level tests
# ===========================================================================

PREFIX = "/api/v1/factset"


class TestProviderHealthCheck:
    async def test_health_check(self, factset_client: FactSetClient):
        result = await factset_client.health_check()
        assert result is True


class TestProviderResolveTicker:
    async def test_resolve_ticker(self, factset_provider: FactSetProvider):
        security = await factset_provider.resolve_ticker(TICKER)
        assert security is not None
        assert isinstance(security, FactSetSecurity)
        assert "AAPL" in security.ticker  # e.g. "AAPL-US"
        assert "Apple" in security.name
        assert security.exchange_code
        assert security.currency

    async def test_resolve_ticker_invalid(self, factset_provider: FactSetProvider):
        result = await factset_provider.resolve_ticker("ZZZZZZZ99")
        assert result is None


class TestProviderSearchSecurities:
    async def test_search_securities(self, factset_provider: FactSetProvider):
        results = await factset_provider.search_securities("Apple", limit=5)
        assert isinstance(results, list)
        assert len(results) <= 5
        assert len(results) > 0
        # At least one result should reference Apple
        names = [s.name for s in results]
        assert any("Apple" in n for n in names)


class TestProviderPrice:
    async def test_get_latest_price(self, factset_provider: FactSetProvider):
        price = await factset_provider.get_price(TICKER)
        assert price is not None
        assert isinstance(price, FactSetPrice)
        assert price.close > 0

    async def test_get_price_specific_date(self, factset_provider: FactSetProvider):
        target = date(2024, 1, 16)  # Known trading day (Tuesday)
        price = await factset_provider.get_price(TICKER, price_date=target)
        assert price is not None
        assert isinstance(price, FactSetPrice)
        assert price.price_date == target

    async def test_get_price_history(self, factset_provider: FactSetProvider):
        start = date(2024, 1, 2)
        end = date(2024, 1, 31)
        prices = await factset_provider.get_price_history(TICKER, start, end)
        assert isinstance(prices, list)
        assert len(prices) > 0
        for p in prices:
            assert p.close > 0
            assert start <= p.price_date <= end

    async def test_get_latest_prices_batch(self, factset_provider: FactSetProvider):
        result = await factset_provider.get_latest_prices(["AAPL", "MSFT"])
        assert isinstance(result, dict)
        assert "AAPL" in result
        assert "MSFT" in result
        assert isinstance(result["AAPL"], FactSetPrice)
        assert isinstance(result["MSFT"], FactSetPrice)


class TestProviderFundamentals:
    async def test_get_fundamentals_annual(self, factset_provider: FactSetProvider):
        data = await factset_provider.get_fundamentals(TICKER, "annual", 4)
        assert isinstance(data, list)
        assert len(data) <= 4
        assert len(data) > 0
        for f in data:
            assert isinstance(f, FactSetFundamentals)
            assert f.eps_diluted is not None

    async def test_get_fundamentals_quarterly(self, factset_provider: FactSetProvider):
        data = await factset_provider.get_fundamentals(TICKER, "quarterly", 4)
        assert isinstance(data, list)
        assert len(data) > 0
        for f in data:
            assert f.period_type == "quarterly"


class TestProviderProfile:
    async def test_get_company_profile(self, factset_provider: FactSetProvider):
        profile = await factset_provider.get_company_profile(TICKER)
        assert profile is not None
        assert isinstance(profile, str)
        assert len(profile) > 0


class TestProviderCorporateActions:
    async def test_get_corporate_actions(self, factset_provider: FactSetProvider):
        actions = await factset_provider.get_corporate_actions(TICKER)
        assert isinstance(actions, list)
        known_types = {"dividend", "split", "rights", "other"}
        for a in actions:
            assert isinstance(a, FactSetCorporateAction)
            assert a.event_type in known_types

    async def test_get_dividends(self, factset_provider: FactSetProvider):
        divs = await factset_provider.get_dividends(TICKER)
        assert isinstance(divs, list)
        assert len(divs) > 0
        for d in divs:
            assert d.event_type == "dividend"

    async def test_get_splits(self, factset_provider: FactSetProvider):
        splits = await factset_provider.get_splits(TICKER)
        assert isinstance(splits, list)
        # AAPL has had splits (7:1 in 2014, 4:1 in 2020)
        assert len(splits) > 0


class TestProviderShares:
    async def test_get_shares_outstanding(self, factset_provider: FactSetProvider):
        shares = await factset_provider.get_shares_outstanding(TICKER)
        assert shares is not None
        assert isinstance(shares, FactSetSharesOutstanding)
        # AAPL has > 15 billion shares outstanding
        assert shares.shares_outstanding > 1_000_000_000


class TestProviderMarketCap:
    async def test_get_market_cap(self, factset_provider: FactSetProvider):
        mcap = await factset_provider.get_market_cap(TICKER)
        assert mcap is not None
        assert isinstance(mcap, float)
        # AAPL market cap > $1 trillion
        assert mcap > 1_000_000_000_000


class TestProviderAdjustedPrices:
    async def test_get_adjusted_price_history(self, factset_provider: FactSetProvider):
        start = date(2024, 1, 2)
        end = date(2024, 1, 31)
        prices = await factset_provider.get_adjusted_price_history(TICKER, start, end)
        assert isinstance(prices, list)
        assert len(prices) > 0
        for p in prices:
            assert p.is_adjusted is True

    async def test_get_adjustment_factors(self, factset_provider: FactSetProvider):
        start = date(2019, 1, 1)
        end = date(2021, 12, 31)
        factors = await factset_provider.get_adjustment_factors(TICKER, start, end)
        assert isinstance(factors, dict)
        for d, f in factors.items():
            assert isinstance(d, date)
            assert isinstance(f, float)
            assert f > 0


# ===========================================================================
# API-level tests
# ===========================================================================


class TestApiSearch:
    async def test_api_search(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/search", params={"query": "Apple"})
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestApiGetSecurity:
    async def test_api_get_security(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}")
        assert r.status_code == 200
        body = r.json()
        assert "name" in body
        assert "ticker" in body
        assert "exchange_code" in body

    async def test_api_get_security_404(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/ZZZZZZZ99")
        assert r.status_code == 404


class TestApiPrice:
    async def test_api_latest_price(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/price")
        assert r.status_code == 200
        body = r.json()
        assert body["close"] > 0

    async def test_api_price_with_date(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/price", params={"date": "2024-01-16"})
        assert r.status_code == 200
        assert r.json()["price_date"] == "2024-01-16"


class TestApiPriceHistory:
    async def test_api_price_history(self, client: httpx.AsyncClient):
        r = await client.get(
            f"{PREFIX}/securities/{TICKER}/price-history",
            params={"start": "2024-01-01", "end": "2024-01-31"},
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0

    async def test_api_adjusted_history(self, client: httpx.AsyncClient):
        r = await client.get(
            f"{PREFIX}/securities/{TICKER}/price-history/adjusted",
            params={"start": "2024-01-01", "end": "2024-01-31"},
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for item in data:
            assert item["is_adjusted"] is True


class TestApiBatchPrices:
    async def test_api_batch_prices(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/prices/batch", params={"tickers": "AAPL,MSFT"})
        assert r.status_code == 200
        body = r.json()
        assert "AAPL" in body
        assert "MSFT" in body


class TestApiFundamentals:
    async def test_api_fundamentals(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/fundamentals")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0


class TestApiProfile:
    async def test_api_profile(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/profile")
        assert r.status_code == 200
        body = r.json()
        assert body["description"]
        assert len(body["description"]) > 0


class TestApiCorporateActions:
    async def test_api_corporate_actions(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/corporate-actions")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0

    async def test_api_dividends(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/dividends")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for item in data:
            assert item["event_type"] == "dividend"

    async def test_api_splits(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/splits")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)


class TestApiShares:
    async def test_api_shares_outstanding(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/shares-outstanding")
        assert r.status_code == 200
        body = r.json()
        assert body["shares_outstanding"] > 0


class TestApiMarketCap:
    async def test_api_market_cap(self, client: httpx.AsyncClient):
        r = await client.get(f"{PREFIX}/securities/{TICKER}/market-cap")
        assert r.status_code == 200
        body = r.json()
        assert body["market_cap"] > 0


class TestApiAdjustmentFactors:
    async def test_api_adjustment_factors(self, client: httpx.AsyncClient):
        r = await client.get(
            f"{PREFIX}/securities/{TICKER}/adjustment-factors",
            params={"start": "2019-01-01", "end": "2021-12-31"},
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
