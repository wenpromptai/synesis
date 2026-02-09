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

    # -- AAPL: current (no date) -----------------------------------------

    async def test_aapl_current_market_cap(self, factset_provider: FactSetProvider):
        """Current AAPL market cap should be $2.5T–$5T."""
        mcap = await factset_provider.get_market_cap("AAPL")
        assert mcap is not None
        assert 2_500_000_000_000 <= mcap <= 5_000_000_000_000, (
            f"AAPL current mcap {mcap / 1e12:.2f}T outside $2.5T–$5T"
        )

    # -- AAPL: historical across two splits ------------------------------

    async def test_aapl_pre_4_1_split(self, factset_provider: FactSetProvider):
        """2020-08-28 (day before 4:1 split): ~$1.8T–$2.3T, not ~$8T."""
        mcap = await factset_provider.get_market_cap("AAPL", date(2020, 8, 28))
        assert mcap is not None
        assert 1_800_000_000_000 <= mcap <= 2_300_000_000_000, (
            f"AAPL 2020-08-28 mcap {mcap / 1e12:.2f}T outside $1.8T–$2.3T (4x inflated?)"
        )

    async def test_aapl_between_splits(self, factset_provider: FactSetProvider):
        """2018-01-02 (between 7:1 and 4:1 splits): ~$800B–$950B."""
        mcap = await factset_provider.get_market_cap("AAPL", date(2018, 1, 2))
        assert mcap is not None
        assert 800_000_000_000 <= mcap <= 950_000_000_000, (
            f"AAPL 2018-01-02 mcap {mcap / 1e12:.2f}T outside $800B–$950B (4x inflated?)"
        )

    async def test_aapl_pre_both_splits(self, factset_provider: FactSetProvider):
        """2014-06-05 (before both splits): ~$500B–$650B, not ~$14T–$18T."""
        mcap = await factset_provider.get_market_cap("AAPL", date(2014, 6, 5))
        assert mcap is not None
        assert 500_000_000_000 <= mcap <= 650_000_000_000, (
            f"AAPL 2014-06-05 mcap {mcap / 1e12:.2f}T outside $500B–$650B (28x inflated?)"
        )

    # -- TSLA: two different split ratios --------------------------------

    async def test_tsla_pre_3_1_split(self, factset_provider: FactSetProvider):
        """2022-08-24 (day before 3:1 split): ~$850B–$1T."""
        mcap = await factset_provider.get_market_cap("TSLA", date(2022, 8, 24))
        assert mcap is not None
        assert 850_000_000_000 <= mcap <= 1_000_000_000_000, (
            f"TSLA 2022-08-24 mcap {mcap / 1e12:.2f}T outside $850B–$1T (3x inflated?)"
        )

    async def test_tsla_pre_both_splits(self, factset_provider: FactSetProvider):
        """2020-08-27 (before both splits): ~$350B–$450B."""
        mcap = await factset_provider.get_market_cap("TSLA", date(2020, 8, 27))
        assert mcap is not None
        assert 350_000_000_000 <= mcap <= 450_000_000_000, (
            f"TSLA 2020-08-27 mcap {mcap / 1e12:.2f}T outside $350B–$450B (15x inflated?)"
        )

    # -- NVDA: large 10:1 split ------------------------------------------

    async def test_nvda_pre_10_1_split(self, factset_provider: FactSetProvider):
        """2024-06-07 (day before 10:1 split): ~$2.5T–$3.5T."""
        mcap = await factset_provider.get_market_cap("NVDA", date(2024, 6, 7))
        assert mcap is not None
        assert 2_500_000_000_000 <= mcap <= 3_500_000_000_000, (
            f"NVDA 2024-06-07 mcap {mcap / 1e12:.2f}T outside $2.5T–$3.5T (10x inflated?)"
        )

    # -- GE: reverse split (1:8) -----------------------------------------

    async def test_ge_pre_reverse_split(self, factset_provider: FactSetProvider):
        """2021-07-30 (before 1:8 reverse split): ~$100B–$140B."""
        mcap = await factset_provider.get_market_cap("GE", date(2021, 7, 30))
        assert mcap is not None
        assert 100_000_000_000 <= mcap <= 140_000_000_000, (
            f"GE 2021-07-30 mcap {mcap / 1e9:.1f}B outside $100B–$140B (reverse split deflated?)"
        )

    async def test_ge_post_reverse_split(self, factset_provider: FactSetProvider):
        """2022-01-03 (after 1:8 reverse split): ~$90B–$120B."""
        mcap = await factset_provider.get_market_cap("GE", date(2022, 1, 3))
        assert mcap is not None
        assert 90_000_000_000 <= mcap <= 120_000_000_000, (
            f"GE 2022-01-03 mcap {mcap / 1e9:.1f}B outside $90B–$120B"
        )

    # -- Sanity check: manual reconstruction -----------------------------

    async def test_market_cap_internal_consistency(self, factset_provider: FactSetProvider):
        """Verify get_market_cap matches adj_price × adj_shares manually."""
        target = date(2020, 8, 28)

        # 1. Unadjusted price
        unadj = await factset_provider.get_price("AAPL", target)
        assert unadj is not None

        # 2. Adjusted price via get_adjusted_price_history
        adj_prices = await factset_provider.get_adjusted_price_history("AAPL", target, target)
        assert len(adj_prices) == 1
        adj_price = adj_prices[0]

        # 3. Market cap from get_market_cap
        mcap = await factset_provider.get_market_cap("AAPL", target)
        assert mcap is not None

        # 4. The adjusted price should be ~1/4 of unadjusted (4:1 split on 2020-08-31)
        ratio = unadj.close / adj_price.close
        assert 3.8 <= ratio <= 4.2, (
            f"Unadj/adj price ratio {ratio:.2f} not ~4.0 (expected from 4:1 split)"
        )

        # 5. Reconstruct market cap: adj_price × adj_shares_outstanding
        #    get_market_cap uses the same math, so result should match exactly
        security = await factset_provider.resolve_ticker("AAPL")
        assert security is not None and security.fsym_security_id is not None
        adj_shares_m = await factset_provider._get_shares_as_of(security.fsym_security_id, target)
        assert adj_shares_m is not None
        reconstructed = adj_price.close * adj_shares_m * 1_000_000

        # Should match within 0.01% (floating point tolerance)
        assert abs(reconstructed - mcap) / mcap < 0.0001, (
            f"Reconstructed {reconstructed / 1e12:.4f}T vs get_market_cap {mcap / 1e12:.4f}T"
        )


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

    async def test_api_current_market_cap_no_date_key(self, client: httpx.AsyncClient):
        """Current market cap response should NOT include a 'date' key."""
        r = await client.get(f"{PREFIX}/securities/{TICKER}/market-cap")
        assert r.status_code == 200
        body = r.json()
        assert "date" not in body

    async def test_api_historical_market_cap(self, client: httpx.AsyncClient):
        """Historical market cap via API: AAPL on 2020-08-28 → $1.8T–$2.3T."""
        r = await client.get(f"{PREFIX}/securities/AAPL/market-cap", params={"date": "2020-08-28"})
        assert r.status_code == 200
        body = r.json()
        assert body["date"] == "2020-08-28"
        mcap = body["market_cap"]
        assert 1_800_000_000_000 <= mcap <= 2_300_000_000_000, (
            f"API AAPL 2020-08-28 mcap {mcap / 1e12:.2f}T outside $1.8T–$2.3T"
        )


class TestApiAdjustmentFactors:
    async def test_api_adjustment_factors(self, client: httpx.AsyncClient):
        r = await client.get(
            f"{PREFIX}/securities/{TICKER}/adjustment-factors",
            params={"start": "2019-01-01", "end": "2021-12-31"},
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
