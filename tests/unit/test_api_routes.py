"""Comprehensive tests for all API route endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.router import api_router
from synesis.core.dependencies import (
    get_agent_state,
    get_db,
    get_factset_provider,
    get_price_provider,
)
from synesis.providers.factset.models import (
    FactSetCorporateAction,
    FactSetFundamentals,
    FactSetPrice,
    FactSetSecurity,
    FactSetSharesOutstanding,
)
from synesis.storage.redis import get_redis


# ---------------------------------------------------------------------------
# Fixtures — mock models
# ---------------------------------------------------------------------------


def _make_security(**overrides: Any) -> FactSetSecurity:
    defaults = {
        "fsym_id": "K7TPSX-R",
        "fsym_security_id": "K7TPSX-S",
        "ticker": "AAPL-US",
        "name": "Apple Inc.",
        "exchange_code": "NAS",
        "security_type": "SHARE",
        "currency": "USD",
        "country": "US",
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }
    defaults.update(overrides)
    return FactSetSecurity(**defaults)


def _make_price(**overrides: Any) -> FactSetPrice:
    defaults = {
        "fsym_id": "K7TPSX-R",
        "price_date": date(2024, 1, 15),
        "close": 185.50,
        "open": 184.00,
        "high": 186.00,
        "low": 183.50,
        "volume": 50_000_000,
    }
    defaults.update(overrides)
    return FactSetPrice(**defaults)


def _make_fundamentals(**overrides: Any) -> FactSetFundamentals:
    defaults = {
        "fsym_id": "K7TPSX-R",
        "period_end": date(2024, 9, 30),
        "fiscal_year": 2024,
        "period_type": "annual",
        "eps_diluted": 6.57,
        "roe": 0.175,
    }
    defaults.update(overrides)
    return FactSetFundamentals(**defaults)


def _make_corporate_action(
    event_type: str = "dividend", **overrides: Any
) -> FactSetCorporateAction:
    defaults = {
        "fsym_id": "K7TPSX-R",
        "event_type": event_type,
        "effective_date": date(2024, 2, 9),
        "dividend_amount": 0.24 if event_type == "dividend" else None,
        "dividend_currency": "USD" if event_type == "dividend" else None,
        "split_factor": 4.0 if event_type == "split" else None,
    }
    defaults.update(overrides)
    return FactSetCorporateAction(**defaults)


def _make_shares(**overrides: Any) -> FactSetSharesOutstanding:
    defaults = {
        "fsym_id": "K7TPSX-R",
        "report_date": date(2024, 1, 15),
        "shares_outstanding": 15_460_000_000,
    }
    defaults.update(overrides)
    return FactSetSharesOutstanding(**defaults)


# ---------------------------------------------------------------------------
# Fixtures — mock dependencies
# ---------------------------------------------------------------------------


@dataclass
class _MockAgentState:
    redis: Any = None
    db: Any = None
    settings: Any = None
    agent_task: asyncio.Task[None] | None = None
    telegram_enabled: bool = False
    reddit_enabled: bool = True
    sentiment_enabled: bool = False
    db_enabled: bool = True
    scheduler: Any = None
    trigger_fns: dict[str, Any] = field(default_factory=dict)
    _background_tasks: list[asyncio.Task[None]] = field(default_factory=list)


@pytest.fixture()
def mock_factset():
    provider = AsyncMock()
    # search
    provider.search_securities.return_value = [_make_security()]
    # resolve
    provider.resolve_ticker.return_value = _make_security()
    # prices
    provider.get_price.return_value = _make_price()
    provider.get_latest_prices.return_value = {
        "AAPL": _make_price(),
        "MSFT": _make_price(fsym_id="MSFT-R", close=380.0),
    }
    provider.get_price_history.return_value = [
        _make_price(price_date=date(2024, 1, d)) for d in range(1, 4)
    ]
    provider.get_adjusted_price_history.return_value = [
        _make_price(price_date=date(2024, 1, d), is_adjusted=True) for d in range(1, 4)
    ]
    # fundamentals
    provider.get_fundamentals.return_value = [_make_fundamentals()]
    # profile
    provider.get_company_profile.return_value = "Apple designs consumer electronics."
    # corporate actions
    provider.get_corporate_actions.return_value = [_make_corporate_action()]
    provider.get_dividends.return_value = [_make_corporate_action("dividend")]
    provider.get_splits.return_value = [_make_corporate_action("split")]
    # shares
    provider.get_shares_outstanding.return_value = _make_shares()
    # market cap
    provider.get_market_cap.return_value = 2_870_000_000_000.0
    # adjustment factors
    provider.get_adjustment_factors.return_value = {
        date(2024, 1, 1): 1.0,
        date(2024, 1, 2): 1.0,
    }
    return provider


@pytest.fixture()
def mock_agent_state():
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True

    running_task = MagicMock()
    running_task.done.return_value = False

    return _MockAgentState(
        redis=mock_redis,
        db=MagicMock(),
        agent_task=running_task,
    )


@pytest.fixture()
def mock_redis_dep():
    """Mock Redis for watchlist endpoints."""
    redis = AsyncMock()
    # smembers returns empty set by default
    redis.smembers.return_value = set()
    redis.sismember.return_value = False
    redis.sadd.return_value = 1
    redis.srem.return_value = 1
    redis.hgetall.return_value = {}
    redis.pipeline.return_value = AsyncMock()
    redis.set.return_value = True
    redis.hset.return_value = 1
    redis.expire.return_value = True
    redis.hincrby.return_value = 2
    redis.delete.return_value = 1
    redis.register_script = MagicMock(return_value=AsyncMock(return_value=0))
    return redis


@pytest.fixture()
def mock_db_dep():
    return MagicMock()


@pytest.fixture()
def app(mock_factset, mock_agent_state, mock_redis_dep, mock_db_dep):
    """Create a FastAPI app with all dependencies overridden."""
    test_app = FastAPI()

    # Mount health/ready directly (mirrors main.py)
    @test_app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @test_app.get("/ready")
    async def ready() -> dict[str, str]:
        checks: dict[str, str] = {}
        state = mock_agent_state
        try:
            await state.redis.ping()
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "error"
        checks["db"] = "ok" if state.db else "disabled"
        checks["agent"] = "ok" if state.agent_task and not state.agent_task.done() else "error"
        status = "ready" if all(v != "error" for v in checks.values()) else "not_ready"
        return {"status": status, **checks}

    test_app.include_router(api_router, prefix="/api/v1")

    # Override all deps
    test_app.dependency_overrides[get_factset_provider] = lambda: mock_factset
    test_app.dependency_overrides[get_agent_state] = lambda: mock_agent_state
    test_app.dependency_overrides[get_redis] = lambda: mock_redis_dep
    test_app.dependency_overrides[get_db] = lambda: mock_db_dep
    test_app.dependency_overrides[get_price_provider] = lambda: AsyncMock()

    return test_app


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ===========================================================================
# Infrastructure endpoints
# ===========================================================================


class TestHealth:
    async def test_health(self, client: httpx.AsyncClient):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestReady:
    async def test_ready_all_ok(self, client: httpx.AsyncClient):
        r = await client.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["redis"] == "ok"
        assert body["agent"] == "ok"

    async def test_ready_redis_down(self, client: httpx.AsyncClient, mock_agent_state):
        mock_agent_state.redis.ping.side_effect = ConnectionError("refused")
        r = await client.get("/ready")
        body = r.json()
        assert body["status"] == "not_ready"
        assert body["redis"] == "error"


# ===========================================================================
# FactSet endpoints  /api/v1/factset/securities/...
# ===========================================================================

PREFIX = "/api/v1/factset"


class TestFactSetSearch:
    async def test_search(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/search", params={"query": "AAPL"})
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["ticker"] == "AAPL-US"
        mock_factset.search_securities.assert_awaited_once_with("AAPL", 20)

    async def test_search_with_limit(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/search", params={"query": "AAPL", "limit": 5})
        assert r.status_code == 200
        mock_factset.search_securities.assert_awaited_once_with("AAPL", 5)


class TestFactSetBatchPrices:
    async def test_batch_prices(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/prices/batch", params={"tickers": "AAPL,MSFT"})
        assert r.status_code == 200
        data = r.json()
        assert "AAPL" in data
        assert "MSFT" in data
        mock_factset.get_latest_prices.assert_awaited_once_with(["AAPL", "MSFT"])


class TestFactSetGetSecurity:
    async def test_get_security(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL")
        assert r.status_code == 200
        assert r.json()["name"] == "Apple Inc."

    async def test_get_security_not_found(self, client: httpx.AsyncClient, mock_factset):
        mock_factset.resolve_ticker.return_value = None
        r = await client.get(f"{PREFIX}/securities/INVALID")
        assert r.status_code == 404


class TestFactSetPrice:
    async def test_get_latest_price(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/price")
        assert r.status_code == 200
        assert r.json()["close"] == 185.5

    async def test_get_price_with_date(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/price", params={"date": "2024-01-15"})
        assert r.status_code == 200
        mock_factset.get_price.assert_awaited_once_with("AAPL", price_date=date(2024, 1, 15))

    async def test_get_price_not_found(self, client: httpx.AsyncClient, mock_factset):
        mock_factset.get_price.return_value = None
        r = await client.get(f"{PREFIX}/securities/INVALID/price")
        assert r.status_code == 404


class TestFactSetPriceHistory:
    async def test_price_history(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(
            f"{PREFIX}/securities/AAPL/price-history",
            params={"start": "2024-01-01", "end": "2024-01-31"},
        )
        assert r.status_code == 200
        assert len(r.json()) == 3

    async def test_adjusted_price_history(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(
            f"{PREFIX}/securities/AAPL/price-history/adjusted",
            params={"start": "2024-01-01", "end": "2024-01-31"},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 3
        assert data[0]["is_adjusted"] is True


class TestFactSetFundamentals:
    async def test_fundamentals_default(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/fundamentals")
        assert r.status_code == 200
        mock_factset.get_fundamentals.assert_awaited_once_with(
            "AAPL", period_type="annual", limit=4
        )

    async def test_fundamentals_quarterly(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(
            f"{PREFIX}/securities/AAPL/fundamentals",
            params={"period": "quarterly", "limit": 8},
        )
        assert r.status_code == 200
        mock_factset.get_fundamentals.assert_awaited_once_with(
            "AAPL", period_type="quarterly", limit=8
        )


class TestFactSetProfile:
    async def test_profile(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/profile")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "AAPL"
        assert "Apple" in body["description"]

    async def test_profile_not_found(self, client: httpx.AsyncClient, mock_factset):
        mock_factset.get_company_profile.return_value = None
        r = await client.get(f"{PREFIX}/securities/INVALID/profile")
        assert r.status_code == 404


class TestFactSetCorporateActions:
    async def test_corporate_actions(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/corporate-actions")
        assert r.status_code == 200
        assert len(r.json()) == 1

    async def test_dividends(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/dividends")
        assert r.status_code == 200
        data = r.json()
        assert data[0]["event_type"] == "dividend"

    async def test_splits(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/splits")
        assert r.status_code == 200
        data = r.json()
        assert data[0]["event_type"] == "split"


class TestFactSetShares:
    async def test_shares_outstanding(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/shares-outstanding")
        assert r.status_code == 200
        assert r.json()["shares_outstanding"] == 15_460_000_000

    async def test_shares_not_found(self, client: httpx.AsyncClient, mock_factset):
        mock_factset.get_shares_outstanding.return_value = None
        r = await client.get(f"{PREFIX}/securities/INVALID/shares-outstanding")
        assert r.status_code == 404


class TestFactSetMarketCap:
    async def test_market_cap(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(f"{PREFIX}/securities/AAPL/market-cap")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "AAPL"
        assert body["market_cap"] == 2_870_000_000_000.0

    async def test_market_cap_not_found(self, client: httpx.AsyncClient, mock_factset):
        mock_factset.get_market_cap.return_value = None
        r = await client.get(f"{PREFIX}/securities/INVALID/market-cap")
        assert r.status_code == 404


class TestFactSetAdjustmentFactors:
    async def test_adjustment_factors(self, client: httpx.AsyncClient, mock_factset):
        r = await client.get(
            f"{PREFIX}/securities/AAPL/adjustment-factors",
            params={"start": "2024-01-01", "end": "2024-01-31"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "2024-01-01" in data
        assert data["2024-01-01"] == 1.0


# ===========================================================================
# Watchlist endpoints  /api/v1/watchlist/...
# ===========================================================================

WL_PREFIX = "/api/v1/watchlist"


class TestWatchlistList:
    async def test_list_empty(self, client: httpx.AsyncClient):
        r = await client.get(f"{WL_PREFIX}/")
        assert r.status_code == 200
        assert r.json() == []

    async def test_list_with_tickers(self, client: httpx.AsyncClient, mock_redis_dep):
        mock_redis_dep.smembers.return_value = {"AAPL", "MSFT"}
        r = await client.get(f"{WL_PREFIX}/")
        assert r.status_code == 200
        data = r.json()
        assert sorted(data) == ["AAPL", "MSFT"]


class TestWatchlistAdd:
    async def test_add_ticker(self, client: httpx.AsyncClient, mock_redis_dep):
        mock_redis_dep.sismember.return_value = False
        r = await client.post(f"{WL_PREFIX}/", json={"ticker": "AAPL", "source": "api"})
        assert r.status_code == 201
        body = r.json()
        assert body["ticker"] == "AAPL"
        assert body["is_new"] is True


class TestWatchlistGetTicker:
    async def test_get_ticker(self, client: httpx.AsyncClient, mock_redis_dep):
        mock_redis_dep.hgetall.return_value = {
            "ticker": "AAPL",
            "source": "api",
            "subreddit": "",
            "added_at": "2024-01-15T10:00:00+00:00",
            "last_seen_at": "2024-01-15T10:00:00+00:00",
            "mention_count": "1",
        }
        r = await client.get(f"{WL_PREFIX}/AAPL")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "AAPL"

    async def test_get_ticker_not_found(self, client: httpx.AsyncClient, mock_redis_dep):
        mock_redis_dep.hgetall.return_value = {}
        r = await client.get(f"{WL_PREFIX}/INVALID")
        assert r.status_code == 404


class TestWatchlistStats:
    async def test_stats(self, client: httpx.AsyncClient, mock_redis_dep):
        # get_stats calls get_all_with_metadata which calls get_all then pipeline
        mock_redis_dep.smembers.return_value = set()
        r = await client.get(f"{WL_PREFIX}/stats")
        assert r.status_code == 200
        body = r.json()
        assert "total_tickers" in body


class TestWatchlistDetailed:
    async def test_detailed_empty(self, client: httpx.AsyncClient, mock_redis_dep):
        mock_redis_dep.smembers.return_value = set()
        r = await client.get(f"{WL_PREFIX}/detailed")
        assert r.status_code == 200
        assert r.json() == []


class TestWatchlistDelete:
    async def test_delete_ticker(self, client: httpx.AsyncClient, mock_redis_dep):
        mock_redis_dep.sismember.return_value = True
        r = await client.delete(f"{WL_PREFIX}/AAPL")
        assert r.status_code == 204

    async def test_delete_not_found(self, client: httpx.AsyncClient, mock_redis_dep):
        mock_redis_dep.sismember.return_value = False
        r = await client.delete(f"{WL_PREFIX}/NOTHERE")
        assert r.status_code == 404


class TestWatchlistCleanup:
    async def test_cleanup(self, client: httpx.AsyncClient, mock_redis_dep):
        mock_redis_dep.smembers.return_value = set()
        r = await client.post(f"{WL_PREFIX}/cleanup")
        assert r.status_code == 200
        assert r.json() == []


# ===========================================================================
# System endpoints  /api/v1/system/...
# ===========================================================================

SYS_PREFIX = "/api/v1/system"


class TestSystemStatus:
    async def test_status(self, client: httpx.AsyncClient, mock_agent_state):
        r = await client.get(f"{SYS_PREFIX}/status")
        assert r.status_code == 200
        body = r.json()
        assert body["telegram"] is False
        assert body["reddit"] is True
        assert body["agent_running"] is True


class TestWatchlistAnalyze:
    async def test_trigger_analysis(self, client: httpx.AsyncClient, mock_agent_state):
        mock_scheduler = MagicMock()
        mock_agent_state.scheduler = mock_scheduler
        mock_agent_state.trigger_fns = {"watchlist_intel": AsyncMock()}
        r = await client.post(f"{WL_PREFIX}/analyze")
        assert r.status_code == 202
        assert r.json() == {"status": "triggered"}
        mock_scheduler.add_job.assert_called_once()

    async def test_trigger_analysis_no_scheduler(self, client: httpx.AsyncClient, mock_agent_state):
        mock_agent_state.scheduler = None
        mock_agent_state.trigger_fns = {}
        r = await client.post(f"{WL_PREFIX}/analyze")
        assert r.status_code == 503
        assert "not enabled" in r.json()["detail"]

    async def test_trigger_analysis_missing_trigger_fn(
        self, client: httpx.AsyncClient, mock_agent_state
    ):
        mock_agent_state.scheduler = MagicMock()
        mock_agent_state.trigger_fns = {}  # no "watchlist_intel" key
        r = await client.post(f"{WL_PREFIX}/analyze")
        assert r.status_code == 503


# ===========================================================================
# Market Intel endpoints  /api/v1/mkt_intel/...
# ===========================================================================

MKT_PREFIX = "/api/v1/mkt_intel"


class TestMktIntelTriggerScan:
    async def test_trigger_scan(self, client: httpx.AsyncClient, mock_agent_state):
        mock_scheduler = MagicMock()
        mock_agent_state.scheduler = mock_scheduler
        mock_agent_state.trigger_fns = {"mkt_intel": AsyncMock()}
        r = await client.post(f"{MKT_PREFIX}/run")
        assert r.status_code == 200
        assert r.json() == {"status": "scan_triggered"}
        mock_scheduler.add_job.assert_called_once()

    async def test_trigger_scan_no_scheduler(self, client: httpx.AsyncClient, mock_agent_state):
        mock_agent_state.scheduler = None
        mock_agent_state.trigger_fns = {}
        r = await client.post(f"{MKT_PREFIX}/run")
        assert r.status_code == 503
        assert "not enabled" in r.json()["detail"]

    async def test_trigger_scan_missing_trigger_fn(
        self, client: httpx.AsyncClient, mock_agent_state
    ):
        mock_agent_state.scheduler = MagicMock()
        mock_agent_state.trigger_fns = {}  # no "mkt_intel" key
        r = await client.post(f"{MKT_PREFIX}/run")
        assert r.status_code == 503


class TestSystemConfig:
    async def test_config(self, client: httpx.AsyncClient):
        with patch("synesis.api.routes.system.get_settings") as mock_settings:
            s = MagicMock()
            s.env = "development"
            s.llm_provider = "anthropic"
            s.telegram_api_id = None
            s.reddit_subreddits = ["wallstreetbets"]
            mock_settings.return_value = s
            r = await client.get(f"{SYS_PREFIX}/config")
        assert r.status_code == 200
        body = r.json()
        assert body["env"] == "development"
        assert body["llm_provider"] == "anthropic"
        assert body["telegram_enabled"] is False
        assert body["reddit_enabled"] is True
