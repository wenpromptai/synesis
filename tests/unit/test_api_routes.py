"""Comprehensive tests for all API route endpoints."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from synesis.api.router import api_router
from synesis.core.dependencies import (
    get_agent_state,
    get_db,
    get_price_provider,
)
from synesis.storage.redis import get_redis


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
    db_enabled: bool = True
    scheduler: Any = None
    trigger_fns: dict[str, Any] = field(default_factory=dict)
    _background_tasks: list[asyncio.Task[None]] = field(default_factory=list)


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
def app(mock_agent_state, mock_redis_dep, mock_db_dep):
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
        assert body["agent_running"] is True


class TestSystemConfig:
    async def test_config(self, client: httpx.AsyncClient):
        with patch("synesis.api.routes.system.get_settings") as mock_settings:
            s = MagicMock()
            s.env = "development"
            s.llm_provider = "anthropic"
            s.telegram_api_id = None
            mock_settings.return_value = s
            r = await client.get(f"{SYS_PREFIX}/config")
        assert r.status_code == 200
        body = r.json()
        assert body["env"] == "development"
        assert body["llm_provider"] == "anthropic"
        assert body["telegram_enabled"] is False
