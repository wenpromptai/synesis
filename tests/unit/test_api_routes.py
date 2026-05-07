"""Comprehensive tests for all API route endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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
    db_enabled: bool = True
    scheduler: Any = None
    trigger_fns: dict[str, Any] = field(default_factory=dict)


@pytest.fixture()
def mock_agent_state():
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True

    mock_scheduler = MagicMock()
    mock_scheduler.running = True

    return _MockAgentState(
        redis=mock_redis,
        db=MagicMock(),
        scheduler=mock_scheduler,
    )


@pytest.fixture()
def mock_redis_dep():
    """Mock Redis for non-watchlist endpoints that still need it."""
    redis = AsyncMock()
    return redis


@pytest.fixture()
def mock_db_dep():
    """Mock Database for watchlist endpoints (DB-only)."""
    db = AsyncMock()
    db.get_active_watchlist = AsyncMock(return_value=[])
    db.get_active_watchlist_with_metadata = AsyncMock(return_value=[])
    db.get_watchlist_metadata = AsyncMock(return_value=None)
    db.get_watchlist_stats = AsyncMock(return_value={"total_tickers": 0, "sources": {}})
    db.upsert_watchlist_ticker = AsyncMock(return_value=True)
    db.remove_watchlist_ticker = AsyncMock(return_value=True)
    db.watchlist_contains = AsyncMock(return_value=False)
    db.deactivate_expired_watchlist = AsyncMock(return_value=[])
    return db


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

    async def test_list_with_tickers(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.get_active_watchlist.return_value = ["AAPL", "MSFT"]
        r = await client.get(f"{WL_PREFIX}/")
        assert r.status_code == 200
        data = r.json()
        assert sorted(data) == ["AAPL", "MSFT"]


class TestWatchlistAdd:
    async def test_add_ticker(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.upsert_watchlist_ticker.return_value = True
        r = await client.post(f"{WL_PREFIX}/", json={"ticker": "AAPL", "source": "api"})
        assert r.status_code == 201
        body = r.json()
        assert body["ticker"] == "AAPL"
        assert body["is_new"] is True

    async def test_add_existing_ticker(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.upsert_watchlist_ticker.return_value = False
        r = await client.post(f"{WL_PREFIX}/", json={"ticker": "AAPL", "source": "api"})
        assert r.status_code == 201
        body = r.json()
        assert body["ticker"] == "AAPL"
        assert body["is_new"] is False

    async def test_add_lowercased_ticker_uppercased_in_response(
        self, client: httpx.AsyncClient, mock_db_dep
    ):
        mock_db_dep.upsert_watchlist_ticker.return_value = True
        r = await client.post(f"{WL_PREFIX}/", json={"ticker": "tsla", "source": "telegram"})
        assert r.status_code == 201
        assert r.json()["ticker"] == "TSLA"


class TestWatchlistGetTicker:
    async def test_get_ticker(self, client: httpx.AsyncClient, mock_db_dep):
        now = datetime.now(UTC)
        mock_db_dep.get_watchlist_metadata.return_value = {
            "ticker": "AAPL",
            "added_by": "api",
            "added_reason": "Signal from api",
            "added_at": now,
            "expires_at": now + timedelta(days=7),
        }
        r = await client.get(f"{WL_PREFIX}/AAPL")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "AAPL"

    async def test_get_ticker_response_fields(self, client: httpx.AsyncClient, mock_db_dep):
        now = datetime.now(UTC)
        expires = now + timedelta(days=7)
        mock_db_dep.get_watchlist_metadata.return_value = {
            "ticker": "NVDA",
            "added_by": "telegram",
            "added_reason": "Signal from telegram",
            "added_at": now,
            "expires_at": expires,
        }
        r = await client.get(f"{WL_PREFIX}/NVDA")
        assert r.status_code == 200
        body = r.json()
        assert body["ticker"] == "NVDA"
        assert body["source"] == "telegram"
        assert "added_at" in body
        assert "expires_at" in body

    async def test_get_ticker_not_found(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.get_watchlist_metadata.return_value = None
        r = await client.get(f"{WL_PREFIX}/INVALID")
        assert r.status_code == 404


class TestWatchlistStats:
    async def test_stats_empty(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.get_watchlist_stats.return_value = {
            "total_tickers": 0,
            "sources": {},
        }
        r = await client.get(f"{WL_PREFIX}/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_tickers"] == 0
        assert body["sources"] == {}

    async def test_stats_populated(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.get_watchlist_stats.return_value = {
            "total_tickers": 5,
            "sources": {"telegram": 3, "api": 2},
        }
        r = await client.get(f"{WL_PREFIX}/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_tickers"] == 5
        assert body["sources"]["telegram"] == 3


class TestWatchlistDetailed:
    async def test_detailed_empty(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.get_active_watchlist_with_metadata.return_value = []
        r = await client.get(f"{WL_PREFIX}/detailed")
        assert r.status_code == 200
        assert r.json() == []

    async def test_detailed_with_records(self, client: httpx.AsyncClient, mock_db_dep):
        now = datetime.now(UTC)
        mock_db_dep.get_active_watchlist_with_metadata.return_value = [
            {
                "ticker": "AAPL",
                "added_by": "telegram",
                "added_at": now,
                "expires_at": now + timedelta(days=7),
            },
            {
                "ticker": "TSLA",
                "added_by": "api",
                "added_at": now,
                "expires_at": now + timedelta(days=3),
            },
        ]
        r = await client.get(f"{WL_PREFIX}/detailed")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        assert data[0]["ticker"] == "AAPL"
        assert data[0]["source"] == "telegram"
        assert data[1]["ticker"] == "TSLA"
        assert data[1]["source"] == "api"


class TestWatchlistDelete:
    async def test_delete_ticker(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.remove_watchlist_ticker.return_value = True
        r = await client.delete(f"{WL_PREFIX}/AAPL")
        assert r.status_code == 204

    async def test_delete_not_found(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.remove_watchlist_ticker.return_value = False
        r = await client.delete(f"{WL_PREFIX}/NOTHERE")
        assert r.status_code == 404


class TestWatchlistCleanup:
    async def test_cleanup_empty(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.deactivate_expired_watchlist.return_value = []
        r = await client.post(f"{WL_PREFIX}/cleanup")
        assert r.status_code == 200
        assert r.json() == []

    async def test_cleanup_with_expired(self, client: httpx.AsyncClient, mock_db_dep):
        mock_db_dep.deactivate_expired_watchlist.return_value = ["AAPL", "GME"]
        r = await client.post(f"{WL_PREFIX}/cleanup")
        assert r.status_code == 200
        assert sorted(r.json()) == ["AAPL", "GME"]


# ===========================================================================
# System endpoints  /api/v1/system/...
# ===========================================================================

SYS_PREFIX = "/api/v1/system"


class TestSystemStatus:
    async def test_status(self, client: httpx.AsyncClient, mock_agent_state):
        r = await client.get(f"{SYS_PREFIX}/status")
        assert r.status_code == 200
        body = r.json()
        assert body["db_enabled"] is True
        assert body["scheduler_running"] is True


class TestSystemConfig:
    async def test_config(self, client: httpx.AsyncClient):
        with patch("synesis.api.routes.system.get_settings") as mock_settings:
            s = MagicMock()
            s.env = "development"
            s.llm_provider = "anthropic"
            mock_settings.return_value = s
            r = await client.get(f"{SYS_PREFIX}/config")
        assert r.status_code == 200
        body = r.json()
        assert body["env"] == "development"
        assert body["llm_provider"] == "anthropic"


# ===========================================================================
# Twitter endpoints  /api/v1/twitter/...
# ===========================================================================

TWITTER_PREFIX = "/api/v1/twitter"


class TestTwitterSearch:
    async def test_search_single_ticker_defaults(self, client: httpx.AsyncClient):
        """Search with default parameters."""
        mock_tweet = MagicMock()
        mock_tweet.tweet_id = "123"
        mock_tweet.username = "trader_joe"
        mock_tweet.text = "Bullish on $NVDA"
        mock_tweet.timestamp.isoformat.return_value = "2026-05-01T12:00:00+00:00"
        mock_tweet.raw = {"likeCount": 250, "retweetCount": 50}

        mock_client = AsyncMock()
        mock_client.search_tweets = AsyncMock(return_value=([mock_tweet], None))
        mock_client.stop = AsyncMock()

        with (
            patch("synesis.api.routes.twitter.get_settings") as mock_settings,
            patch("synesis.api.routes.twitter.TwitterClient", return_value=mock_client),
        ):
            s = MagicMock()
            s.twitterapi_api_key.get_secret_value.return_value = "test-key"
            s.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value = s

            r = await client.get(f"{TWITTER_PREFIX}/search?tickers=nvda")

        assert r.status_code == 200
        body = r.json()
        nvda = body["results"]["NVDA"]
        assert "min_faves:200" in nvda["query"]
        assert "-filter:replies" in nvda["query"]
        assert "since_time:" in nvda["query"]
        assert " since:" not in nvda["query"]
        assert nvda["count"] == 1
        assert nvda["tweets"][0]["username"] == "trader_joe"
        assert nvda["tweets"][0]["likes"] == 250
        call_args = mock_client.search_tweets.call_args
        assert call_args.kwargs["query_type"] == "Top"
        mock_client.stop.assert_awaited_once()

    async def test_search_multi_ticker(self, client: httpx.AsyncClient):
        """Search multiple tickers in one request."""
        def _make_mock(tid: str, user: str, text: str, likes: int):
            m = MagicMock()
            m.tweet_id = tid
            m.username = user
            m.text = text
            m.timestamp.isoformat.return_value = "2026-05-06T10:00:00+00:00"
            m.raw = {"likeCount": likes, "retweetCount": 10}
            return m

        mock_client = AsyncMock()
        mock_client.search_tweets = AsyncMock(side_effect=[
            ([_make_mock("1", "user_a", "$NVDA up", 300)], None),
            ([_make_mock("2", "user_b", "$AMD down", 150)], None),
        ])
        mock_client.stop = AsyncMock()

        with (
            patch("synesis.api.routes.twitter.get_settings") as mock_settings,
            patch("synesis.api.routes.twitter.TwitterClient", return_value=mock_client),
        ):
            s = MagicMock()
            s.twitterapi_api_key.get_secret_value.return_value = "test-key"
            s.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value = s

            r = await client.get(f"{TWITTER_PREFIX}/search?tickers=NVDA,AMD")

        assert r.status_code == 200
        body = r.json()
        assert "NVDA" in body["results"]
        assert "AMD" in body["results"]
        assert body["results"]["NVDA"]["count"] == 1
        assert body["results"]["AMD"]["count"] == 1
        assert body["results"]["NVDA"]["tweets"][0]["text"] == "$NVDA up"
        assert body["results"]["AMD"]["tweets"][0]["text"] == "$AMD down"
        assert mock_client.search_tweets.await_count == 2
        mock_client.stop.assert_awaited_once()

    async def test_search_custom_params(self, client: httpx.AsyncClient):
        """Search with custom min_faves, since_days, and query_type."""
        mock_tweet = MagicMock()
        mock_tweet.tweet_id = "456"
        mock_tweet.username = "analyst_jane"
        mock_tweet.text = "Top tweet about $AAPL"
        mock_tweet.timestamp.isoformat.return_value = "2026-05-06T10:00:00+00:00"
        mock_tweet.raw = {"likeCount": 1000, "retweetCount": 200}

        mock_client = AsyncMock()
        mock_client.search_tweets = AsyncMock(return_value=([mock_tweet], None))
        mock_client.stop = AsyncMock()

        with (
            patch("synesis.api.routes.twitter.get_settings") as mock_settings,
            patch("synesis.api.routes.twitter.TwitterClient", return_value=mock_client),
        ):
            s = MagicMock()
            s.twitterapi_api_key.get_secret_value.return_value = "test-key"
            s.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value = s

            r = await client.get(
                f"{TWITTER_PREFIX}/search?tickers=aapl"
                "&min_faves=1000&since_days=2&query_type=Top&exclude_replies=false"
            )

        assert r.status_code == 200
        body = r.json()
        aapl = body["results"]["AAPL"]
        assert "min_faves:1000" in aapl["query"]
        assert "since_time:" in aapl["query"]
        assert "-filter:replies" not in aapl["query"]
        assert aapl["count"] == 1
        call_args = mock_client.search_tweets.call_args
        assert call_args.kwargs["query_type"] == "Top"
        mock_client.stop.assert_awaited_once()

    async def test_search_missing_api_key(self, client: httpx.AsyncClient):
        """503 when TWITTERAPI_API_KEY is not configured."""
        with patch("synesis.api.routes.twitter.get_settings") as mock_settings:
            s = MagicMock()
            s.twitterapi_api_key = None
            mock_settings.return_value = s

            r = await client.get(f"{TWITTER_PREFIX}/search?tickers=TSLA")

        assert r.status_code == 503
        assert "not configured" in r.json()["detail"]

    async def test_search_partial_failure(self, client: httpx.AsyncClient):
        """One ticker fails but others still return."""
        mock_tweet = MagicMock()
        mock_tweet.tweet_id = "1"
        mock_tweet.username = "user"
        mock_tweet.text = "ok"
        mock_tweet.timestamp.isoformat.return_value = "2026-05-06T10:00:00+00:00"
        mock_tweet.raw = {"likeCount": 100, "retweetCount": 5}

        mock_client = AsyncMock()
        mock_client.search_tweets = AsyncMock(side_effect=[
            ([mock_tweet], None),
            Exception("rate limited"),
        ])
        mock_client.stop = AsyncMock()

        with (
            patch("synesis.api.routes.twitter.get_settings") as mock_settings,
            patch("synesis.api.routes.twitter.TwitterClient", return_value=mock_client),
        ):
            s = MagicMock()
            s.twitterapi_api_key.get_secret_value.return_value = "test-key"
            s.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value = s

            r = await client.get(f"{TWITTER_PREFIX}/search?tickers=META,AMZN")

        assert r.status_code == 200
        body = r.json()
        assert body["results"]["META"]["count"] == 1
        assert body["results"]["AMZN"]["error"] == "rate limited"
        assert body["results"]["AMZN"]["count"] == 0
        mock_client.stop.assert_awaited_once()

    async def test_search_empty_results(self, client: httpx.AsyncClient):
        """Handle empty search results gracefully."""
        mock_client = AsyncMock()
        mock_client.search_tweets = AsyncMock(return_value=([], None))
        mock_client.stop = AsyncMock()

        with (
            patch("synesis.api.routes.twitter.get_settings") as mock_settings,
            patch("synesis.api.routes.twitter.TwitterClient", return_value=mock_client),
        ):
            s = MagicMock()
            s.twitterapi_api_key.get_secret_value.return_value = "test-key"
            s.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value = s

            r = await client.get(f"{TWITTER_PREFIX}/search?tickers=XYZ")

        assert r.status_code == 200
        body = r.json()
        assert body["results"]["XYZ"]["count"] == 0
        assert body["results"]["XYZ"]["tweets"] == []

    async def test_search_no_tickers(self, client: httpx.AsyncClient):
        """422 when no tickers provided."""
        with (
            patch("synesis.api.routes.twitter.get_settings") as mock_settings,
        ):
            s = MagicMock()
            s.twitterapi_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value = s

            r = await client.get(f"{TWITTER_PREFIX}/search")

        assert r.status_code == 422
        # FastAPI auto-validates missing required query param
        detail = r.json()["detail"]
        assert isinstance(detail, list)
        assert detail[0]["loc"] == ["query", "tickers"]

    async def test_search_empty_tickers(self, client: httpx.AsyncClient):
        """422 when tickers param is empty string."""
        with (
            patch("synesis.api.routes.twitter.get_settings") as mock_settings,
        ):
            s = MagicMock()
            s.twitterapi_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value = s

            r = await client.get(f"{TWITTER_PREFIX}/search?tickers=")

        assert r.status_code == 422
        assert "No valid tickers" in r.json()["detail"]

    async def test_search_rejects_advanced_search_injection(self, client: httpx.AsyncClient):
        """422 when a ticker value contains Twitter search operators."""
        r = await client.get(f"{TWITTER_PREFIX}/search?tickers=NVDA%20OR%20TSLA")

        assert r.status_code == 422
        assert "Invalid ticker" in r.json()["detail"]

    async def test_search_too_many_tickers(self, client: httpx.AsyncClient):
        """422 when more than 10 tickers."""
        with (
            patch("synesis.api.routes.twitter.get_settings") as mock_settings,
        ):
            s = MagicMock()
            s.twitterapi_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value = s

            r = await client.get(f"{TWITTER_PREFIX}/search?tickers=A,B,C,D,E,F,G,H,I,J,K")

        assert r.status_code == 422
        assert "Max 10" in r.json()["detail"]


# ===========================================================================
# Intelligence endpoints
# ===========================================================================
