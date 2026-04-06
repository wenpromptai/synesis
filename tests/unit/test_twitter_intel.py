"""Tests for Twitter data collection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from synesis.ingestion.twitterapi import Tweet
from synesis.processing.common.watchlist import WatchlistManager
from synesis.processing.intelligence.specialists.social_sentiment.x_accounts import (
    ACCOUNT_PROFILES,
    get_profile,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


_tweet_counter = 0


def _make_tweet(
    username: str = "aleabitoreddit",
    text: str = "AAOI crushing earnings, photonics demand through the roof",
    hours_ago: float = 2.0,
) -> Tweet:
    global _tweet_counter
    _tweet_counter += 1
    return Tweet(
        tweet_id=f"tw_{_tweet_counter}",
        user_id=f"uid_{username}",
        username=username,
        text=text,
        timestamp=datetime.now(UTC) - timedelta(hours=hours_ago),
        raw={"url": f"https://x.com/{username}/status/tw_{_tweet_counter}"},
    )


# ---------------------------------------------------------------------------
# Account profiles
# ---------------------------------------------------------------------------


class TestAccountProfiles:
    def test_all_24_accounts_have_profiles(self) -> None:
        assert len(ACCOUNT_PROFILES) == 24

    def test_case_insensitive_lookup(self) -> None:
        assert get_profile("NickTimiraos") is not None
        assert get_profile("nicktimiraos") is not None
        assert get_profile("NICKTIMIRAOS") is not None

    def test_short_sellers_flagged_as_biased(self) -> None:
        for handle in ["fuzzypandashort", "muddywatersre"]:
            profile = get_profile(handle)
            assert profile is not None
            assert "BIAS" in profile.description

    def test_fed_reporter_flagged(self) -> None:
        profile = get_profile("NickTimiraos")
        assert profile is not None
        assert profile.category == "fed_reporter"
        assert "near-official" in profile.description

    def test_unknown_handle_returns_none(self) -> None:
        assert get_profile("nonexistent_handle_12345") is None


# ---------------------------------------------------------------------------
# Job: data collection
# ---------------------------------------------------------------------------


class TestTwitterAgentJob:
    @pytest.mark.asyncio
    async def test_skips_when_no_api_key(self) -> None:
        """Job exits early when Twitter API key is not configured."""
        with patch("synesis.processing.twitter.job.get_settings") as mock_settings:
            mock_settings.return_value.twitterapi_api_key = None
            mock_settings.return_value.twitter_accounts = []

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job()  # Should not raise

    @pytest.mark.asyncio
    async def test_skips_when_no_db(self) -> None:
        """Job exits early when no database is configured."""
        with patch("synesis.processing.twitter.job.get_settings") as mock_settings:
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_accounts = ["testaccount"]

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(db=None)  # Should not raise

    @pytest.mark.asyncio
    async def test_skips_when_no_tweets(self) -> None:
        """Job exits early when no tweets found."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["testaccount"]

            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([], None))

            mock_db = AsyncMock()

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(db=mock_db)

        mock_db.store_raw_tweets.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_raw_tweets(self) -> None:
        """Job should persist raw tweets to DB."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_db = AsyncMock()
            mock_db.store_raw_tweets = AsyncMock(return_value=1)

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(db=mock_db)

        mock_db.store_raw_tweets.assert_called_once()
        stored_tweets = mock_db.store_raw_tweets.call_args.args[0]
        assert len(stored_tweets) == 1
        assert stored_tweets[0]["account_username"] == tweet.username
        assert stored_tweets[0]["tweet_id"] == tweet.tweet_id
        assert stored_tweets[0]["tweet_url"] == tweet.raw.get("url")

    @pytest.mark.asyncio
    async def test_db_store_failure_raises(self) -> None:
        """If DB store raises, job should propagate the exception."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_db = AsyncMock()
            mock_db.store_raw_tweets = AsyncMock(side_effect=RuntimeError("DB connection lost"))

            from synesis.processing.twitter.job import twitter_agent_job

            with pytest.raises(RuntimeError, match="DB connection lost"):
                await twitter_agent_job(db=mock_db)

    @pytest.mark.asyncio
    async def test_stores_all_tweets_no_time_filter(self) -> None:
        """Job stores all fetched tweets — DB handles dedup, no Python-side filtering."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]

            # One recent, one old — both should be stored
            tweets = [
                _make_tweet(hours_ago=1.0),
                _make_tweet(hours_ago=48.0),
            ]
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=(tweets, None))

            mock_db = AsyncMock()
            mock_db.store_raw_tweets = AsyncMock(return_value=2)

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(db=mock_db)

        stored_tweets = mock_db.store_raw_tweets.call_args.args[0]
        assert len(stored_tweets) == 2


# ---------------------------------------------------------------------------
# Job: partial fetch failure
# ---------------------------------------------------------------------------


class TestPartialFetchFailure:
    @pytest.mark.asyncio
    async def test_one_account_fails_others_succeed(self) -> None:
        """If one account fetch fails, tweets from others are still stored."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["good_account", "bad_account"]

            tweet = _make_tweet(username="good_account", hours_ago=1.0)

            async def _side_effect(username: str) -> tuple[list[Tweet], str | None]:
                if username == "bad_account":
                    raise httpx.RequestError("Connection refused")
                return [tweet], None

            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(side_effect=_side_effect)

            mock_db = AsyncMock()
            mock_db.store_raw_tweets = AsyncMock(return_value=1)

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(db=mock_db)

        stored_tweets = mock_db.store_raw_tweets.call_args.args[0]
        assert len(stored_tweets) == 1
        assert stored_tweets[0]["account_username"] == "good_account"

    @pytest.mark.asyncio
    async def test_all_accounts_fail_skips_storage(self) -> None:
        """If all account fetches fail, job exits without storing."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["acc1", "acc2"]

            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(
                side_effect=httpx.RequestError("Connection refused")
            )

            mock_db = AsyncMock()

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(db=mock_db)

        mock_db.store_raw_tweets.assert_not_called()


# ---------------------------------------------------------------------------
# _make_tweet unique IDs
# ---------------------------------------------------------------------------


class TestMakeTweetUniqueIds:
    def test_unique_tweet_ids(self) -> None:
        """Each call to _make_tweet produces a unique tweet_id."""
        t1 = _make_tweet()
        t2 = _make_tweet()
        assert t1.tweet_id != t2.tweet_id


# ---------------------------------------------------------------------------
# API route
# ---------------------------------------------------------------------------


@dataclass
class _FakeAgentState:
    trigger_fns: dict[str, Any] = field(default_factory=dict)


class TestTwitterAPIRoute:
    @pytest.fixture()
    def app_no_trigger(self) -> FastAPI:
        """App where twitter_agent trigger is NOT configured."""
        from synesis.api.routes.twitter import router
        from synesis.core.dependencies import get_agent_state

        app = FastAPI()
        app.include_router(router, prefix="/twitter")

        state = _FakeAgentState(trigger_fns={})
        app.dependency_overrides[get_agent_state] = lambda: state
        return app

    @pytest.fixture()
    def app_with_trigger(self) -> FastAPI:
        """App where twitter_agent trigger IS configured."""
        from synesis.api.routes.twitter import router
        from synesis.core.dependencies import get_agent_state

        app = FastAPI()
        app.include_router(router, prefix="/twitter")

        async def _noop() -> None:
            pass

        state = _FakeAgentState(trigger_fns={"twitter_agent": _noop})
        app.dependency_overrides[get_agent_state] = lambda: state
        return app

    @pytest.mark.asyncio
    async def test_503_when_not_configured(self, app_no_trigger: FastAPI) -> None:
        """POST /twitter/analyze returns 503 when trigger not registered."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_no_trigger), base_url="http://test"
        ) as client:
            r = await client.post("/twitter/analyze")
            assert r.status_code == 503

    @pytest.mark.asyncio
    async def test_200_when_configured(self, app_with_trigger: FastAPI) -> None:
        """POST /twitter/analyze returns 200 when trigger is registered."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_with_trigger), base_url="http://test"
        ) as client:
            r = await client.post("/twitter/analyze")
            assert r.status_code == 200
            assert r.json()["status"] == "triggered"


# ---------------------------------------------------------------------------
# Watchlist (independent of twitter — tests WatchlistManager directly)
# ---------------------------------------------------------------------------


class TestWatchlistAddReason:
    @pytest.mark.asyncio
    async def test_add_ticker_uses_custom_reason(self) -> None:
        mock_db = AsyncMock()
        mock_db.upsert_watchlist_ticker = AsyncMock(return_value=True)

        wl = WatchlistManager(db=mock_db)
        await wl.add_ticker(
            "AAOI",
            source="@aleabitoreddit",
            added_reason="Q4 revenue beat 20%, backlog growing",
        )

        mock_db.upsert_watchlist_ticker.assert_called_once()
        call_kwargs = mock_db.upsert_watchlist_ticker.call_args.kwargs
        assert call_kwargs["added_by"] == "@aleabitoreddit"
        assert call_kwargs["added_reason"] == "Q4 revenue beat 20%, backlog growing"

    @pytest.mark.asyncio
    async def test_add_ticker_fallback_reason(self) -> None:
        mock_db = AsyncMock()
        mock_db.upsert_watchlist_ticker = AsyncMock(return_value=True)

        wl = WatchlistManager(db=mock_db)
        await wl.add_ticker("AAOI", source="@aleabitoreddit")

        call_kwargs = mock_db.upsert_watchlist_ticker.call_args.kwargs
        assert call_kwargs["added_reason"] == "Signal from @aleabitoreddit"

    @pytest.mark.asyncio
    async def test_bulk_add_passes_reason(self) -> None:
        mock_db = AsyncMock()
        mock_db.upsert_watchlist_ticker = AsyncMock(return_value=True)

        wl = WatchlistManager(db=mock_db)
        added, refreshed = await wl.bulk_add(
            ["AAOI", "LITE"],
            source="@aleabitoreddit",
            added_reason="Photonics demand surge",
        )

        assert added == ["AAOI", "LITE"]
        assert mock_db.upsert_watchlist_ticker.call_count == 2

        for call in mock_db.upsert_watchlist_ticker.call_args_list:
            assert call.kwargs["added_reason"] == "Photonics demand surge"
            assert call.kwargs["added_by"] == "@aleabitoreddit"
