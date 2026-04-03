"""Tests for Twitter agent daily digest pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from synesis.ingestion.twitterapi import Tweet
from synesis.notifications.discord import (
    COLOR_BEARISH,
    COLOR_BULLISH,
    COLOR_NEUTRAL,
    format_twitter_agent_embed,
    format_twitter_agent_embeds,
)
from synesis.processing.common.watchlist import WatchlistManager
from synesis.processing.twitter.accounts import ACCOUNT_PROFILES, get_profile
from synesis.processing.twitter.models import (
    Theme,
    TickerMention,
    TwitterAgentAnalysis,
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
        raw={},
    )


def _make_analysis(**kwargs: object) -> TwitterAgentAnalysis:
    from synesis.processing.twitter.models import AccountSummary

    defaults: dict = {
        "market_overview": "Markets rallied on strong earnings. Photonics sector led gains.",
        "account_summaries": [
            AccountSummary(
                username="aleabitoreddit",
                posted_about="Posted about AAOI earnings beat and photonics demand.",
                theses=["Long AAOI on supply constraints"],
            ),
        ],
        "themes": [
            Theme(
                title="Photonics supply chain bottleneck",
                summary="Multiple accounts flagged AAOI earnings beat and supply constraints.",
                category="earnings",
                sources=["aleabitoreddit", "unusual_whales"],
                tickers=[
                    TickerMention(
                        ticker="AAOI",
                        direction="bullish",
                        reasoning="Q4 revenue beat 20%, backlog growing",
                        current_price=145.0,
                        price_context="Trading at $145, +8% today, above 50d MA ($130)",
                        trade_idea="Buy Jun $140 calls, IV at 35% is low vs 60d realized",
                        time_horizon="weeks",
                        conviction="high",
                    ),
                    TickerMention(
                        ticker="LITE",
                        direction="bullish",
                        reasoning="Competitor benefiting from same demand wave",
                    ),
                ],
                risk_factors=["Customer concentration risk", "Tariff exposure"],
                verified=True,
                verification_notes="AAOI Q4 revenue confirmed at $140M vs $117M est",
                conviction="high",
                research_notes="Confirmed AAOI Q4 beat via SEC filing. DRAM supply tightening per TrendForce.",
            ),
            Theme(
                title="Korean memory selloff",
                summary="EWY dumping on Samsung guidance cut.",
                category="sector",
                sources=["aleabitoreddit"],
                tickers=[
                    TickerMention(
                        ticker="EWY",
                        direction="bearish",
                        reasoning="Short-term sector rotation out of Korea",
                    ),
                    TickerMention(
                        ticker="MU",
                        direction="bullish",
                        reasoning="US memory names benefit from Korea weakness",
                    ),
                ],
                risk_factors=["Broad EM contagion"],
                verified=False,
                verification_notes="Samsung guidance not yet public",
                conviction="medium",
            ),
        ],
        "raw_tweet_count": 15,
    }
    defaults.update(kwargs)
    return TwitterAgentAnalysis(**defaults)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestTwitterAgentModels:
    def test_ticker_mention_per_ticker_direction(self) -> None:
        """Direction is per-ticker, not per-theme."""
        theme = _make_analysis().themes[1]  # Korean memory selloff
        directions = {tm.ticker: tm.direction for tm in theme.tickers}
        assert directions["EWY"] == "bearish"
        assert directions["MU"] == "bullish"

    def test_analysis_raw_tweet_count(self) -> None:
        a = _make_analysis(raw_tweet_count=42)
        assert a.raw_tweet_count == 42

    def test_theme_categories(self) -> None:
        a = _make_analysis()
        cats = [t.category for t in a.themes]
        assert "earnings" in cats
        assert "sector" in cats


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
# Discord embed formatter
# ---------------------------------------------------------------------------


class TestFormatTwitterAgentEmbeds:
    """Tests for the multi-message format (one message per theme)."""

    def test_header_plus_theme_messages(self) -> None:
        """Should produce 1 header + 1 per theme = 3 messages."""
        messages = format_twitter_agent_embeds(_make_analysis())
        assert len(messages) == 3  # header + 2 themes

    def test_header_contains_overview(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        header = messages[0][0]
        assert "Daily X Brief" in header["title"]
        assert "Markets rallied" in header["description"]
        assert header["color"] == COLOR_NEUTRAL

    def test_header_shows_counts(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        header = messages[0][0]
        field_values = {f["name"]: f["value"] for f in header["fields"]}
        assert field_values["Themes"] == "2"
        assert field_values["Tweets Analyzed"] == "15"

    def test_theme_messages_have_correct_color(self) -> None:
        """Bullish theme → green, bearish-leaning → red."""
        messages = format_twitter_agent_embeds(_make_analysis())
        # Theme 1 (Photonics): AAOI bullish + LITE bullish → green
        assert messages[1][0]["color"] == COLOR_BULLISH
        # Theme 2 (Korean memory): EWY bearish, MU bullish → tied → neutral
        assert messages[2][0]["color"] == COLOR_NEUTRAL

    def test_theme_tickers_in_fields(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        # Theme 1 has AAOI and LITE
        theme1_fields = " ".join(f["value"] for f in messages[1][0]["fields"])
        assert "$AAOI" in theme1_fields
        assert "$LITE" in theme1_fields
        # Theme 2 has EWY and MU
        theme2_fields = " ".join(f["value"] for f in messages[2][0]["fields"])
        assert "$EWY" in theme2_fields
        assert "$MU" in theme2_fields

    def test_verification_badge_in_title(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        # Theme 1 is verified
        assert "\u2705" in messages[1][0]["title"]
        # Theme 2 is not verified
        assert "\u2753" in messages[2][0]["title"]

    def test_sources_in_theme(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        theme1_fields = " ".join(f["value"] for f in messages[1][0]["fields"])
        assert "@aleabitoreddit" in theme1_fields

    def test_risk_factors_shown(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        theme1_fields = " ".join(f["value"] for f in messages[1][0]["fields"])
        assert "Tariff exposure" in theme1_fields

    def test_theme_footer_shows_position(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        assert "Theme 1/2" in messages[1][0]["footer"]["text"]
        assert "Theme 2/2" in messages[2][0]["footer"]["text"]

    def test_empty_themes_produces_header_only(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis(themes=[]))
        assert len(messages) == 1
        assert "Daily X Brief" in messages[0][0]["title"]

    def test_legacy_flat_wrapper_still_works(self) -> None:
        """format_twitter_agent_embed returns flat list of all embeds."""
        embeds = format_twitter_agent_embed(_make_analysis())
        assert len(embeds) == 4  # header + account summaries + 2 themes
        assert "Daily X Brief" in embeds[0]["title"]


# ---------------------------------------------------------------------------
# Watchlist add_ticker with reason
# ---------------------------------------------------------------------------


class TestWatchlistAddReason:
    @pytest.mark.asyncio
    async def test_add_ticker_uses_custom_reason(self) -> None:
        """add_ticker should pass the provided reason, not generate a generic one."""
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
        """Without added_reason, falls back to 'Signal from {source}'."""
        mock_db = AsyncMock()
        mock_db.upsert_watchlist_ticker = AsyncMock(return_value=True)

        wl = WatchlistManager(db=mock_db)
        await wl.add_ticker("AAOI", source="@aleabitoreddit")

        call_kwargs = mock_db.upsert_watchlist_ticker.call_args.kwargs
        assert call_kwargs["added_reason"] == "Signal from @aleabitoreddit"

    @pytest.mark.asyncio
    async def test_bulk_add_passes_reason(self) -> None:
        """bulk_add should forward added_reason to each add_ticker call."""
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


# ---------------------------------------------------------------------------
# Job: tweet filtering and watchlist integration
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
    async def test_skips_when_no_tweets(self) -> None:
        """Job exits early when no tweets found in 24hr window."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["testaccount"]
            mock_settings.return_value.discord_twitter_webhook_url = None

            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([], None))

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job()  # Should not raise or send Discord

    @pytest.mark.asyncio
    async def test_adds_tickers_to_watchlist_with_reason(self) -> None:
        """Job adds mentioned tickers with theme-specific reason."""
        analysis = _make_analysis()

        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
            patch("synesis.processing.twitter.job.send_discord", new_callable=AsyncMock),
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]
            mock_settings.return_value.discord_twitter_webhook_url = SecretStr("https://hook")

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_tweets = AsyncMock(return_value=analysis)

            mock_watchlist = AsyncMock(spec=WatchlistManager)
            mock_watchlist.add_ticker = AsyncMock(return_value=True)

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(watchlist=mock_watchlist)

        # Should have called add_ticker for each ticker across all themes
        assert mock_watchlist.add_ticker.call_count == 4  # AAOI, LITE, EWY, MU

        # Check first call (AAOI from first theme)
        first_call = mock_watchlist.add_ticker.call_args_list[0]
        assert first_call.args[0] == "AAOI"
        assert "@aleabitoreddit" in first_call.kwargs["source"]
        assert "Q4 revenue beat" in first_call.kwargs["added_reason"]

    @pytest.mark.asyncio
    async def test_discord_send_count_tracked(self) -> None:
        """Job should track actual send successes, not just message count."""
        analysis = _make_analysis()

        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
            patch(
                "synesis.processing.twitter.job.send_discord",
                new_callable=AsyncMock,
                side_effect=[True, False, True],  # 1st ok, 2nd fails, 3rd ok
            ) as mock_send,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]
            mock_settings.return_value.discord_twitter_webhook_url = SecretStr("https://hook")

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_tweets = AsyncMock(return_value=analysis)

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job()

        assert mock_send.call_count == 3  # header + 2 themes

    @pytest.mark.asyncio
    async def test_watchlist_error_does_not_block_discord(self) -> None:
        """If watchlist DB errors, Discord notification should still be sent."""
        analysis = _make_analysis()

        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
            patch(
                "synesis.processing.twitter.job.send_discord",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]
            mock_settings.return_value.discord_twitter_webhook_url = SecretStr("https://hook")

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_tweets = AsyncMock(return_value=analysis)

            mock_watchlist = AsyncMock(spec=WatchlistManager)
            mock_watchlist.add_ticker = AsyncMock(side_effect=RuntimeError("DB connection lost"))

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(watchlist=mock_watchlist)

        # Discord should still be called despite watchlist failure
        assert mock_send.call_count == 3

    @pytest.mark.asyncio
    async def test_job_saves_to_diary(self) -> None:
        """Job persists analysis to diary with source='twitter'."""
        analysis = _make_analysis()

        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
            patch("synesis.processing.twitter.job.send_discord", new_callable=AsyncMock),
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]
            mock_settings.return_value.discord_twitter_webhook_url = SecretStr("https://hook")

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_tweets = AsyncMock(return_value=analysis)

            mock_db = AsyncMock()

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(db=mock_db)

        mock_db.upsert_diary_entry.assert_called_once()
        call_kwargs = mock_db.upsert_diary_entry.call_args[1]
        assert call_kwargs["source"] == "twitter"
        assert call_kwargs["payload"] == analysis.model_dump(mode="json")

    @pytest.mark.asyncio
    async def test_job_diary_failure_does_not_crash(self) -> None:
        """If diary upsert raises, Discord send still happens."""
        analysis = _make_analysis()

        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
            patch(
                "synesis.processing.twitter.job.send_discord",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]
            mock_settings.return_value.discord_twitter_webhook_url = SecretStr("https://hook")

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_tweets = AsyncMock(return_value=analysis)

            mock_db = AsyncMock()
            mock_db.upsert_diary_entry = AsyncMock(side_effect=RuntimeError("DB down"))

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(db=mock_db)

        # Discord should still be called despite diary failure
        assert mock_send.call_count == 3

    @pytest.mark.asyncio
    async def test_job_no_db_skips_diary(self) -> None:
        """When db=None (default), diary is skipped without error."""
        analysis = _make_analysis()

        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
            patch("synesis.processing.twitter.job.send_discord", new_callable=AsyncMock),
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]
            mock_settings.return_value.discord_twitter_webhook_url = SecretStr("https://hook")

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_tweets = AsyncMock(return_value=analysis)

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job()  # db=None by default, should not raise


# ---------------------------------------------------------------------------
# Analyzer: edge cases
# ---------------------------------------------------------------------------


class TestTwitterAgentAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_empty_tweets_returns_none(self) -> None:
        """analyze_tweets([]) should return None without calling the LLM."""
        from synesis.processing.twitter.analyzer import TwitterAgentAnalyzer

        analyzer = TwitterAgentAnalyzer()
        result = await analyzer.analyze_tweets([])
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_tweets_exception_returns_none(self) -> None:
        """analyze_tweets should return None if the LLM agent raises."""
        from synesis.processing.twitter.analyzer import TwitterAgentAnalyzer

        analyzer = TwitterAgentAnalyzer()

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        analyzer._agent = mock_agent  # type: ignore[assignment]
        result = await analyzer.analyze_tweets([_make_tweet()])

        assert result is None


# ---------------------------------------------------------------------------
# Discord: bearish-dominant theme color
# ---------------------------------------------------------------------------


class TestThemeColor:
    def test_bearish_dominant_theme_gets_red(self) -> None:
        """A theme with more bearish than bullish tickers should be red."""
        analysis = _make_analysis(
            themes=[
                Theme(
                    title="Tech selloff",
                    summary="Broad tech weakness",
                    category="sector",
                    sources=["testaccount"],
                    tickers=[
                        TickerMention(
                            ticker="AAPL", direction="bearish", reasoning="Weak guidance"
                        ),
                        TickerMention(
                            ticker="MSFT", direction="bearish", reasoning="Cloud slowdown"
                        ),
                        TickerMention(ticker="NVDA", direction="bullish", reasoning="AI demand"),
                    ],
                    risk_factors=[],
                    verified=False,
                    verification_notes="",
                    conviction="medium",
                ),
            ],
        )
        messages = format_twitter_agent_embeds(analysis)
        assert messages[1][0]["color"] == COLOR_BEARISH


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
# TwitterAgentDeps.accounts deduplication
# ---------------------------------------------------------------------------


class TestTwitterAgentDeps:
    def test_accounts_deduplication(self) -> None:
        """accounts property returns unique sorted usernames."""
        from synesis.processing.twitter.analyzer import TwitterAgentDeps

        tweets = [
            _make_tweet(username="alice"),
            _make_tweet(username="bob"),
            _make_tweet(username="alice"),
            _make_tweet(username="bob"),
            _make_tweet(username="charlie"),
        ]
        deps = TwitterAgentDeps(tweets=tweets)
        assert deps.accounts == ["alice", "bob", "charlie"]


# ---------------------------------------------------------------------------
# API route: 503 when not configured, 200 when configured
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
# Job: partial fetch failure (one account fails, others succeed)
# ---------------------------------------------------------------------------


class TestPartialFetchFailure:
    @pytest.mark.asyncio
    async def test_one_account_fails_others_succeed(self) -> None:
        """If one account fetch fails, tweets from others are still processed."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["good_account", "bad_account"]
            mock_settings.return_value.discord_twitter_webhook_url = None

            tweet = _make_tweet(username="good_account", hours_ago=1.0)

            async def _side_effect(username: str) -> tuple[list[Tweet], str | None]:
                if username == "bad_account":
                    raise httpx.RequestError("Connection refused")
                return [tweet], None

            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(side_effect=_side_effect)

            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_tweets = AsyncMock(return_value=_make_analysis())

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job()

        # Analyzer should have been called with the one successful tweet
        mock_analyzer.analyze_tweets.assert_called_once()
        tweets_passed = mock_analyzer.analyze_tweets.call_args.args[0]
        assert len(tweets_passed) == 1
        assert tweets_passed[0].username == "good_account"

    @pytest.mark.asyncio
    async def test_all_accounts_fail_logs_error(self) -> None:
        """If all account fetches fail, job exits early without calling analyzer."""
        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["acc1", "acc2"]
            mock_settings.return_value.discord_twitter_webhook_url = None

            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(
                side_effect=httpx.RequestError("Connection refused")
            )

            mock_analyzer = mock_analyzer_cls.return_value

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job()

        # Analyzer should NOT have been called
        mock_analyzer.analyze_tweets.assert_not_called()

    @pytest.mark.asyncio
    async def test_job_passes_yfinance(self) -> None:
        """Job should forward yfinance to analyzer."""
        analysis = _make_analysis()

        with (
            patch("synesis.processing.twitter.job.get_settings") as mock_settings,
            patch("synesis.processing.twitter.job.TwitterClient") as mock_client_cls,
            patch("synesis.processing.twitter.job.TwitterAgentAnalyzer") as mock_analyzer_cls,
            patch("synesis.processing.twitter.job.send_discord", new_callable=AsyncMock),
        ):
            from pydantic import SecretStr

            mock_settings.return_value.twitterapi_api_key = SecretStr("key")
            mock_settings.return_value.twitter_api_base_url = "https://api.twitterapi.io"
            mock_settings.return_value.twitter_accounts = ["aleabitoreddit"]
            mock_settings.return_value.discord_twitter_webhook_url = None

            tweet = _make_tweet(hours_ago=1.0)
            mock_instance = mock_client_cls.return_value
            mock_instance.get_user_tweets = AsyncMock(return_value=([tweet], None))

            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_tweets = AsyncMock(return_value=analysis)

            fake_yf = object()

            from synesis.processing.twitter.job import twitter_agent_job

            await twitter_agent_job(yfinance=fake_yf)

        call_kwargs = mock_analyzer.analyze_tweets.call_args.kwargs
        assert call_kwargs["yfinance"] is fake_yf


# ---------------------------------------------------------------------------
# Expanded model fields
# ---------------------------------------------------------------------------


class TestExpandedModelFields:
    def test_ticker_mention_new_fields_default_none(self) -> None:
        """New optional fields default to None for backward compat."""
        tm = TickerMention(ticker="AAPL", direction="bullish", reasoning="Strong earnings")
        assert tm.current_price is None
        assert tm.price_context is None
        assert tm.trade_idea is None
        assert tm.time_horizon is None
        assert tm.conviction is None

    def test_ticker_mention_with_all_fields(self) -> None:
        tm = TickerMention(
            ticker="AAPL",
            direction="bullish",
            reasoning="Strong earnings",
            current_price=195.50,
            price_context="Trading at $195.50, +2% today, above 50d MA",
            trade_idea="Buy Apr $190 calls",
            time_horizon="weeks",
            conviction="high",
        )
        assert tm.current_price == 195.50
        assert tm.time_horizon == "weeks"
        assert tm.conviction == "high"

    def test_theme_research_notes_default_none(self) -> None:
        theme = Theme(
            title="Test",
            summary="Test",
            category="macro",
            sources=["test"],
            tickers=[],
            risk_factors=[],
            verified=False,
            verification_notes="",
            conviction="low",
        )
        assert theme.research_notes is None

    def test_theme_with_research_notes(self) -> None:
        theme = Theme(
            title="Test",
            summary="Test",
            category="macro",
            sources=["test"],
            tickers=[],
            risk_factors=[],
            verified=True,
            verification_notes="Confirmed",
            conviction="high",
            research_notes="Web search confirmed CPI came in hot at 3.5%",
        )
        assert "CPI" in theme.research_notes  # type: ignore[operator]

    def test_expanded_model_serializes_correctly(self) -> None:
        """Full analysis with new fields round-trips through JSON."""
        analysis = _make_analysis()
        data = analysis.model_dump(mode="json")
        restored = TwitterAgentAnalysis.model_validate(data)
        assert restored.themes[0].tickers[0].current_price == 145.0
        assert restored.themes[0].research_notes is not None
        # Second theme has no new fields
        assert restored.themes[1].tickers[0].current_price is None
        assert restored.themes[1].research_notes is None


# ---------------------------------------------------------------------------
# Discord: new fields rendering
# ---------------------------------------------------------------------------


class TestDiscordNewFields:
    def test_price_context_shown_in_ticker(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        theme1_fields = " ".join(f["value"] for f in messages[1][0]["fields"])
        assert "$145" in theme1_fields
        assert "above 50d MA" in theme1_fields

    def test_trade_idea_shown_in_ticker(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        theme1_fields = " ".join(f["value"] for f in messages[1][0]["fields"])
        assert "Idea:" in theme1_fields
        assert "Jun $140 calls" in theme1_fields

    def test_research_notes_shown_as_field(self) -> None:
        messages = format_twitter_agent_embeds(_make_analysis())
        field_names = [f["name"] for f in messages[1][0]["fields"]]
        assert "Research" in field_names
        research_field = next(f for f in messages[1][0]["fields"] if f["name"] == "Research")
        assert "DRAM supply tightening" in research_field["value"]

    def test_no_research_field_when_none(self) -> None:
        """Theme without research_notes should not have Research field."""
        messages = format_twitter_agent_embeds(_make_analysis())
        # Theme 2 has no research_notes
        field_names = [f["name"] for f in messages[2][0]["fields"]]
        assert "Research" not in field_names


# ---------------------------------------------------------------------------
# Analyzer deps with new fields
# ---------------------------------------------------------------------------


class TestTwitterAgentDepsExpanded:
    def test_deps_with_providers(self) -> None:
        from synesis.processing.twitter.analyzer import TwitterAgentDeps

        tweets = [_make_tweet()]
        deps = TwitterAgentDeps(tweets=tweets, yfinance=None)
        assert deps.yfinance is None
        assert deps.accounts == [tweets[0].username]
