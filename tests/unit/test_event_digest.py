"""Tests for Event Radar digest (two-part daily digest) and surprise detection."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from synesis.processing.events.digest import (
    _enrich_with_outcomes,
    _format_whats_coming_embeds,
    _format_yesterday_brief_fallback,
    _format_yesterday_brief_rich,
    _get_13f_deadline_reminder,
    _get_crawled_outcome,
    _get_economic_data_outcome,
    _get_fomc_minutes_outcome,
    _get_yesterday_13f_briefs,
    _split_content,
    send_event_digest,
)
from synesis.processing.events.models import (
    Actionable,
    MarketSnapshot,
    SubAnalysis,
    YesterdayBriefAnalysis,
    YesterdayTheme,
)
from synesis.processing.events.yesterday.surprise import detect_surprise_events
from synesis.processing.events.yesterday import synthesize_yesterday_brief
from synesis.processing.events.yesterday.earnings import _format_earnings_events
from synesis.processing.events.yesterday.events import (
    _format_calendar_events,
    _format_surprise_events,
)
from synesis.processing.events.yesterday.filings import _format_filing_briefs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event_row(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": 1,
        "title": "FOMC Rate Decision",
        "description": "Federal Reserve interest rate decision",
        "event_date": date(2026, 3, 19),
        "event_end_date": None,
        "category": "fed",
        "sector": None,
        "region": ["US"],
        "tickers": [],
        "importance": 9,
        "source_urls": [],
        "confidence": 0.99,
    }
    defaults.update(overrides)
    return defaults


class _FakeRecord(dict):
    """Mimics asyncpg.Record."""

    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def _record(**kwargs: Any) -> _FakeRecord:
    return _FakeRecord(_make_event_row(**kwargs))


def _make_yesterday_analysis() -> YesterdayBriefAnalysis:
    return YesterdayBriefAnalysis(
        headline="Markets rallied on Fed dovish pivot",
        market_snapshot=MarketSnapshot(
            summary="Risk-on session driven by dovish Fed. Tech led, yields fell.",
            equities="SPY +1.2%, QQQ +1.8%, IWM +0.9%",
            rates_fx="TLT +1.5%, UUP -0.3%",
            commodities="GLD +0.5%, USO -0.8%",
            volatility="VIX 14.2 (-1.8)",
            sector_performance="XLK +2.1%, XLF +0.8%, XLE -0.3%",
        ),
        themes=[
            YesterdayTheme(
                title="Fed Dovish Pivot",
                category="macro",
                sentiment="bullish",
                source="calendar",
                outcome="Fed held rates at 5.25-5.50%, dot plot shifted to 3 cuts in 2026",
                analysis="The FOMC surprised markets with dovish language.",
                key_events=["FOMC held rates steady", "Dot plot shifted down"],
                tickers=["SPY", "TLT"],
                market_reaction="Bond yields fell sharply, risk assets rallied",
            ),
            YesterdayTheme(
                title="GPT-5.4 Launch",
                category="tech",
                sentiment="bullish",
                source="surprise",
                outcome="OpenAI launched GPT-5.4 with 2x context and native tool use",
                analysis="OpenAI announced GPT-5.4 with major capability improvements.",
                key_events=["GPT-5.4 announced"],
                tickers=["MSFT"],
                market_reaction="AI infrastructure stocks surged",
            ),
        ],
        synthesis="Risk-on sentiment likely to continue into the week.",
        actionables=[
            Actionable(
                action="Buy TLT on rate-cut repricing",
                rationale="Dovish dot plot supports duration",
                tickers=["TLT"],
                direction="long",
                timeframe="this_week",
            ),
        ],
        risk_radar=["PCE Friday could confirm inflation trend"],
        top_movers=["SPY", "TLT", "MSFT", "NVDA"],
    )


# ---------------------------------------------------------------------------
# Tests: _split_content
# ---------------------------------------------------------------------------


class TestSplitContent:
    def test_short_content_no_split(self) -> None:
        assert _split_content("hello", 100) == ["hello"]

    def test_long_content_splits(self) -> None:
        lines = [f"Line {i}" for i in range(100)]
        text = "\n".join(lines)
        chunks = _split_content(text, 200)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_empty_string(self) -> None:
        assert _split_content("", 100) == [""]


# ---------------------------------------------------------------------------
# Tests: _get_13f_deadline_reminder
# ---------------------------------------------------------------------------


class TestGet13FDeadlineReminder:
    def test_deadline_within_window(self) -> None:
        # Q4 deadline is Feb 14 — test with today = Feb 1
        result = _get_13f_deadline_reminder(date(2026, 2, 1), 14)
        assert result is not None
        assert "13F Filing Deadline" in result
        assert "Feb 14" in result
        assert "Q4" in result
        assert "Berkshire Hathaway" in result

    def test_no_deadline_in_window(self) -> None:
        # No deadline near Apr 15
        result = _get_13f_deadline_reminder(date(2026, 4, 15), 14)
        assert result is None

    def test_deadline_exactly_on_boundary(self) -> None:
        # Q1 deadline is May 15 — test with today = May 1
        result = _get_13f_deadline_reminder(date(2026, 5, 1), 14)
        assert result is not None
        assert "May 15" in result


# ---------------------------------------------------------------------------
# Tests: _format_whats_coming_embeds
# ---------------------------------------------------------------------------


class TestFormatWhatsComingEmbeds:
    def test_groups_by_date(self) -> None:
        rows = [
            _record(id=1, event_date=date(2026, 3, 19), title="FOMC"),
            _record(id=2, event_date=date(2026, 3, 20), title="CPI Release"),
        ]
        messages = _format_whats_coming_embeds(
            rows, set(), None, datetime.now(timezone.utc).isoformat(), date(2026, 3, 18)
        )

        assert len(messages) >= 1
        desc = messages[0][0]["description"]
        assert "Mar 19" in desc
        assert "Mar 20" in desc
        assert "FOMC" in desc
        assert "CPI Release" in desc

    def test_new_badge_applied(self) -> None:
        rows = [_record(id=42, title="New Event")]
        messages = _format_whats_coming_embeds(
            rows, {42}, None, datetime.now(timezone.utc).isoformat(), date(2026, 3, 18)
        )
        desc = messages[0][0]["description"]
        assert "\U0001f195" in desc  # NEW emoji

    def test_deadline_reminder_included(self) -> None:
        rows = [_record()]
        reminder = "\u23f0 **13F Filing Deadline: Feb 14 (Q4 2025)**"
        messages = _format_whats_coming_embeds(
            rows, set(), reminder, datetime.now(timezone.utc).isoformat(), date(2026, 3, 18)
        )
        desc = messages[0][0]["description"]
        assert "13F Filing Deadline" in desc

    def test_title_set_correctly(self) -> None:
        rows = [_record()]
        messages = _format_whats_coming_embeds(
            rows, set(), None, datetime.now(timezone.utc).isoformat(), date(2026, 3, 18)
        )
        assert "What's Coming" in messages[0][0]["title"]


# ---------------------------------------------------------------------------
# Tests: _format_yesterday_brief_rich
# ---------------------------------------------------------------------------


class TestFormatYesterdayBriefRich:
    def test_header_embed(self) -> None:
        analysis = _make_yesterday_analysis()
        messages = _format_yesterday_brief_rich(
            analysis, date(2026, 3, 5), datetime.now(timezone.utc).isoformat()
        )

        assert len(messages) >= 1
        header = messages[0][0]
        assert "Yesterday's Brief" in header["title"]
        assert analysis.headline in header["description"]
        # Snapshot data is in fields, not description
        field_names = [f["name"] for f in header["fields"]]
        assert "\U0001f4c8 Equities" in field_names
        assert "\U0001f4b5 Rates / FX" in field_names
        assert "\U0001f6e2\ufe0f Commodities" in field_names
        assert "\U0001f4ca Sectors" in field_names
        assert "Top Movers" in field_names

    def test_theme_embeds(self) -> None:
        analysis = _make_yesterday_analysis()
        messages = _format_yesterday_brief_rich(
            analysis, date(2026, 3, 5), datetime.now(timezone.utc).isoformat()
        )

        # 1 header + 2 theme embeds + 1 synthesis embed
        assert len(messages) == 4

    def test_surprise_indicator(self) -> None:
        analysis = _make_yesterday_analysis()
        messages = _format_yesterday_brief_rich(
            analysis, date(2026, 3, 5), datetime.now(timezone.utc).isoformat()
        )
        # Second theme (GPT-5.4) should be marked as surprise
        surprise_embed = messages[2][0]
        source_field = next(f for f in surprise_embed["fields"] if f["name"] == "Source")
        assert "Surprise" in source_field["value"]

    def test_analysis_source_label(self) -> None:
        analysis = _make_yesterday_analysis()
        # Change first theme to analysis source
        analysis.themes[0].source = "analysis"
        messages = _format_yesterday_brief_rich(
            analysis, date(2026, 3, 5), datetime.now(timezone.utc).isoformat()
        )
        analysis_embed = messages[1][0]
        source_field = next(f for f in analysis_embed["fields"] if f["name"] == "Source")
        assert "Analysis" in source_field["value"]

    def test_calendar_source_label(self) -> None:
        analysis = _make_yesterday_analysis()
        # First theme already has source="calendar"
        messages = _format_yesterday_brief_rich(
            analysis, date(2026, 3, 5), datetime.now(timezone.utc).isoformat()
        )
        calendar_embed = messages[1][0]
        source_field = next(f for f in calendar_embed["fields"] if f["name"] == "Source")
        assert "Calendar" in source_field["value"]

    def test_synthesis_embed_content(self) -> None:
        analysis = _make_yesterday_analysis()
        messages = _format_yesterday_brief_rich(
            analysis, date(2026, 3, 5), datetime.now(timezone.utc).isoformat()
        )
        # Synthesis embed is the last message
        synth_embed = messages[-1][0]
        desc = synth_embed["description"]
        assert "Summary" in desc
        assert analysis.synthesis in desc
        # Actionable with DIRECTION_ICON
        assert "LONG" in desc
        assert "TLT" in desc
        # Risk radar
        assert "Risk Radar" in desc
        assert "PCE Friday" in desc


# ---------------------------------------------------------------------------
# Tests: _format_yesterday_brief_fallback
# ---------------------------------------------------------------------------


class TestFormatYesterdayBriefFallback:
    def test_fallback_with_events(self) -> None:
        events = [{"title": "FOMC", "category": "fed"}]
        surprises = [{"title": "GPT-5.4 launched"}]
        messages = _format_yesterday_brief_fallback(
            events, surprises, date(2026, 3, 5), datetime.now(timezone.utc).isoformat()
        )
        assert len(messages) == 1
        desc = messages[0][0]["description"]
        assert "FOMC" in desc
        assert "GPT-5.4" in desc

    def test_fallback_empty(self) -> None:
        messages = _format_yesterday_brief_fallback(
            [], [], date(2026, 3, 5), datetime.now(timezone.utc).isoformat()
        )
        assert "No notable events" in messages[0][0]["description"]


# ---------------------------------------------------------------------------
# Tests: sub-analyzer formatters
# ---------------------------------------------------------------------------


class TestFormatEarningsEvents:
    def test_formats_earnings(self) -> None:
        events = [
            {
                "title": "AAPL Q1 Earnings",
                "description": "Apple reports Q1",
                "tickers": ["AAPL"],
                "outcome": "EPS $2.18 vs $2.10 expected",
            }
        ]
        result = _format_earnings_events(events)
        assert "AAPL Q1 Earnings" in result
        assert "$AAPL" in result
        assert "OUTCOME: EPS $2.18" in result

    def test_empty_list(self) -> None:
        result = _format_earnings_events([])
        assert result == ""


class TestFormatCalendarEvents:
    def test_formats_events(self) -> None:
        events = [{"title": "CPI Release", "category": "economic_data", "tickers": ["SPY"]}]
        result = _format_calendar_events(events)
        assert "CALENDAR EVENTS" in result
        assert "CPI Release" in result
        assert "$SPY" in result

    def test_includes_outcome(self) -> None:
        events = [
            {
                "title": "FOMC Decision",
                "category": "fed",
                "tickers": [],
                "outcome": "Rate held steady",
            }
        ]
        result = _format_calendar_events(events)
        assert "OUTCOME: Rate held steady" in result


class TestFormatSurpriseEvents:
    def test_formats_surprises(self) -> None:
        surprises = [{"title": "Big news", "snippet": "Details here", "url": "https://example.com"}]
        result = _format_surprise_events(surprises)
        assert "SURPRISE EVENTS" in result
        assert "Big news" in result
        assert "Details here" in result


class TestFormatFilingBriefs:
    def test_formats_filings(self) -> None:
        filings = [
            {
                "fund_name": "Berkshire",
                "new_positions": [{"name_of_issuer": "AAPL INC"}],
                "exited_positions": [{"name_of_issuer": "TSLA INC"}],
                "increased": [{"name_of_issuer": "MSFT CORP", "change_pct": 25.3}],
                "decreased": [{"name_of_issuer": "GOOG LLC", "change_pct": -15.7}],
                "total_value_current": 312000,
            }
        ]
        result = _format_filing_briefs(filings)
        assert "13F FILINGS" in result
        assert "Berkshire" in result
        assert "AAPL INC" in result
        assert "TSLA INC" in result
        assert "MSFT CORP" in result
        assert "+25%" in result
        assert "GOOG LLC" in result
        assert "-16%" in result


# ---------------------------------------------------------------------------
# Tests: synthesize_yesterday_brief
# ---------------------------------------------------------------------------


def _make_sub_analysis(**overrides: Any) -> SubAnalysis:
    defaults: dict[str, Any] = {
        "themes": [
            YesterdayTheme(
                title="Test Theme",
                category="macro",
                sentiment="bullish",
                source="calendar",
                outcome="Test outcome",
                analysis="Test analysis",
                key_events=["Event 1"],
                tickers=["SPY"],
                market_reaction="Markets rallied",
            ),
        ],
        "key_takeaways": ["Key point 1", "Key point 2"],
        "tickers_affected": ["SPY"],
    }
    defaults.update(overrides)
    return SubAnalysis(**defaults)


class TestSynthesizeYesterdayBrief:
    @pytest.mark.asyncio
    async def test_returns_none_on_empty_input(self) -> None:
        result = await synthesize_yesterday_brief([], [], [])
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_all_sub_analyzers_fail(self) -> None:
        with (
            patch(
                "synesis.processing.events.yesterday.analyze_events",
                AsyncMock(return_value=None),
            ),
        ):
            result = await synthesize_yesterday_brief(
                [{"title": "Test", "category": "other", "tickers": []}], [], []
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_analysis_on_success(self) -> None:
        analysis = _make_yesterday_analysis()
        sub = _make_sub_analysis()

        with (
            patch(
                "synesis.processing.events.yesterday.analyze_events",
                AsyncMock(return_value=sub),
            ),
            patch(
                "synesis.processing.events.yesterday.consolidate",
                AsyncMock(return_value=analysis),
            ),
        ):
            result = await synthesize_yesterday_brief(
                [{"title": "FOMC", "category": "fed", "tickers": []}], [], []
            )

        assert result is not None
        assert result.headline == analysis.headline
        assert len(result.themes) == 2

    @pytest.mark.asyncio
    async def test_no_earnings_skips_earnings_analyzer(self) -> None:
        """When no earnings events, analyze_earnings should not be called."""
        analysis = _make_yesterday_analysis()
        sub = _make_sub_analysis()

        with (
            patch(
                "synesis.processing.events.yesterday.analyze_earnings",
                AsyncMock(return_value=sub),
            ) as mock_earnings,
            patch(
                "synesis.processing.events.yesterday.analyze_events",
                AsyncMock(return_value=sub),
            ),
            patch(
                "synesis.processing.events.yesterday.consolidate",
                AsyncMock(return_value=analysis),
            ),
        ):
            await synthesize_yesterday_brief(
                [{"title": "FOMC", "category": "fed", "tickers": []}], [], []
            )

        mock_earnings.assert_not_called()

    @pytest.mark.asyncio
    async def test_earnings_events_routed_to_earnings_analyzer(self) -> None:
        """Earnings events should be routed to analyze_earnings."""
        analysis = _make_yesterday_analysis()
        sub = _make_sub_analysis()

        with (
            patch(
                "synesis.processing.events.yesterday.analyze_earnings",
                AsyncMock(return_value=sub),
            ) as mock_earnings,
            patch(
                "synesis.processing.events.yesterday.analyze_events",
                AsyncMock(return_value=None),
            ),
            patch(
                "synesis.processing.events.yesterday.consolidate",
                AsyncMock(return_value=analysis),
            ),
        ):
            await synthesize_yesterday_brief(
                [{"title": "AAPL Earnings", "category": "earnings", "tickers": ["AAPL"]}],
                [],
                [],
            )

        mock_earnings.assert_called_once()

    @pytest.mark.asyncio
    async def test_filings_routed_to_filings_analyzer(self) -> None:
        """13F filing briefs should be passed to analyze_filings."""
        analysis = _make_yesterday_analysis()
        sub = _make_sub_analysis()

        with (
            patch(
                "synesis.processing.events.yesterday.analyze_filings",
                AsyncMock(return_value=sub),
            ) as mock_filings,
            patch(
                "synesis.processing.events.yesterday.analyze_events",
                AsyncMock(return_value=None),
            ),
            patch(
                "synesis.processing.events.yesterday.consolidate",
                AsyncMock(return_value=analysis),
            ),
        ):
            await synthesize_yesterday_brief(
                [],
                [],
                [{"fund_name": "Berkshire"}],
            )

        mock_filings.assert_called_once()
        call_args = mock_filings.call_args[0]
        assert call_args[0] == [{"fund_name": "Berkshire"}]


# ---------------------------------------------------------------------------
# Tests: detect_surprise_events
# ---------------------------------------------------------------------------


class TestDetectSurpriseEvents:
    @pytest.mark.asyncio
    async def test_returns_cached(self) -> None:
        redis = AsyncMock()
        cached = [{"title": "Cached surprise", "snippet": "test"}]
        redis.get = AsyncMock(return_value=orjson.dumps(cached))

        result = await detect_surprise_events(redis)

        assert len(result) == 1
        assert result[0]["title"] == "Cached surprise"

    @pytest.mark.asyncio
    async def test_searches_and_caches(self) -> None:
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        search_results = [
            {"title": "Breaking: Fed cuts rates", "snippet": "...", "url": "https://example.com"},
            {"title": "AI company raises $1B", "snippet": "...", "url": "https://example2.com"},
        ]

        with patch(
            "synesis.processing.events.yesterday.surprise.search_market_impact",
            AsyncMock(return_value=search_results),
        ):
            result = await detect_surprise_events(redis)

        assert len(result) >= 2
        redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplicates_titles(self) -> None:
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        # Same title appearing in two different search queries
        results = [
            {"title": "Fed cuts rates by 50bps", "snippet": "...", "url": ""},
            {"title": "Fed cuts rates by 50bps", "snippet": "different", "url": ""},
        ]

        with patch(
            "synesis.processing.events.yesterday.surprise.search_market_impact",
            AsyncMock(return_value=results),
        ):
            result = await detect_surprise_events(redis)

        titles = [r["title"] for r in result]
        assert titles.count("Fed cuts rates by 50bps") == 1

    @pytest.mark.asyncio
    async def test_handles_search_failure(self) -> None:
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        from synesis.processing.common.web_search import SearchProvidersExhaustedError

        with patch(
            "synesis.processing.events.yesterday.surprise.search_market_impact",
            AsyncMock(side_effect=SearchProvidersExhaustedError("No providers")),
        ):
            result = await detect_surprise_events(redis)

        assert result == []

    @pytest.mark.asyncio
    async def test_limits_to_max_results(self) -> None:
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        results = [{"title": f"Event {i}", "snippet": "", "url": ""} for i in range(20)]

        with patch(
            "synesis.processing.events.yesterday.surprise.search_market_impact",
            AsyncMock(return_value=results),
        ):
            result = await detect_surprise_events(redis)

        assert len(result) <= 10


# ---------------------------------------------------------------------------
# Tests: _get_yesterday_13f_briefs (category filter)
# ---------------------------------------------------------------------------


class TestGetYesterday13fBriefs:
    @pytest.mark.asyncio
    async def test_matches_13f_filing_category(self) -> None:
        """13F events stored as category=13f_filing."""
        rows = [
            _record(category="13f_filing", title="13F Filing: Berkshire Hathaway"),
            _record(category="regulatory", title="SEC Form D: Some Company"),
            _record(category="earnings", title="13F Filing: Should Not Match"),
        ]
        sec_edgar = AsyncMock()
        sec_edgar.compare_13f_quarters = AsyncMock(return_value=None)

        with patch("synesis.processing.events.digest.load_hedge_fund_registry") as mock_reg:
            mock_reg.return_value = ({"0001067983": "Berkshire Hathaway"}, {})
            await _get_yesterday_13f_briefs(sec_edgar, rows)

        # Only the first row matches (category=13f_filing)
        sec_edgar.compare_13f_quarters.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_non_13f_regulatory(self) -> None:
        """Non-13F regulatory events should be skipped."""
        rows = [
            _record(category="regulatory", title="SEC Enforcement Action"),
        ]
        sec_edgar = AsyncMock()

        with patch("synesis.processing.events.digest.load_hedge_fund_registry") as mock_reg:
            mock_reg.return_value = ({"0001067983": "Berkshire Hathaway"}, {})
            briefs = await _get_yesterday_13f_briefs(sec_edgar, rows)

        assert briefs == []
        sec_edgar.compare_13f_quarters.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: send_event_digest (orchestrator)
# ---------------------------------------------------------------------------


class TestSendEventDigest:
    @pytest.mark.asyncio
    async def test_returns_false_no_webhook(self) -> None:
        db = AsyncMock()
        with patch("synesis.processing.events.digest.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                discord_events_webhook_url=None,
                discord_webhook_url=None,
            )
            result = await send_event_digest(db)

        assert result is False

    @pytest.mark.asyncio
    async def test_sends_both_messages(self) -> None:
        db = AsyncMock()
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()

        with (
            patch("synesis.processing.events.digest.get_settings") as mock_settings,
            patch(
                "synesis.processing.events.digest._send_whats_coming", AsyncMock(return_value=True)
            ) as mock_coming,
            patch(
                "synesis.processing.events.digest._send_yesterday_brief",
                AsyncMock(return_value=True),
            ) as mock_brief,
        ):
            mock_settings.return_value = MagicMock(
                discord_events_webhook_url=MagicMock(get_secret_value=lambda: "https://webhook"),
                discord_webhook_url=None,
            )
            result = await send_event_digest(db, redis=redis)

        assert result is True
        mock_coming.assert_called_once()
        mock_brief.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_true_if_either_sent(self) -> None:
        db = AsyncMock()

        with (
            patch("synesis.processing.events.digest.get_settings") as mock_settings,
            patch(
                "synesis.processing.events.digest._send_whats_coming", AsyncMock(return_value=False)
            ),
            patch(
                "synesis.processing.events.digest._send_yesterday_brief",
                AsyncMock(return_value=True),
            ),
        ):
            mock_settings.return_value = MagicMock(
                discord_events_webhook_url=MagicMock(get_secret_value=lambda: "https://webhook"),
                discord_webhook_url=None,
            )
            result = await send_event_digest(db)

        assert result is True


# ---------------------------------------------------------------------------
# Tests: _enrich_with_outcomes
# ---------------------------------------------------------------------------


class TestEnrichWithOutcomes:
    @pytest.mark.asyncio
    async def test_earnings_uses_sec_press_release(self) -> None:
        """Earnings events should be enriched with SEC 8-K content, not web search."""
        events = [
            {"title": "Earnings: NVIDIA (NVDA)", "category": "earnings", "tickers": ["NVDA"]},
        ]
        sec_edgar = AsyncMock()
        sec_edgar.get_earnings_releases = AsyncMock(
            return_value=[
                MagicMock(
                    content="Revenue $35.1B vs $33.2B expected. EPS $0.89 vs $0.84.",
                    filed_date=date(2026, 3, 5),
                )
            ]
        )

        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, sec_edgar)

        mock_search.assert_not_called()
        assert "Revenue $35.1B" in result[0].get("outcome", "")

    @pytest.mark.asyncio
    async def test_earnings_no_ticker_falls_through(self) -> None:
        """Earnings without tickers skip SEC enrichment gracefully."""
        events = [
            {"title": "Earnings: Unknown Co", "category": "earnings", "tickers": []},
        ]
        result = await _enrich_with_outcomes(events, None, None)
        assert result[0].get("outcome", "") == ""

    @pytest.mark.asyncio
    async def test_13f_skipped(self) -> None:
        """13F events should not be enriched at all."""
        events = [
            {"title": "13F Filing: Berkshire", "category": "13f_filing", "tickers": []},
        ]
        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[]),
        ) as mock_search:
            await _enrich_with_outcomes(events, None, None)

        mock_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_category_web_searches(self) -> None:
        """Categories without specialized sources should use web search."""
        events = [
            {"title": "Trade Agreement Signed", "category": "geopolitical", "tickers": []},
        ]
        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[{"snippet": "Trade deal signed"}]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, None)

        mock_search.assert_called_once()
        assert "Trade deal signed" in result[0].get("outcome", "")

    @pytest.mark.asyncio
    async def test_economic_data_uses_fred(self) -> None:
        """economic_data events should use FRED API."""
        events = [
            {"title": "CPI Release", "category": "economic_data", "tickers": []},
        ]
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(
                observations=[
                    MagicMock(value=3.2),
                    MagicMock(value=3.1),
                ]
            )
        )

        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, None, fred=fred)

        mock_search.assert_not_called()
        assert "CPI" in result[0].get("outcome", "")
        assert "3.2" in result[0].get("outcome", "")

    @pytest.mark.asyncio
    async def test_fed_uses_crawler(self) -> None:
        """fed events should use Crawl4AI."""
        events = [
            {
                "title": "FOMC Decision",
                "category": "fed",
                "tickers": [],
                "source_urls": ["https://fed.gov/statement"],
                "event_date": date(2026, 3, 19),
            },
        ]
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(
            return_value=MagicMock(success=True, markdown="Fed held rates at 5.25-5.50%")
        )

        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, None, crawler=crawler)

        mock_search.assert_not_called()
        assert "Fed held rates" in result[0].get("outcome", "")

    @pytest.mark.asyncio
    async def test_other_category_uses_web_search_for_outcome(self) -> None:
        """Categories without specialized sources should use web search."""
        events = [
            {
                "title": "EIA Petroleum Status Report",
                "category": "other",
                "tickers": [],
            },
        ]
        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[{"snippet": "Crude inventories fell 2.1M barrels"}]),
        ):
            result = await _enrich_with_outcomes(events, None, None)

        assert "Crude inventories" in result[0].get("outcome", "")


# ---------------------------------------------------------------------------
# Tests: _get_economic_data_outcome
# ---------------------------------------------------------------------------


class TestGetEconomicDataOutcome:
    @pytest.mark.asyncio
    async def test_matches_cpi_and_returns_formatted(self) -> None:
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=3.2), MagicMock(value=3.1)])
        )

        result = await _get_economic_data_outcome(
            {"title": "CPI Release", "category": "economic_data"}, fred
        )

        assert "CPI" in result
        assert "3.2%" in result
        assert "prev 3.1%" in result
        fred.get_observations.assert_called_once_with(
            "CPIAUCSL", sort_order="desc", limit=2, units="pc1"
        )

    @pytest.mark.asyncio
    async def test_prefers_longer_key_match(self) -> None:
        """'Core CPI' should match before 'CPI'."""
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=2.8), MagicMock(value=2.7)])
        )

        result = await _get_economic_data_outcome(
            {"title": "Core CPI Release", "category": "economic_data"}, fred
        )

        assert "Core CPI" in result
        fred.get_observations.assert_called_once_with(
            "CPILFESL", sort_order="desc", limit=2, units="pc1"
        )

    @pytest.mark.asyncio
    async def test_unemployment_no_percent_suffix(self) -> None:
        """Unemployment uses 'lin' units — value is already a percentage level."""
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=4.1), MagicMock(value=4.0)])
        )

        result = await _get_economic_data_outcome(
            {"title": "Unemployment Rate", "category": "economic_data"}, fred
        )

        assert "Unemployment" in result
        # 'lin' should not append % after the value
        assert "4.1" in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_fred(self) -> None:
        result = await _get_economic_data_outcome(
            {"title": "CPI Release", "category": "economic_data"}, None
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_match(self) -> None:
        fred = AsyncMock()
        result = await _get_economic_data_outcome(
            {"title": "Unknown Indicator XYZ", "category": "economic_data"}, fred
        )
        assert result == ""
        fred.get_observations.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_observation(self) -> None:
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=3.5)])
        )

        result = await _get_economic_data_outcome(
            {"title": "GDP Report", "category": "economic_data"}, fred
        )

        assert "GDP" in result
        assert "3.5%" in result
        assert "prev" not in result


# ---------------------------------------------------------------------------
# Tests: _get_crawled_outcome
# ---------------------------------------------------------------------------


class TestGetCrawledOutcome:
    @pytest.mark.asyncio
    async def test_fed_constructs_date_specific_url(self) -> None:
        """fed events should construct a date-specific Fed statement URL."""
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(
            return_value=MagicMock(success=True, markdown="Fed statement content here")
        )

        result = await _get_crawled_outcome(
            {
                "source_urls": ["https://fed.gov/calendar"],
                "category": "fed",
                "event_date": date(2026, 3, 19),
            },
            crawler,
        )

        assert "Fed statement content here" in result
        crawler.crawl.assert_called_once_with(
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260319a.htm"
        )

    @pytest.mark.asyncio
    async def test_fed_fallback_no_date(self) -> None:
        """fed with no event_date and no source_urls falls back to press releases."""
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(
            return_value=MagicMock(success=True, markdown="Latest Fed press releases")
        )

        result = await _get_crawled_outcome({"source_urls": [], "category": "fed"}, crawler)

        assert "Latest Fed press releases" in result
        crawler.crawl.assert_called_once_with(
            "https://www.federalreserve.gov/newsevents/pressreleases.htm"
        )

    @pytest.mark.asyncio
    async def test_returns_empty_without_crawler(self) -> None:
        result = await _get_crawled_outcome(
            {"source_urls": ["https://example.com"], "category": "other"}, None
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_on_no_source_urls_non_fed(self) -> None:
        crawler = AsyncMock()
        result = await _get_crawled_outcome({"source_urls": [], "category": "other"}, crawler)
        assert result == ""
        crawler.crawl.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncates_to_500_chars(self) -> None:
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(return_value=MagicMock(success=True, markdown="x" * 1000))

        result = await _get_crawled_outcome(
            {"source_urls": ["https://example.com"], "category": "fed"}, crawler
        )

        assert len(result) == 500

    @pytest.mark.asyncio
    async def test_handles_crawl_failure(self) -> None:
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(return_value=MagicMock(success=False, markdown=""))

        result = await _get_crawled_outcome(
            {"source_urls": ["https://example.com"], "category": "fed"}, crawler
        )

        assert result == ""


# ---------------------------------------------------------------------------
# Tests: FOMC E2E (discovery → enrichment → analysis routing)
# ---------------------------------------------------------------------------


class TestFomcEndToEnd:
    """Verify FOMC rate decisions AND minutes flow through the full pipeline."""

    @pytest.mark.asyncio
    async def test_fomc_decision_outcome_constructs_statement_url(self) -> None:
        """FOMC rate decision → crawl date-specific Fed statement URL."""
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(
            return_value=MagicMock(
                success=True,
                markdown="The Federal Open Market Committee decided to maintain the target range "
                "for the federal funds rate at 5-1/4 to 5-1/2 percent.",
            )
        )

        result = await _get_crawled_outcome(
            {
                "title": "FOMC Rate Decision",
                "category": "fed",
                "event_date": date(2026, 3, 18),
                "source_urls": ["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"],
            },
            crawler,
        )

        # Should construct date-specific monetary statement URL
        crawler.crawl.assert_called_once_with(
            "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260318a.htm"
        )
        assert "federal funds rate" in result

    @pytest.mark.asyncio
    async def test_fomc_minutes_uses_crawler_with_db_lookup(self) -> None:
        """FOMC minutes release → DB lookup for meeting date → crawl minutes URL."""
        events = [
            {
                "title": "FOMC Minutes Released",
                "category": "fed",
                "tickers": [],
                "event_date": date(2026, 2, 18),
                "source_urls": [],
            },
        ]
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(
            return_value=MagicMock(
                success=True,
                markdown="Minutes showed officials discussed cutting rates at upcoming meetings",
            )
        )

        db = AsyncMock()
        db.get_last_fomc_meeting_date = AsyncMock(return_value=date(2026, 1, 28))

        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, None, crawler=crawler, db=db)

        # Crawler used via DB lookup, NOT web search
        mock_search.assert_not_called()
        db.get_last_fomc_meeting_date.assert_called_once_with(date(2026, 2, 18))
        crawler.crawl.assert_called_once_with(
            "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260128.htm"
        )
        assert "cutting rates" in result[0].get("outcome", "")

    @pytest.mark.asyncio
    async def test_fomc_minutes_outcome_no_db_returns_empty(self) -> None:
        """FOMC minutes without DB returns empty string."""
        ev = {
            "title": "FOMC Minutes Released",
            "category": "fed",
            "event_date": date(2026, 2, 18),
        }
        crawler = AsyncMock()
        result = await _get_fomc_minutes_outcome(ev, None, crawler)
        assert result == ""

    @pytest.mark.asyncio
    async def test_fomc_minutes_outcome_no_meeting_found(self) -> None:
        """FOMC minutes with no matching meeting date returns empty string."""
        ev = {
            "title": "FOMC Minutes Released",
            "category": "fed",
            "event_date": date(2026, 2, 18),
        }
        db = AsyncMock()
        db.get_last_fomc_meeting_date = AsyncMock(return_value=None)
        crawler = AsyncMock()
        result = await _get_fomc_minutes_outcome(ev, db, crawler)
        assert result == ""
        crawler.crawl.assert_not_called()

    @pytest.mark.asyncio
    async def test_fomc_events_routed_to_macro_analyzer(self) -> None:
        """Both FOMC decisions and minutes should route to analyze_macro()."""
        from synesis.processing.events.yesterday.macro import MACRO_CATEGORIES

        assert "fed" in MACRO_CATEGORIES

        # Simulate the routing logic from synthesize_yesterday_brief
        events = [
            {"title": "FOMC Rate Decision", "category": "fed", "tickers": []},
            {"title": "FOMC Minutes Released", "category": "fed", "tickers": []},
            {"title": "CPI Release", "category": "economic_data", "tickers": []},
            {"title": "NVIDIA Earnings", "category": "earnings", "tickers": ["NVDA"]},
        ]

        macro_events = [e for e in events if e.get("category") in MACRO_CATEGORIES]
        earnings_events = [e for e in events if e.get("category") == "earnings"]

        assert len(macro_events) == 3  # both FOMC + CPI
        assert len(earnings_events) == 1
        assert all(e["category"] in ("fed", "economic_data") for e in macro_events)

    @pytest.mark.asyncio
    async def test_fomc_enrichment_dispatches_to_crawler(self) -> None:
        """fed events should use _get_crawled_outcome, not web search."""
        events = [
            {
                "title": "FOMC Rate Decision",
                "category": "fed",
                "tickers": [],
                "source_urls": [],
                "event_date": date(2026, 3, 18),
            },
        ]
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(
            return_value=MagicMock(success=True, markdown="Fed held rates steady")
        )

        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, None, crawler=crawler)

        # Web search should NOT be called for fed
        mock_search.assert_not_called()
        assert "Fed held rates" in result[0].get("outcome", "")


# ---------------------------------------------------------------------------
# Tests: FRED E2E (discovery → enrichment → analysis routing)
# ---------------------------------------------------------------------------


class TestFredEndToEnd:
    """Verify FRED economic data events flow through the full pipeline."""

    @pytest.mark.asyncio
    async def test_cpi_outcome_matches_fred_series(self) -> None:
        """'CPI Release' title → matches FRED series CPIAUCSL with YoY% units."""
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=3.2), MagicMock(value=3.1)])
        )

        result = await _get_economic_data_outcome(
            {"title": "CPI Release", "category": "economic_data"}, fred
        )

        assert "CPI: 3.2%" in result
        assert "prev 3.1%" in result
        fred.get_observations.assert_called_once_with(
            "CPIAUCSL", sort_order="desc", limit=2, units="pc1"
        )

    @pytest.mark.asyncio
    async def test_core_cpi_matches_before_cpi(self) -> None:
        """'Core CPI' should match CPILFESL, not CPIAUCSL (longest match first)."""
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=2.8), MagicMock(value=2.7)])
        )

        result = await _get_economic_data_outcome(
            {"title": "Core CPI Release", "category": "economic_data"}, fred
        )

        assert "Core CPI" in result
        fred.get_observations.assert_called_once_with(
            "CPILFESL", sort_order="desc", limit=2, units="pc1"
        )

    @pytest.mark.asyncio
    async def test_nfp_matches_employment_situation_title(self) -> None:
        """'Employment Situation (NFP) Release' → matches 'NFP' keyword → PAYEMS."""
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=256.0), MagicMock(value=187.0)])
        )

        result = await _get_economic_data_outcome(
            {"title": "Employment Situation (NFP) Release", "category": "economic_data"},
            fred,
        )

        # "Non-Farm Payroll" (16 chars) is longer than "NFP" (3 chars)
        # but "Non-Farm Payroll" is NOT a substring of "Employment Situation (NFP) Release"
        # so it falls through to "NFP" which IS a substring
        assert "NFP" in result
        fred.get_observations.assert_called_once_with(
            "PAYEMS", sort_order="desc", limit=2, units="chg"
        )

    @pytest.mark.asyncio
    async def test_pce_title_matches_personal_income_first(self) -> None:
        """'PCE / Personal Income Release' → 'Personal Income' matched first (longer)."""
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=2.5), MagicMock(value=2.4)])
        )

        result = await _get_economic_data_outcome(
            {"title": "PCE / Personal Income Release", "category": "economic_data"},
            fred,
        )

        # "Personal Income" (15 chars) is longer than "PCE" (3 chars) and "Core PCE" (8 chars)
        # "Personal Income" IS a substring of "PCE / Personal Income Release"
        assert "Personal Income" in result
        fred.get_observations.assert_called_once_with("PI", sort_order="desc", limit=2, units="pch")

    @pytest.mark.asyncio
    async def test_economic_data_enrichment_uses_fred_not_web_search(self) -> None:
        """economic_data events should use FRED API, not web search."""
        events = [
            {"title": "CPI Release", "category": "economic_data", "tickers": []},
        ]
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=3.2), MagicMock(value=3.1)])
        )

        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, None, fred=fred)

        mock_search.assert_not_called()
        assert "CPI" in result[0].get("outcome", "")

    @pytest.mark.asyncio
    async def test_economic_data_routed_to_macro_analyzer(self) -> None:
        """economic_data events should be routed to analyze_macro(), not analyze_events()."""
        from synesis.processing.events.yesterday.macro import MACRO_CATEGORIES

        assert "economic_data" in MACRO_CATEGORIES

        events = [
            {"title": "CPI Release", "category": "economic_data", "tickers": []},
            {"title": "GDP Release", "category": "economic_data", "tickers": []},
            {"title": "PPI Release", "category": "economic_data", "tickers": []},
        ]

        macro_events = [e for e in events if e.get("category") in MACRO_CATEGORIES]
        other_events = [
            e
            for e in events
            if e.get("category") != "earnings" and e.get("category") not in MACRO_CATEGORIES
        ]

        assert len(macro_events) == 3
        assert len(other_events) == 0

    @pytest.mark.asyncio
    async def test_pce_outcome_does_not_match_core_pce(self) -> None:
        """Verify 'PCE / Personal Income Release' doesn't match 'Core PCE'."""
        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=2.5), MagicMock(value=2.4)])
        )

        # "Core PCE" (8 chars) is checked before "PCE" (3 chars) by length
        # but "Core PCE" is NOT a substring of "PCE / Personal Income Release"
        # so it correctly falls through
        result = await _get_economic_data_outcome(
            {"title": "PCE / Personal Income Release", "category": "economic_data"},
            fred,
        )

        assert "Core PCE" not in result


# ---------------------------------------------------------------------------
# Tests: Full category matrix — every category through enrichment + routing
# ---------------------------------------------------------------------------


class TestAllCategoriesEnrichment:
    """Run every category through _enrich_with_outcomes and verify the correct
    enrichment path is taken for each one."""

    @pytest.mark.asyncio
    async def test_all_categories_take_correct_enrichment_path(self) -> None:
        """Create one event per category, run all through enrichment, verify paths."""
        events = [
            {
                "title": "Earnings: NVIDIA (NVDA)",
                "category": "earnings",
                "tickers": ["NVDA"],
                "source_urls": [],
            },
            {
                "title": "CPI Release",
                "category": "economic_data",
                "tickers": [],
                "source_urls": [],
            },
            {
                "title": "FOMC Rate Decision",
                "category": "fed",
                "tickers": [],
                "event_date": date(2026, 3, 18),
                "source_urls": [],
            },
            {
                "title": "FOMC Minutes Released",
                "category": "fed",
                "tickers": [],
                "event_date": date(2026, 4, 9),
                "source_urls": [],
            },
            {
                "title": "13F Filing: Berkshire Hathaway (Q4 2025)",
                "category": "13f_filing",
                "tickers": [],
                "source_urls": [],
            },
            {
                "title": "NVIDIA GTC Keynote",
                "category": "conference",
                "tickers": ["NVDA"],
                "source_urls": [],
            },
            {
                "title": "GPT-5.5 Released",
                "category": "release",
                "tickers": ["MSFT"],
                "source_urls": [],
            },
            {
                "title": "EU AI Act Enforcement Begins",
                "category": "regulatory",
                "tickers": [],
                "source_urls": [],
            },
            {
                "title": "Unexpected Trade Agreement",
                "category": "other",
                "tickers": [],
                "source_urls": [],
            },
        ]

        # Mock all enrichment sources
        sec_edgar = AsyncMock()
        sec_edgar.get_earnings_releases = AsyncMock(
            return_value=[MagicMock(content="NVDA Revenue $44.1B, EPS $0.96 vs $0.89")]
        )

        fred = AsyncMock()
        fred.get_observations = AsyncMock(
            return_value=MagicMock(observations=[MagicMock(value=3.2), MagicMock(value=3.1)])
        )

        crawler = AsyncMock()
        crawler.crawl = AsyncMock(
            return_value=MagicMock(
                success=True,
                markdown="The Committee decided to maintain the target range at 5.25-5.50%",
            )
        )

        db = AsyncMock()
        db.get_last_fomc_meeting_date = AsyncMock(return_value=date(2026, 3, 18))

        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[{"snippet": "Web search result for event"}]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, sec_edgar, crawler, fred, db)

        # 1. earnings → SEC EDGAR 8-K press release
        assert "NVDA Revenue $44.1B" in result[0].get("outcome", "")
        sec_edgar.get_earnings_releases.assert_called_once_with("NVDA", limit=1)

        # 2. economic_data → FRED observations
        assert "CPI: 3.2%" in result[1].get("outcome", "")
        fred.get_observations.assert_called_once_with(
            "CPIAUCSL", sort_order="desc", limit=2, units="pc1"
        )

        # 3. fed (rate decision) → Crawl4AI Fed statement
        assert "maintain the target range" in result[2].get("outcome", "")

        # 4. fed (minutes) → DB lookup + Crawl4AI minutes URL
        assert "maintain the target range" in result[3].get("outcome", "")
        db.get_last_fomc_meeting_date.assert_called_once_with(date(2026, 4, 9))

        # 5. 13f_filing → skipped (no outcome set)
        assert result[4].get("outcome") is None

        # 6. conference → web search
        assert "Web search result" in result[5].get("outcome", "")

        # 7. release → web search
        assert "Web search result" in result[6].get("outcome", "")

        # 8. regulatory → web search
        assert "Web search result" in result[7].get("outcome", "")

        # 9. other → web search
        assert "Web search result" in result[8].get("outcome", "")

        # Web search called for: conference + release + regulatory + other = 4 (not minutes)
        assert mock_search.call_count == 4

        # Crawler called twice: rate decision + minutes
        assert crawler.crawl.call_count == 2


class TestAllCategoriesRouting:
    """Verify every category routes to the correct sub-analyzer."""

    @pytest.mark.asyncio
    async def test_all_categories_route_to_correct_analyzer(self) -> None:
        """Simulate synthesize_yesterday_brief routing with all categories."""
        from synesis.processing.events.yesterday.macro import MACRO_CATEGORIES

        events = [
            {"title": "NVDA Earnings", "category": "earnings", "tickers": ["NVDA"]},
            {"title": "CPI Release", "category": "economic_data", "tickers": []},
            {"title": "GDP Release", "category": "economic_data", "tickers": []},
            {"title": "FOMC Rate Decision", "category": "fed", "tickers": []},
            {"title": "FOMC Minutes Released", "category": "fed", "tickers": []},
            {"title": "NVIDIA GTC Keynote", "category": "conference", "tickers": ["NVDA"]},
            {"title": "GPT-5.5 Released", "category": "release", "tickers": ["MSFT"]},
            {"title": "EU AI Act", "category": "regulatory", "tickers": []},
            {"title": "Trade Deal", "category": "other", "tickers": []},
        ]

        # Replicate exact routing logic from synthesize_yesterday_brief
        earnings_events = [e for e in events if e.get("category") == "earnings"]
        macro_events = [e for e in events if e.get("category") in MACRO_CATEGORIES]
        other_events = [
            e
            for e in events
            if e.get("category") not in {"earnings", "13f_filing"}
            and e.get("category") not in MACRO_CATEGORIES
        ]

        # earnings → analyze_earnings()
        assert len(earnings_events) == 1
        assert earnings_events[0]["title"] == "NVDA Earnings"

        # economic_data + fed → analyze_macro()
        assert len(macro_events) == 4
        macro_titles = {e["title"] for e in macro_events}
        assert macro_titles == {
            "CPI Release",
            "GDP Release",
            "FOMC Rate Decision",
            "FOMC Minutes Released",
        }

        # conference + release + regulatory + other → analyze_events()
        assert len(other_events) == 4
        other_titles = {e["title"] for e in other_events}
        assert other_titles == {
            "NVIDIA GTC Keynote",
            "GPT-5.5 Released",
            "EU AI Act",
            "Trade Deal",
        }

        # No event should be in multiple groups (13f_filing excluded from all three)
        all_routed = earnings_events + macro_events + other_events
        assert len(all_routed) == len(events)

    @pytest.mark.asyncio
    async def test_13f_filings_route_via_filing_briefs(self) -> None:
        """13F events are excluded from all sub-analyzers — they flow via filing_briefs."""
        from synesis.processing.events.yesterday.macro import MACRO_CATEGORIES

        # 13f_filing events are in calendar_events but their analysis data
        # comes from _get_yesterday_13f_briefs → filing_briefs → analyze_filings()
        events = [
            {"title": "13F Filing: Berkshire", "category": "13f_filing", "tickers": []},
            {"title": "CPI Release", "category": "economic_data", "tickers": []},
        ]

        earnings_events = [e for e in events if e.get("category") == "earnings"]
        macro_events = [e for e in events if e.get("category") in MACRO_CATEGORIES]
        other_events = [
            e
            for e in events
            if e.get("category") not in {"earnings", "13f_filing"}
            and e.get("category") not in MACRO_CATEGORIES
        ]

        # 13f_filing is excluded from all three groups
        assert "13f_filing" not in MACRO_CATEGORIES
        assert len(earnings_events) == 0
        assert len(macro_events) == 1  # only CPI
        assert len(other_events) == 0  # 13F excluded, CPI in macro
