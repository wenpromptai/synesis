"""Integration tests for Event Radar pipeline.

Uses REAL APIs (FRED, SEC EDGAR, NASDAQ, LLM) to verify end-to-end flows.
Each category is tested: earnings, economic_data, fed (decision + minutes),
13f_filing, and release (AI model extraction from crawled content).

Run with: pytest -m integration tests/integration/test_event_radar_e2e.py -v
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from synesis.processing.events.crawler import (
    FRED_MACRO_RELEASES,
    fetch_fred_macro_events,
    fetch_nasdaq_earnings_events,
)
from synesis.processing.events.digest import (
    _enrich_with_outcomes,
    _get_crawled_outcome,
    _get_economic_data_outcome,
    _get_earnings_outcome,
)
from synesis.processing.events.extractor import extract_events_from_markdown
from synesis.processing.events.models import CalendarEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_calendar_event(
    event: CalendarEvent,
    *,
    category: str,
    min_importance: int = 1,
    has_title: bool = True,
) -> None:
    """Assert basic CalendarEvent validity."""
    assert event.category == category, f"Expected {category}, got {event.category}"
    assert event.importance >= min_importance
    assert event.event_date >= date.today() - timedelta(days=1)
    if has_title:
        assert len(event.title) > 0
    assert event.confidence > 0.0
    assert len(event.region) > 0


# ---------------------------------------------------------------------------
# 1. EARNINGS — NASDAQ discovery + SEC EDGAR outcome
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEarningsE2E:
    """Earnings: NASDAQ API → CalendarEvent → SEC 8-K outcome enrichment."""

    @pytest.mark.asyncio
    async def test_nasdaq_discovers_upcoming_earnings(self) -> None:
        """Fetch real NASDAQ earnings calendar — should find >$10B companies."""
        from synesis.providers.nasdaq import NasdaqClient

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        client = NasdaqClient(redis=redis)

        events = await fetch_nasdaq_earnings_events(client, days_ahead=14)

        assert len(events) > 0, "No earnings found — NASDAQ API may be down"
        for ev in events:
            _assert_calendar_event(ev, category="earnings", min_importance=6)
            assert len(ev.tickers) > 0, f"Earnings event missing ticker: {ev.title}"
            assert "Earnings:" in ev.title

    @pytest.mark.asyncio
    async def test_sec_earnings_outcome_for_known_ticker(self) -> None:
        """Fetch SEC 8-K earnings press release for a mega-cap (AAPL)."""
        from synesis.providers.sec_edgar.client import SECEdgarClient

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        sec = SECEdgarClient(redis=redis)
        ev = {"title": "Earnings: Apple (AAPL)", "category": "earnings", "tickers": ["AAPL"]}

        outcome = await _get_earnings_outcome(ev, sec)

        # AAPL always has earnings releases on file
        assert len(outcome) > 0, "No SEC earnings release found for AAPL"
        assert len(outcome) <= 800  # Truncated to 800 chars


# ---------------------------------------------------------------------------
# 2. ECONOMIC DATA — FRED discovery + FRED outcome
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEconomicDataE2E:
    """Economic data: FRED release dates → CalendarEvent → FRED observations."""

    @pytest.mark.asyncio
    async def test_fred_discovers_macro_releases(self) -> None:
        """Fetch real FRED release dates — should find upcoming CPI, GDP, NFP, etc."""
        from synesis.providers.fred import FREDClient

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        client = FREDClient(redis=redis)

        events = await fetch_fred_macro_events(client, days_ahead=60)

        assert len(events) > 0, "No FRED macro events found — API may be down"

        # Verify we get the major releases
        titles = {ev.title for ev in events}
        found_categories = set()
        for title in titles:
            for release_id, (name, _) in FRED_MACRO_RELEASES.items():
                if name in title:
                    found_categories.add(name)

        assert "CPI" in found_categories, f"CPI not found in {titles}"
        assert "GDP" in found_categories or "GDP (2nd)" in found_categories, (
            f"GDP not found in {titles}"
        )

        for ev in events:
            _assert_calendar_event(ev, category="economic_data", min_importance=7)

    @pytest.mark.asyncio
    async def test_fred_outcome_for_cpi(self) -> None:
        """Fetch real FRED CPI observations for outcome enrichment."""
        from synesis.providers.fred import FREDClient

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        client = FREDClient(redis=redis)

        ev = {"title": "CPI Release", "category": "economic_data"}
        outcome = await _get_economic_data_outcome(ev, client)

        assert "CPI" in outcome, f"Expected CPI in outcome, got: {outcome}"
        assert "%" in outcome, f"Expected percentage in outcome, got: {outcome}"
        assert "prev" in outcome, f"Expected previous value in outcome, got: {outcome}"

    @pytest.mark.asyncio
    async def test_fred_outcome_for_all_series(self) -> None:
        """Verify FRED outcome enrichment works for every FRED_MACRO_RELEASES title."""
        from synesis.providers.fred import FREDClient

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        client = FREDClient(redis=redis)

        for release_id, (name, importance) in FRED_MACRO_RELEASES.items():
            title = f"{name} Release"
            ev = {"title": title, "category": "economic_data"}
            outcome = await _get_economic_data_outcome(ev, client)

            # Every FRED_MACRO_RELEASES title should match a FRED_OUTCOME_SERIES key
            assert len(outcome) > 0, (
                f"No outcome for '{title}' — FRED_OUTCOME_SERIES missing keyword match"
            )


# ---------------------------------------------------------------------------
# 3. CENTRAL BANK — FOMC rate decision + minutes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFedE2E:
    """Fed: FOMC calendar crawl → extract decisions + minutes → outcome."""

    # Use dates far enough in the future that the extractor won't filter them out
    FOMC_CALENDAR_MARKDOWN = """\
# FOMC Calendars

## 2027

### March 16-17, 2027
- Statement, Implementation Note, Press Conference
- Projection materials
- Minutes released April 7, 2027

### April 27-28, 2027
- Statement, Implementation Note, Press Conference
- Minutes released May 19, 2027

### June 15-16, 2027
- Statement, Implementation Note, Press Conference
- Projection materials
- Minutes released July 7, 2027
"""

    @pytest.mark.asyncio
    async def test_llm_extracts_fomc_meetings_and_minutes(self) -> None:
        """Feed realistic FOMC calendar markdown to LLM extractor.

        Should extract meeting dates (and optionally minutes release dates)
        as fed events.
        """
        events = await extract_events_from_markdown(
            self.FOMC_CALENDAR_MARKDOWN,
            source_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            source_name="Fed FOMC Calendar",
            default_region="US",
        )

        assert len(events) >= 3, (
            f"Expected at least 3 FOMC events, got {len(events)}: {[e.title for e in events]}"
        )

        # All should be fed category
        for ev in events:
            assert ev.category == "fed", (
                f"FOMC event '{ev.title}' has wrong category: {ev.category}"
            )
            assert "US" in ev.region

    @pytest.mark.asyncio
    async def test_fomc_decision_outcome_crawls_real_statement(self) -> None:
        """Crawl a real past Fed statement URL to verify the URL pattern works."""
        from synesis.providers.crawler.crawl4ai import Crawl4AICrawlerProvider

        crawler = Crawl4AICrawlerProvider()

        # Use a known past FOMC statement date (Dec 18, 2024)
        ev: dict[str, Any] = {
            "title": "FOMC Rate Decision",
            "category": "fed",
            "event_date": date(2024, 12, 18),
            "source_urls": [],
        }

        outcome = await _get_crawled_outcome(ev, crawler)

        assert len(outcome) > 0, (
            "Crawled FOMC statement returned empty — "
            "URL pattern monetary{YYYYMMDD}a.htm may be wrong or Crawl4AI down"
        )
        # The statement should contain Fed-specific language
        outcome_lower = outcome.lower()
        assert any(
            term in outcome_lower for term in ["federal", "committee", "rate", "percent", "fomc"]
        ), f"Statement doesn't look like a Fed statement: {outcome[:200]}"


# ---------------------------------------------------------------------------
# 4. 13F FILING — SEC EDGAR discovery + skip enrichment
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestThirteenFE2E:
    """13F: SEC EDGAR filing detection → QoQ diff → enrichment skip."""

    @pytest.mark.asyncio
    async def test_sec_13f_filings_found_for_berkshire(self) -> None:
        """Fetch real 13F-HR filings for Berkshire Hathaway (CIK 1067983)."""
        from synesis.providers.sec_edgar.client import SECEdgarClient

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        sec = SECEdgarClient(redis=redis)
        filings = await sec.get_13f_filings("1067983", limit=2)

        assert len(filings) > 0, "No 13F filings found for Berkshire Hathaway"
        assert filings[0].form == "13F-HR"
        assert filings[0].filed_date is not None

    @pytest.mark.asyncio
    async def test_sec_13f_qoq_diff_returns_positions(self) -> None:
        """Compare QoQ 13F positions for Berkshire Hathaway."""
        from synesis.providers.sec_edgar.client import SECEdgarClient

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        sec = SECEdgarClient(redis=redis)
        diff = await sec.compare_13f_quarters("1067983", "Berkshire Hathaway")

        if diff is None:
            pytest.skip("13F QoQ diff returned None — may need >1 filing on record")

        # Should have position change data
        assert "total_value_current" in diff
        # Should have at least some positions
        has_positions = (
            len(diff.get("new_positions", [])) > 0
            or len(diff.get("exited_positions", [])) > 0
            or len(diff.get("increased", [])) > 0
            or len(diff.get("decreased", [])) > 0
        )
        assert has_positions, f"13F diff has no position changes: {list(diff.keys())}"

    @pytest.mark.asyncio
    async def test_13f_enrichment_is_skipped(self) -> None:
        """13F events should NOT be enriched (already have QoQ diff inline)."""
        events = [
            {
                "title": "13F Filing: Berkshire Hathaway (Q4 2025)",
                "category": "13f_filing",
                "tickers": [],
            },
        ]

        from unittest.mock import patch

        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[{"snippet": "should not be called"}]),
        ) as mock_search:
            result = await _enrich_with_outcomes(events, None, None)

        mock_search.assert_not_called()
        assert result[0].get("outcome") is None


# ---------------------------------------------------------------------------
# 5. RELEASE — LLM extraction of AI model launches from crawled content
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReleaseE2E:
    """Release: LLM extracts AI model releases from crawled blog content."""

    @pytest.mark.asyncio
    async def test_extracts_openai_model_release(self) -> None:
        """Simulate crawled OpenAI blog post announcing a new model."""
        markdown = f"""\
# Introducing GPT-5.4

Published: {date.today().isoformat()}

Today we're releasing GPT-5.4, our most capable model yet.

GPT-5.4 achieves state-of-the-art results on major benchmarks and introduces
native tool use, 2x context window (256K tokens), and significantly improved
reasoning capabilities.

GPT-5.4 is available today via the API and in ChatGPT Plus.

Key improvements:
- MMLU: 92.1% (up from 88.7%)
- HumanEval: 95.3%
- 2x faster inference
- Native function calling and tool use
- Available in API at $5/M input, $15/M output tokens
"""

        events = await extract_events_from_markdown(
            markdown,
            source_url="https://openai.com/index/gpt-5-4",
            source_name="OpenAI Blog",
            default_region="US",
        )

        assert len(events) > 0, "LLM failed to extract GPT-5.4 release event"
        release_events = [e for e in events if e.category == "release"]
        assert len(release_events) > 0, (
            f"No 'release' category events. Got: {[(e.title, e.category) for e in events]}"
        )

        ev = release_events[0]
        assert "gpt" in ev.title.lower() or "5.4" in ev.title
        assert ev.importance >= 7, f"GPT-5.4 should be high importance, got {ev.importance}"

    @pytest.mark.asyncio
    async def test_extracts_deepseek_model_release(self) -> None:
        """Simulate crawled DeepSeek blog announcing a new model."""
        markdown = f"""\
# DeepSeek-R2 Release

Date: {date.today().isoformat()}

We are excited to announce DeepSeek-R2, our next-generation reasoning model.

DeepSeek-R2 achieves breakthrough results on mathematical reasoning and code
generation benchmarks, surpassing previous state-of-the-art models.

Model highlights:
- AIME 2025: 85.2% (new SOTA)
- Codeforces: 2100+ rating
- Available as open-weights under MIT license
- 671B MoE architecture with 37B active parameters
"""

        events = await extract_events_from_markdown(
            markdown,
            source_url="https://www.deepseek.com/en/blog/deepseek-r2",
            source_name="DeepSeek Blog",
            default_region="global",
        )

        assert len(events) > 0, "LLM failed to extract DeepSeek-R2 release event"
        release_events = [e for e in events if e.category == "release"]
        assert len(release_events) > 0, (
            f"No 'release' category events. Got: {[(e.title, e.category) for e in events]}"
        )

        ev = release_events[0]
        assert "deepseek" in ev.title.lower() or "r2" in ev.title.lower()
        assert "global" in ev.region

    @pytest.mark.asyncio
    async def test_extracts_qwen_model_release(self) -> None:
        """Simulate crawled Alibaba/Qwen blog announcing a new model."""
        markdown = f"""\
# Qwen3 Release

Published: {date.today().isoformat()}

Alibaba Cloud's Qwen team is proud to release Qwen3, our latest foundation model
series spanning 0.6B to 235B parameters.

Qwen3 represents a major leap in multilingual capabilities and long-context
understanding. Available immediately on ModelScope and Hugging Face.

Qwen3-235B-A22B:
- MMLU-Pro: 79.7%
- LiveCodeBench: 70.7%
- AIME 2025: 81.5%
- Supports 100+ languages
"""

        events = await extract_events_from_markdown(
            markdown,
            source_url="https://qwenlm.github.io/blog/qwen3",
            source_name="Alibaba Qwen Blog",
            default_region="global",
            default_tickers=["BABA"],
        )

        assert len(events) > 0, "LLM failed to extract Qwen3 release event"
        release_events = [e for e in events if e.category == "release"]
        assert len(release_events) > 0, (
            f"No 'release' category events. Got: {[(e.title, e.category) for e in events]}"
        )

        ev = release_events[0]
        assert "qwen" in ev.title.lower() or "alibaba" in ev.title.lower()
        # default_tickers should be applied
        assert "BABA" in ev.tickers

    @pytest.mark.asyncio
    async def test_extracts_anthropic_model_release(self) -> None:
        """Simulate crawled Anthropic news post announcing a new model."""
        markdown = f"""\
# Introducing Claude 5 Opus

{date.today().isoformat()}

Today we're releasing Claude 5 Opus, our most intelligent model to date.

Claude 5 Opus pushes the frontier on complex reasoning, agentic coding,
and extended thinking. It achieves new state-of-the-art results across
a wide range of evaluations.

Available now via the Anthropic API and on claude.ai.

Performance:
- SWE-bench Verified: 72.0%
- GPQA Diamond: 84.1%
- MATH: 96.4%
"""

        events = await extract_events_from_markdown(
            markdown,
            source_url="https://www.anthropic.com/news/claude-5-opus",
            source_name="Anthropic News",
            default_region="US",
        )

        assert len(events) > 0, "LLM failed to extract Claude 5 Opus release event"
        release_events = [e for e in events if e.category == "release"]
        assert len(release_events) > 0, (
            f"No 'release' category events. Got: {[(e.title, e.category) for e in events]}"
        )

        ev = release_events[0]
        assert "claude" in ev.title.lower() or "opus" in ev.title.lower()
        assert ev.importance >= 7


# ---------------------------------------------------------------------------
# 6. FULL ENRICHMENT MATRIX — all categories through _enrich_with_outcomes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEnrichmentMatrixE2E:
    """Run realistic events for each category through _enrich_with_outcomes
    using real FRED + SEC EDGAR APIs."""

    @pytest.mark.asyncio
    async def test_enrichment_with_real_apis(self) -> None:
        """Each category should hit the right API and return meaningful outcome."""
        from unittest.mock import patch

        from synesis.providers.fred import FREDClient
        from synesis.providers.sec_edgar.client import SECEdgarClient

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock()
        fred = FREDClient(redis=redis)
        sec_edgar = SECEdgarClient(redis=redis)

        events: list[dict[str, Any]] = [
            # 1. earnings → SEC 8-K
            {
                "title": "Earnings: Apple (AAPL)",
                "category": "earnings",
                "tickers": ["AAPL"],
            },
            # 2. economic_data → FRED
            {
                "title": "CPI Release",
                "category": "economic_data",
                "tickers": [],
            },
            # 3. fed (decision) → crawler (mock since no Crawl4AI in CI)
            {
                "title": "FOMC Rate Decision",
                "category": "fed",
                "tickers": [],
                "event_date": date(2024, 12, 18),
                "source_urls": [],
            },
            # 4. fed (minutes) → DB lookup + crawler
            {
                "title": "FOMC Minutes Released",
                "category": "fed",
                "tickers": [],
                "event_date": date(2025, 1, 8),
                "source_urls": [],
            },
            # 5. 13f_filing → skipped
            {
                "title": "13F Filing: Berkshire Hathaway",
                "category": "13f_filing",
                "tickers": [],
            },
            # 6. conference → web search
            {
                "title": "NVIDIA GTC 2026 Keynote",
                "category": "conference",
                "tickers": ["NVDA"],
            },
            # 7. release → web search
            {
                "title": "GPT-5.4 Released by OpenAI",
                "category": "release",
                "tickers": ["MSFT"],
            },
            # 8. regulatory → web search
            {
                "title": "EU AI Act Enforcement",
                "category": "regulatory",
                "tickers": [],
            },
            # 9. other → web search
            {
                "title": "Major Trade Agreement Signed",
                "category": "other",
                "tickers": [],
            },
        ]

        # Mock crawler (Crawl4AI may not be running)
        crawler = AsyncMock()
        crawler.crawl = AsyncMock(
            return_value=AsyncMock(
                success=True,
                markdown="The Committee decided to lower the target range for the federal "
                "funds rate by 1/4 percentage point to 4-1/4 to 4-1/2 percent.",
            )
        )

        # Mock DB for FOMC minutes lookup
        mock_db = AsyncMock()
        mock_db.get_last_fomc_meeting_date = AsyncMock(return_value=date(2024, 12, 18))

        # Mock web search (SearXNG may not be running)
        with patch(
            "synesis.processing.common.web_search.search_market_impact",
            AsyncMock(return_value=[{"snippet": "Web search outcome for event"}]),
        ):
            result = await _enrich_with_outcomes(events, redis, sec_edgar, crawler, fred, mock_db)

        # 1. earnings → real SEC data
        earnings_outcome = result[0].get("outcome", "")
        assert len(earnings_outcome) > 0, "Earnings outcome empty — SEC EDGAR may be down"

        # 2. economic_data → real FRED data
        econ_outcome = result[1].get("outcome", "")
        assert "CPI" in econ_outcome, f"Expected CPI in outcome, got: {econ_outcome}"
        assert "%" in econ_outcome

        # 3. fed (decision) → mocked crawler
        decision_outcome = result[2].get("outcome", "")
        assert "federal funds rate" in decision_outcome.lower()

        # 4. fed (minutes) → DB lookup + mocked crawler
        minutes_outcome = result[3].get("outcome", "")
        assert "federal funds rate" in minutes_outcome.lower()

        # 5. 13f_filing → skipped
        assert result[4].get("outcome") is None

        # 6-9. conference/release/regulatory/other → web search
        for i in range(5, 9):
            outcome = result[i].get("outcome", "")
            assert "Web search outcome" in outcome, (
                f"Event '{result[i]['title']}' missing web search outcome"
            )
