"""Integration tests for Event Radar pipeline.

Uses REAL APIs (FRED, SEC EDGAR, NASDAQ, LLM) to verify end-to-end flows.
Categories tested: earnings, economic_data, fed (decision + minutes), 13f_filing.

Run with: pytest -m integration tests/integration/test_event_radar_e2e.py -v
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from synesis.processing.events.fetchers import (
    FRED_MACRO_RELEASES,
    fetch_fomc_events,
    fetch_fred_macro_events,
    fetch_nasdaq_earnings_events,
)
from synesis.processing.intelligence.strategists.macro import (
    _enrich_events_with_outcomes,
    _get_crawled_outcome,
    _get_economic_data_outcome,
    _get_earnings_outcome,
)
from synesis.processing.events.models import CalendarEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_calendar_event(
    event: CalendarEvent,
    *,
    category: str,
    has_title: bool = True,
) -> None:
    """Assert basic CalendarEvent validity."""
    assert event.category == category, f"Expected {category}, got {event.category}"
    assert event.event_date >= date.today() - timedelta(days=1)
    if has_title:
        assert len(event.title) > 0
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
            _assert_calendar_event(ev, category="earnings")
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
            for release_id, name in FRED_MACRO_RELEASES.items():
                if name in title:
                    found_categories.add(name)

        assert "CPI" in found_categories, f"CPI not found in {titles}"
        assert "GDP" in found_categories or "GDP (2nd)" in found_categories, (
            f"GDP not found in {titles}"
        )

        for ev in events:
            _assert_calendar_event(ev, category="economic_data")

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

        for release_id, name in FRED_MACRO_RELEASES.items():
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
    """Fed: FOMC structured crawler (no LLM) → rate decisions + minutes → outcome."""

    @pytest.mark.asyncio
    async def test_fomc_crawler_fetches_real_decisions(self) -> None:
        """fetch_fomc_events() hits the real Fed calendar — no LLM, pure regex.

        Verifies the Fed page HTML structure hasn't changed and we still parse correctly.
        """
        events = await fetch_fomc_events(days_ahead=365)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        assert len(decisions) >= 4, (
            f"Expected ≥4 FOMC Rate Decision events in the next year, got {len(decisions)}"
        )
        for ev in decisions:
            _assert_calendar_event(ev, category="fed")

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
            result = await _enrich_events_with_outcomes(
                events, AsyncMock(), AsyncMock(), None, AsyncMock()
            )

        mock_search.assert_not_called()
        assert result[0].get("outcome") is None


# ---------------------------------------------------------------------------
# 5. FULL ENRICHMENT MATRIX — all categories through _enrich_events_with_outcomes
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEnrichmentMatrixE2E:
    """Run all structured categories through _enrich_events_with_outcomes
    using real FRED + SEC EDGAR APIs."""

    @pytest.mark.asyncio
    async def test_enrichment_with_real_apis(self) -> None:
        """Each structured category hits the right API and returns a meaningful outcome."""
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
            # 3. fed (decision) → crawler (mocked — Crawl4AI not in CI)
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
            # 5. 13f_filing → skipped (already has QoQ diff inline)
            {
                "title": "13F Filing: Berkshire Hathaway",
                "category": "13f_filing",
                "tickers": [],
            },
        ]

        # Mock crawler (Crawl4AI not running locally)
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

        result = await _enrich_events_with_outcomes(events, sec_edgar, fred, crawler, mock_db)

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
