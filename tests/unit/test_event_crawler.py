"""Unit tests for event fetchers — fetch_fomc_events, fetch_fred_macro_events,
fetch_nasdaq_earnings_events, fetch_13f_events.

All tests are offline (no network calls). External clients are mocked.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.processing.events.fetchers import (
    fetch_13f_events,
    fetch_fomc_events,
    fetch_fred_macro_events,
    fetch_nasdaq_earnings_events,
)


# ---------------------------------------------------------------------------
# HTML builder helpers
# ---------------------------------------------------------------------------


def _fomc_html(year: int, meetings: list[dict]) -> str:
    """Build minimal FOMC calendar HTML matching the real Fed page structure."""
    blocks = []
    for m in meetings:
        minutes_inner = m.get("minutes", "\n\n")
        blocks.append(
            f'<div class="row fomc-meeting">\n'
            f'  <div class="fomc-meeting__month col-xs-3 col-sm-3">'
            f"<strong>{m['month']}</strong></div>\n"
            f'  <div class="fomc-meeting__date col-xs-3">{m["dates"]}</div>\n'
            f'  <div class="col-xs-12 col-md-4 col-lg-4 fomc-meeting__minutes">'
            f"{minutes_inner}</div>\n"
            f"</div>"
        )
    return f"<html>\n{year} FOMC Meetings\n" + "\n".join(blocks) + "\n</html>"


def _patch_httpx(html: str):
    """Patch httpx.AsyncClient to return given HTML without network access."""
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    return patch("httpx.AsyncClient", mock_cls)


# ---------------------------------------------------------------------------
# fetch_fomc_events — Rate Decision
# ---------------------------------------------------------------------------


class TestFomcRateDecision:
    """Rate decision event parsing from FOMC HTML."""

    @pytest.mark.asyncio
    async def test_rate_decision_in_window(self) -> None:
        """A future meeting within days_ahead creates an FOMC Rate Decision event."""
        html = _fomc_html(2099, [{"month": "March", "dates": "17-18"}])
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        assert len(decisions) == 1
        ev = decisions[0]
        assert ev.event_date == date(2099, 3, 18)
        assert ev.category == "fed"
        assert "US" in ev.region

    @pytest.mark.asyncio
    async def test_rate_decision_past_filtered_out(self) -> None:
        """A past meeting date is not returned."""
        html = _fomc_html(2020, [{"month": "March", "dates": "17-18"}])
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=365)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        assert len(decisions) == 0

    @pytest.mark.asyncio
    async def test_rate_decision_beyond_cutoff_filtered_out(self) -> None:
        """A meeting date beyond the cutoff is not returned."""
        html = _fomc_html(2099, [{"month": "June", "dates": "16-17"}])
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=7)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        assert len(decisions) == 0

    @pytest.mark.asyncio
    async def test_rate_decision_month_straddle(self) -> None:
        """'October 31-1' → decision day is November 1 (next month)."""
        html = _fomc_html(2099, [{"month": "October", "dates": "31-1"}])
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        assert len(decisions) == 1
        assert decisions[0].event_date == date(2099, 11, 1)

    @pytest.mark.asyncio
    async def test_rate_decision_december_straddle_bumps_year(self) -> None:
        """'December 31-1' → decision day is January 1 of the following year."""
        html = _fomc_html(2099, [{"month": "December", "dates": "31-1"}])
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        assert len(decisions) == 1
        assert decisions[0].event_date == date(2100, 1, 1)

    @pytest.mark.asyncio
    async def test_multiple_meetings_all_returned(self) -> None:
        """Multiple meetings in the window all produce Rate Decision events."""
        html = _fomc_html(
            2099,
            [
                {"month": "March", "dates": "17-18"},
                {"month": "April", "dates": "28-29"},
                {"month": "June", "dates": "15-16"},
            ],
        )
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        assert len(decisions) == 3
        dates = {e.event_date for e in decisions}
        assert date(2099, 3, 18) in dates
        assert date(2099, 4, 29) in dates
        assert date(2099, 6, 16) in dates

    @pytest.mark.asyncio
    async def test_multiple_year_blocks(self) -> None:
        """Meetings from two year blocks are both processed."""
        html = (
            "<html>\n"
            "2099 FOMC Meetings\n"
            '<div class="row fomc-meeting">\n'
            '  <div class="fomc-meeting__month col-xs-3"><strong>March</strong></div>\n'
            '  <div class="fomc-meeting__date col-xs-3">17-18</div>\n'
            '  <div class="col-xs-12 col-md-4 col-lg-4 fomc-meeting__minutes">\n\n</div>\n'
            "</div>\n"
            "2100 FOMC Meetings\n"
            '<div class="row fomc-meeting">\n'
            '  <div class="fomc-meeting__month col-xs-3"><strong>January</strong></div>\n'
            '  <div class="fomc-meeting__date col-xs-3">26-27</div>\n'
            '  <div class="col-xs-12 col-md-4 col-lg-4 fomc-meeting__minutes">\n\n</div>\n'
            "</div>\n"
            "</html>"
        )
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        assert len(decisions) == 2
        dates = {e.event_date for e in decisions}
        assert date(2099, 3, 18) in dates
        assert date(2100, 1, 27) in dates


# ---------------------------------------------------------------------------
# fetch_fomc_events — Minutes Release
# ---------------------------------------------------------------------------


class TestFomcMinutesRelease:
    """Minutes release date parsing from FOMC HTML."""

    @pytest.mark.asyncio
    async def test_minutes_in_window_creates_event(self) -> None:
        """A minutes release date within the window creates a Minutes Release event."""
        html = _fomc_html(
            2099,
            [
                {
                    "month": "February",
                    "dates": "24-25",
                    "minutes": "\n    (Released March 18, 2099)\n  ",
                }
            ],
        )
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        minutes_events = [e for e in events if e.title == "FOMC Minutes Release"]
        assert len(minutes_events) == 1
        ev = minutes_events[0]
        assert ev.event_date == date(2099, 3, 18)
        assert ev.category == "fed"
        assert "US" in ev.region
        assert "February 24-25" in ev.description

    @pytest.mark.asyncio
    async def test_minutes_past_filtered_out(self) -> None:
        """A past minutes release date is not returned."""
        html = _fomc_html(
            2020,
            [
                {
                    "month": "January",
                    "dates": "28-29",
                    "minutes": "\n    (Released February 19, 2020)\n  ",
                }
            ],
        )
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=365)

        minutes_events = [e for e in events if e.title == "FOMC Minutes Release"]
        assert len(minutes_events) == 0

    @pytest.mark.asyncio
    async def test_minutes_empty_div_no_event(self) -> None:
        """An empty minutes div (no release date yet) produces no minutes event."""
        html = _fomc_html(2099, [{"month": "March", "dates": "17-18", "minutes": "\n\n"}])
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        minutes_events = [e for e in events if e.title == "FOMC Minutes Release"]
        assert len(minutes_events) == 0

    @pytest.mark.asyncio
    async def test_minutes_beyond_cutoff_filtered_out(self) -> None:
        """A minutes release beyond the cutoff is not returned."""
        html = _fomc_html(
            2099,
            [
                {
                    "month": "February",
                    "dates": "24-25",
                    "minutes": "\n    (Released March 18, 2099)\n  ",
                }
            ],
        )
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=7)

        minutes_events = [e for e in events if e.title == "FOMC Minutes Release"]
        assert len(minutes_events) == 0

    @pytest.mark.asyncio
    async def test_meeting_produces_decision_and_minutes(self) -> None:
        """One meeting block in window produces both a Rate Decision and a Minutes Release."""
        html = _fomc_html(
            2099,
            [
                {
                    "month": "February",
                    "dates": "24-25",
                    "minutes": "\n    (Released March 18, 2099)\n  ",
                }
            ],
        )
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        minutes_events = [e for e in events if e.title == "FOMC Minutes Release"]
        assert len(decisions) == 1
        assert len(minutes_events) == 1

    @pytest.mark.asyncio
    async def test_mixed_meetings_partial_minutes(self) -> None:
        """Two meetings: one with minutes in window, one without."""
        html = _fomc_html(
            2099,
            [
                {
                    "month": "February",
                    "dates": "24-25",
                    "minutes": "\n    (Released March 18, 2099)\n  ",
                },
                {"month": "April", "dates": "28-29", "minutes": "\n\n"},
            ],
        )
        with _patch_httpx(html):
            events = await fetch_fomc_events(days_ahead=36500)

        decisions = [e for e in events if e.title == "FOMC Rate Decision"]
        minutes_events = [e for e in events if e.title == "FOMC Minutes Release"]
        assert len(decisions) == 2
        assert len(minutes_events) == 1
        assert minutes_events[0].event_date == date(2099, 3, 18)


# ---------------------------------------------------------------------------
# fetch_fred_macro_events
# ---------------------------------------------------------------------------


def _make_fred_release(release_date: date, release_name: str = "CPI") -> MagicMock:
    rd = MagicMock()
    rd.date = release_date
    rd.release_name = release_name
    return rd


class TestFetchFredMacroEvents:
    """Tests for fetch_fred_macro_events."""

    @pytest.mark.asyncio
    async def test_in_window_event_returned(self) -> None:
        """A release date within today..cutoff creates a CalendarEvent."""
        future_date = date.today() + timedelta(days=10)
        fred = AsyncMock()
        fred.get_release_dates = AsyncMock(return_value=[_make_fred_release(future_date, "CPI")])

        events = await fetch_fred_macro_events(fred, days_ahead=30)

        cpi_events = [e for e in events if "CPI" in e.title]
        assert len(cpi_events) >= 1
        assert cpi_events[0].event_date == future_date
        assert cpi_events[0].category == "economic_data"

    @pytest.mark.asyncio
    async def test_past_date_filtered(self) -> None:
        """A release date in the past is not returned."""
        past_date = date.today() - timedelta(days=1)
        fred = AsyncMock()
        fred.get_release_dates = AsyncMock(return_value=[_make_fred_release(past_date)])

        events = await fetch_fred_macro_events(fred, days_ahead=30)

        assert all(e.event_date >= date.today() for e in events)

    @pytest.mark.asyncio
    async def test_beyond_cutoff_filtered(self) -> None:
        """A release date beyond the cutoff is not returned."""
        far_future = date.today() + timedelta(days=100)
        fred = AsyncMock()
        fred.get_release_dates = AsyncMock(return_value=[_make_fred_release(far_future)])

        events = await fetch_fred_macro_events(fred, days_ahead=30)

        assert all(e.event_date <= date.today() + timedelta(days=30) for e in events)

    @pytest.mark.asyncio
    async def test_one_release_fails_others_returned(self) -> None:
        """An exception for one release_id does not abort the rest."""
        future_date = date.today() + timedelta(days=5)

        call_count = 0

        async def get_release_dates(release_id: int, **kwargs) -> list:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("FRED API error")
            return [_make_fred_release(future_date)]

        fred = AsyncMock()
        fred.get_release_dates = AsyncMock(side_effect=get_release_dates)

        events = await fetch_fred_macro_events(fred, days_ahead=30)

        # At least some events from other release IDs should be returned
        assert len(events) > 0


# ---------------------------------------------------------------------------
# fetch_nasdaq_earnings_events
# ---------------------------------------------------------------------------


def _make_earnings(
    ticker: str = "AAPL",
    company_name: str = "Apple Inc.",
    market_cap: float | None = 300_000_000_000,
    fiscal_quarter: int = 1,
    earnings_date: date | None = None,
    time: str | None = "AMC",
    eps_forecast: float | None = 1.50,
) -> MagicMock:
    e = MagicMock()
    e.ticker = ticker
    e.company_name = company_name
    e.market_cap = market_cap
    e.fiscal_quarter = fiscal_quarter
    e.earnings_date = earnings_date or (date.today() + timedelta(days=3))
    e.time = time
    e.eps_forecast = eps_forecast
    return e


class TestFetchNasdaqEarningsEvents:
    """Tests for fetch_nasdaq_earnings_events."""

    @pytest.mark.asyncio
    async def test_above_min_cap_creates_event(self) -> None:
        """Earnings with market_cap >= $10B creates a CalendarEvent."""
        nasdaq = AsyncMock()
        nasdaq.get_earnings_by_date = AsyncMock(return_value=[_make_earnings(market_cap=300e9)])

        events = await fetch_nasdaq_earnings_events(nasdaq, days_ahead=1)

        assert len(events) == 1
        assert "AAPL" in events[0].title
        assert events[0].category == "earnings"

    @pytest.mark.asyncio
    async def test_below_min_cap_filtered(self) -> None:
        """Earnings with market_cap < $10B are not included."""
        nasdaq = AsyncMock()
        nasdaq.get_earnings_by_date = AsyncMock(
            return_value=[_make_earnings(market_cap=5_000_000_000)]
        )

        events = await fetch_nasdaq_earnings_events(nasdaq, days_ahead=1)

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_null_market_cap_filtered(self) -> None:
        """Earnings with market_cap=None are excluded (no division, no exception)."""
        nasdaq = AsyncMock()
        nasdaq.get_earnings_by_date = AsyncMock(return_value=[_make_earnings(market_cap=None)])

        events = await fetch_nasdaq_earnings_events(nasdaq, days_ahead=1)

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_one_day_fails_others_returned(self) -> None:
        """An exception for one date does not abort all other dates."""
        call_count = 0

        async def get_earnings(target_date: date) -> list:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("NASDAQ unavailable")
            return [_make_earnings(earnings_date=target_date)]

        nasdaq = AsyncMock()
        nasdaq.get_earnings_by_date = AsyncMock(side_effect=get_earnings)

        events = await fetch_nasdaq_earnings_events(nasdaq, days_ahead=3)

        assert len(events) > 0  # days 2 and 3 still returned


# ---------------------------------------------------------------------------
# fetch_13f_events
# ---------------------------------------------------------------------------


class TestFetch13fEvents:
    """Tests for fetch_13f_events."""

    def _make_filing(
        self,
        accession: str = "0001234-25-000001",
        filed_date: date | None = None,
        items: str = "2025-03-31",
        url: str = "https://sec.gov/filing",
    ) -> MagicMock:
        f = MagicMock()
        f.accession_number = accession
        f.filed_date = filed_date or date.today()
        f.items = items
        f.url = url
        return f

    def _make_clients(
        self,
        filings: list,
        seen_value: bytes | None = None,
        diff: dict | None = None,
        top_tier_cik: str | None = None,
    ) -> tuple[AsyncMock, AsyncMock]:
        sec_edgar = AsyncMock()
        sec_edgar.get_13f_filings = AsyncMock(return_value=filings)
        sec_edgar.compare_13f_quarters = AsyncMock(return_value=diff or {})

        redis = AsyncMock()
        redis.get = AsyncMock(return_value=seen_value)
        redis.set = AsyncMock()

        return sec_edgar, redis

    @pytest.mark.asyncio
    async def test_new_filing_creates_event(self) -> None:
        """A new (unseen) filing creates a CalendarEvent and marks it seen."""
        filing = self._make_filing()
        sec_edgar, redis = self._make_clients(filings=[filing], seen_value=None)

        with patch(
            "synesis.processing.events.fetchers.load_hedge_fund_registry",
            return_value=({"0001234": "Test Fund"}, set()),
        ):
            events = await fetch_13f_events(sec_edgar, redis)

        assert len(events) == 1
        assert "Test Fund" in events[0].title
        redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_already_seen_filing_skipped(self) -> None:
        """A filing whose accession number is already in Redis is not returned."""
        filing = self._make_filing()
        sec_edgar, redis = self._make_clients(filings=[filing], seen_value=b"1")

        with patch(
            "synesis.processing.events.fetchers.load_hedge_fund_registry",
            return_value=({"0001234": "Test Fund"}, set()),
        ):
            events = await fetch_13f_events(sec_edgar, redis)

        assert len(events) == 0
        redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_redis_error_does_not_abort_other_funds(self) -> None:
        """A Redis error on one fund check does not block the rest (return_exceptions=True)."""
        filing = self._make_filing(accession="0000333")
        sec_edgar = AsyncMock()
        sec_edgar.get_13f_filings = AsyncMock(return_value=[filing])
        sec_edgar.compare_13f_quarters = AsyncMock(return_value={})

        call_count = 0

        async def redis_get(key: str) -> bytes | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Redis down")
            return None

        redis = AsyncMock()
        redis.get = AsyncMock(side_effect=redis_get)
        redis.set = AsyncMock()

        with patch(
            "synesis.processing.events.fetchers.load_hedge_fund_registry",
            return_value=(
                {"0000111": "Fund A", "0000333": "Fund B"},
                set(),
            ),
        ):
            events = await fetch_13f_events(sec_edgar, redis)

        # Fund B should still be processed even if Fund A hit a Redis error
        assert any("Fund B" in e.title for e in events)
