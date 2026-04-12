"""Unit tests for macro strategist enrichment functions.

Tests the event outcome enrichment, 13F brief fetching, event context formatting,
and benchmark context formatting that were migrated from events/digest.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from synesis.processing.intelligence.strategists.macro import (
    _enrich_events_with_outcomes,
    _fetch_13f_briefs,
    _format_benchmark_context,
    _format_event_context,
)


# ── _format_benchmark_context ────────────────────────────────────


class TestFormatBenchmarkContext:
    def test_empty_data(self) -> None:
        result = _format_benchmark_context({})
        assert "No benchmark data available" in result

    def test_basic_equity_benchmark(self) -> None:
        data = {
            "SPY": {"last": 500.0, "prev_close": 495.0, "avg_50d": 490.0, "avg_200d": 480.0},
        }
        result = _format_benchmark_context(data)
        assert "S&P 500" in result
        assert "SPY" in result
        assert "$500.00" in result
        assert "above 50d MA" in result
        assert "above 200d MA" in result
        assert "1d:" in result

    def test_below_moving_averages(self) -> None:
        data = {
            "SPY": {"last": 470.0, "prev_close": 475.0, "avg_50d": 490.0, "avg_200d": 480.0},
        }
        result = _format_benchmark_context(data)
        assert "below 50d MA" in result
        assert "below 200d MA" in result

    def test_missing_last_price_skipped(self) -> None:
        data = {
            "SPY": {"last": None, "prev_close": 495.0, "avg_50d": 490.0, "avg_200d": 480.0},
        }
        result = _format_benchmark_context(data)
        assert "S&P 500" not in result

    def test_missing_moving_averages(self) -> None:
        data = {
            "SPY": {"last": 500.0, "prev_close": 495.0, "avg_50d": None, "avg_200d": None},
        }
        result = _format_benchmark_context(data)
        assert "S&P 500" in result
        assert "MA" not in result

    def test_multiple_categories(self) -> None:
        data = {
            "SPY": {"last": 500.0, "prev_close": 495.0, "avg_50d": None, "avg_200d": None},
            "GLD": {"last": 200.0, "prev_close": 198.0, "avg_50d": None, "avg_200d": None},
            "TLT": {"last": 90.0, "prev_close": 91.0, "avg_50d": None, "avg_200d": None},
        }
        result = _format_benchmark_context(data)
        assert "Equities" in result
        assert "Commodities" in result
        assert "Treasuries" in result

    def test_unknown_ticker_uses_other_category(self) -> None:
        data = {
            "ZZZZZ": {"last": 10.0, "prev_close": 9.0, "avg_50d": None, "avg_200d": None},
        }
        result = _format_benchmark_context(data)
        # Unknown ticker won't be in _BENCHMARKS, so it won't appear
        assert "ZZZZZ" not in result


# ── _format_event_context ────────────────────────────────────────


class TestFormatEventContext:
    def test_empty_inputs(self) -> None:
        result = _format_event_context([], [], [])
        assert "No event data available" in result

    def test_recent_events_formatted(self) -> None:
        recent = [
            {
                "category": "earnings",
                "title": "AAPL Q1 Earnings",
                "tickers": ["AAPL"],
                "description": "Apple beat estimates",
            },
        ]
        result = _format_event_context([], recent, [])
        assert "Yesterday's Events" in result
        assert "[earnings]" in result
        assert "[AAPL]" in result
        assert "AAPL Q1 Earnings" in result
        assert "Apple beat estimates" in result

    def test_outcome_included(self) -> None:
        recent = [
            {
                "category": "earnings",
                "title": "NVDA Earnings",
                "tickers": ["NVDA"],
                "outcome": "Revenue beat by 15%",
            },
        ]
        result = _format_event_context([], recent, [])
        assert "**Outcome:**" in result
        assert "Revenue beat by 15%" in result

    def test_long_outcome_truncated(self) -> None:
        recent = [
            {
                "category": "fed",
                "title": "Fed Minutes",
                "tickers": [],
                "outcome": "x" * 5000,
            },
        ]
        result = _format_event_context([], recent, [])
        assert "[...truncated]" in result
        # The 2000-char truncation at the format layer
        assert len(result) < 5500

    def test_upcoming_events_formatted(self) -> None:
        upcoming = [
            {"category": "earnings", "title": "MSFT Earnings", "event_date": "2026-04-15"},
            {"category": "fed", "title": "FOMC Decision", "event_date": "2026-04-16"},
        ]
        result = _format_event_context(upcoming, [], [])
        assert "Upcoming Catalysts" in result
        assert "MSFT Earnings" in result
        assert "FOMC Decision" in result

    def test_13f_briefs_formatted(self) -> None:
        briefs = [
            {
                "fund_name": "Bridgewater Associates",
                "current_report_date": "2026-03-31",
                "previous_report_date": "2025-12-31",
                "total_value_current": 150_000_000_000,
                "total_value_previous": 140_000_000_000,
                "new_positions": [{"name_of_issuer": "NVDA"}, {"name_of_issuer": "TSLA"}],
                "exited_positions": [{"name_of_issuer": "META"}],
                "increased": [],
                "decreased": [],
            },
        ]
        result = _format_event_context([], [], briefs)
        assert "13F Hedge Fund Position Changes" in result
        assert "Bridgewater Associates" in result
        assert "NVDA" in result
        assert "META" in result

    def test_all_sections_combined(self) -> None:
        upcoming = [{"category": "earnings", "title": "GOOG", "event_date": "2026-04-15"}]
        recent = [{"category": "macro", "title": "GDP Release", "tickers": []}]
        briefs = [
            {
                "fund_name": "Renaissance",
                "current_report_date": "2026-03-31",
                "previous_report_date": "2025-12-31",
            },
        ]
        result = _format_event_context(upcoming, recent, briefs)
        assert "Yesterday's Events" in result
        assert "13F Hedge Fund" in result
        assert "Upcoming Catalysts" in result


# ── _enrich_events_with_outcomes ─────────────────────────────────


class TestEnrichEventsWithOutcomes:
    @pytest.mark.asyncio
    async def test_empty_events(self) -> None:
        result = await _enrich_events_with_outcomes([], AsyncMock(), AsyncMock(), None, AsyncMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_earnings_event_enriched(self) -> None:
        mock_sec = AsyncMock()
        mock_release = AsyncMock()
        mock_release.content = "Revenue: $100B, beat estimates"
        mock_sec.get_earnings_releases = AsyncMock(return_value=[mock_release])

        events: list[dict[str, Any]] = [
            {"category": "earnings", "title": "AAPL Earnings", "tickers": ["AAPL"]},
        ]
        result = await _enrich_events_with_outcomes(
            events, mock_sec, AsyncMock(), None, AsyncMock()
        )
        assert result[0].get("outcome") == "Revenue: $100B, beat estimates"

    @pytest.mark.asyncio
    async def test_13f_events_skipped(self) -> None:
        """13F events are handled by _fetch_13f_briefs, not enrichment."""
        events: list[dict[str, Any]] = [
            {"category": "13f_filing", "title": "Bridgewater 13F"},
        ]
        result = await _enrich_events_with_outcomes(
            events, AsyncMock(), AsyncMock(), None, AsyncMock()
        )
        assert "outcome" not in result[0]

    @pytest.mark.asyncio
    async def test_economic_data_enriched(self) -> None:
        mock_fred = AsyncMock()
        obs1 = AsyncMock()
        obs1.value = 3.5
        obs2 = AsyncMock()
        obs2.value = 3.4
        mock_fred.get_observations = AsyncMock(return_value=AsyncMock(observations=[obs1, obs2]))

        events: list[dict[str, Any]] = [
            {"category": "economic_data", "title": "CPI Report"},
        ]
        result = await _enrich_events_with_outcomes(
            events, AsyncMock(), mock_fred, None, AsyncMock()
        )
        assert result[0].get("outcome")
        assert "CPI" in result[0]["outcome"]

    @pytest.mark.asyncio
    async def test_unknown_category_no_outcome(self) -> None:
        events: list[dict[str, Any]] = [
            {"category": "other", "title": "Some Event"},
        ]
        result = await _enrich_events_with_outcomes(
            events, AsyncMock(), AsyncMock(), None, AsyncMock()
        )
        assert "outcome" not in result[0]

    @pytest.mark.asyncio
    async def test_exception_gracefully_handled(self) -> None:
        mock_sec = AsyncMock()
        mock_sec.get_earnings_releases = AsyncMock(side_effect=RuntimeError("API down"))

        events: list[dict[str, Any]] = [
            {"category": "earnings", "title": "AAPL Earnings", "tickers": ["AAPL"]},
        ]
        result = await _enrich_events_with_outcomes(
            events, mock_sec, AsyncMock(), None, AsyncMock()
        )
        # Should not crash; event returned without outcome
        assert "outcome" not in result[0]

    @pytest.mark.asyncio
    async def test_multiple_events_enriched_concurrently(self) -> None:
        mock_sec = AsyncMock()
        mock_release = AsyncMock()
        mock_release.content = "Beat estimates"
        mock_sec.get_earnings_releases = AsyncMock(return_value=[mock_release])

        events: list[dict[str, Any]] = [
            {"category": "earnings", "title": "AAPL", "tickers": ["AAPL"]},
            {"category": "earnings", "title": "NVDA", "tickers": ["NVDA"]},
            {"category": "other", "title": "Other"},
        ]
        result = await _enrich_events_with_outcomes(
            events, mock_sec, AsyncMock(), None, AsyncMock()
        )
        assert result[0].get("outcome") == "Beat estimates"
        assert result[1].get("outcome") == "Beat estimates"
        assert "outcome" not in result[2]


# ── _fetch_13f_briefs ────────────────────────────────────────────


class TestFetch13fBriefs:
    @pytest.mark.asyncio
    async def test_empty_events(self) -> None:
        result = await _fetch_13f_briefs(AsyncMock(), [])
        assert result == []

    @pytest.mark.asyncio
    async def test_non_13f_events_skipped(self) -> None:
        events = [
            {"category": "earnings", "title": "AAPL Earnings"},
            {"category": "fed", "title": "Fed Decision"},
        ]
        result = await _fetch_13f_briefs(AsyncMock(), events)
        assert result == []

    @pytest.mark.asyncio
    async def test_13f_event_matched_and_fetched(self) -> None:
        mock_sec = AsyncMock()
        mock_sec.compare_13f_quarters = AsyncMock(
            return_value={"new_positions": [], "exited_positions": []}
        )

        # Mock the fund registry to return a known fund
        with patch(
            "synesis.processing.intelligence.strategists.macro.load_hedge_fund_registry",
            return_value=({"0001336528": "Bridgewater Associates"}, {}),
        ):
            events = [
                {"category": "13f_filing", "title": "13F Filing: Bridgewater Associates"},
            ]
            result = await _fetch_13f_briefs(mock_sec, events)

        assert len(result) == 1
        assert result[0]["fund_name"] == "Bridgewater Associates"
        mock_sec.compare_13f_quarters.assert_called_once_with(
            "0001336528", "Bridgewater Associates"
        )

    @pytest.mark.asyncio
    async def test_13f_no_match_in_registry(self) -> None:
        with patch(
            "synesis.processing.intelligence.strategists.macro.load_hedge_fund_registry",
            return_value=({"0001336528": "Bridgewater Associates"}, {}),
        ):
            events = [
                {"category": "13f_filing", "title": "13F Filing: Unknown Fund LLC"},
            ]
            result = await _fetch_13f_briefs(AsyncMock(), events)

        assert result == []

    @pytest.mark.asyncio
    async def test_13f_api_failure_graceful(self) -> None:
        mock_sec = AsyncMock()
        mock_sec.compare_13f_quarters = AsyncMock(side_effect=RuntimeError("SEC down"))

        with patch(
            "synesis.processing.intelligence.strategists.macro.load_hedge_fund_registry",
            return_value=({"0001336528": "Bridgewater Associates"}, {}),
        ):
            events = [
                {"category": "13f_filing", "title": "13F Filing: Bridgewater Associates"},
            ]
            result = await _fetch_13f_briefs(mock_sec, events)

        # Should not crash; returns empty
        assert result == []

    @pytest.mark.asyncio
    async def test_13f_null_result_skipped(self) -> None:
        mock_sec = AsyncMock()
        mock_sec.compare_13f_quarters = AsyncMock(return_value=None)

        with patch(
            "synesis.processing.intelligence.strategists.macro.load_hedge_fund_registry",
            return_value=({"0001336528": "Bridgewater Associates"}, {}),
        ):
            events = [
                {"category": "13f_filing", "title": "13F Filing: Bridgewater Associates"},
            ]
            result = await _fetch_13f_briefs(mock_sec, events)

        assert result == []
