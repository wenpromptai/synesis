"""Tests for the market movers pipeline (processing/market/)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synesis.processing.market.discord_format import (
    _format_movers,
    _format_sector_lines,
    _format_ticker_lines,
    format_market_movers_embeds,
)
from synesis.processing.market.models import MarketMoversData, TickerChange
from synesis.processing.market.snapshot import fetch_market_movers
from synesis.providers.yfinance.models import MarketMover, MarketMovers


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 9, 15, 0, 0, tzinfo=UTC)


def _make_movers(**overrides: Any) -> MarketMovers:
    defaults: dict[str, Any] = {
        "gainers": [
            MarketMover(
                ticker="NVDA",
                name="NVIDIA Corp",
                price=950.0,
                change_pct=8.5,
                sector="Technology",
            ),
            MarketMover(
                ticker="TSLA",
                name="Tesla Inc",
                price=220.0,
                change_pct=5.2,
                sector="Consumer Cyclical",
            ),
        ],
        "losers": [
            MarketMover(
                ticker="META",
                name="Meta Platforms",
                price=480.0,
                change_pct=-4.3,
                sector="Communication Services",
            ),
        ],
        "most_actives": [
            MarketMover(
                ticker="AAPL",
                name="Apple Inc",
                price=178.0,
                change_pct=0.5,
                sector="Technology",
                volume=120_000_000,
            ),
        ],
        "fetched_at": _NOW,
    }
    defaults.update(overrides)
    return MarketMovers(**defaults)


def _make_data(**overrides: Any) -> MarketMoversData:
    defaults: dict[str, Any] = {
        "equities": [
            TickerChange(ticker="SPY", last=520.0, prev_close=515.0, change_pct=0.97),
            TickerChange(ticker="QQQ", last=445.0, prev_close=440.0, change_pct=1.14),
            TickerChange(ticker="IWM", last=210.0, prev_close=212.0, change_pct=-0.94),
        ],
        "rates_fx": [
            TickerChange(ticker="TLT", last=95.0, prev_close=94.5, change_pct=0.53),
            TickerChange(ticker="UUP", last=28.0, prev_close=28.1, change_pct=-0.36),
        ],
        "commodities": [
            TickerChange(ticker="GLD", last=215.0, prev_close=213.0, change_pct=0.94),
            TickerChange(ticker="USO", last=72.0, prev_close=73.0, change_pct=-1.37),
        ],
        "volatility": TickerChange(ticker="^VIX", last=14.5, prev_close=15.2, change_pct=None),
        "sectors": [
            TickerChange(ticker="XLK", label="Tech", last=200.0, prev_close=197.0, change_pct=1.52),
            TickerChange(
                ticker="XLE", label="Energy", last=85.0, prev_close=86.0, change_pct=-1.16
            ),
        ],
        "movers": _make_movers(),
        "fetched_at": _NOW,
    }
    defaults.update(overrides)
    return MarketMoversData(**defaults)


# ---------------------------------------------------------------------------
# TickerChange model tests
# ---------------------------------------------------------------------------


class TestTickerChangeModel:
    def test_basic(self) -> None:
        tc = TickerChange(ticker="SPY", last=520.0, prev_close=515.0, change_pct=0.97)
        assert tc.ticker == "SPY"
        assert tc.change_pct == 0.97
        assert tc.label is None
        assert tc.name is None

    def test_with_label_and_name(self) -> None:
        tc = TickerChange(
            ticker="XLK", label="Tech", name="Technology Select Sector", change_pct=1.5
        )
        assert tc.label == "Tech"
        assert tc.name == "Technology Select Sector"

    def test_optional_fields(self) -> None:
        tc = TickerChange(ticker="UNKNOWN")
        assert tc.last is None
        assert tc.prev_close is None
        assert tc.change_pct is None
        assert tc.label is None
        assert tc.name is None


class TestMarketMoversDataModel:
    def test_round_trip(self) -> None:
        data = _make_data()
        dumped = data.model_dump(mode="json")
        restored = MarketMoversData.model_validate(dumped)
        assert len(restored.equities) == 3
        assert restored.movers.gainers[0].ticker == "NVDA"

    def test_empty_movers(self) -> None:
        movers = MarketMovers(gainers=[], losers=[], most_actives=[], fetched_at=_NOW)
        data = _make_data(movers=movers)
        assert len(data.movers.gainers) == 0

    def test_no_volatility(self) -> None:
        data = _make_data(volatility=None)
        assert data.volatility is None


# ---------------------------------------------------------------------------
# snapshot.py tests
# ---------------------------------------------------------------------------


class TestFetchMarketMovers:
    @pytest.mark.asyncio
    async def test_fetches_quotes_and_movers(self) -> None:
        """fetch_market_movers returns structured data from YFinanceClient."""
        mock_redis = AsyncMock()

        mock_quote = MagicMock()
        mock_quote.name = "SPDR S&P 500"
        mock_quote.last = 520.0
        mock_quote.prev_close = 515.0

        mock_movers = _make_movers()

        with (
            patch("synesis.processing.market.snapshot.YFinanceClient") as MockClient,
        ):
            instance = MockClient.return_value
            instance.get_quote = AsyncMock(return_value=mock_quote)
            instance.get_market_movers = AsyncMock(return_value=mock_movers)

            data = await fetch_market_movers(mock_redis)

        assert len(data.equities) == 3
        assert len(data.rates_fx) == 2
        assert len(data.commodities) == 2
        assert data.volatility is not None
        assert data.volatility.ticker == "^VIX"
        assert data.equities[0].name == "SPDR S&P 500"
        assert len(data.sectors) == 11  # All sector tickers
        assert data.movers.gainers[0].ticker == "NVDA"

    @pytest.mark.asyncio
    async def test_movers_failure_graceful(self) -> None:
        """If get_market_movers fails, data still returns with empty movers."""
        mock_redis = AsyncMock()

        mock_quote = MagicMock()
        mock_quote.name = "Test"
        mock_quote.last = 100.0
        mock_quote.prev_close = 99.0

        with patch("synesis.processing.market.snapshot.YFinanceClient") as MockClient:
            instance = MockClient.return_value
            instance.get_quote = AsyncMock(return_value=mock_quote)
            instance.get_market_movers = AsyncMock(side_effect=Exception("screener down"))

            data = await fetch_market_movers(mock_redis)

        assert data.movers.gainers == []
        assert data.movers.losers == []
        assert data.movers.most_actives == []

    @pytest.mark.asyncio
    async def test_quote_failure_graceful(self) -> None:
        """If individual quote fails, it returns None values."""
        mock_redis = AsyncMock()

        async def _mock_get_quote(ticker: str) -> MagicMock:
            if ticker == "SPY":
                raise Exception("timeout")
            q = MagicMock()
            q.name = "Test ETF"
            q.last = 100.0
            q.prev_close = 99.0
            return q

        mock_movers = _make_movers(gainers=[], losers=[], most_actives=[])

        with patch("synesis.processing.market.snapshot.YFinanceClient") as MockClient:
            instance = MockClient.return_value
            instance.get_quote = AsyncMock(side_effect=_mock_get_quote)
            instance.get_market_movers = AsyncMock(return_value=mock_movers)

            data = await fetch_market_movers(mock_redis)

        # SPY should have None values
        spy = next(tc for tc in data.equities if tc.ticker == "SPY")
        assert spy.last is None
        assert spy.name is None
        assert spy.change_pct is None

    @pytest.mark.asyncio
    async def test_sectors_sorted_by_performance(self) -> None:
        """Sectors should be sorted best-to-worst performance."""
        mock_redis = AsyncMock()

        # Map tickers to different returns
        perf_map = {
            "XLK": (200.0, 195.0),  # +2.56%
            "XLE": (85.0, 88.0),  # -3.41%
            "XLF": (40.0, 39.0),  # +2.56%
        }

        async def _mock_get_quote(ticker: str) -> MagicMock:
            q = MagicMock()
            q.name = f"{ticker} ETF"
            if ticker in perf_map:
                q.last, q.prev_close = perf_map[ticker]
            else:
                q.last = 100.0
                q.prev_close = 100.0
            return q

        mock_movers = _make_movers(gainers=[], losers=[], most_actives=[])

        with patch("synesis.processing.market.snapshot.YFinanceClient") as MockClient:
            instance = MockClient.return_value
            instance.get_quote = AsyncMock(side_effect=_mock_get_quote)
            instance.get_market_movers = AsyncMock(return_value=mock_movers)

            data = await fetch_market_movers(mock_redis)

        # XLE should be last (worst perf)
        sector_tickers = [s.ticker for s in data.sectors]
        assert sector_tickers.index("XLE") > sector_tickers.index("XLK")


# ---------------------------------------------------------------------------
# discord_format.py tests
# ---------------------------------------------------------------------------


class TestFormatTickerLines:
    def test_includes_price_and_pct(self) -> None:
        tickers = [
            TickerChange(ticker="SPY", last=520.0, change_pct=1.2),
        ]
        result = _format_ticker_lines(tickers)
        assert "`$SPY`" in result
        assert "**$520.00**" in result
        assert "+1.2%" in result

    def test_multiple_tickers_on_separate_lines(self) -> None:
        tickers = [
            TickerChange(ticker="SPY", last=520.0, change_pct=1.2),
            TickerChange(ticker="QQQ", last=445.0, change_pct=-0.5),
        ]
        result = _format_ticker_lines(tickers)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "SPY" in lines[0]
        assert "QQQ" in lines[1]

    def test_none_values_skipped(self) -> None:
        tickers = [
            TickerChange(ticker="SPY", last=None, change_pct=None),
            TickerChange(ticker="QQQ", last=445.0, change_pct=1.0),
        ]
        result = _format_ticker_lines(tickers)
        assert "SPY" not in result
        assert "QQQ" in result

    def test_price_only_no_pct(self) -> None:
        tickers = [TickerChange(ticker="SPY", last=520.0, change_pct=None)]
        result = _format_ticker_lines(tickers)
        assert "**$520.00**" in result
        assert "%" not in result

    def test_empty_list(self) -> None:
        assert _format_ticker_lines([]) == ""


class TestFormatSectorLines:
    def test_includes_label_price_pct(self) -> None:
        sectors = [
            TickerChange(ticker="XLK", label="Tech", last=200.0, change_pct=1.5),
        ]
        result = _format_sector_lines(sectors)
        assert "Tech" in result
        assert "`$XLK`" in result
        assert "**$200.00**" in result
        assert "+1.5%" in result

    def test_each_sector_own_row(self) -> None:
        sectors = [
            TickerChange(ticker="XLK", label="Tech", last=200.0, change_pct=1.5),
            TickerChange(ticker="XLE", label="Energy", last=85.0, change_pct=-1.2),
            TickerChange(ticker="XLF", label="Financials", last=40.0, change_pct=0.3),
        ]
        result = _format_sector_lines(sectors)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "Tech" in lines[0]
        assert "Energy" in lines[1]
        assert "Financials" in lines[2]

    def test_empty_list(self) -> None:
        assert _format_sector_lines([]) == ""


class TestFormatMovers:
    def test_includes_name_price_sector(self) -> None:
        movers = [
            MarketMover(
                ticker="NVDA", name="NVIDIA Corp", price=950.0, change_pct=8.5, sector="Technology"
            ),
        ]
        result = _format_movers(movers)
        assert "`$NVDA`" in result
        assert "NVIDIA Corp" in result
        assert "**$950.00**" in result
        assert "+8.5%" in result
        assert "(Technology)" in result

    def test_no_name(self) -> None:
        movers = [MarketMover(ticker="ABC", price=10.0, change_pct=2.0, sector="Tech")]
        result = _format_movers(movers)
        assert "`$ABC`" in result
        assert "**$10.00**" in result
        # No stray empty name
        assert "  " not in result or "`$ABC` **$10.00**" in result

    def test_no_sector(self) -> None:
        movers = [
            MarketMover(ticker="ABC", name="ABC Inc", price=10.0, change_pct=2.0, sector=None)
        ]
        result = _format_movers(movers)
        assert "`$ABC`" in result
        assert "()" not in result

    def test_no_price(self) -> None:
        movers = [MarketMover(ticker="ABC", name="ABC Inc", price=None, change_pct=2.0)]
        result = _format_movers(movers)
        assert "`$ABC`" in result
        assert "ABC Inc" in result
        assert "+2.0%" in result
        assert "$" not in result.split("+2.0%")[0].split("ABC Inc")[1]

    def test_none_change_pct_skipped(self) -> None:
        movers = [MarketMover(ticker="XYZ", change_pct=None)]
        result = _format_movers(movers)
        assert result == ""

    def test_empty_list(self) -> None:
        assert _format_movers([]) == ""


class TestFormatMarketMoversEmbeds:
    def test_returns_single_message(self) -> None:
        data = _make_data()
        messages = format_market_movers_embeds(data)
        assert len(messages) >= 1
        assert len(messages[0]) == 1  # one embed per message

    def test_embed_structure(self) -> None:
        data = _make_data()
        messages = format_market_movers_embeds(data)
        embed = messages[0][0]
        assert "Market Movers" in embed["title"]
        assert "color" in embed
        assert "fields" in embed
        assert "timestamp" in embed

    def test_footer_on_last_message(self) -> None:
        data = _make_data()
        messages = format_market_movers_embeds(data)
        last_embed = messages[-1][0]
        assert last_embed["footer"]["text"] == "Synesis Market Movers"

    def test_equity_field_has_prices(self) -> None:
        data = _make_data()
        messages = format_market_movers_embeds(data)
        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        eq_field = next(f for f in all_fields if "Equities" in f["name"])
        assert "**$520.00**" in eq_field["value"]
        assert "+1.0%" in eq_field["value"]

    def test_sector_field_has_rows(self) -> None:
        data = _make_data()
        messages = format_market_movers_embeds(data)
        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        sector_field = next(f for f in all_fields if "Sectors" in f["name"])
        lines = sector_field["value"].strip().split("\n")
        assert len(lines) == 2  # Tech and Energy
        assert "Tech" in lines[0]
        assert "**$200.00**" in lines[0]
        assert "Energy" in lines[1]

    def test_volatility_field_has_price(self) -> None:
        data = _make_data()
        messages = format_market_movers_embeds(data)
        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        vol_field = next(f for f in all_fields if "Volatility" in f["name"])
        assert "**14.50**" in vol_field["value"]
        assert "(-0.70)" in vol_field["value"]

    def test_no_volatility_when_none(self) -> None:
        data = _make_data(volatility=None)
        messages = format_market_movers_embeds(data)
        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        field_names = [f["name"] for f in all_fields]
        assert not any("Volatility" in name for name in field_names)

    def test_movers_have_company_name(self) -> None:
        data = _make_data()
        messages = format_market_movers_embeds(data)
        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        gainers_field = next(f for f in all_fields if "Gainers" in f["name"])
        assert "NVIDIA Corp" in gainers_field["value"]
        assert "**$950.00**" in gainers_field["value"]
        assert "(Technology)" in gainers_field["value"]

    def test_no_movers_when_empty(self) -> None:
        movers = MarketMovers(gainers=[], losers=[], most_actives=[], fetched_at=_NOW)
        data = _make_data(movers=movers)
        messages = format_market_movers_embeds(data)
        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        field_names = [f["name"] for f in all_fields]
        assert not any("Gainers" in name for name in field_names)
        assert not any("Losers" in name for name in field_names)

    def test_field_values_truncated_to_1024(self) -> None:
        """Field values exceeding 1024 chars should be truncated."""
        huge_movers = [
            MarketMover(
                ticker=f"T{i:03d}",
                name="A" * 50,
                price=100.0,
                change_pct=float(i),
                sector="A" * 100,
            )
            for i in range(20)
        ]
        movers = _make_movers(gainers=huge_movers, losers=[], most_actives=[])
        data = _make_data(movers=movers)
        messages = format_market_movers_embeds(data)
        for msg in messages:
            for embed in msg:
                for field in embed.get("fields", []):
                    assert len(field["value"]) <= 1024

    def test_splits_when_exceeding_embed_limit(self) -> None:
        """Should split into multiple messages if total embed size > 6000."""
        from synesis.processing.market.discord_format import _embed_size, _split_into_messages

        # Directly test the splitter with fields that exceed 6000 chars total
        fields = [
            {"name": f"Field {i}", "value": "x" * 900, "inline": False}
            for i in range(8)  # 8 x ~906 = ~7248 chars in fields alone
        ]
        messages = _split_into_messages(fields, "Test Title", "2026-01-01T00:00:00Z")

        assert len(messages) >= 2
        assert "title" in messages[0][0]
        assert "footer" in messages[-1][0]
        for msg in messages[:-1]:
            assert "footer" not in msg[0]

        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        assert len(all_fields) == 8

        for msg in messages:
            for embed in msg:
                assert _embed_size(embed) <= 6000

    def test_each_embed_under_6000_chars(self) -> None:
        """Each individual embed should stay under Discord's 6000-char limit."""
        from synesis.processing.market.discord_format import _embed_size

        big_movers = [
            MarketMover(
                ticker=f"TICK{i}",
                name=f"Company {i} Inc",
                price=100.0 + i,
                change_pct=float(i),
                sector="Communication Services And Technology",
            )
            for i in range(25)
        ]
        movers = _make_movers(
            gainers=big_movers[:10],
            losers=big_movers[10:20],
            most_actives=big_movers[20:25],
        )
        data = _make_data(movers=movers)
        messages = format_market_movers_embeds(data)
        for msg in messages:
            for embed in msg:
                assert _embed_size(embed) <= 6000

    def test_each_embed_under_25_fields(self) -> None:
        """Each individual embed should have at most 25 fields."""
        big_movers = [
            MarketMover(
                ticker=f"T{i:03d}",
                change_pct=float(i),
                sector="Sector",
            )
            for i in range(25)
        ]
        movers = _make_movers(
            gainers=big_movers[:10],
            losers=big_movers[10:20],
            most_actives=big_movers[20:25],
        )
        data = _make_data(movers=movers)
        messages = format_market_movers_embeds(data)
        for msg in messages:
            for embed in msg:
                assert len(embed.get("fields", [])) <= 25

    def test_volatility_no_prev_close(self) -> None:
        """VIX with last but no prev_close should still show."""
        data = _make_data(volatility=TickerChange(ticker="^VIX", last=16.0, prev_close=None))
        messages = format_market_movers_embeds(data)
        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        vol_field = next(f for f in all_fields if "Volatility" in f["name"])
        assert "**16.00**" in vol_field["value"]
        assert "(" not in vol_field["value"]  # no change shown

    def test_volatility_last_none(self) -> None:
        """VIX with None last should not appear."""
        data = _make_data(volatility=TickerChange(ticker="^VIX", last=None, prev_close=15.0))
        messages = format_market_movers_embeds(data)
        all_fields = [f for msg in messages for e in msg for f in e.get("fields", [])]
        field_names = [f["name"] for f in all_fields]
        assert not any("Volatility" in name for name in field_names)

    def test_split_embeds_continuation_has_no_title(self) -> None:
        """When embeds split, only the first embed gets a title."""
        from synesis.processing.market.discord_format import _split_into_messages

        fields = [{"name": f"Field {i}", "value": "x" * 900, "inline": False} for i in range(8)]
        messages = _split_into_messages(fields, "Test Title", "2026-01-01T00:00:00Z")

        assert len(messages) >= 2
        assert "title" in messages[0][0]
        for msg in messages[1:]:
            assert "title" not in msg[0]


# ---------------------------------------------------------------------------
# job.py tests
# ---------------------------------------------------------------------------


class TestMarketMoversJob:
    @pytest.mark.asyncio
    async def test_job_sends_to_discord(self) -> None:
        from synesis.processing.market.job import market_movers_job

        data = _make_data()
        mock_redis = AsyncMock()

        with (
            patch(
                "synesis.processing.market.job.fetch_market_movers",
                new=AsyncMock(return_value=data),
            ),
            patch(
                "synesis.processing.market.job.send_discord",
                new=AsyncMock(return_value=True),
            ) as mock_send,
            patch("synesis.processing.market.job.get_settings") as mock_settings,
        ):
            mock_settings.return_value.discord_brief_webhook_url = MagicMock()
            mock_settings.return_value.discord_webhook_url = None
            await market_movers_job(mock_redis)

        mock_send.assert_called_once()
        embeds = mock_send.call_args[0][0]
        assert any("Market Movers" in str(e.get("title", "")) for e in embeds)

    @pytest.mark.asyncio
    async def test_job_multi_message_sends_each(self) -> None:
        """When format_market_movers_embeds returns 2+ messages, each is sent."""
        from synesis.processing.market.job import market_movers_job

        data = _make_data()
        mock_redis = AsyncMock()
        two_messages = [[{"title": "msg1"}], [{"title": "msg2"}]]

        with (
            patch(
                "synesis.processing.market.job.fetch_market_movers",
                new=AsyncMock(return_value=data),
            ),
            patch(
                "synesis.processing.market.job.format_market_movers_embeds",
                return_value=two_messages,
            ),
            patch(
                "synesis.processing.market.job.send_discord",
                new=AsyncMock(return_value=True),
            ) as mock_send,
            patch("synesis.processing.market.job.get_settings") as mock_settings,
        ):
            mock_settings.return_value.discord_brief_webhook_url = MagicMock()
            mock_settings.return_value.discord_webhook_url = None
            await market_movers_job(mock_redis)

        assert mock_send.call_count == 2

    @pytest.mark.asyncio
    async def test_job_no_webhook_skips(self) -> None:
        from synesis.processing.market.job import market_movers_job

        mock_redis = AsyncMock()

        with (
            patch("synesis.processing.market.job.get_settings") as mock_settings,
            patch(
                "synesis.processing.market.job.send_discord",
                new=AsyncMock(),
            ) as mock_send,
        ):
            mock_settings.return_value.discord_brief_webhook_url = None
            mock_settings.return_value.discord_webhook_url = None
            await market_movers_job(mock_redis)

        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_job_falls_back_to_default_webhook(self) -> None:
        from synesis.processing.market.job import market_movers_job

        data = _make_data()
        mock_redis = AsyncMock()
        fallback_webhook = MagicMock()

        with (
            patch(
                "synesis.processing.market.job.fetch_market_movers",
                new=AsyncMock(return_value=data),
            ),
            patch(
                "synesis.processing.market.job.send_discord",
                new=AsyncMock(return_value=True),
            ) as mock_send,
            patch("synesis.processing.market.job.get_settings") as mock_settings,
        ):
            mock_settings.return_value.discord_brief_webhook_url = None
            mock_settings.return_value.discord_webhook_url = fallback_webhook
            await market_movers_job(mock_redis)

        mock_send.assert_called_once()
        assert mock_send.call_args[1]["webhook_url_override"] == fallback_webhook


# ---------------------------------------------------------------------------
# Scheduler registration test
# ---------------------------------------------------------------------------


class TestSchedulerRegistration:
    def test_market_movers_job_in_scheduler(self) -> None:
        """market_movers_job is exported from scheduler module."""
        from synesis.agent.scheduler import market_movers_job

        assert callable(market_movers_job)

    def test_market_movers_in_init(self) -> None:
        """market_movers_job is re-exported from processing.market."""
        from synesis.processing.market import market_movers_job

        assert callable(market_movers_job)


# ---------------------------------------------------------------------------
# API route tests for POST /market/movers
# ---------------------------------------------------------------------------


class TestMarketMoversAPI:
    @pytest.fixture()
    def _app(self):
        """Minimal FastAPI app with market router and overridden deps."""
        from dataclasses import dataclass, field as dfield

        from fastapi import FastAPI

        from synesis.api.router import api_router
        from synesis.core.dependencies import get_agent_state
        from synesis.storage.redis import get_redis

        @dataclass
        class _State:
            redis: Any = None
            db: Any = None
            settings: Any = None
            db_enabled: bool = True
            scheduler: Any = None
            trigger_fns: dict[str, Any] = dfield(default_factory=dict)

        self._state = _State(redis=AsyncMock())
        app = FastAPI()
        app.include_router(api_router, prefix="/api/v1")
        app.dependency_overrides[get_agent_state] = lambda: self._state
        app.dependency_overrides[get_redis] = lambda: AsyncMock()
        return app

    @pytest.fixture()
    async def client(self, _app):
        import httpx

        transport = httpx.ASGITransport(app=_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_trigger_happy_path(self, _app, client) -> None:
        """POST /market/movers returns triggered when trigger_fn exists."""
        self._state.trigger_fns["market_movers"] = AsyncMock()
        r = await client.post("/api/v1/market/movers")
        assert r.status_code == 200
        assert r.json()["status"] == "triggered"

    @pytest.mark.asyncio
    async def test_trigger_503_when_missing(self, _app, client) -> None:
        """POST /market/movers returns 503 when trigger is not configured."""
        self._state.trigger_fns.clear()
        r = await client.post("/api/v1/market/movers")
        assert r.status_code == 503
