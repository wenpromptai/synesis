"""Market snapshot — fetch benchmark/sector quotes and top movers.

Extracted from events/digest.py:_get_market_data().
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from synesis.core.constants import BENCHMARK_TICKERS, SECTOR_LABELS, SECTOR_TICKERS
from synesis.core.logging import get_logger
from synesis.processing.market.models import MarketBriefData, TickerChange
from synesis.providers.yfinance.client import YFinanceClient
from synesis.providers.yfinance.models import MarketMover, MarketMovers

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


async def fetch_market_brief(redis: Redis) -> MarketBriefData:
    """Fetch benchmark + sector quotes and top movers.

    Returns structured MarketBriefData with all market snapshot data.
    """
    client = YFinanceClient(redis=redis)
    sem = asyncio.Semaphore(10)

    async def _fetch_one(
        ticker: str,
    ) -> tuple[str, str | None, float | None, float | None]:
        async with sem:
            try:
                quote = await client.get_quote(ticker)
                return (ticker, quote.name, quote.last, quote.prev_close)
            except Exception:
                logger.warning("Market data fetch failed", ticker=ticker, exc_info=True)
                return (ticker, None, None, None)

    # Fetch all quotes concurrently
    all_tickers = BENCHMARK_TICKERS + SECTOR_TICKERS
    results = await asyncio.gather(*[_fetch_one(t) for t in all_tickers])
    quotes = {t: (name, last, prev) for t, name, last, prev in results}

    def _pct(last: float | None, prev: float | None) -> float | None:
        if last is None or prev is None or prev == 0:
            return None
        return ((last - prev) / prev) * 100

    def _make_tc(ticker: str) -> TickerChange:
        name, last, prev = quotes[ticker]
        return TickerChange(
            ticker=ticker,
            name=name,
            label=SECTOR_LABELS.get(ticker),
            last=last,
            prev_close=prev,
            change_pct=_pct(last, prev),
        )

    # Sort sectors by performance (best first)
    def _sector_sort_key(ticker: str) -> float:
        _name, last, prev = quotes[ticker]
        pct = _pct(last, prev)
        return pct if pct is not None else -999.0

    sorted_sectors = sorted(SECTOR_TICKERS, key=_sector_sort_key, reverse=True)

    # VIX
    vix_tc = _make_tc("^VIX")

    # Fetch top movers
    try:
        movers = await client.get_market_movers()
    except Exception:
        logger.warning("Market movers fetch failed", exc_info=True)
        movers = MarketMovers(gainers=[], losers=[], most_actives=[], fetched_at=datetime.now(UTC))

    return MarketBriefData(
        equities=[_make_tc(t) for t in ["SPY", "QQQ", "IWM"]],
        rates_fx=[_make_tc(t) for t in ["TLT", "UUP"]],
        commodities=[_make_tc(t) for t in ["GLD", "USO"]],
        volatility=vix_tc,
        sectors=[_make_tc(t) for t in sorted_sectors],
        movers=movers,
        fetched_at=datetime.now(UTC),
    )


def _fmt_mover_section(movers_list: list[MarketMover], label: str) -> str | None:
    """Format a single movers category (gainers/losers/actives) for LLM text."""
    if not movers_list:
        return None
    parts = [
        f"{m.ticker} {m.change_pct:+.1f}%{f' ({m.sector})' if m.sector else ''}"
        for m in movers_list[:10]
        if m.change_pct is not None
    ]
    return f"{label}: {', '.join(parts)}" if parts else None


def format_market_data_for_llm(brief: MarketBriefData) -> tuple[str, str]:
    """Format MarketBriefData as text for LLM prompt.

    Returns (llm_prompt_text, formatted_sector_string) — same contract as
    the old events/digest.py:_get_market_data().
    """

    def _fmt(tc: TickerChange) -> str | None:
        if tc.ticker == "^VIX":
            if tc.last is None or tc.prev_close is None:
                return None
            change = tc.last - tc.prev_close
            sign = "+" if change >= 0 else ""
            return f"VIX {tc.last:.1f} ({sign}{change:.1f})"
        if tc.change_pct is None:
            return None
        sign = "+" if tc.change_pct >= 0 else ""
        if tc.label:
            return f"{tc.label}: {tc.ticker} {sign}{tc.change_pct:.1f}%"
        return f"{tc.ticker} {sign}{tc.change_pct:.1f}%"

    def _collect(tickers: list[TickerChange]) -> list[str]:
        return [s for tc in tickers if (s := _fmt(tc)) is not None]

    equity_parts = _collect(brief.equities)
    rates_parts = _collect(brief.rates_fx)
    comm_parts = _collect(brief.commodities)
    vix_part = _fmt(brief.volatility) if brief.volatility else None
    sector_parts = _collect(brief.sectors)

    lines = ["## MARKET DATA (yesterday's close)"]
    if equity_parts:
        lines.append(f"Equities: {', '.join(equity_parts)}")
    if rates_parts:
        lines.append(f"Rates/FX: {', '.join(rates_parts)}")
    if comm_parts:
        lines.append(f"Commodities: {', '.join(comm_parts)}")
    if vix_part:
        lines.append(f"Volatility: {vix_part}")
    sector_display = ", ".join(sector_parts)
    if sector_parts:
        lines.append(f"Sectors: {sector_display}")

    # Append top movers section for LLM context
    movers = brief.movers
    if movers.gainers or movers.losers or movers.most_actives:
        lines.append("")
        lines.append("## TOP MOVERS")
        for label, mover_list in [
            ("Gainers", movers.gainers),
            ("Losers", movers.losers),
            ("Most Active", movers.most_actives),
        ]:
            line = _fmt_mover_section(mover_list, label)
            if line:
                lines.append(line)

    return "\n".join(lines), sector_display
