"""Market snapshot — fetch benchmark/sector quotes and top movers."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from synesis.core.constants import BENCHMARK_TICKERS, SECTOR_LABELS, SECTOR_TICKERS
from synesis.core.logging import get_logger
from synesis.processing.market.models import MarketMoversData, TickerChange
from synesis.providers.yfinance.client import YFinanceClient
from synesis.providers.yfinance.models import MarketMovers

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


async def fetch_market_movers(redis: Redis) -> MarketMoversData:
    """Fetch benchmark + sector quotes and top movers.

    Returns structured MarketMoversData with all market snapshot data.
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

    return MarketMoversData(
        equities=[_make_tc(t) for t in ["SPY", "QQQ", "IWM"]],
        rates_fx=[_make_tc(t) for t in ["TLT", "UUP"]],
        commodities=[_make_tc(t) for t in ["GLD", "USO"]],
        volatility=vix_tc,
        sectors=[_make_tc(t) for t in sorted_sectors],
        movers=movers,
        fetched_at=datetime.now(UTC),
    )
