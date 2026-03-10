"""yfinance API endpoints — quotes, history, FX rates, and options."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from starlette.requests import Request

from synesis.core.dependencies import YFinanceClientDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter

logger = get_logger(__name__)

router = APIRouter()


@router.get("/quote/{ticker}")
@limiter.limit("30/minute")
async def get_quote(
    request: Request,
    ticker: str,
    client: YFinanceClientDep,
) -> dict[str, Any]:
    """Snapshot quote (last, prev_close, open, high, low, market cap, moving averages)."""
    quote = await client.get_quote(ticker)
    return quote.model_dump(mode="json")


@router.get("/history/{ticker}")
@limiter.limit("30/minute")
async def get_history(
    request: Request,
    ticker: str,
    client: YFinanceClientDep,
    period: str = Query(
        "1mo", description="Period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"
    ),
    interval: str = Query(
        "1d", description="Interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo"
    ),
) -> dict[str, Any]:
    """OHLCV history bars."""
    bars = await client.get_history(ticker, period=period, interval=interval)
    return {
        "ticker": ticker.upper(),
        "period": period,
        "interval": interval,
        "bars": [b.model_dump(mode="json") for b in bars],
        "count": len(bars),
    }


@router.get("/fx/{pair}")
@limiter.limit("30/minute")
async def get_fx_rate(
    request: Request,
    pair: str,
    client: YFinanceClientDep,
) -> dict[str, Any]:
    """FX spot rate (e.g. pair=EURUSD=X)."""
    rate = await client.get_fx_rate(pair)
    return rate.model_dump(mode="json")


@router.get("/options/{ticker}/expirations")
@limiter.limit("30/minute")
async def get_options_expirations(
    request: Request,
    ticker: str,
    client: YFinanceClientDep,
) -> dict[str, Any]:
    """List available options expiration dates."""
    expirations = await client.get_options_expirations(ticker)
    return {
        "ticker": ticker.upper(),
        "expirations": expirations,
        "count": len(expirations),
    }


@router.get("/options/{ticker}/chain")
@limiter.limit("10/minute")
async def get_options_chain(
    request: Request,
    ticker: str,
    client: YFinanceClientDep,
    expiration: str = Query(..., description="Expiration date (YYYY-MM-DD)"),
    greeks: bool = Query(False, description="Compute Black-Scholes Greeks"),
) -> dict[str, Any]:
    """Full options chain (calls + puts) for a given expiration."""
    chain = await client.get_options_chain(ticker, expiration=expiration, greeks=greeks)
    return chain.model_dump(mode="json")


@router.get("/options/{ticker}/snapshot")
@limiter.limit("10/minute")
async def get_options_snapshot(
    request: Request,
    ticker: str,
    client: YFinanceClientDep,
    greeks: bool = Query(True, description="Compute Black-Scholes Greeks"),
) -> dict[str, Any]:
    """Options snapshot: spot, 30d realized vol, nearest valid expiry, ATM ±10 strikes."""
    snapshot = await client.get_options_snapshot(ticker, greeks=greeks)
    return snapshot.model_dump(mode="json")


@router.get("/movers")
@limiter.limit("10/minute")
async def get_market_movers(
    request: Request,
    client: YFinanceClientDep,
    size: int = Query(25, ge=1, le=50, description="Number of results per category"),
) -> dict[str, Any]:
    """Top market movers: gainers, losers, and most-actives with sector/industry."""
    movers = await client.get_market_movers(size=size)
    return movers.model_dump(mode="json")
