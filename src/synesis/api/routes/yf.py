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
    """Snapshot quote for a single ticker via yfinance.

    **Path params:**
    - `ticker` (str): equity/ETF symbol, case-insensitive (e.g. `NVDA`).

    **Returns:** `EquityQuote` — `ticker`, `name`, `currency`, `exchange`,
    `last`, `prev_close`, `open`, `high`, `low`, `volume`, `market_cap`,
    `avg_50d`, `avg_200d`. All numeric fields can be null.

    **Example:** `curl http://localhost:7337/api/v1/yf/quote/NVDA`
    """
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
    """OHLCV history bars from yfinance.

    **Path params:**
    - `ticker` (str): symbol.

    **Query params:**
    - `period` (str, default `1mo`): `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`,
      `2y`, `5y`, `10y`, `ytd`, `max`.
    - `interval` (str, default `1d`): `1m`, `2m`, `5m`, `15m`, `30m`, `60m`,
      `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`. Sub-day intervals are only
      available for short periods (yfinance limitation).

    **Returns:**
    - `ticker` (str), `period` (str), `interval` (str)
    - `bars` (list[`OHLCBar`]): each `{date, open, high, low, close, volume}`.
    - `count` (int)

    **Example:** `curl 'http://localhost:7337/api/v1/yf/history/AAPL?period=3mo&interval=1d'`
    """
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
    """FX spot rate for a yfinance pair symbol.

    **Path params:**
    - `pair` (str): yfinance FX symbol — typically `XXXYYY=X` (e.g. `EURUSD=X`,
      `GBPUSD=X`, `USDJPY=X`).

    **Returns:** `FXRate` — `pair`, `rate`, `bid`, `ask` (numeric fields nullable).

    **Example:** `curl http://localhost:7337/api/v1/yf/fx/EURUSD=X`
    """
    rate = await client.get_fx_rate(pair)
    return rate.model_dump(mode="json")


@router.get("/options/{ticker}/expirations")
@limiter.limit("30/minute")
async def get_options_expirations(
    request: Request,
    ticker: str,
    client: YFinanceClientDep,
) -> dict[str, Any]:
    """Available options expiration dates for a ticker.

    **Path params:**
    - `ticker` (str): symbol.

    **Returns:**
    - `ticker` (str), `expirations` (list[str], YYYY-MM-DD), `count` (int).

    **Example:** `curl http://localhost:7337/api/v1/yf/options/SPY/expirations`
    """
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
    """Full options chain (calls + puts) for one expiration.

    **Path params:**
    - `ticker` (str).

    **Query params:**
    - `expiration` (YYYY-MM-DD, required): one of the dates from `/expirations`.
    - `greeks` (bool, default `false`): when true, computes Black–Scholes
      delta/gamma/theta/vega/rho per leg using risk-free rate + 30d realized vol.

    **Returns:** `OptionsChain` —
    - `ticker` (str), `expiration` (str, YYYY-MM-DD),
    - `calls` (list[`OptionsContract`]), `puts` (list[`OptionsContract`]).
      Each contract: `contract_symbol`, `strike`, `last_price`, `bid`, `ask`,
      `volume`, `open_interest`, `implied_volatility`, `in_the_money`,
      `greeks` (object with `delta, gamma, theta, vega, rho, implied_volatility`,
      populated only when `greeks=true`).

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/yf/options/AAPL/chain?expiration=2026-06-19&greeks=true'
    ```
    """
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
    """Compact options snapshot at the nearest valid expiration.

    Designed for the intelligence pipeline — picks the soonest expiration and
    returns the ATM-region chain. Lighter payload than `/chain`.

    **Path params:**
    - `ticker` (str).

    **Query params:**
    - `greeks` (bool, default `true`): include Black–Scholes Greeks per leg.

    **Returns:** `OptionsSnapshot` —
    - `ticker` (str), `spot` (float | null), `realized_vol_30d` (float | null),
    - `expiration` (str), `days_to_expiry` (int),
    - `calls` (list[`OptionsContract`]), `puts` (list[`OptionsContract`]).

    **Example:** `curl http://localhost:7337/api/v1/yf/options/NVDA/snapshot`
    """
    snapshot = await client.get_options_snapshot(ticker, greeks=greeks)
    return snapshot.model_dump(mode="json")


@router.get("/analyst/{ticker}")
@limiter.limit("30/minute")
async def get_analyst_ratings(
    request: Request,
    ticker: str,
    client: YFinanceClientDep,
    limit: int = Query(25, ge=1, le=100, description="Max upgrade/downgrade records"),
) -> dict[str, Any]:
    """Analyst ratings rollup for a ticker.

    Aggregates yfinance recommendation data — current consensus + recent rating
    actions + price target distribution.

    **Path params:**
    - `ticker` (str).

    **Query params:**
    - `limit` (int, 1–100, default 25): max upgrade/downgrade history rows.

    **Returns:** `AnalystRatings` —
    - `ticker` (str).
    - `recommendations` (list[`RecommendationTrend`]): monthly aggregated
      counts — each `{period, strong_buy, buy, hold, sell, strong_sell}`.
    - `upgrades_downgrades` (list[`UpgradeDowngrade`]): each
      `{date, firm, to_grade, from_grade, action,
       price_target_action, current_price_target, prior_price_target}`.
      Capped by the `limit` query param.
    - `price_targets` (`AnalystPriceTargets` | null):
      `{current, high, low, mean, median}`.

    **Example:** `curl http://localhost:7337/api/v1/yf/analyst/NVDA`
    """
    ratings = await client.get_analyst_ratings(ticker, limit=limit)
    return ratings.model_dump(mode="json")


@router.get("/movers")
@limiter.limit("10/minute")
async def get_market_movers(
    request: Request,
    client: YFinanceClientDep,
    size: int = Query(25, ge=1, le=50, description="Number of results per category"),
) -> dict[str, Any]:
    """Top US market movers — gainers, losers, most actives.

    **Query params:**
    - `size` (int, 1–50, default 25): items per category.

    **Returns:** `MarketMovers` —
    - `gainers` (list[`MarketMover`]), `losers` (list[`MarketMover`]),
      `most_actives` (list[`MarketMover`]),
    - `fetched_at` (datetime).
      Each `MarketMover`: `ticker, name, price, change_pct, change_abs,
      volume, avg_volume_3m, volume_ratio, market_cap, sector, industry`.

    **Example:** `curl 'http://localhost:7337/api/v1/yf/movers?size=10'`
    """
    movers = await client.get_market_movers(size=size)
    return movers.model_dump(mode="json")
