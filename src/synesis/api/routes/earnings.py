"""Earnings calendar API endpoints (NASDAQ data)."""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Query
from starlette.requests import Request

from synesis.core.dependencies import DbDep, NasdaqClientDep
from synesis.core.logging import get_logger
from synesis.core.rate_limit import limiter
from synesis.processing.common.watchlist import WatchlistManager

logger = get_logger(__name__)

router = APIRouter()


@router.get("/calendar")
@limiter.limit("30/minute")
async def get_earnings_calendar(
    request: Request,
    client: NasdaqClientDep,
    target_date: date = Query(
        default_factory=date.today,
        alias="date",
        description="Date to get earnings for (YYYY-MM-DD)",
    ),
) -> dict[str, Any]:
    """All earnings reports scheduled on a given date (NASDAQ).

    Pulls from the NASDAQ public earnings calendar — every company reporting
    on the date, not filtered by watchlist.

    **Query params:**
    - `date` (YYYY-MM-DD, default = today): the calendar day to fetch.

    **Returns:**
    - `date` (str): echoed ISO date.
    - `earnings` (list[`EarningsEvent`]): each event has
      `ticker`, `company_name`, `earnings_date`, `time`
      (`pre-market` | `after-hours` | `during-market`),
      `eps_forecast`, `num_estimates`, `market_cap`, `fiscal_quarter`.
    - `count` (int): number of events.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/earnings/calendar?date=2026-05-08'
    ```
    """
    events = await client.get_earnings_by_date(target_date)
    return {
        "date": target_date.isoformat(),
        "earnings": [e.model_dump(mode="json") for e in events],
        "count": len(events),
    }


@router.get("/upcoming")
@limiter.limit("30/minute")
async def get_upcoming_earnings(
    request: Request,
    client: NasdaqClientDep,
    db: DbDep,
    days: int = Query(14, ge=1, le=90, description="Days to look ahead"),
) -> dict[str, Any]:
    """Upcoming earnings for the user's watchlist tickers.

    Resolves the active watchlist from the DB, then asks NASDAQ for any
    earnings within the next `days` window.

    **Query params:**
    - `days` (int, 1–90, default 14): lookahead window in days.

    **Returns:**
    - `tickers_checked` (int): watchlist size at fetch time.
    - `days` (int): echoed window.
    - `earnings` (list[`EarningsEvent`]): one entry per upcoming report.
      Same shape as `/calendar` (ticker, company_name, earnings_date, time, …).
    - `count` (int): number of events found.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/earnings/upcoming?days=7'
    ```

    **Empty case:** if the watchlist is empty, returns
    `{"tickers_checked": 0, "earnings": [], "count": 0}` without hitting NASDAQ.
    """
    watchlist = WatchlistManager(db)
    tickers = await watchlist.get_all()
    if not tickers:
        return {"tickers_checked": 0, "earnings": [], "count": 0}

    events = await client.get_upcoming_earnings(tickers, days=days)
    return {
        "tickers_checked": len(tickers),
        "days": days,
        "earnings": [e.model_dump(mode="json") for e in events],
        "count": len(events),
    }


@router.get("/upcoming/{ticker}")
@limiter.limit("30/minute")
async def get_upcoming_earnings_for_ticker(
    request: Request,
    ticker: str,
    client: NasdaqClientDep,
    days: int = Query(14, ge=1, le=90, description="Days to look ahead"),
) -> dict[str, Any]:
    """Next earnings event for a specific ticker.

    **Path params:**
    - `ticker` (str): symbol (case-insensitive — uppercased server-side).

    **Query params:**
    - `days` (int, 1–90, default 14): how far ahead to look.

    **Returns:**
    - `ticker` (str): uppercase ticker.
    - `next_earnings` (`EarningsEvent` | null): the soonest event, or null
      if none in window.
    - `all_in_range` (list[`EarningsEvent`]): every event for this ticker.

    Each `EarningsEvent`: `ticker, company_name, earnings_date, time
    (pre-market|after-hours|during-market), eps_forecast, num_estimates,
    market_cap, fiscal_quarter`.

    **Example:**
    ```bash
    curl 'http://localhost:7337/api/v1/earnings/upcoming/NVDA?days=30'
    ```
    """
    events = await client.get_upcoming_earnings([ticker], days=days)
    if events:
        return {
            "ticker": ticker.upper(),
            "next_earnings": events[0].model_dump(mode="json"),
            "all_in_range": [e.model_dump(mode="json") for e in events],
        }
    return {
        "ticker": ticker.upper(),
        "next_earnings": None,
        "all_in_range": [],
    }
