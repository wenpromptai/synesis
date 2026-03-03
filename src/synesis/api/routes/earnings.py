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
    """Get all earnings reports for a specific date."""
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
    """Get upcoming earnings for watchlist tickers."""
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
    """Get next earnings date for a specific ticker."""
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
